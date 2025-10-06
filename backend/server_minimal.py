#!/usr/bin/env python3
"""
Server minimal pour tester la source du problème CPU
"""
from fastapi import FastAPI
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minimal Test Server")

@app.get("/test")
async def test_endpoint():
    return {"status": "ok", "message": "Minimal server working"}

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Minimal server starting...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
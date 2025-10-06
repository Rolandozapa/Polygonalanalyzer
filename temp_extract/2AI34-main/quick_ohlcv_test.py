#!/usr/bin/env python3
"""
Quick Enhanced OHLCV Integration Test
"""

import requests
import json

# Get backend URL
try:
    with open('/app/frontend/.env', 'r') as f:
        for line in f:
            if line.startswith('REACT_APP_BACKEND_URL='):
                backend_url = line.split('=')[1].strip()
                break
        else:
            backend_url = "http://localhost:8001"
except Exception:
    backend_url = "http://localhost:8001"

api_url = f"{backend_url}/api"

print("🚀 Quick Enhanced OHLCV Integration Test")
print("=" * 60)

# Test 1: Check opportunities endpoint (Scout system)
print("\n📊 TEST 1: Scout System (/api/opportunities)")
try:
    response = requests.get(f"{api_url}/opportunities", timeout=30)
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            opportunities = data.get('opportunities', [])
            print(f"✅ Scout system working: {len(opportunities)} opportunities found")
            
            if opportunities:
                sample = opportunities[0]
                print(f"   📈 Sample opportunity: {sample.get('symbol', 'N/A')}")
                print(f"   💰 Price: ${sample.get('current_price', 0):.6f}")
                print(f"   📊 Volume 24h: ${sample.get('volume_24h', 0):,.0f}")
                print(f"   📈 Market Cap: ${sample.get('market_cap', 0):,.0f}")
                print(f"   🎯 Volatility: {sample.get('volatility', 0):.2%}")
        else:
            print(f"❌ Scout system failed: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ Scout API failed: HTTP {response.status_code}")
except Exception as e:
    print(f"❌ Scout test error: {e}")

# Test 2: Check IA1 cycle with BTCUSDT
print("\n📊 TEST 2: IA1 Cycle OHLCV Integration (/api/run-ia1-cycle)")
try:
    response = requests.post(f"{api_url}/run-ia1-cycle", json={"symbol": "BTCUSDT"}, timeout=45)
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            analysis = data.get('analysis', {})
            print(f"✅ IA1 cycle working for BTCUSDT")
            print(f"   💰 Entry Price: ${analysis.get('entry_price', 0):.6f}")
            print(f"   💰 Current Price: ${analysis.get('current_price', 0):.6f}")
            print(f"   🎯 Confidence: {analysis.get('confidence', 0):.1%}")
            print(f"   📊 Signal: {analysis.get('recommendation', 'N/A')}")
            
            # Check technical indicators
            indicators = {
                'RSI': analysis.get('rsi_signal', 'unknown'),
                'MACD': analysis.get('macd_trend', 'unknown'),
                'Stochastic': analysis.get('stochastic_signal', 'unknown'),
                'MFI': analysis.get('mfi_signal', 'unknown'),
                'VWAP': analysis.get('vwap_signal', 'unknown')
            }
            
            meaningful_count = sum(1 for v in indicators.values() if v not in ['unknown', 'neutral', ''])
            print(f"   🔧 Technical Indicators ({meaningful_count}/5 meaningful):")
            for name, value in indicators.items():
                status = "✅" if value not in ['unknown', 'neutral', ''] else "❌"
                print(f"      {status} {name}: {value}")
                
        else:
            print(f"❌ IA1 cycle failed: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ IA1 API failed: HTTP {response.status_code}")
except Exception as e:
    print(f"❌ IA1 test error: {e}")

# Test 3: Check if Enhanced OHLCV fetcher is working by testing a direct call
print("\n📊 TEST 3: Enhanced OHLCV System Status")
try:
    # Test multiple symbols to see if we get realistic prices
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    realistic_prices = 0
    
    for symbol in test_symbols:
        try:
            response = requests.post(f"{api_url}/run-ia1-cycle", json={"symbol": symbol}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    analysis = data.get('analysis', {})
                    entry_price = analysis.get('entry_price', 0)
                    
                    # Check if price is realistic for major cryptos
                    if symbol == "BTCUSDT" and entry_price > 20000:
                        realistic_prices += 1
                        print(f"   ✅ {symbol}: Realistic price ${entry_price:,.2f}")
                    elif symbol == "ETHUSDT" and entry_price > 1000:
                        realistic_prices += 1
                        print(f"   ✅ {symbol}: Realistic price ${entry_price:,.2f}")
                    elif symbol == "SOLUSDT" and entry_price > 50:
                        realistic_prices += 1
                        print(f"   ✅ {symbol}: Realistic price ${entry_price:,.2f}")
                    else:
                        print(f"   ❌ {symbol}: Suspicious price ${entry_price:.6f}")
        except Exception as e:
            print(f"   ❌ {symbol}: Error {e}")
    
    print(f"\n🎯 Enhanced OHLCV Status: {realistic_prices}/3 symbols with realistic prices")
    
    if realistic_prices >= 2:
        print("✅ Enhanced OHLCV system appears to be working with real market data")
    elif realistic_prices >= 1:
        print("⚠️ Enhanced OHLCV system partially working - some real data detected")
    else:
        print("❌ Enhanced OHLCV system may not be providing real market data")
        
except Exception as e:
    print(f"❌ Enhanced OHLCV test error: {e}")

print("\n" + "=" * 60)
print("🏁 Quick test completed")
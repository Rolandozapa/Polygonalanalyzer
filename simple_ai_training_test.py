#!/usr/bin/env python3
"""
Simple AI Training System Test
Focus: Quick validation of AI Training System endpoints
"""

import requests
import json
import time
import sys

def test_endpoint(url, method='GET', data=None, timeout=30, description=""):
    """Test a single endpoint with proper error handling"""
    print(f"\n🔍 Testing: {description}")
    print(f"   URL: {url}")
    print(f"   Method: {method}")
    
    try:
        start_time = time.time()
        
        if method == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        
        elapsed = time.time() - start_time
        
        print(f"   Status: {response.status_code}")
        print(f"   Time: {elapsed:.1f}s")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   Response type: {type(result)}")
                
                if isinstance(result, dict):
                    print(f"   Keys: {list(result.keys())[:5]}")  # Show first 5 keys
                elif isinstance(result, list):
                    print(f"   List length: {len(result)}")
                
                return True, result
            except:
                print(f"   Response: {response.text[:200]}...")
                return True, response.text
        else:
            print(f"   Error: {response.text[:200]}")
            return False, response.text
            
    except requests.exceptions.Timeout:
        print(f"   ❌ TIMEOUT after {timeout}s")
        return False, "Timeout"
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False, str(e)

def main():
    base_url = "https://bingx-fusion.preview.emergentagent.com/api"
    
    print("🚀 AI Training System Quick Test")
    print("=" * 60)
    
    tests = [
        # AI Training endpoints
        (f"{base_url}/ai-training/status", "GET", None, 60, "AI Training Status"),
        (f"{base_url}/ai-training/run", "POST", None, 180, "AI Training Run"),
        (f"{base_url}/ai-training/results/market-conditions", "GET", None, 30, "Market Conditions Results"),
        (f"{base_url}/ai-training/results/pattern-training", "GET", None, 30, "Pattern Training Results"),
        (f"{base_url}/ai-training/results/ia1-enhancements", "GET", None, 30, "IA1 Enhancements Results"),
        (f"{base_url}/ai-training/results/ia2-enhancements", "GET", None, 30, "IA2 Enhancements Results"),
        
        # Adaptive Context endpoints
        (f"{base_url}/adaptive-context/status", "GET", None, 30, "Adaptive Context Status"),
        (f"{base_url}/adaptive-context/load-training", "POST", None, 60, "Load Training Data"),
        (f"{base_url}/adaptive-context/analyze", "POST", {
            "symbols": {
                "BTCUSDT": {
                    "price_change_24h": 3.5,
                    "volatility": 8.2,
                    "volume_ratio": 1.3,
                    "rsi": 65,
                    "macd_signal": 0.002
                }
            }
        }, 30, "Analyze Market Data"),
    ]
    
    results = []
    
    for url, method, data, timeout, description in tests:
        success, response = test_endpoint(url, method, data, timeout, description)
        results.append((description, success, response))
        
        # Add delay between tests to avoid overwhelming the system
        time.sleep(2)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for description, success, response in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {description}")
        if success:
            passed += 1
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed")
    
    # Detailed analysis
    if passed > 0:
        print("\n✅ WORKING ENDPOINTS:")
        for description, success, response in results:
            if success:
                print(f"   • {description}")
    
    if passed < total:
        print("\n❌ FAILED ENDPOINTS:")
        for description, success, response in results:
            if not success:
                print(f"   • {description}: {response}")
    
    return passed, total

if __name__ == "__main__":
    passed, total = main()
    sys.exit(0 if passed > total * 0.5 else 1)  # Pass if more than 50% work
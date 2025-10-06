#!/usr/bin/env python3
"""
Simple test script to check if DYNAMIC RR logging is working
"""

import requests
import time

def test_dynamic_rr_logging():
    """Test if the DYNAMIC RR log line appears"""
    
    api_url = "https://tradingbot-ultra.preview.emergentagent.com/api"
    
    # Test symbols
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    print("🔍 Testing Dynamic RR Integration...")
    
    for symbol in test_symbols:
        print(f"\n📞 Testing {symbol}...")
        
        try:
            response = requests.post(
                f"{api_url}/force-ia1-analysis",
                json={"symbol": symbol},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {symbol} analysis successful")
                
                # Check if new fields are present
                ia1_analysis = data.get('ia1_analysis', {})
                trade_type = ia1_analysis.get('trade_type')
                minimum_rr_threshold = ia1_analysis.get('minimum_rr_threshold')
                
                print(f"   trade_type: {trade_type}")
                print(f"   minimum_rr_threshold: {minimum_rr_threshold}")
                
                if trade_type and minimum_rr_threshold:
                    print(f"   ✅ New fields present!")
                else:
                    print(f"   ❌ New fields missing")
                    
            else:
                print(f"❌ {symbol} failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {symbol} error: {e}")
        
        # Wait between requests
        time.sleep(5)
    
    print("\n🔍 Check backend logs for 'DYNAMIC RR:' line...")

if __name__ == "__main__":
    test_dynamic_rr_logging()
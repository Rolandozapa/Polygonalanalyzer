#!/usr/bin/env python3
"""
MACD & Fibonacci Analysis - Detailed Investigation
Analyze current API data to identify specific issues with MACD and Fibonacci integration
"""

import requests
import json
from pymongo import MongoClient

def analyze_current_data():
    """Analyze current API data for MACD and Fibonacci issues"""
    
    print("🔍 MACD & FIBONACCI INTEGRATION ANALYSIS")
    print("=" * 60)
    
    # Get current analyses
    try:
        response = requests.get("https://cryptobot-pro-11.preview.emergentagent.com/api/analyses", timeout=30)
        if response.status_code == 200:
            data = response.json()
            analyses = data.get('analyses', [])
            
            if analyses:
                print(f"✅ Found {len(analyses)} analyses")
                latest = analyses[0]
                
                print(f"\n📊 LATEST ANALYSIS: {latest.get('symbol', 'N/A')}")
                print(f"   ID: {latest.get('id', 'N/A')}")
                print(f"   Timestamp: {latest.get('timestamp', 'N/A')}")
                
                # MACD Analysis
                print(f"\n🔍 MACD ANALYSIS:")
                macd_signal = latest.get('macd_signal', 'N/A')
                macd_line = latest.get('macd_line', 'N/A')
                macd_histogram = latest.get('macd_histogram', 'N/A')
                macd_trend = latest.get('macd_trend', 'N/A')
                
                print(f"   macd_signal: {macd_signal}")
                print(f"   macd_line: {macd_line}")
                print(f"   macd_histogram: {macd_histogram}")
                print(f"   macd_trend: {macd_trend}")
                
                # Check if MACD values are zeros (the issue)
                macd_issues = []
                if macd_signal == 0:
                    macd_issues.append("macd_signal is 0")
                if macd_line == 0:
                    macd_issues.append("macd_line is 0")
                if macd_histogram == 0:
                    macd_issues.append("macd_histogram is 0")
                if macd_trend in ['neutral', 'unknown']:
                    macd_issues.append(f"macd_trend is {macd_trend}")
                
                if macd_issues:
                    print(f"   ❌ MACD ISSUES FOUND: {', '.join(macd_issues)}")
                else:
                    print(f"   ✅ MACD values look good")
                
                # Fibonacci Analysis
                print(f"\n🔍 FIBONACCI ANALYSIS:")
                fib_signal_strength = latest.get('fibonacci_signal_strength', 'N/A')
                fib_signal_direction = latest.get('fibonacci_signal_direction', 'N/A')
                fib_key_level_proximity = latest.get('fibonacci_key_level_proximity', 'N/A')
                fib_level = latest.get('fibonacci_level', 'N/A')
                fib_nearest_level = latest.get('fibonacci_nearest_level', 'N/A')
                fib_trend_direction = latest.get('fibonacci_trend_direction', 'N/A')
                
                print(f"   fibonacci_signal_strength: {fib_signal_strength}")
                print(f"   fibonacci_signal_direction: {fib_signal_direction}")
                print(f"   fibonacci_key_level_proximity: {fib_key_level_proximity}")
                print(f"   fibonacci_level: {fib_level}")
                print(f"   fibonacci_nearest_level: {fib_nearest_level}")
                print(f"   fibonacci_trend_direction: {fib_trend_direction}")
                
                # Check Fibonacci integration
                fibonacci_issues = []
                if fib_signal_strength == 0:
                    fibonacci_issues.append("fibonacci_signal_strength is 0")
                if fib_signal_direction == 'neutral':
                    fibonacci_issues.append("fibonacci_signal_direction is neutral")
                if fib_key_level_proximity == False:
                    fibonacci_issues.append("fibonacci_key_level_proximity is False")
                
                if fibonacci_issues:
                    print(f"   ⚠️ FIBONACCI OBSERVATIONS: {', '.join(fibonacci_issues)}")
                else:
                    print(f"   ✅ Fibonacci values look good")
                
                # Support/Resistance Analysis
                support_levels = latest.get('support_levels', [])
                resistance_levels = latest.get('resistance_levels', [])
                print(f"\n📈 SUPPORT/RESISTANCE:")
                print(f"   Support levels: {support_levels}")
                print(f"   Resistance levels: {resistance_levels}")
                
                # Overall Assessment
                print(f"\n🎯 OVERALL ASSESSMENT:")
                
                # MACD Fix Status
                if all(val == 0 for val in [macd_signal, macd_line, macd_histogram]):
                    print(f"   ❌ MACD CALCULATION FIX: NOT WORKING - All MACD values are 0")
                    print(f"      Issue: The fix to use real MACD values instead of defaults is not working")
                else:
                    print(f"   ✅ MACD CALCULATION FIX: WORKING - Real MACD values found")
                
                # Fibonacci Integration Status
                if fib_signal_strength is not None and fib_signal_direction is not None:
                    print(f"   ✅ FIBONACCI INTEGRATION: WORKING - Fibonacci fields are present")
                    if fib_signal_strength > 0 or fib_signal_direction != 'neutral':
                        print(f"      Status: Fibonacci analysis is providing meaningful signals")
                    else:
                        print(f"      Status: Fibonacci analysis is working but showing neutral/low signals")
                else:
                    print(f"   ❌ FIBONACCI INTEGRATION: NOT WORKING - Missing Fibonacci fields")
                
                # Check multiple analyses for consistency
                if len(analyses) > 1:
                    print(f"\n🔍 CONSISTENCY CHECK (checking {min(5, len(analyses))} analyses):")
                    macd_zero_count = 0
                    fibonacci_present_count = 0
                    
                    for i, analysis in enumerate(analyses[:5]):
                        macd_vals = [analysis.get('macd_signal', 0), analysis.get('macd_line', 0), analysis.get('macd_histogram', 0)]
                        if all(val == 0 for val in macd_vals):
                            macd_zero_count += 1
                        
                        if analysis.get('fibonacci_signal_strength') is not None:
                            fibonacci_present_count += 1
                    
                    print(f"   MACD zeros in {macd_zero_count}/{min(5, len(analyses))} analyses")
                    print(f"   Fibonacci present in {fibonacci_present_count}/{min(5, len(analyses))} analyses")
                    
                    if macd_zero_count == min(5, len(analyses)):
                        print(f"   ❌ CRITICAL: All recent analyses have MACD values of 0")
                    elif macd_zero_count > 0:
                        print(f"   ⚠️ WARNING: Some analyses have MACD values of 0")
                    else:
                        print(f"   ✅ GOOD: No analyses with all-zero MACD values")
                
            else:
                print("❌ No analyses found in API response")
        else:
            print(f"❌ API request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error analyzing API data: {e}")
    
    # Database Analysis
    print(f"\n🗄️ DATABASE ANALYSIS:")
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["myapp"]
        
        # Get latest analysis from database
        latest_db = db.technical_analyses.find_one({}, sort=[("timestamp", -1)])
        
        if latest_db:
            print(f"   ✅ Latest DB analysis: {latest_db.get('symbol', 'N/A')}")
            
            # Check MACD in database
            db_macd_signal = latest_db.get('macd_signal', 'N/A')
            db_macd_line = latest_db.get('macd_line', 'N/A')
            db_macd_histogram = latest_db.get('macd_histogram', 'N/A')
            
            print(f"   DB MACD: signal={db_macd_signal}, line={db_macd_line}, histogram={db_macd_histogram}")
            
            # Check Fibonacci in database
            db_fib_strength = latest_db.get('fibonacci_signal_strength', 'N/A')
            db_fib_direction = latest_db.get('fibonacci_signal_direction', 'N/A')
            
            print(f"   DB Fibonacci: strength={db_fib_strength}, direction={db_fib_direction}")
            
        else:
            print("   ❌ No analyses found in database")
            
    except Exception as e:
        print(f"   ❌ Database analysis error: {e}")

if __name__ == "__main__":
    analyze_current_data()
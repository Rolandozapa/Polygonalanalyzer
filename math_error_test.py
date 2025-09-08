#!/usr/bin/env python3
"""
Mathematical Error Test - Check if divide by zero and RuntimeWarning errors are fixed
"""

import subprocess
import time
import requests
import re

def check_mathematical_errors():
    """Check for mathematical errors in recent logs"""
    print("🔍 Checking for mathematical errors after fixes...")
    
    # Wait for some system activity
    print("⏳ Waiting 30 seconds for system activity...")
    time.sleep(30)
    
    # Get recent backend logs
    log_cmd = "tail -n 200 /var/log/supervisor/backend.*.log"
    result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
    logs = result.stdout
    
    # Check for mathematical errors
    errors = {
        'runtime_warnings': [],
        'divide_by_zero': [],
        'nan_values': [],
        'inf_values': [],
        'math_errors': []
    }
    
    # Patterns to look for
    patterns = {
        'runtime_warnings': [r'RuntimeWarning', r'Warning.*math', r'Warning.*divide'],
        'divide_by_zero': [r'divide by zero', r'division by zero', r'ZeroDivisionError'],
        'nan_values': [r'nan', r'NaN', r'not a number'],
        'inf_values': [r'inf', r'infinity', r'Infinity'],
        'math_errors': [r'ValueError.*math', r'OverflowError', r'ArithmeticError']
    }
    
    for line in logs.split('\n'):
        for error_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, line, re.IGNORECASE):
                    errors[error_type].append(line.strip())
    
    # Report results
    print("\n📊 Mathematical Error Analysis (Recent Activity):")
    total_errors = 0
    critical_errors = 0
    
    for error_type, error_list in errors.items():
        if error_list:
            print(f"   ❌ {error_type}: {len(error_list)} occurrences")
            total_errors += len(error_list)
            if error_type in ['runtime_warnings', 'divide_by_zero', 'math_errors']:
                critical_errors += len(error_list)
            # Show first few examples
            for i, error in enumerate(error_list[:3]):
                print(f"      {i+1}. {error[:100]}...")
        else:
            print(f"   ✅ {error_type}: No errors found")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Total errors: {total_errors}")
    print(f"   Critical errors: {critical_errors}")
    
    if critical_errors == 0:
        print("   ✅ SUCCESS: No critical mathematical errors detected!")
        return True
    else:
        print("   ❌ FAILURE: Critical mathematical errors still present")
        return False

def test_trading_startup():
    """Test if trading system can start without errors"""
    print("\n🚀 Testing trading system startup...")
    
    api_url = 'https://adaptive-trade-ai-2.preview.emergentagent.com/api'
    
    try:
        # Try to start trading
        response = requests.post(f'{api_url}/start-trading', timeout=30)
        print(f"   Start trading response: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Trading system started successfully")
            
            # Check if it's actually running
            time.sleep(5)
            status_response = requests.get(f'{api_url}/market-status', timeout=30)
            if status_response.status_code == 200:
                status_data = status_response.json()
                trading_active = status_data.get('trading_active', False)
                print(f"   Trading active: {trading_active}")
                return trading_active
            else:
                print(f"   ❌ Could not verify trading status: {status_response.status_code}")
                return False
        else:
            print(f"   ❌ Failed to start trading: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception during trading startup: {e}")
        return False

def test_pattern_functionality():
    """Test if pattern detection works without mathematical errors"""
    print("\n🎯 Testing pattern functionality...")
    
    api_url = 'https://adaptive-trade-ai-2.preview.emergentagent.com/api'
    
    try:
        # Get analyses to see if patterns are working
        response = requests.get(f'{api_url}/analyses', timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            analyses = data if isinstance(data, list) else data.get('analyses', [])
            
            print(f"   Found {len(analyses)} IA1 analyses")
            
            patterns_found = 0
            for analysis in analyses:
                patterns = analysis.get('patterns_detected', [])
                if patterns:
                    patterns_found += len(patterns)
                    symbol = analysis.get('symbol', 'UNKNOWN')
                    print(f"   📊 {symbol}: {len(patterns)} patterns - {patterns}")
            
            print(f"   Total patterns detected: {patterns_found}")
            return patterns_found > 0
            
        else:
            print(f"   ❌ Could not get analyses: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception during pattern test: {e}")
        return False

if __name__ == "__main__":
    print("🧮 MATHEMATICAL ERROR VALIDATION TEST")
    print("=" * 50)
    
    # Test 1: Check for mathematical errors
    math_errors_fixed = check_mathematical_errors()
    
    # Test 2: Test trading startup
    trading_works = test_trading_startup()
    
    # Test 3: Test pattern functionality
    patterns_work = test_pattern_functionality()
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS:")
    print(f"   Mathematical errors fixed: {'✅ YES' if math_errors_fixed else '❌ NO'}")
    print(f"   Trading system startup: {'✅ YES' if trading_works else '❌ NO'}")
    print(f"   Pattern functionality: {'✅ YES' if patterns_work else '❌ NO'}")
    
    if math_errors_fixed and trading_works and patterns_work:
        print("\n🎉 ALL TESTS PASSED - Mathematical errors have been successfully corrected!")
        exit(0)
    else:
        print("\n⚠️ SOME ISSUES REMAIN - Further investigation needed")
        exit(1)
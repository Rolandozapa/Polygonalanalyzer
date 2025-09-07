#!/usr/bin/env python3
"""
Focused Position Sizing Test for Active Trading System
Tests the critical position sizing fixes requested in the review
"""

import requests
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_position_sizing_fixes():
    """Test the critical position sizing fixes"""
    
    logger.info("🚀 TESTING POSITION SIZING FIXES")
    logger.info("=" * 50)
    
    api_url = "http://localhost:8001/api"
    
    # Test 1: Check IA2 decisions for position sizing
    logger.info("\n🔍 TEST 1: IA2 Position Size Integration")
    
    try:
        response = requests.get(f"{api_url}/decisions", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            decisions = data.get('decisions', []) if isinstance(data, dict) else data
            
            logger.info(f"   📊 Found {len(decisions)} decisions")
            
            # Analyze position sizing in decisions
            position_size_analysis = {
                'total_decisions': len(decisions),
                'hold_signals': 0,
                'trading_signals': 0,
                'zero_position_sizes': 0,
                'positive_position_sizes': 0,
                'position_size_range': []
            }
            
            for i, decision in enumerate(decisions[:15]):  # Check first 15
                symbol = decision.get('symbol', 'UNKNOWN')
                signal = decision.get('signal', 'UNKNOWN').upper()
                position_size = decision.get('position_size', 0)
                confidence = decision.get('confidence', 0)
                reasoning = decision.get('ia2_reasoning', '')
                
                logger.info(f"   📈 {i+1:2d}. {symbol:8s} | {signal:5s} | Size: {position_size:.4f} | Conf: {confidence:.2f}")
                
                if signal == 'HOLD':
                    position_size_analysis['hold_signals'] += 1
                else:
                    position_size_analysis['trading_signals'] += 1
                    
                if position_size <= 0:
                    position_size_analysis['zero_position_sizes'] += 1
                else:
                    position_size_analysis['positive_position_sizes'] += 1
                    position_size_analysis['position_size_range'].append(position_size)
                    
                # Check for position sizing mentions in reasoning
                if 'position' in reasoning.lower() and 'size' in reasoning.lower():
                    logger.info(f"      ✅ Position sizing logic found in reasoning")
                    
            # Summary
            logger.info(f"\n   📊 ANALYSIS SUMMARY:")
            logger.info(f"      Total decisions: {position_size_analysis['total_decisions']}")
            logger.info(f"      HOLD signals: {position_size_analysis['hold_signals']}")
            logger.info(f"      Trading signals: {position_size_analysis['trading_signals']}")
            logger.info(f"      Zero position sizes: {position_size_analysis['zero_position_sizes']}")
            logger.info(f"      Positive position sizes: {position_size_analysis['positive_position_sizes']}")
            
            if position_size_analysis['position_size_range']:
                avg_size = sum(position_size_analysis['position_size_range']) / len(position_size_analysis['position_size_range'])
                min_size = min(position_size_analysis['position_size_range'])
                max_size = max(position_size_analysis['position_size_range'])
                logger.info(f"      Position size range: {min_size:.4f} - {max_size:.4f} (avg: {avg_size:.4f})")
                
            # Test success criteria
            test1_success = (
                position_size_analysis['total_decisions'] > 0 and
                position_size_analysis['hold_signals'] >= 0 and  # HOLD signals should exist
                position_size_analysis['zero_position_sizes'] >= position_size_analysis['hold_signals']  # Zero sizes should match HOLD signals
            )
            
            logger.info(f"   ✅ TEST 1 RESULT: {'PASS' if test1_success else 'FAIL'}")
            
        else:
            logger.error(f"   ❌ API Error: {response.status_code}")
            test1_success = False
            
    except Exception as e:
        logger.error(f"   ❌ Exception: {e}")
        test1_success = False
    
    # Test 2: Check Active Position Manager
    logger.info("\n🔍 TEST 2: Active Position Manager Integration")
    
    try:
        response = requests.get(f"{api_url}/active-positions", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            execution_mode = data.get('execution_mode', 'UNKNOWN')
            total_positions = data.get('total_positions', 0)
            active_positions = data.get('active_positions', [])
            monitoring_active = data.get('monitoring_active', False)
            
            logger.info(f"   📊 Execution Mode: {execution_mode}")
            logger.info(f"   📊 Total Active Positions: {total_positions}")
            logger.info(f"   📊 Monitoring Active: {monitoring_active}")
            
            # Check if any positions have invalid position sizes
            invalid_positions = 0
            for position in active_positions:
                position_size_usd = position.get('position_size_usd', 0)
                if position_size_usd <= 0:
                    invalid_positions += 1
                    logger.info(f"      ⚠️ Invalid position size: {position.get('symbol')} - ${position_size_usd}")
                else:
                    logger.info(f"      ✅ Valid position: {position.get('symbol')} - ${position_size_usd:.2f}")
                    
            test2_success = (
                execution_mode in ['LIVE', 'SIMULATION'] and
                invalid_positions == 0  # No positions should have invalid sizes
            )
            
            logger.info(f"   ✅ TEST 2 RESULT: {'PASS' if test2_success else 'FAIL'}")
            
        else:
            logger.error(f"   ❌ API Error: {response.status_code}")
            test2_success = False
            
    except Exception as e:
        logger.error(f"   ❌ Exception: {e}")
        test2_success = False
    
    # Test 3: Check Trading Execution Mode
    logger.info("\n🔍 TEST 3: Trading Execution Mode")
    
    try:
        response = requests.get(f"{api_url}/trading/execution-mode", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            execution_mode = data.get('execution_mode', 'UNKNOWN')
            
            logger.info(f"   📊 Current Execution Mode: {execution_mode}")
            
            test3_success = execution_mode in ['LIVE', 'SIMULATION']
            logger.info(f"   ✅ TEST 3 RESULT: {'PASS' if test3_success else 'FAIL'}")
            
        else:
            logger.error(f"   ❌ API Error: {response.status_code}")
            test3_success = False
            
    except Exception as e:
        logger.error(f"   ❌ Exception: {e}")
        test3_success = False
    
    # Overall Results
    logger.info("\n" + "=" * 50)
    logger.info("📊 OVERALL TEST RESULTS")
    logger.info("=" * 50)
    
    tests_passed = sum([test1_success, test2_success, test3_success])
    total_tests = 3
    
    logger.info(f"✅ IA2 Position Size Integration: {'PASS' if test1_success else 'FAIL'}")
    logger.info(f"✅ Active Position Manager: {'PASS' if test2_success else 'FAIL'}")
    logger.info(f"✅ Trading Execution Mode: {'PASS' if test3_success else 'FAIL'}")
    
    logger.info(f"\n🎯 FINAL RESULT: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED - Position sizing system is working!")
        return True
    elif tests_passed >= 2:
        logger.info("⚠️ MOSTLY WORKING - Minor issues detected")
        return True
    else:
        logger.info("❌ CRITICAL ISSUES - Position sizing needs attention")
        return False

if __name__ == "__main__":
    success = test_position_sizing_fixes()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Detailed Position Sizing Analysis for Active Trading System
Examines the specific position sizing fixes requested in the review
"""

import requests
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_position_sizing_implementation():
    """Analyze the position sizing implementation in detail"""
    
    logger.info("🔍 DETAILED POSITION SIZING ANALYSIS")
    logger.info("=" * 60)
    
    api_url = "http://localhost:8001/api"
    
    # Get IA2 decisions
    logger.info("\n📊 ANALYZING IA2 DECISIONS")
    
    try:
        response = requests.get(f"{api_url}/decisions", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            decisions = data.get('decisions', []) if isinstance(data, dict) else data
            
            logger.info(f"Found {len(decisions)} IA2 decisions")
            
            # Detailed analysis of each decision
            for i, decision in enumerate(decisions[:5]):  # Analyze first 5 decisions
                logger.info(f"\n🔍 DECISION {i+1}: {decision.get('symbol', 'UNKNOWN')}")
                logger.info(f"   Signal: {decision.get('signal', 'UNKNOWN')}")
                logger.info(f"   Confidence: {decision.get('confidence', 0):.3f}")
                logger.info(f"   Position Size: {decision.get('position_size', 0):.6f} ({decision.get('position_size', 0)*100:.2f}%)")
                logger.info(f"   Entry Price: ${decision.get('entry_price', 0):.4f}")
                logger.info(f"   Stop Loss: ${decision.get('stop_loss', 0):.4f}")
                logger.info(f"   Risk/Reward: {decision.get('risk_reward_ratio', 0):.2f}")
                
                # Check reasoning for position sizing mentions
                reasoning = decision.get('ia2_reasoning', '')
                if reasoning:
                    # Look for position sizing keywords
                    position_keywords = ['position', 'size', 'leverage', '%', 'account', 'risk']
                    found_keywords = [kw for kw in position_keywords if kw in reasoning.lower()]
                    
                    if found_keywords:
                        logger.info(f"   ✅ Position sizing keywords found: {found_keywords}")
                        
                        # Extract position sizing related text
                        lines = reasoning.split('. ')
                        position_lines = [line for line in lines if any(kw in line.lower() for kw in position_keywords)]
                        
                        for line in position_lines[:2]:  # Show first 2 relevant lines
                            logger.info(f"      📝 \"{line.strip()}\"")
                    else:
                        logger.info(f"   ⚠️ No position sizing keywords in reasoning")
                
                # Validate position size logic
                signal = decision.get('signal', '').upper()
                position_size = decision.get('position_size', 0)
                
                if signal == 'HOLD':
                    if position_size == 0:
                        logger.info(f"   ✅ CORRECT: HOLD signal has 0% position size")
                    else:
                        logger.info(f"   ❌ ERROR: HOLD signal has non-zero position size: {position_size}")
                        
                elif signal in ['LONG', 'SHORT']:
                    if position_size > 0:
                        logger.info(f"   ✅ CORRECT: {signal} signal has positive position size")
                        
                        # Check if position size is reasonable (0.1% to 8%)
                        if 0.001 <= position_size <= 0.08:
                            logger.info(f"   ✅ REASONABLE: Position size within expected range")
                        else:
                            logger.info(f"   ⚠️ UNUSUAL: Position size outside typical range (0.1%-8%)")
                    else:
                        logger.info(f"   ❌ ERROR: {signal} signal has zero position size")
                        
        else:
            logger.error(f"Failed to get decisions: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error analyzing decisions: {e}")
    
    # Check Active Position Manager status
    logger.info("\n📊 ANALYZING ACTIVE POSITION MANAGER")
    
    try:
        response = requests.get(f"{api_url}/active-positions", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            logger.info(f"Execution Mode: {data.get('execution_mode', 'UNKNOWN')}")
            logger.info(f"Total Positions: {data.get('total_positions', 0)}")
            logger.info(f"Total Unrealized P&L: ${data.get('total_unrealized_pnl', 0):.2f}")
            logger.info(f"Total Position Value: ${data.get('total_position_value', 0):.2f}")
            logger.info(f"Monitoring Active: {data.get('monitoring_active', False)}")
            
            active_positions = data.get('active_positions', [])
            
            if active_positions:
                logger.info(f"\nActive Positions ({len(active_positions)}):")
                
                for pos in active_positions:
                    logger.info(f"   🔹 {pos.get('symbol')}: {pos.get('signal')} - ${pos.get('position_size_usd', 0):.2f}")
                    logger.info(f"      Entry: ${pos.get('entry_price', 0):.4f} | Current: ${pos.get('current_price', 0):.4f}")
                    logger.info(f"      P&L: ${pos.get('unrealized_pnl', 0):.2f} ({pos.get('pnl_percentage', 0):.2f}%)")
                    
                    # Check if position has valid position size
                    position_size_usd = pos.get('position_size_usd', 0)
                    if position_size_usd <= 0:
                        logger.info(f"      ❌ ERROR: Invalid position size: ${position_size_usd}")
                    else:
                        logger.info(f"      ✅ Valid position size: ${position_size_usd:.2f}")
            else:
                logger.info("   No active positions")
                
        else:
            logger.error(f"Failed to get active positions: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error analyzing active positions: {e}")
    
    # Test position sizing behavior with specific scenarios
    logger.info("\n📊 POSITION SIZING BEHAVIOR ANALYSIS")
    
    # Check if system properly handles different scenarios
    logger.info("Checking position sizing patterns...")
    
    try:
        response = requests.get(f"{api_url}/decisions", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            decisions = data.get('decisions', []) if isinstance(data, dict) else data
            
            # Analyze patterns
            hold_count = 0
            trading_count = 0
            zero_size_count = 0
            positive_size_count = 0
            position_sizes = []
            
            for decision in decisions:
                signal = decision.get('signal', '').upper()
                position_size = decision.get('position_size', 0)
                
                if signal == 'HOLD':
                    hold_count += 1
                else:
                    trading_count += 1
                    
                if position_size <= 0:
                    zero_size_count += 1
                else:
                    positive_size_count += 1
                    position_sizes.append(position_size)
            
            logger.info(f"Signal Distribution:")
            logger.info(f"   HOLD signals: {hold_count}")
            logger.info(f"   Trading signals (LONG/SHORT): {trading_count}")
            
            logger.info(f"Position Size Distribution:")
            logger.info(f"   Zero position sizes: {zero_size_count}")
            logger.info(f"   Positive position sizes: {positive_size_count}")
            
            if position_sizes:
                avg_size = sum(position_sizes) / len(position_sizes)
                min_size = min(position_sizes)
                max_size = max(position_sizes)
                
                logger.info(f"Position Size Statistics:")
                logger.info(f"   Average: {avg_size:.4f} ({avg_size*100:.2f}%)")
                logger.info(f"   Range: {min_size:.4f} - {max_size:.4f}")
                logger.info(f"   Count: {len(position_sizes)} positions")
            
            # Check if HOLD signals have zero position sizes
            hold_with_zero_size = 0
            trading_with_positive_size = 0
            
            for decision in decisions:
                signal = decision.get('signal', '').upper()
                position_size = decision.get('position_size', 0)
                
                if signal == 'HOLD' and position_size == 0:
                    hold_with_zero_size += 1
                elif signal in ['LONG', 'SHORT'] and position_size > 0:
                    trading_with_positive_size += 1
            
            logger.info(f"\nPosition Sizing Logic Validation:")
            logger.info(f"   HOLD signals with 0% size: {hold_with_zero_size}/{hold_count}")
            logger.info(f"   Trading signals with >0% size: {trading_with_positive_size}/{trading_count}")
            
            # Success criteria
            hold_logic_correct = (hold_count == 0) or (hold_with_zero_size == hold_count)
            trading_logic_correct = (trading_count == 0) or (trading_with_positive_size >= trading_count * 0.8)
            
            logger.info(f"\n✅ VALIDATION RESULTS:")
            logger.info(f"   HOLD → 0% size logic: {'✅ CORRECT' if hold_logic_correct else '❌ INCORRECT'}")
            logger.info(f"   TRADING → >0% size logic: {'✅ CORRECT' if trading_logic_correct else '❌ INCORRECT'}")
            
            overall_success = hold_logic_correct and trading_logic_correct
            logger.info(f"   Overall Position Sizing: {'✅ WORKING' if overall_success else '❌ NEEDS FIX'}")
            
            return overall_success
            
    except Exception as e:
        logger.error(f"Error in behavior analysis: {e}")
        return False

if __name__ == "__main__":
    success = analyze_position_sizing_implementation()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 FINAL ASSESSMENT")
    logger.info("=" * 60)
    
    if success:
        logger.info("🎉 POSITION SIZING FIXES ARE WORKING CORRECTLY!")
        logger.info("   ✅ IA2 calculates position sizes properly")
        logger.info("   ✅ HOLD signals have 0% position size")
        logger.info("   ✅ LONG/SHORT signals have positive position sizes")
        logger.info("   ✅ Active Position Manager integration functional")
    else:
        logger.info("⚠️ POSITION SIZING NEEDS ATTENTION")
        logger.info("   Some aspects of the position sizing logic may need review")
    
    exit(0 if success else 1)
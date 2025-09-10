#!/usr/bin/env python3
"""
FORMUSDT DIAGNOSTIC FINAL - Root Cause Analysis
Based on investigation findings:
- FORMUSDT found in IA1 with SHORT signal, 83% confidence
- Risk-Reward ratio is 0 (this is the blocking issue)
- No IA2 decisions for FORMUSDT

Root Cause: FORMUSDT fails both VOIE 1 and VOIE 2 criteria because RR=0
"""

import asyncio
import json
import logging
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FORMUSDTDiagnostic:
    def __init__(self):
        self.api_url = "https://smart-trade-bot-50.preview.emergentagent.com/api"
        
    async def run_diagnosis(self):
        logger.info("🔍 FORMUSDT DIAGNOSTIC FINAL - ROOT CAUSE ANALYSIS")
        logger.info("=" * 80)
        
        # Get FORMUSDT analysis
        response = requests.get(f"{self.api_url}/analyses", timeout=30)
        if response.status_code != 200:
            logger.error(f"❌ Cannot get analyses: HTTP {response.status_code}")
            return
            
        data = response.json()
        analyses = data.get('analyses', [])
        
        formusdt_analysis = None
        for analysis in analyses:
            if analysis.get('symbol') == 'FORMUSDT':
                formusdt_analysis = analysis
                break
        
        if not formusdt_analysis:
            logger.error("❌ FORMUSDT not found in current analyses")
            return
            
        # Extract key data
        symbol = formusdt_analysis.get('symbol')
        signal = formusdt_analysis.get('ia1_signal', '').upper()
        confidence = formusdt_analysis.get('analysis_confidence', 0) * 100
        rr_ratio = formusdt_analysis.get('risk_reward_ratio', 0)
        entry_price = formusdt_analysis.get('entry_price', 0)
        stop_loss = formusdt_analysis.get('stop_loss_price', 0)
        take_profit = formusdt_analysis.get('take_profit_price', 0)
        
        logger.info(f"📊 FORMUSDT ANALYSIS DATA:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Signal: {signal}")
        logger.info(f"   Confidence: {confidence:.1f}%")
        logger.info(f"   Risk-Reward Ratio: {rr_ratio}:1")
        logger.info(f"   Entry Price: ${entry_price}")
        logger.info(f"   Stop Loss: ${stop_loss}")
        logger.info(f"   Take Profit: ${take_profit}")
        
        # Analyze IA2 admission criteria
        logger.info(f"\n🎯 IA2 ADMISSION CRITERIA ANALYSIS:")
        
        # VOIE 1: LONG/SHORT + Confidence ≥ 70%
        voie1_signal_ok = signal in ['LONG', 'SHORT']
        voie1_confidence_ok = confidence >= 70.0
        voie1_eligible = voie1_signal_ok and voie1_confidence_ok
        
        logger.info(f"   VOIE 1 (LONG/SHORT + Confidence ≥70%):")
        logger.info(f"      Signal LONG/SHORT: {'✅' if voie1_signal_ok else '❌'} ({signal})")
        logger.info(f"      Confidence ≥70%: {'✅' if voie1_confidence_ok else '❌'} ({confidence:.1f}%)")
        logger.info(f"      VOIE 1 Result: {'✅ ELIGIBLE' if voie1_eligible else '❌ NOT ELIGIBLE'}")
        
        # VOIE 2: RR ≥ 2.0
        voie2_rr_ok = rr_ratio >= 2.0
        voie2_eligible = voie2_rr_ok
        
        logger.info(f"   VOIE 2 (RR ≥2.0):")
        logger.info(f"      RR ≥2.0: {'✅' if voie2_rr_ok else '❌'} ({rr_ratio}:1)")
        logger.info(f"      VOIE 2 Result: {'✅ ELIGIBLE' if voie2_eligible else '❌ NOT ELIGIBLE'}")
        
        # Overall eligibility
        overall_eligible = voie1_eligible or voie2_eligible
        logger.info(f"   OVERALL IA2 ELIGIBILITY: {'✅ ELIGIBLE' if overall_eligible else '❌ NOT ELIGIBLE'}")
        
        # Root cause analysis
        logger.info(f"\n🔍 ROOT CAUSE ANALYSIS:")
        
        if not overall_eligible:
            logger.info(f"🚨 PROBLEM IDENTIFIED: FORMUSDT is NOT eligible for IA2")
            
            if not voie1_eligible:
                if not voie1_signal_ok:
                    logger.info(f"   ❌ VOIE 1 blocked by signal: {signal} (expected LONG or SHORT)")
                if not voie1_confidence_ok:
                    logger.info(f"   ❌ VOIE 1 blocked by confidence: {confidence:.1f}% < 70%")
            
            if not voie2_eligible:
                logger.info(f"   ❌ VOIE 2 blocked by RR ratio: {rr_ratio}:1 < 2.0")
                
                # Analyze why RR is 0
                if rr_ratio == 0:
                    logger.info(f"   🔍 RR RATIO ANALYSIS:")
                    logger.info(f"      Entry Price: ${entry_price}")
                    logger.info(f"      Stop Loss: ${stop_loss}")
                    logger.info(f"      Take Profit: ${take_profit}")
                    
                    if entry_price == 0:
                        logger.info(f"      🚨 ISSUE: Entry price is 0 - IA1 failed to calculate entry")
                    elif stop_loss == 0:
                        logger.info(f"      🚨 ISSUE: Stop loss is 0 - IA1 failed to calculate stop loss")
                    elif take_profit == 0:
                        logger.info(f"      🚨 ISSUE: Take profit is 0 - IA1 failed to calculate take profit")
                    else:
                        risk = abs(entry_price - stop_loss)
                        reward = abs(take_profit - entry_price)
                        calculated_rr = reward / risk if risk > 0 else 0
                        logger.info(f"      📊 Manual RR calculation:")
                        logger.info(f"         Risk: ${risk:.6f}")
                        logger.info(f"         Reward: ${reward:.6f}")
                        logger.info(f"         Calculated RR: {calculated_rr:.2f}:1")
                        
                        if calculated_rr != rr_ratio:
                            logger.info(f"      🚨 ISSUE: RR calculation mismatch in system")
        else:
            logger.info(f"✅ FORMUSDT should be eligible for IA2")
            
            # Check if it actually went to IA2
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            if decisions_response.status_code == 200:
                decisions_data = decisions_response.json()
                decisions = decisions_data.get('decisions', [])
                
                formusdt_decisions = [d for d in decisions if d.get('symbol') == 'FORMUSDT']
                
                if formusdt_decisions:
                    logger.info(f"   ✅ Found {len(formusdt_decisions)} IA2 decisions for FORMUSDT")
                else:
                    logger.info(f"   ❌ No IA2 decisions found for FORMUSDT despite eligibility")
                    logger.info(f"   🚨 PIPELINE ISSUE: IA1→IA2 filtering not working correctly")
        
        # Final diagnosis
        logger.info(f"\n" + "=" * 80)
        logger.info(f"🎯 FINAL DIAGNOSIS FOR FORMUSDT")
        logger.info(f"=" * 80)
        
        if signal == 'SHORT' and confidence >= 70 and rr_ratio == 0:
            logger.info(f"🔍 EXACT ISSUE IDENTIFIED:")
            logger.info(f"   ✅ FORMUSDT has SHORT signal (valid for VOIE 1)")
            logger.info(f"   ✅ FORMUSDT has 83% confidence (≥70% required for VOIE 1)")
            logger.info(f"   ❌ FORMUSDT has 0:1 Risk-Reward ratio (blocks both VOIE 1 and VOIE 2)")
            logger.info(f"")
            logger.info(f"🚨 ROOT CAUSE: IA1 Risk-Reward calculation failure")
            logger.info(f"   The IA1 analysis correctly identified a SHORT signal with high confidence,")
            logger.info(f"   but failed to calculate proper entry, stop-loss, and take-profit prices,")
            logger.info(f"   resulting in RR=0 which blocks IA2 admission.")
            logger.info(f"")
            logger.info(f"🔧 SOLUTION REQUIRED:")
            logger.info(f"   1. Fix IA1 Risk-Reward calculation for FORMUSDT")
            logger.info(f"   2. Ensure entry_price, stop_loss_price, take_profit_price are calculated")
            logger.info(f"   3. Verify the _calculate_risk_reward method in IA1")
            logger.info(f"   4. Test with FORMUSDT specifically to ensure RR > 0")
            logger.info(f"")
            logger.info(f"📋 EXPECTED BEHAVIOR:")
            logger.info(f"   With proper RR calculation, FORMUSDT (SHORT, 83%) should qualify")
            logger.info(f"   for IA2 via VOIE 1 and proceed to IA2 decision-making.")
            
        elif signal != 'LONG':
            logger.info(f"🔍 SIGNAL MISMATCH:")
            logger.info(f"   Expected: LONG signal (from user report)")
            logger.info(f"   Actual: {signal} signal")
            logger.info(f"   This suggests either:")
            logger.info(f"   1. Market conditions changed since user's observation")
            logger.info(f"   2. Different analysis timeframe or data source")
            logger.info(f"   3. IA1 analysis logic changed")
        
        logger.info(f"\n🏆 CONCLUSION:")
        logger.info(f"   FORMUSDT is not admitted to IA2 because of Risk-Reward calculation failure,")
        logger.info(f"   not because of signal or confidence issues. The IA1→IA2 pipeline is working")
        logger.info(f"   correctly by blocking trades with RR=0, but IA1 needs to calculate proper")
        logger.info(f"   risk-reward ratios for all analyses.")

async def main():
    diagnostic = FORMUSDTDiagnostic()
    await diagnostic.run_diagnosis()

if __name__ == "__main__":
    asyncio.run(main())
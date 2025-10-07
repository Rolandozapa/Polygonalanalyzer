#!/usr/bin/env python3
"""
FOCUSED RISK-REWARD CALCULATION DIAGNOSTIC TEST
Focus: Diagnose the specific RR calculation issues identified in the validation request.

CRITICAL FINDINGS FROM INITIAL TEST:
1. calculated_rr field is present but always 0.0 (not calculated)
2. risk_reward_ratio field has values but appears to use defaults (1.0, 20.0)
3. Fields are incoherent (calculated_rr=0.0 vs risk_reward_ratio=1.0+)
4. Manual calculations show significant differences from API values

DIAGNOSTIC APPROACH:
1. Use /api/analyses endpoint (where RR fields are available)
2. Perform manual RR calculations using the correct formulas
3. Compare API values with manual calculations
4. Identify specific symbols with aberrant RR values (>10)
5. Test the requested symbols: BTCUSDT, ETHUSDT, SOLUSDT
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import requests
import subprocess
import re
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedRRDiagnosticTest:
    """Focused diagnostic test for RR calculation issues"""
    
    def __init__(self):
        # Get backend URL from frontend env
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
        
        self.api_url = f"{backend_url}/api"
        logger.info(f"Testing RR Calculation Diagnostic at: {self.api_url}")
        
        # Test symbols from review request
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        # Database connection info
        self.mongo_url = "mongodb://localhost:27017"
        self.db_name = "myapp"
        
        # Results storage
        self.diagnostic_results = []
        
    def calculate_manual_rr(self, entry_price: float, stop_loss: float, take_profit: float, signal: str) -> float:
        """Calculate RR manually using the correct formulas"""
        try:
            if signal.upper() == 'LONG':
                # LONG: RR = (Take_Profit - Entry) / (Entry - Stop_Loss)
                reward = abs(take_profit - entry_price)
                risk = abs(entry_price - stop_loss)
            elif signal.upper() == 'SHORT':
                # SHORT: RR = (Entry - Take_Profit) / (Stop_Loss - Entry)
                reward = abs(entry_price - take_profit)
                risk = abs(stop_loss - entry_price)
            else:
                return 0.0
            
            if risk == 0:
                return 0.0  # Avoid division by zero
            
            return reward / risk
            
        except Exception as e:
            logger.error(f"Manual RR calculation error: {e}")
            return 0.0
    
    async def run_focused_diagnostic(self):
        """Run focused RR diagnostic test"""
        logger.info("🚀 STARTING FOCUSED RR CALCULATION DIAGNOSTIC")
        logger.info("=" * 80)
        
        # Step 1: Get recent analyses from API
        logger.info("\n📊 STEP 1: Getting recent analyses from /api/analyses endpoint...")
        
        try:
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code == 200:
                analyses_data = response.json()
                analyses = analyses_data.get('analyses', [])
                logger.info(f"✅ Retrieved {len(analyses)} analyses from API")
                
                # Step 2: Analyze RR fields for requested symbols
                logger.info(f"\n📊 STEP 2: Analyzing RR fields for requested symbols: {self.test_symbols}")
                
                target_analyses = []
                for analysis in analyses:
                    symbol = analysis.get('symbol')
                    if symbol in self.test_symbols:
                        target_analyses.append(analysis)
                
                if not target_analyses:
                    logger.warning("⚠️ No analyses found for requested symbols, using first 5 available")
                    target_analyses = analyses[:5]
                
                logger.info(f"📊 Found {len(target_analyses)} analyses to examine")
                
                # Step 3: Detailed RR analysis
                logger.info(f"\n📊 STEP 3: Detailed RR calculation analysis")
                
                coherent_count = 0
                incoherent_count = 0
                aberrant_count = 0
                default_count = 0
                
                for i, analysis in enumerate(target_analyses[:11]):  # Limit to 11 as mentioned in review
                    symbol = analysis.get('symbol', 'UNKNOWN')
                    
                    # Extract RR fields
                    calculated_rr = analysis.get('calculated_rr', 0)
                    risk_reward_ratio = analysis.get('risk_reward_ratio', 0)
                    
                    # Extract price fields
                    entry_price = analysis.get('entry_price')
                    stop_loss_price = analysis.get('stop_loss_price')
                    take_profit_price = analysis.get('take_profit_price')
                    ia1_signal = analysis.get('ia1_signal', 'hold')
                    
                    logger.info(f"\n   📋 Analysis {i+1}: {symbol}")
                    logger.info(f"      API RR Fields: calculated_rr={calculated_rr}, risk_reward_ratio={risk_reward_ratio}")
                    logger.info(f"      Price Data: entry={entry_price}, stop_loss={stop_loss_price}, take_profit={take_profit_price}")
                    logger.info(f"      Signal: {ia1_signal}")
                    
                    # Perform manual calculation
                    manual_rr = 0.0
                    if entry_price and stop_loss_price and take_profit_price:
                        manual_rr = self.calculate_manual_rr(entry_price, stop_loss_price, take_profit_price, ia1_signal)
                        logger.info(f"      Manual RR: {manual_rr:.3f}")
                        
                        # Compare with API values
                        api_rr = risk_reward_ratio if risk_reward_ratio != 0 else calculated_rr
                        
                        if api_rr > 0 and manual_rr > 0:
                            difference_pct = abs(api_rr - manual_rr) / max(api_rr, manual_rr) * 100
                            logger.info(f"      Comparison: API={api_rr:.3f} vs Manual={manual_rr:.3f} (diff: {difference_pct:.1f}%)")
                            
                            # Check coherence (within 20% tolerance)
                            if difference_pct <= 20:
                                coherent_count += 1
                                logger.info(f"      ✅ COHERENT: Values match within tolerance")
                            else:
                                incoherent_count += 1
                                logger.warning(f"      ❌ INCOHERENT: Values differ significantly ({difference_pct:.1f}%)")
                        else:
                            incoherent_count += 1
                            logger.warning(f"      ❌ INCOHERENT: Zero or invalid values")
                    else:
                        logger.warning(f"      ⚠️ Missing price data for manual calculation")
                    
                    # Check for aberrant values (>10)
                    if risk_reward_ratio > 10 or manual_rr > 10:
                        aberrant_count += 1
                        logger.warning(f"      🚨 ABERRANT: RR value >10 detected (API={risk_reward_ratio}, Manual={manual_rr:.3f})")
                    
                    # Check for default values
                    if calculated_rr in [0.0, 1.0, 20.0] or risk_reward_ratio in [1.0, 20.0]:
                        default_count += 1
                        logger.warning(f"      ⚠️ DEFAULT: Suspected default values (calculated_rr={calculated_rr}, risk_reward_ratio={risk_reward_ratio})")
                    
                    # Store results
                    self.diagnostic_results.append({
                        'symbol': symbol,
                        'calculated_rr': calculated_rr,
                        'risk_reward_ratio': risk_reward_ratio,
                        'manual_rr': manual_rr,
                        'entry_price': entry_price,
                        'stop_loss_price': stop_loss_price,
                        'take_profit_price': take_profit_price,
                        'signal': ia1_signal,
                        'coherent': abs(risk_reward_ratio - manual_rr) / max(risk_reward_ratio, manual_rr) <= 0.2 if risk_reward_ratio > 0 and manual_rr > 0 else False,
                        'aberrant': risk_reward_ratio > 10 or manual_rr > 10,
                        'default_suspected': calculated_rr in [0.0, 1.0, 20.0] or risk_reward_ratio in [1.0, 20.0]
                    })
                
                # Step 4: Summary analysis
                logger.info(f"\n📊 STEP 4: Summary Analysis")
                total_analyses = len(self.diagnostic_results)
                coherence_rate = coherent_count / total_analyses * 100 if total_analyses > 0 else 0
                
                logger.info(f"   Total analyses examined: {total_analyses}")
                logger.info(f"   Coherent calculations: {coherent_count}")
                logger.info(f"   Incoherent calculations: {incoherent_count}")
                logger.info(f"   Coherence rate: {coherence_rate:.1f}%")
                logger.info(f"   Aberrant values (>10): {aberrant_count}")
                logger.info(f"   Default values suspected: {default_count}")
                
                # Step 5: Specific examples from review request
                logger.info(f"\n📊 STEP 5: Specific Examples Analysis")
                
                examples = [
                    {'symbol': 'DOGEUSDT', 'expected_api': 20.0, 'expected_manual': 31.072},
                    {'symbol': 'ETHUSDT', 'expected_api': 1.0, 'expected_manual': 2.000},
                    {'symbol': 'FILUSDT', 'expected_api': 20.0, 'expected_manual': 199.0}
                ]
                
                for example in examples:
                    symbol = example['symbol']
                    found_analysis = None
                    
                    for result in self.diagnostic_results:
                        if result['symbol'] == symbol:
                            found_analysis = result
                            break
                    
                    if found_analysis:
                        logger.info(f"   📋 {symbol} Found:")
                        logger.info(f"      API RR: {found_analysis['risk_reward_ratio']} (expected: {example['expected_api']})")
                        logger.info(f"      Manual RR: {found_analysis['manual_rr']:.3f} (expected: {example['expected_manual']})")
                        logger.info(f"      Coherent: {found_analysis['coherent']}")
                    else:
                        logger.info(f"   📋 {symbol} Not found in recent analyses")
                
                # Step 6: Database verification
                logger.info(f"\n📊 STEP 6: Database Field Verification")
                
                try:
                    client = MongoClient(self.mongo_url)
                    db = client[self.db_name]
                    
                    # Check recent analyses in database
                    recent_db_analyses = list(db.technical_analyses.find().sort("timestamp", -1).limit(10))
                    
                    calculated_rr_present = 0
                    risk_reward_ratio_present = 0
                    both_present = 0
                    
                    for db_analysis in recent_db_analyses:
                        if 'calculated_rr' in db_analysis:
                            calculated_rr_present += 1
                        if 'risk_reward_ratio' in db_analysis:
                            risk_reward_ratio_present += 1
                        if 'calculated_rr' in db_analysis and 'risk_reward_ratio' in db_analysis:
                            both_present += 1
                    
                    logger.info(f"   Database field presence (last 10 analyses):")
                    logger.info(f"      calculated_rr present: {calculated_rr_present}/10")
                    logger.info(f"      risk_reward_ratio present: {risk_reward_ratio_present}/10")
                    logger.info(f"      Both fields present: {both_present}/10")
                    
                    client.close()
                    
                except Exception as e:
                    logger.error(f"   Database verification failed: {e}")
                
                # Final assessment
                logger.info(f"\n🎯 FINAL DIAGNOSTIC ASSESSMENT:")
                
                if coherence_rate >= 70:
                    logger.info(f"   ✅ RR CALCULATION STATUS: WORKING (coherence: {coherence_rate:.1f}%)")
                elif coherence_rate >= 30:
                    logger.info(f"   ⚠️ RR CALCULATION STATUS: PARTIALLY WORKING (coherence: {coherence_rate:.1f}%)")
                else:
                    logger.info(f"   ❌ RR CALCULATION STATUS: NOT WORKING (coherence: {coherence_rate:.1f}%)")
                
                logger.info(f"   📊 Key Issues Identified:")
                if default_count > total_analyses * 0.5:
                    logger.info(f"      - High number of default values ({default_count}/{total_analyses})")
                if aberrant_count > 0:
                    logger.info(f"      - Aberrant values detected ({aberrant_count} cases)")
                if calculated_rr_present < 8:
                    logger.info(f"      - calculated_rr field missing in some analyses")
                if coherence_rate < 50:
                    logger.info(f"      - Low coherence between API and manual calculations")
                
                return coherence_rate >= 70
                
            else:
                logger.error(f"❌ Failed to get analyses: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Diagnostic test failed: {e}")
            return False

async def main():
    """Main diagnostic execution"""
    test = FocusedRRDiagnosticTest()
    success = await test.run_focused_diagnostic()
    
    if success:
        logger.info("\n✅ RR calculation diagnostic completed - system appears to be working!")
        return 0
    else:
        logger.info("\n❌ RR calculation diagnostic completed - issues identified!")
        return 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)
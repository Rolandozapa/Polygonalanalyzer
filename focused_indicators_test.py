#!/usr/bin/env python3
"""
Focused IA1 Technical Indicators Test - Post-Correction Analysis
Focus on analyzing the current state of technical indicators after corrections
"""

import requests
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_technical_indicators():
    """Analyze current technical indicators in IA1 analyses"""
    
    # Get backend URL
    api_url = "https://dual-ai-trader-4.preview.emergentagent.com/api"
    
    logger.info("🔍 FOCUSED TECHNICAL INDICATORS ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Get IA1 analyses
        response = requests.get(f"{api_url}/analyses", timeout=30)
        
        if response.status_code != 200:
            logger.error(f"❌ Failed to get analyses: HTTP {response.status_code}")
            return
        
        data = response.json()
        analyses = data.get('analyses', [])
        
        if not analyses:
            logger.error("❌ No IA1 analyses found")
            return
        
        logger.info(f"📊 Found {len(analyses)} IA1 analyses")
        
        # Analyze indicators
        default_values = {'rsi': 50.0, 'macd': 0.0, 'stochastic': 50.0, 'bollinger': 0.0}
        
        results = {
            'total': 0,
            'with_calculated_rsi': 0,
            'with_calculated_macd': 0,
            'with_calculated_stochastic': 0,
            'with_calculated_bollinger': 0,
            'fallback_analyses': 0,
            'real_analyses': 0,
            'symbols_with_calculated': [],
            'symbols_with_defaults': []
        }
        
        logger.info("\n📋 DETAILED ANALYSIS BY SYMBOL:")
        logger.info("-" * 60)
        
        for analysis in analyses[:20]:  # Check first 20
            symbol = analysis.get('symbol', 'Unknown')
            reasoning = analysis.get('ia1_reasoning', '')
            
            results['total'] += 1
            
            # Check if it's a fallback analysis
            if 'Fallback ultra professional analysis' in reasoning:
                results['fallback_analyses'] += 1
                results['symbols_with_defaults'].append(symbol)
                logger.info(f"❌ {symbol}: FALLBACK ANALYSIS (all default values)")
                continue
            else:
                results['real_analyses'] += 1
            
            # Check each indicator
            rsi = analysis.get('rsi', 50.0)
            macd = analysis.get('macd_signal', 0.0)
            stochastic = analysis.get('stochastic', 50.0)
            bollinger = analysis.get('bollinger_position', 0.0)
            
            calculated_indicators = []
            
            # Check RSI
            if abs(rsi - default_values['rsi']) > 0.001:
                results['with_calculated_rsi'] += 1
                calculated_indicators.append('RSI')
            
            # Check MACD
            if abs(macd - default_values['macd']) > 0.001:
                results['with_calculated_macd'] += 1
                calculated_indicators.append('MACD')
            
            # Check Stochastic
            if abs(stochastic - default_values['stochastic']) > 0.001:
                results['with_calculated_stochastic'] += 1
                calculated_indicators.append('Stochastic')
            
            # Check Bollinger
            if abs(bollinger - default_values['bollinger']) > 0.001:
                results['with_calculated_bollinger'] += 1
                calculated_indicators.append('Bollinger')
            
            if calculated_indicators:
                results['symbols_with_calculated'].append(symbol)
                logger.info(f"✅ {symbol}: CALCULATED - {', '.join(calculated_indicators)}")
                logger.info(f"   RSI: {rsi:.2f}, MACD: {macd:.6f}, Stoch: {stochastic:.2f}, BB: {bollinger:.2f}")
            else:
                results['symbols_with_defaults'].append(symbol)
                logger.info(f"❌ {symbol}: ALL DEFAULT VALUES")
                logger.info(f"   RSI: {rsi}, MACD: {macd}, Stoch: {stochastic}, BB: {bollinger}")
        
        # Summary statistics
        logger.info("\n" + "=" * 60)
        logger.info("📊 TECHNICAL INDICATORS SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Total analyses: {results['total']}")
        logger.info(f"Real analyses (not fallback): {results['real_analyses']}")
        logger.info(f"Fallback analyses: {results['fallback_analyses']}")
        
        if results['real_analyses'] > 0:
            rsi_pct = (results['with_calculated_rsi'] / results['real_analyses']) * 100
            macd_pct = (results['with_calculated_macd'] / results['real_analyses']) * 100
            stoch_pct = (results['with_calculated_stochastic'] / results['real_analyses']) * 100
            bb_pct = (results['with_calculated_bollinger'] / results['real_analyses']) * 100
            
            logger.info(f"\nINDICATOR CALCULATION SUCCESS RATES (excluding fallbacks):")
            logger.info(f"✅ RSI calculated: {results['with_calculated_rsi']}/{results['real_analyses']} ({rsi_pct:.1f}%)")
            logger.info(f"✅ MACD calculated: {results['with_calculated_macd']}/{results['real_analyses']} ({macd_pct:.1f}%)")
            logger.info(f"✅ Stochastic calculated: {results['with_calculated_stochastic']}/{results['real_analyses']} ({stoch_pct:.1f}%)")
            logger.info(f"✅ Bollinger calculated: {results['with_calculated_bollinger']}/{results['real_analyses']} ({bb_pct:.1f}%)")
        
        logger.info(f"\nSYMBOLS WITH CALCULATED INDICATORS:")
        for symbol in results['symbols_with_calculated']:
            logger.info(f"  ✅ {symbol}")
        
        logger.info(f"\nSYMBOLS WITH DEFAULT/FALLBACK VALUES:")
        for symbol in results['symbols_with_defaults']:
            logger.info(f"  ❌ {symbol}")
        
        # Verdict
        logger.info("\n" + "=" * 60)
        logger.info("🎯 VERDICT")
        logger.info("=" * 60)
        
        if results['real_analyses'] == 0:
            logger.info("❌ CRITICAL: All analyses are fallback - no real IA1 processing")
        elif results['with_calculated_rsi'] > 0 or results['with_calculated_macd'] > 0:
            logger.info("✅ PARTIAL SUCCESS: Some indicators are being calculated")
            logger.info("🔍 Issue: Mix of calculated and default values indicates partial fix")
        else:
            logger.info("❌ FAILED: No calculated indicators found - corrections not effective")
        
        # Trigger new analysis
        logger.info("\n🚀 TRIGGERING NEW ANALYSIS TO TEST CORRECTIONS...")
        try:
            start_response = requests.post(f"{api_url}/trading/start-trading", timeout=60)
            if start_response.status_code in [200, 201]:
                logger.info("✅ New analysis triggered successfully")
            else:
                logger.warning(f"⚠️ Start trading returned HTTP {start_response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to trigger new analysis: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error in analysis: {e}")

if __name__ == "__main__":
    analyze_technical_indicators()
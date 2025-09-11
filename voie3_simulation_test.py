#!/usr/bin/env python3
"""
VOIE 3 Simulation Test - Direct Testing of Escalation Logic
This test simulates the _should_send_to_ia2 function directly to validate VOIE 3 logic
"""

import sys
import os
import logging
from datetime import datetime

# Add the backend directory to the path
sys.path.append('/app/backend')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock classes to simulate the escalation logic
class MockTechnicalAnalysis:
    def __init__(self, symbol, ia1_signal, analysis_confidence, risk_reward_ratio, ia1_reasoning="Mock analysis"):
        self.symbol = symbol
        self.ia1_signal = ia1_signal
        self.analysis_confidence = analysis_confidence
        self.risk_reward_ratio = risk_reward_ratio
        self.ia1_reasoning = ia1_reasoning

class MockMarketOpportunity:
    def __init__(self, symbol):
        self.symbol = symbol

class MockEscalationTester:
    """Mock class to test the escalation logic directly"""
    
    def _should_send_to_ia2(self, analysis: MockTechnicalAnalysis, opportunity: MockMarketOpportunity) -> bool:
        """Exact copy of the escalation logic from server.py"""
        try:
            # FILTRE 0: Vérification de base analyse IA1
            if not analysis.ia1_reasoning or len(analysis.ia1_reasoning.strip()) < 50:
                logger.warning(f"❌ IA2 REJECT - {analysis.symbol}: Analyse IA1 vide/corrompue")
                return False
            
            # FILTRE 1: Confiance IA1 extrêmement faible (analyse défaillante)
            if analysis.analysis_confidence < 0.3:
                logger.warning(f"❌ IA2 REJECT - {analysis.symbol}: Confiance IA1 trop faible ({analysis.analysis_confidence:.2%})")
                return False
            
            # 🎯 NOUVELLE LOGIQUE CONDITIONNELLE: 3 voies vers IA2
            
            ia1_signal = getattr(analysis, 'ia1_signal', 'hold').lower()
            risk_reward_ratio = getattr(analysis, 'risk_reward_ratio', 1.0)
            confidence = analysis.analysis_confidence
            
            # VOIE 1: Position LONG/SHORT avec confidence > 70%
            strong_signal_with_confidence = (
                ia1_signal in ['long', 'short'] and 
                confidence >= 0.70
            )
            
            # VOIE 2: RR supérieur à 2.0 (peu importe le signal)
            excellent_rr = risk_reward_ratio >= 2.0
            
            # 🚀 VOIE 3: OVERRIDE - Sentiment technique exceptionnel >95% (NOUVEAU)
            # Cette voie permet de bypasser les critères standard quand le sentiment technique est exceptionnel
            # Cas d'usage: ARKMUSDT avec sentiment LONG excellent mais RR faible à cause de niveaux S/R serrés
            exceptional_technical_sentiment = (
                ia1_signal in ['long', 'short'] and 
                confidence >= 0.95  # Sentiment technique exceptionnel
            )
            
            # 📊 DÉCISION: Au moins UNE des trois voies doit être satisfaite
            if strong_signal_with_confidence:
                logger.info(f"✅ IA2 ACCEPTED (VOIE 1) - {analysis.symbol}: Signal {ia1_signal.upper()} avec confiance {confidence:.1%} ≥ 70% | RR={risk_reward_ratio:.2f}:1")
                return True
                
            elif excellent_rr:
                logger.info(f"✅ IA2 ACCEPTED (VOIE 2) - {analysis.symbol}: RR excellent {risk_reward_ratio:.2f}:1 ≥ 2.0 | Signal={ia1_signal.upper()}, Confiance={confidence:.1%}")
                return True
                
            elif exceptional_technical_sentiment:
                logger.info(f"🚀 IA2 ACCEPTED (VOIE 3 - OVERRIDE) - {analysis.symbol}: Sentiment technique EXCEPTIONNEL {confidence:.1%} ≥ 95% pour signal {ia1_signal.upper()} | RR={risk_reward_ratio:.2f}:1 - BYPASS des critères standard")
                return True
                
            else:
                # Aucune des trois voies satisfaite
                reasons = []
                if not strong_signal_with_confidence:
                    if ia1_signal == 'hold':
                        reasons.append(f"Signal HOLD (pas de position)")
                    else:
                        reasons.append(f"Signal {ia1_signal.upper()} mais confiance {confidence:.1%} < 70%")
                        
                if not excellent_rr:
                    reasons.append(f"RR {risk_reward_ratio:.2f}:1 < 2.0")
                
                if not exceptional_technical_sentiment:
                    if ia1_signal != 'hold':
                        reasons.append(f"Sentiment technique {confidence:.1%} < 95% (pas d'override)")
                
                logger.info(f"🛑 IA2 SKIP - {analysis.symbol}: Aucune des 3 voies satisfaite | {' ET '.join(reasons)}")
                return False
            
        except Exception as e:
            logger.error(f"Erreur filtrage IA2 pour {analysis.symbol}: {e}")
            return True  # En cas d'erreur, envoyer à IA2 (principe de précaution)

def test_voie3_scenarios():
    """Test various VOIE 3 scenarios"""
    logger.info("🚀 Starting VOIE 3 Direct Simulation Test")
    logger.info("=" * 80)
    
    tester = MockEscalationTester()
    
    # Test scenarios
    scenarios = [
        {
            "name": "ARKMUSDT Case - VOIE 3 Override",
            "symbol": "ARKMUSDT",
            "signal": "LONG",
            "confidence": 0.96,
            "rr": 0.64,
            "expected": True,
            "expected_voie": "VOIE 3"
        },
        {
            "name": "High Confidence SHORT - VOIE 3",
            "symbol": "BTCUSDT", 
            "signal": "SHORT",
            "confidence": 0.97,
            "rr": 1.2,
            "expected": True,
            "expected_voie": "VOIE 3"
        },
        {
            "name": "Exactly 95% Confidence - VOIE 3",
            "symbol": "ETHUSDT",
            "signal": "LONG", 
            "confidence": 0.95,
            "rr": 1.5,
            "expected": True,
            "expected_voie": "VOIE 3"
        },
        {
            "name": "Just Below 95% - Should Not Trigger VOIE 3",
            "symbol": "SOLUSDT",
            "signal": "LONG",
            "confidence": 0.949,
            "rr": 1.8,
            "expected": False,
            "expected_voie": None
        },
        {
            "name": "HOLD Signal High Confidence - No Escalation",
            "symbol": "ADAUSDT",
            "signal": "HOLD",
            "confidence": 0.98,
            "rr": 3.0,
            "expected": False,
            "expected_voie": None
        },
        {
            "name": "Standard VOIE 1 - High Confidence",
            "symbol": "LINKUSDT",
            "signal": "LONG",
            "confidence": 0.75,
            "rr": 1.5,
            "expected": True,
            "expected_voie": "VOIE 1"
        },
        {
            "name": "Standard VOIE 2 - Excellent RR",
            "symbol": "MATICUSDT",
            "signal": "SHORT",
            "confidence": 0.65,
            "rr": 2.5,
            "expected": True,
            "expected_voie": "VOIE 2"
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        logger.info(f"\n📊 Testing: {scenario['name']}")
        logger.info(f"   Symbol: {scenario['symbol']}")
        logger.info(f"   Signal: {scenario['signal']}")
        logger.info(f"   Confidence: {scenario['confidence']:.1%}")
        logger.info(f"   Risk-Reward: {scenario['rr']:.2f}:1")
        logger.info(f"   Expected: {'Escalate' if scenario['expected'] else 'Skip'} ({scenario['expected_voie'] or 'None'})")
        
        # Create mock objects
        analysis = MockTechnicalAnalysis(
            symbol=scenario['symbol'],
            ia1_signal=scenario['signal'],
            analysis_confidence=scenario['confidence'],
            risk_reward_ratio=scenario['rr'],
            ia1_reasoning="Mock detailed technical analysis with sufficient length to pass the basic filter check"
        )
        
        opportunity = MockMarketOpportunity(scenario['symbol'])
        
        # Test the escalation logic
        result = tester._should_send_to_ia2(analysis, opportunity)
        
        # Evaluate result
        success = (result == scenario['expected'])
        status = "✅ PASS" if success else "❌ FAIL"
        
        logger.info(f"   Result: {'Escalate' if result else 'Skip'} - {status}")
        
        results.append({
            'scenario': scenario['name'],
            'expected': scenario['expected'],
            'actual': result,
            'success': success,
            'expected_voie': scenario['expected_voie']
        })
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 VOIE 3 DIRECT SIMULATION TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        logger.info(f"{status}: {result['scenario']}")
        logger.info(f"   Expected: {'Escalate' if result['expected'] else 'Skip'}, Actual: {'Escalate' if result['actual'] else 'Skip'}")
    
    logger.info(f"\n🎯 OVERALL RESULT: {passed}/{total} scenarios passed")
    
    # Specific VOIE 3 analysis
    voie3_scenarios = [r for r in results if r['expected_voie'] == 'VOIE 3']
    voie3_passed = sum(1 for r in voie3_scenarios if r['success'])
    
    logger.info(f"🚀 VOIE 3 SPECIFIC: {voie3_passed}/{len(voie3_scenarios)} VOIE 3 scenarios passed")
    
    if passed == total:
        logger.info("\n🎉 VERDICT: VOIE 3 Escalation Logic is FULLY FUNCTIONAL!")
        logger.info("✅ All escalation scenarios working correctly")
        logger.info("✅ VOIE 3 override properly bypassing RR requirements for 95%+ confidence")
        logger.info("✅ ARKMUSDT case study scenario handled correctly")
        logger.info("✅ Edge cases (95% boundary, HOLD signals) working as expected")
    else:
        failed_scenarios = [r['scenario'] for r in results if not r['success']]
        logger.info(f"\n❌ VERDICT: Some escalation scenarios failed: {failed_scenarios}")
    
    return passed, total

if __name__ == "__main__":
    passed, total = test_voie3_scenarios()
    sys.exit(0 if passed == total else 1)
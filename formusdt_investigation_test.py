#!/usr/bin/env python3
"""
FORMUSDT INVESTIGATION SPÉCIFIQUE - Test Suite
Analyser pourquoi FORMUSDT n'a pas été admis en IA2

CONTEXTE:
L'utilisateur demande pourquoi le symbole FORMUSDT n'a pas été traité par IA2. 
D'après les logs précédents, FORMUSDT était éligible avec Signal=LONG et Confidence=83% 
ce qui devrait satisfaire les critères VOIE 1 (LONG/SHORT + Confidence ≥ 70%).

OBJECTIFS DE TEST:
1. Statut Actuel IA1 pour FORMUSDT
2. Critères d'Admission IA2 pour FORMUSDT
3. Pipeline IA1→IA2 pour FORMUSDT
4. Logs d'Erreur FORMUSDT
5. Validation Système
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FORMUSDTInvestigationSuite:
    """Test suite spécifique pour l'investigation FORMUSDT"""
    
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
        logger.info(f"🔍 FORMUSDT Investigation at: {self.api_url}")
        
        # Test results
        self.test_results = []
        self.formusdt_data = {}
        
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"   Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    async def test_1_formusdt_ia1_current_status(self):
        """Test 1: Vérifier le statut actuel IA1 pour FORMUSDT"""
        logger.info("\n🔍 TEST 1: Statut Actuel IA1 pour FORMUSDT")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("FORMUSDT IA1 Status", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            # Search for FORMUSDT analysis
            formusdt_analysis = None
            for analysis in analyses:
                if analysis.get('symbol', '').upper() == 'FORMUSDT':
                    formusdt_analysis = analysis
                    break
            
            if formusdt_analysis:
                symbol = formusdt_analysis.get('symbol', 'Unknown')
                signal = formusdt_analysis.get('recommendation', '').upper()
                confidence = formusdt_analysis.get('confidence', 0)
                rr_ratio = formusdt_analysis.get('risk_reward_ratio', 0)
                reasoning = formusdt_analysis.get('reasoning', '')
                timestamp = formusdt_analysis.get('timestamp', '')
                
                # Store FORMUSDT data for other tests
                self.formusdt_data = {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'rr_ratio': rr_ratio,
                    'reasoning': reasoning,
                    'timestamp': timestamp,
                    'analysis': formusdt_analysis
                }
                
                logger.info(f"   📊 FORMUSDT trouvé dans IA1:")
                logger.info(f"      Symbol: {symbol}")
                logger.info(f"      Signal: {signal}")
                logger.info(f"      Confidence: {confidence}%")
                logger.info(f"      Risk-Reward: {rr_ratio}:1")
                logger.info(f"      Timestamp: {timestamp}")
                logger.info(f"      Reasoning: {reasoning[:200]}...")
                
                # Check if it matches expected criteria (LONG + 83%)
                expected_signal = signal == 'LONG'
                expected_confidence = abs(confidence - 83.0) <= 5.0  # Allow 5% tolerance
                
                success = expected_signal and expected_confidence
                details = f"Signal: {signal} (expected LONG), Confidence: {confidence}% (expected ~83%)"
                
                self.log_test_result("FORMUSDT IA1 Status", success, details)
                
            else:
                logger.info("   ❌ FORMUSDT non trouvé dans les analyses IA1 actuelles")
                logger.info("   🔍 Recherche dans toutes les analyses disponibles...")
                
                # Search in all analyses for any FORM-related symbols
                form_related = []
                for analysis in analyses:
                    symbol = analysis.get('symbol', '').upper()
                    if 'FORM' in symbol:
                        form_related.append({
                            'symbol': symbol,
                            'signal': analysis.get('recommendation', '').upper(),
                            'confidence': analysis.get('confidence', 0),
                            'timestamp': analysis.get('timestamp', '')
                        })
                
                if form_related:
                    logger.info(f"   📊 Symboles FORM-related trouvés: {len(form_related)}")
                    for item in form_related:
                        logger.info(f"      {item['symbol']}: {item['signal']} {item['confidence']}% ({item['timestamp']})")
                else:
                    logger.info("   ❌ Aucun symbole FORM-related trouvé")
                
                self.log_test_result("FORMUSDT IA1 Status", False, "FORMUSDT non trouvé dans les analyses IA1")
                
        except Exception as e:
            self.log_test_result("FORMUSDT IA1 Status", False, f"Exception: {str(e)}")
    
    async def test_2_formusdt_ia2_admission_criteria(self):
        """Test 2: Vérifier les critères d'admission IA2 pour FORMUSDT"""
        logger.info("\n🔍 TEST 2: Critères d'Admission IA2 pour FORMUSDT")
        
        try:
            if not self.formusdt_data:
                self.log_test_result("FORMUSDT IA2 Criteria", False, "Pas de données FORMUSDT disponibles")
                return
            
            signal = self.formusdt_data.get('signal', '')
            confidence = self.formusdt_data.get('confidence', 0)
            rr_ratio = self.formusdt_data.get('rr_ratio', 0)
            
            logger.info(f"   📊 Analyse des critères d'admission IA2 pour FORMUSDT:")
            logger.info(f"      Signal: {signal}")
            logger.info(f"      Confidence: {confidence}%")
            logger.info(f"      Risk-Reward: {rr_ratio}:1")
            
            # VOIE 1: LONG/SHORT + Confidence ≥ 70%
            voie1_signal_ok = signal in ['LONG', 'SHORT']
            voie1_confidence_ok = confidence >= 70.0
            voie1_eligible = voie1_signal_ok and voie1_confidence_ok
            
            logger.info(f"   🎯 VOIE 1 (LONG/SHORT + Confidence ≥70%):")
            logger.info(f"      Signal LONG/SHORT: {'✅' if voie1_signal_ok else '❌'} ({signal})")
            logger.info(f"      Confidence ≥70%: {'✅' if voie1_confidence_ok else '❌'} ({confidence}%)")
            logger.info(f"      VOIE 1 Éligible: {'✅' if voie1_eligible else '❌'}")
            
            # VOIE 2: RR ≥ 2.0
            voie2_rr_ok = rr_ratio >= 2.0
            voie2_eligible = voie2_rr_ok
            
            logger.info(f"   🎯 VOIE 2 (RR ≥2.0):")
            logger.info(f"      RR ≥2.0: {'✅' if voie2_rr_ok else '❌'} ({rr_ratio}:1)")
            logger.info(f"      VOIE 2 Éligible: {'✅' if voie2_eligible else '❌'}")
            
            # Overall eligibility
            overall_eligible = voie1_eligible or voie2_eligible
            
            logger.info(f"   🏆 ÉLIGIBILITÉ GLOBALE IA2: {'✅' if overall_eligible else '❌'}")
            
            if overall_eligible:
                if voie1_eligible:
                    logger.info(f"      ✅ FORMUSDT devrait être admis en IA2 via VOIE 1")
                if voie2_eligible:
                    logger.info(f"      ✅ FORMUSDT devrait être admis en IA2 via VOIE 2")
            else:
                logger.info(f"      ❌ FORMUSDT ne satisfait aucun critère d'admission IA2")
                logger.info(f"         VOIE 1: Signal={signal}, Confidence={confidence}%")
                logger.info(f"         VOIE 2: RR={rr_ratio}:1")
            
            success = overall_eligible
            details = f"VOIE 1: {voie1_eligible} (Signal: {signal}, Conf: {confidence}%), VOIE 2: {voie2_eligible} (RR: {rr_ratio}:1)"
            
            self.log_test_result("FORMUSDT IA2 Criteria", success, details)
            
        except Exception as e:
            self.log_test_result("FORMUSDT IA2 Criteria", False, f"Exception: {str(e)}")
    
    async def test_3_formusdt_ia2_decisions_check(self):
        """Test 3: Vérifier si FORMUSDT a des décisions IA2"""
        logger.info("\n🔍 TEST 3: Vérifier les décisions IA2 pour FORMUSDT")
        
        try:
            # Get IA2 decisions
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("FORMUSDT IA2 Decisions", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            decisions = data.get('decisions', [])
            
            # Search for FORMUSDT decisions
            formusdt_decisions = []
            for decision in decisions:
                if decision.get('symbol', '').upper() == 'FORMUSDT':
                    formusdt_decisions.append(decision)
            
            logger.info(f"   📊 Décisions IA2 pour FORMUSDT: {len(formusdt_decisions)} trouvées")
            
            if formusdt_decisions:
                for i, decision in enumerate(formusdt_decisions):
                    signal = decision.get('signal', '').upper()
                    confidence = decision.get('confidence', 0)
                    timestamp = decision.get('timestamp', '')
                    reasoning = decision.get('reasoning', '')
                    
                    logger.info(f"      Décision {i+1}:")
                    logger.info(f"         Signal: {signal}")
                    logger.info(f"         Confidence: {confidence}%")
                    logger.info(f"         Timestamp: {timestamp}")
                    logger.info(f"         Reasoning: {reasoning[:150]}...")
                
                success = True
                details = f"{len(formusdt_decisions)} décisions IA2 trouvées pour FORMUSDT"
            else:
                logger.info("   ❌ Aucune décision IA2 trouvée pour FORMUSDT")
                
                # Check for FORM-related decisions
                form_decisions = []
                for decision in decisions:
                    symbol = decision.get('symbol', '').upper()
                    if 'FORM' in symbol:
                        form_decisions.append({
                            'symbol': symbol,
                            'signal': decision.get('signal', '').upper(),
                            'confidence': decision.get('confidence', 0),
                            'timestamp': decision.get('timestamp', '')
                        })
                
                if form_decisions:
                    logger.info(f"   📊 Décisions FORM-related trouvées: {len(form_decisions)}")
                    for item in form_decisions:
                        logger.info(f"      {item['symbol']}: {item['signal']} {item['confidence']}% ({item['timestamp']})")
                
                success = False
                details = "Aucune décision IA2 pour FORMUSDT - c'est le problème principal"
            
            self.log_test_result("FORMUSDT IA2 Decisions", success, details)
            
        except Exception as e:
            self.log_test_result("FORMUSDT IA2 Decisions", False, f"Exception: {str(e)}")
    
    async def test_4_formusdt_pipeline_logs(self):
        """Test 4: Chercher les logs spécifiques à FORMUSDT dans le pipeline IA1→IA2"""
        logger.info("\n🔍 TEST 4: Logs Pipeline IA1→IA2 pour FORMUSDT")
        
        try:
            # Get backend logs
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "10000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "5000", "/var/log/supervisor/backend.err.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            if not backend_logs:
                self.log_test_result("FORMUSDT Pipeline Logs", False, "Impossible de récupérer les logs backend")
                return
            
            # Search for FORMUSDT-specific logs
            formusdt_logs = []
            lines = backend_logs.split('\n')
            
            for line in lines:
                if 'FORMUSDT' in line.upper():
                    formusdt_logs.append(line.strip())
            
            logger.info(f"   📊 Logs FORMUSDT trouvés: {len(formusdt_logs)}")
            
            if formusdt_logs:
                logger.info("   📋 Logs FORMUSDT récents:")
                for i, log in enumerate(formusdt_logs[-10:]):  # Show last 10
                    logger.info(f"      {i+1}. {log}")
                
                # Analyze log patterns
                ia1_analysis_logs = sum(1 for log in formusdt_logs if 'IA1' in log)
                ia2_filter_logs = sum(1 for log in formusdt_logs if any(pattern in log for pattern in ['IA2 FILTER', 'IA2 ACCEPTED', 'IA2 SKIP']))
                voie1_logs = sum(1 for log in formusdt_logs if 'VOIE 1' in log)
                voie2_logs = sum(1 for log in formusdt_logs if 'VOIE 2' in log)
                error_logs = sum(1 for log in formusdt_logs if any(pattern in log for pattern in ['ERROR', 'FAILED', 'Exception']))
                
                logger.info(f"   📊 Analyse des logs FORMUSDT:")
                logger.info(f"      IA1 Analysis logs: {ia1_analysis_logs}")
                logger.info(f"      IA2 Filter logs: {ia2_filter_logs}")
                logger.info(f"      VOIE 1 logs: {voie1_logs}")
                logger.info(f"      VOIE 2 logs: {voie2_logs}")
                logger.info(f"      Error logs: {error_logs}")
                
                # Look for specific filtering reasons
                skip_reasons = []
                for log in formusdt_logs:
                    if 'IA2 SKIP' in log or 'SKIP' in log:
                        skip_reasons.append(log)
                
                if skip_reasons:
                    logger.info(f"   🚫 Raisons de skip FORMUSDT:")
                    for reason in skip_reasons:
                        logger.info(f"      - {reason}")
                
                success = len(formusdt_logs) > 0
                details = f"{len(formusdt_logs)} logs trouvés, IA1: {ia1_analysis_logs}, IA2 Filter: {ia2_filter_logs}, Errors: {error_logs}"
                
            else:
                logger.info("   ❌ Aucun log spécifique à FORMUSDT trouvé")
                
                # Search for general IA2 filtering logs
                ia2_logs = []
                for line in lines:
                    if any(pattern in line for pattern in ['IA2 FILTER', 'IA2 ACCEPTED', 'IA2 SKIP', 'VOIE 1', 'VOIE 2']):
                        ia2_logs.append(line.strip())
                
                logger.info(f"   📊 Logs IA2 généraux trouvés: {len(ia2_logs)}")
                if ia2_logs:
                    logger.info("   📋 Exemples de logs IA2 récents:")
                    for i, log in enumerate(ia2_logs[-5:]):  # Show last 5
                        logger.info(f"      {i+1}. {log}")
                
                success = False
                details = "Aucun log FORMUSDT spécifique - le symbole n'est peut-être pas traité"
            
            self.log_test_result("FORMUSDT Pipeline Logs", success, details)
            
        except Exception as e:
            self.log_test_result("FORMUSDT Pipeline Logs", False, f"Exception: {str(e)}")
    
    async def test_5_formusdt_error_analysis(self):
        """Test 5: Chercher les erreurs spécifiques à FORMUSDT"""
        logger.info("\n🔍 TEST 5: Analyse des erreurs FORMUSDT")
        
        try:
            # Get backend error logs
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "5000", "/var/log/supervisor/backend.err.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            try:
                log_result = subprocess.run(
                    ["grep", "-i", "error", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            if not backend_logs:
                self.log_test_result("FORMUSDT Error Analysis", False, "Impossible de récupérer les logs d'erreur")
                return
            
            # Search for FORMUSDT-related errors
            formusdt_errors = []
            lines = backend_logs.split('\n')
            
            for line in lines:
                if 'FORMUSDT' in line.upper() and any(error_word in line.upper() for error_word in ['ERROR', 'EXCEPTION', 'FAILED', 'TRACEBACK']):
                    formusdt_errors.append(line.strip())
            
            logger.info(f"   📊 Erreurs FORMUSDT trouvées: {len(formusdt_errors)}")
            
            if formusdt_errors:
                logger.info("   🚨 Erreurs FORMUSDT spécifiques:")
                for i, error in enumerate(formusdt_errors):
                    logger.info(f"      {i+1}. {error}")
                
                success = False  # Errors found is actually bad
                details = f"{len(formusdt_errors)} erreurs spécifiques à FORMUSDT trouvées"
                
            else:
                logger.info("   ✅ Aucune erreur spécifique à FORMUSDT trouvée")
                
                # Look for general IA2 processing errors
                ia2_errors = []
                for line in lines:
                    if any(pattern in line.upper() for pattern in ['IA2', 'DECISION']) and any(error_word in line.upper() for error_word in ['ERROR', 'EXCEPTION', 'FAILED']):
                        ia2_errors.append(line.strip())
                
                logger.info(f"   📊 Erreurs IA2 générales: {len(ia2_errors)}")
                if ia2_errors:
                    logger.info("   🚨 Exemples d'erreurs IA2:")
                    for i, error in enumerate(ia2_errors[-3:]):  # Show last 3
                        logger.info(f"      {i+1}. {error}")
                
                success = True  # No FORMUSDT-specific errors is good
                details = f"Aucune erreur FORMUSDT spécifique, {len(ia2_errors)} erreurs IA2 générales"
            
            self.log_test_result("FORMUSDT Error Analysis", success, details)
            
        except Exception as e:
            self.log_test_result("FORMUSDT Error Analysis", False, f"Exception: {str(e)}")
    
    async def test_6_trigger_fresh_analysis(self):
        """Test 6: Déclencher une nouvelle analyse pour tester le système"""
        logger.info("\n🔍 TEST 6: Déclencher une analyse fraîche et vérifier FORMUSDT")
        
        try:
            logger.info("   🚀 Déclenchement d'une nouvelle analyse via /api/trading/start-trading...")
            
            # Trigger fresh analysis
            start_response = requests.post(f"{self.api_url}/trading/start-trading", timeout=180)
            
            if start_response.status_code not in [200, 201]:
                logger.warning(f"   ⚠️ Start trading returned HTTP {start_response.status_code}")
                logger.info(f"   Response: {start_response.text[:500]}")
            else:
                logger.info("   ✅ Analyse déclenchée avec succès")
            
            # Wait for processing
            logger.info("   ⏳ Attente 45 secondes pour le traitement...")
            await asyncio.sleep(45)
            
            # Check for new FORMUSDT analysis
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Fresh Analysis Trigger", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            # Look for recent FORMUSDT analysis
            recent_formusdt = None
            cutoff_time = datetime.now() - timedelta(minutes=10)  # Last 10 minutes
            
            for analysis in analyses:
                if analysis.get('symbol', '').upper() == 'FORMUSDT':
                    timestamp_str = analysis.get('timestamp', '')
                    try:
                        if 'T' in timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.fromisoformat(timestamp_str)
                        
                        if timestamp.tzinfo:
                            timestamp = timestamp.replace(tzinfo=None)
                        
                        if timestamp > cutoff_time:
                            recent_formusdt = analysis
                            break
                    except:
                        continue
            
            if recent_formusdt:
                signal = recent_formusdt.get('recommendation', '').upper()
                confidence = recent_formusdt.get('confidence', 0)
                rr_ratio = recent_formusdt.get('risk_reward_ratio', 0)
                
                logger.info(f"   ✅ Nouvelle analyse FORMUSDT trouvée:")
                logger.info(f"      Signal: {signal}")
                logger.info(f"      Confidence: {confidence}%")
                logger.info(f"      Risk-Reward: {rr_ratio}:1")
                
                # Check if it should go to IA2
                should_go_to_ia2 = (signal in ['LONG', 'SHORT'] and confidence >= 70.0) or rr_ratio >= 2.0
                
                logger.info(f"   🎯 Devrait aller en IA2: {'✅' if should_go_to_ia2 else '❌'}")
                
                # Check if it actually went to IA2
                await asyncio.sleep(30)  # Wait for IA2 processing
                
                decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
                if decisions_response.status_code == 200:
                    decisions_data = decisions_response.json()
                    decisions = decisions_data.get('decisions', [])
                    
                    recent_formusdt_decision = None
                    for decision in decisions:
                        if decision.get('symbol', '').upper() == 'FORMUSDT':
                            decision_timestamp_str = decision.get('timestamp', '')
                            try:
                                if 'T' in decision_timestamp_str:
                                    decision_timestamp = datetime.fromisoformat(decision_timestamp_str.replace('Z', '+00:00'))
                                else:
                                    decision_timestamp = datetime.fromisoformat(decision_timestamp_str)
                                
                                if decision_timestamp.tzinfo:
                                    decision_timestamp = decision_timestamp.replace(tzinfo=None)
                                
                                if decision_timestamp > cutoff_time:
                                    recent_formusdt_decision = decision
                                    break
                            except:
                                continue
                    
                    if recent_formusdt_decision:
                        logger.info(f"   ✅ Décision IA2 FORMUSDT trouvée - Le pipeline fonctionne!")
                        success = True
                        details = f"Pipeline fonctionnel: IA1 ({signal}, {confidence}%) → IA2"
                    else:
                        logger.info(f"   ❌ Pas de décision IA2 FORMUSDT - Pipeline bloqué")
                        success = False
                        details = f"Pipeline bloqué: IA1 ({signal}, {confidence}%) → ❌ IA2"
                else:
                    success = False
                    details = "Impossible de vérifier les décisions IA2"
                
            else:
                logger.info("   ❌ Aucune nouvelle analyse FORMUSDT générée")
                success = False
                details = "Aucune nouvelle analyse FORMUSDT après déclenchement"
            
            self.log_test_result("Fresh Analysis Trigger", success, details)
            
        except Exception as e:
            self.log_test_result("Fresh Analysis Trigger", False, f"Exception: {str(e)}")
    
    async def run_formusdt_investigation(self):
        """Exécuter l'investigation complète FORMUSDT"""
        logger.info("🚀 DÉBUT DE L'INVESTIGATION FORMUSDT")
        logger.info("=" * 80)
        logger.info("🔍 FORMUSDT INVESTIGATION SPÉCIFIQUE")
        logger.info("🎯 Objectif: Comprendre pourquoi FORMUSDT n'a pas été admis en IA2")
        logger.info("🎯 Critères attendus: Signal=LONG, Confidence=83% → VOIE 1 éligible")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_formusdt_ia1_current_status()
        await self.test_2_formusdt_ia2_admission_criteria()
        await self.test_3_formusdt_ia2_decisions_check()
        await self.test_4_formusdt_pipeline_logs()
        await self.test_5_formusdt_error_analysis()
        await self.test_6_trigger_fresh_analysis()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 RÉSUMÉ DE L'INVESTIGATION FORMUSDT")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        logger.info(f"\n🎯 RÉSULTAT GLOBAL: {passed_tests}/{total_tests} tests réussis")
        
        # Diagnostic analysis
        logger.info("\n" + "=" * 80)
        logger.info("🔍 DIAGNOSTIC FORMUSDT")
        logger.info("=" * 80)
        
        # Analyze the root cause
        if self.formusdt_data:
            signal = self.formusdt_data.get('signal', '')
            confidence = self.formusdt_data.get('confidence', 0)
            rr_ratio = self.formusdt_data.get('rr_ratio', 0)
            
            logger.info(f"📊 DONNÉES FORMUSDT TROUVÉES:")
            logger.info(f"   Signal: {signal}")
            logger.info(f"   Confidence: {confidence}%")
            logger.info(f"   Risk-Reward: {rr_ratio}:1")
            
            # Determine why it wasn't admitted
            voie1_eligible = signal in ['LONG', 'SHORT'] and confidence >= 70.0
            voie2_eligible = rr_ratio >= 2.0
            
            if voie1_eligible or voie2_eligible:
                logger.info("🎯 CONCLUSION: FORMUSDT DEVRAIT ÊTRE ADMIS EN IA2")
                if voie1_eligible:
                    logger.info(f"   ✅ VOIE 1: {signal} + {confidence}% ≥ 70%")
                if voie2_eligible:
                    logger.info(f"   ✅ VOIE 2: RR {rr_ratio}:1 ≥ 2.0")
                
                logger.info("🚨 PROBLÈME IDENTIFIÉ: Pipeline IA1→IA2 ne fonctionne pas pour FORMUSDT")
                logger.info("🔧 SOLUTIONS POSSIBLES:")
                logger.info("   1. Vérifier les logs de filtrage IA2")
                logger.info("   2. Vérifier les erreurs de traitement IA2")
                logger.info("   3. Vérifier la méthode _should_send_to_ia2()")
                logger.info("   4. Vérifier les critères VOIE 1 et VOIE 2 dans le code")
                
            else:
                logger.info("🎯 CONCLUSION: FORMUSDT NE SATISFAIT PAS LES CRITÈRES IA2")
                logger.info(f"   ❌ VOIE 1: {signal} + {confidence}% < 70%")
                logger.info(f"   ❌ VOIE 2: RR {rr_ratio}:1 < 2.0")
                logger.info("🔧 SOLUTION: Améliorer l'analyse IA1 pour FORMUSDT")
        else:
            logger.info("🚨 PROBLÈME MAJEUR: FORMUSDT NON TROUVÉ DANS IA1")
            logger.info("🔧 SOLUTIONS POSSIBLES:")
            logger.info("   1. Vérifier si FORMUSDT est dans la liste des symboles analysés")
            logger.info("   2. Vérifier si FORMUSDT est tradable sur BingX")
            logger.info("   3. Vérifier les filtres du Scout")
            logger.info("   4. Vérifier les sources de données de marché")
        
        # Final recommendations
        logger.info("\n" + "=" * 80)
        logger.info("🎯 RECOMMANDATIONS POUR RÉSOUDRE LE PROBLÈME FORMUSDT")
        logger.info("=" * 80)
        
        if passed_tests < total_tests * 0.5:
            logger.info("🚨 PROBLÈME CRITIQUE: Plusieurs composants ne fonctionnent pas")
            logger.info("🔧 Actions prioritaires:")
            logger.info("   1. Vérifier que FORMUSDT est analysé par IA1")
            logger.info("   2. Vérifier les critères d'admission IA2")
            logger.info("   3. Déboguer le pipeline IA1→IA2")
            logger.info("   4. Vérifier les logs d'erreur système")
        else:
            logger.info("⚠️ PROBLÈME SPÉCIFIQUE: Le système fonctionne mais FORMUSDT est bloqué")
            logger.info("🔧 Actions ciblées:")
            logger.info("   1. Analyser les logs de filtrage spécifiques à FORMUSDT")
            logger.info("   2. Vérifier les critères VOIE 1/VOIE 2 pour ce symbole")
            logger.info("   3. Tester manuellement le pipeline avec FORMUSDT")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = FORMUSDTInvestigationSuite()
    passed, total = await test_suite.run_formusdt_investigation()
    
    # Exit with appropriate code
    if passed >= total * 0.7:  # 70% success rate acceptable for investigation
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
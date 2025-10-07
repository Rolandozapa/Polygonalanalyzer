#!/usr/bin/env python3
"""
DIAGNOSTIC APPROFONDI DU CALCUL RISK-REWARD (RR) IA1

PROBLÈME SIGNALÉ: L'utilisateur trouve que les valeurs RR calculées pour IA1 paraissent suspectes 
et veut vérifier la cohérence du système de calcul.

TESTS À EFFECTUER:
1. **Validation des Calculs RR**: Tester plusieurs symboles et vérifier manuellement les calculs RR avec les formules:
   - LONG: RR = (Take_Profit - Entry) / (Entry - Stop_Loss)
   - SHORT: RR = (Entry - Take_Profit) / (Stop_Loss - Entry)

2. **Vérification des Prix d'Entrée/Sortie**: S'assurer que:
   - Entry price est cohérent avec le prix actuel du marché
   - Stop-loss est logique (plus bas pour LONG, plus haut pour SHORT)
   - Take-profit est logique (plus haut pour LONG, plus bas pour SHORT)

3. **Analyse des Erreurs**: Chercher spécifiquement:
   - "ERROR:risk_reward_calculator:Error in optimal_rr_setup: float division by zero"
   - Valeurs RR aberrantes (trop élevées >10 ou négatives)
   - Cohérence entre calculated_rr et risk_reward_ratio

4. **Test de Calculs Manuels**: Pour chaque analyse:
   - Extraire Entry, Stop-Loss, Take-Profit depuis l'API
   - Calculer manuellement le RR
   - Comparer avec la valeur retournée par le système

5. **Vérification Base de Données**: Contrôler les valeurs RR stockées en MongoDB et leur cohérence

SYMBOLES À TESTER: BTCUSDT, ETHUSDT, SOLUSDT (au minimum)
"""

import asyncio
import json
import logging
import os
import sys
import time
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import requests
import subprocess
import re
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RRCalculationDiagnosticSuite:
    """Suite de diagnostic approfondi pour les calculs Risk-Reward IA1"""
    
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
        logger.info(f"🎯 Diagnostic RR Calculation à: {self.api_url}")
        
        # Test results
        self.test_results = []
        self.rr_analyses = []
        self.manual_calculations = []
        self.error_logs = []
        
        # Symboles à tester (priorité aux symboles demandés)
        self.priority_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.test_symbols = []
        
        # Database connection info
        self.mongo_url = "mongodb://localhost:27017"
        self.db_name = "myapp"
        
        # RR calculation thresholds for validation
        self.rr_validation_thresholds = {
            'min_valid_rr': 0.1,    # RR minimum acceptable
            'max_valid_rr': 15.0,   # RR maximum acceptable (au-delà = suspect)
            'suspicious_rr': 10.0,  # RR suspect (nécessite investigation)
            'negative_rr_tolerance': -0.1  # Tolérance pour RR légèrement négatif
        }
        
        # Formules RR pour validation manuelle
        self.rr_formulas = {
            'LONG': lambda entry, stop_loss, take_profit: (take_profit - entry) / (entry - stop_loss) if (entry - stop_loss) != 0 else 0,
            'SHORT': lambda entry, stop_loss, take_profit: (entry - take_profit) / (stop_loss - entry) if (stop_loss - entry) != 0 else 0
        }
    
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
    
    async def _capture_backend_logs(self):
        """Capture backend logs for RR calculation error analysis"""
        try:
            # Try to capture supervisor backend logs
            result = subprocess.run(
                ['tail', '-n', '500', '/var/log/supervisor/backend.out.log'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.split('\n')
            else:
                # Try alternative log location
                result = subprocess.run(
                    ['tail', '-n', '500', '/var/log/supervisor/backend.err.log'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and result.stdout:
                    return result.stdout.split('\n')
                else:
                    return []
                    
        except Exception as e:
            logger.warning(f"Could not capture backend logs: {e}")
            return []
    
    def _validate_price_logic(self, entry: float, stop_loss: float, take_profit: float, signal: str) -> Dict[str, Any]:
        """Valider la logique des prix selon la direction du signal"""
        validation = {
            'entry_valid': entry > 0,
            'stop_loss_valid': stop_loss > 0,
            'take_profit_valid': take_profit > 0,
            'price_logic_valid': False,
            'price_logic_errors': []
        }
        
        if signal.upper() == 'LONG':
            # Pour LONG: Stop-Loss < Entry < Take-Profit
            if stop_loss >= entry:
                validation['price_logic_errors'].append(f"LONG: Stop-Loss ({stop_loss}) should be < Entry ({entry})")
            if take_profit <= entry:
                validation['price_logic_errors'].append(f"LONG: Take-Profit ({take_profit}) should be > Entry ({entry})")
            
            validation['price_logic_valid'] = stop_loss < entry < take_profit
            
        elif signal.upper() == 'SHORT':
            # Pour SHORT: Take-Profit < Entry < Stop-Loss
            if take_profit >= entry:
                validation['price_logic_errors'].append(f"SHORT: Take-Profit ({take_profit}) should be < Entry ({entry})")
            if stop_loss <= entry:
                validation['price_logic_errors'].append(f"SHORT: Stop-Loss ({stop_loss}) should be > Entry ({entry})")
            
            validation['price_logic_valid'] = take_profit < entry < stop_loss
        
        return validation
    
    def _calculate_manual_rr(self, entry: float, stop_loss: float, take_profit: float, signal: str) -> Dict[str, Any]:
        """Calculer manuellement le RR selon les formules"""
        calculation = {
            'manual_rr': 0.0,
            'formula_used': '',
            'calculation_valid': False,
            'calculation_error': None,
            'risk': 0.0,
            'reward': 0.0
        }
        
        try:
            if signal.upper() == 'LONG':
                # LONG: RR = (Take_Profit - Entry) / (Entry - Stop_Loss)
                risk = entry - stop_loss
                reward = take_profit - entry
                calculation['formula_used'] = 'LONG: (TP - Entry) / (Entry - SL)'
                
            elif signal.upper() == 'SHORT':
                # SHORT: RR = (Entry - Take_Profit) / (Stop_Loss - Entry)
                risk = stop_loss - entry
                reward = entry - take_profit
                calculation['formula_used'] = 'SHORT: (Entry - TP) / (SL - Entry)'
            
            else:
                calculation['calculation_error'] = f"Signal invalide: {signal}"
                return calculation
            
            calculation['risk'] = risk
            calculation['reward'] = reward
            
            if risk != 0:
                calculation['manual_rr'] = reward / risk
                calculation['calculation_valid'] = True
            else:
                calculation['calculation_error'] = "Division par zéro: Risk = 0"
                
        except Exception as e:
            calculation['calculation_error'] = f"Erreur calcul: {str(e)}"
        
        return calculation
    
    def _analyze_rr_consistency(self, api_rr: float, manual_rr: float, tolerance: float = 0.1) -> Dict[str, Any]:
        """Analyser la cohérence entre RR API et RR manuel"""
        analysis = {
            'consistent': False,
            'difference': 0.0,
            'percentage_difference': 0.0,
            'tolerance_met': False,
            'analysis_notes': []
        }
        
        try:
            analysis['difference'] = abs(api_rr - manual_rr)
            
            if manual_rr != 0:
                analysis['percentage_difference'] = (analysis['difference'] / abs(manual_rr)) * 100
            
            analysis['tolerance_met'] = analysis['difference'] <= tolerance
            analysis['consistent'] = analysis['tolerance_met']
            
            if analysis['consistent']:
                analysis['analysis_notes'].append(f"✅ RR cohérent: API={api_rr:.3f}, Manuel={manual_rr:.3f}")
            else:
                analysis['analysis_notes'].append(f"❌ RR incohérent: API={api_rr:.3f}, Manuel={manual_rr:.3f}, Diff={analysis['difference']:.3f}")
            
            # Analyse des cas spéciaux
            if api_rr == 1.0 and manual_rr != 1.0:
                analysis['analysis_notes'].append("⚠️ API retourne RR=1.0 (valeur par défaut?)")
            
            if api_rr == 0.0:
                analysis['analysis_notes'].append("⚠️ API retourne RR=0.0 (calcul échoué?)")
            
            if manual_rr < 0:
                analysis['analysis_notes'].append("⚠️ RR manuel négatif (logique prix inversée?)")
                
        except Exception as e:
            analysis['analysis_notes'].append(f"❌ Erreur analyse cohérence: {str(e)}")
        
        return analysis
    
    async def test_1_get_available_symbols(self):
        """Test 1: Obtenir les symboles disponibles et prioriser BTCUSDT, ETHUSDT, SOLUSDT"""
        logger.info("\n🔍 TEST 1: Obtention des symboles disponibles")
        
        try:
            logger.info("   📞 Récupération des opportunités disponibles...")
            
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            
            if response.status_code == 200:
                opportunities = response.json()
                if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                    opportunities = opportunities['opportunities']
                
                # Extraire tous les symboles disponibles
                available_symbols = [opp.get('symbol') for opp in opportunities if opp.get('symbol')]
                logger.info(f"      ✅ {len(available_symbols)} symboles disponibles")
                
                # Prioriser les symboles demandés
                selected_symbols = []
                
                # Ajouter les symboles prioritaires s'ils sont disponibles
                for priority_symbol in self.priority_symbols:
                    if priority_symbol in available_symbols:
                        selected_symbols.append(priority_symbol)
                        logger.info(f"      ✅ Symbole prioritaire trouvé: {priority_symbol}")
                
                # Compléter avec d'autres symboles si nécessaire
                for symbol in available_symbols:
                    if symbol not in selected_symbols and len(selected_symbols) < 5:
                        selected_symbols.append(symbol)
                
                self.test_symbols = selected_symbols[:5]  # Limiter à 5 symboles max
                
                logger.info(f"      📊 Symboles sélectionnés pour test RR: {self.test_symbols}")
                
                # Vérifier si on a au moins les symboles prioritaires
                priority_found = sum(1 for symbol in self.priority_symbols if symbol in self.test_symbols)
                
                self.log_test_result("Obtention Symboles Disponibles", True, 
                                   f"{len(self.test_symbols)} symboles sélectionnés, {priority_found}/3 symboles prioritaires trouvés")
                
            else:
                logger.error(f"      ❌ Erreur récupération opportunités: HTTP {response.status_code}")
                # Fallback vers symboles par défaut
                self.test_symbols = self.priority_symbols
                self.log_test_result("Obtention Symboles Disponibles", False, 
                                   f"Utilisation symboles par défaut: {self.test_symbols}")
                
        except Exception as e:
            logger.error(f"      ❌ Exception récupération symboles: {e}")
            self.test_symbols = self.priority_symbols
            self.log_test_result("Obtention Symboles Disponibles", False, f"Exception: {str(e)}")
    
    async def test_2_rr_calculation_validation(self):
        """Test 2: Validation approfondie des calculs RR pour chaque symbole"""
        logger.info("\n🔍 TEST 2: Validation Calculs Risk-Reward")
        
        rr_validation_results = {
            'analyses_attempted': 0,
            'analyses_successful': 0,
            'rr_calculations_valid': 0,
            'price_logic_valid': 0,
            'manual_calculations_successful': 0,
            'consistency_checks_passed': 0,
            'suspicious_rr_values': 0,
            'division_by_zero_errors': 0,
            'detailed_analyses': []
        }
        
        try:
            logger.info(f"   🚀 Test calculs RR pour {len(self.test_symbols)} symboles...")
            
            for symbol in self.test_symbols:
                logger.info(f"\n   📞 Analyse RR pour {symbol}...")
                rr_validation_results['analyses_attempted'] += 1
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        rr_validation_results['analyses_successful'] += 1
                        
                        logger.info(f"      ✅ {symbol} analyse réussie ({response_time:.2f}s)")
                        
                        # Extraire les données IA1
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        if not isinstance(ia1_analysis, dict):
                            ia1_analysis = {}
                        
                        # Extraire les prix et RR
                        entry_price = ia1_analysis.get('entry_price', 0.0)
                        stop_loss_price = ia1_analysis.get('stop_loss_price', 0.0)
                        take_profit_price = ia1_analysis.get('take_profit_price', 0.0)
                        
                        # Essayer différents champs pour RR
                        api_rr = (ia1_analysis.get('calculated_rr') or 
                                 ia1_analysis.get('risk_reward_ratio') or 
                                 ia1_analysis.get('rr_ratio') or 0.0)
                        
                        signal = ia1_analysis.get('ia1_signal', 'hold').upper()
                        current_price = ia1_analysis.get('current_price', 0.0)
                        
                        logger.info(f"         📊 Données extraites:")
                        logger.info(f"            Entry: {entry_price}, SL: {stop_loss_price}, TP: {take_profit_price}")
                        logger.info(f"            API RR: {api_rr}, Signal: {signal}, Prix actuel: {current_price}")
                        
                        # Validation de la logique des prix
                        price_validation = self._validate_price_logic(entry_price, stop_loss_price, take_profit_price, signal)
                        
                        if price_validation['price_logic_valid']:
                            rr_validation_results['price_logic_valid'] += 1
                            logger.info(f"         ✅ Logique prix valide pour {signal}")
                        else:
                            logger.warning(f"         ⚠️ Logique prix invalide: {price_validation['price_logic_errors']}")
                        
                        # Calcul manuel du RR
                        manual_calculation = self._calculate_manual_rr(entry_price, stop_loss_price, take_profit_price, signal)
                        
                        if manual_calculation['calculation_valid']:
                            rr_validation_results['manual_calculations_successful'] += 1
                            manual_rr = manual_calculation['manual_rr']
                            logger.info(f"         ✅ Calcul manuel RR: {manual_rr:.3f} ({manual_calculation['formula_used']})")
                            logger.info(f"            Risk: {manual_calculation['risk']:.6f}, Reward: {manual_calculation['reward']:.6f}")
                        else:
                            logger.error(f"         ❌ Calcul manuel échoué: {manual_calculation['calculation_error']}")
                            manual_rr = 0.0
                            
                            # Vérifier si c'est une division par zéro
                            if "Division par zéro" in str(manual_calculation['calculation_error']):
                                rr_validation_results['division_by_zero_errors'] += 1
                        
                        # Analyse de cohérence API vs Manuel
                        consistency_analysis = self._analyze_rr_consistency(api_rr, manual_rr)
                        
                        if consistency_analysis['consistent']:
                            rr_validation_results['consistency_checks_passed'] += 1
                            logger.info(f"         ✅ Cohérence RR validée")
                        else:
                            logger.warning(f"         ⚠️ Incohérence RR détectée")
                        
                        for note in consistency_analysis['analysis_notes']:
                            logger.info(f"            {note}")
                        
                        # Vérification des valeurs suspectes
                        if (api_rr > self.rr_validation_thresholds['suspicious_rr'] or 
                            manual_rr > self.rr_validation_thresholds['suspicious_rr'] or
                            api_rr < self.rr_validation_thresholds['negative_rr_tolerance']):
                            rr_validation_results['suspicious_rr_values'] += 1
                            logger.warning(f"         🚨 Valeur RR suspecte détectée: API={api_rr:.3f}, Manuel={manual_rr:.3f}")
                        
                        # Validation globale du RR
                        rr_valid = (
                            price_validation['price_logic_valid'] and
                            manual_calculation['calculation_valid'] and
                            consistency_analysis['consistent'] and
                            self.rr_validation_thresholds['min_valid_rr'] <= api_rr <= self.rr_validation_thresholds['max_valid_rr']
                        )
                        
                        if rr_valid:
                            rr_validation_results['rr_calculations_valid'] += 1
                        
                        # Stocker l'analyse détaillée
                        detailed_analysis = {
                            'symbol': symbol,
                            'response_time': response_time,
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss_price,
                            'take_profit_price': take_profit_price,
                            'current_price': current_price,
                            'signal': signal,
                            'api_rr': api_rr,
                            'manual_rr': manual_rr,
                            'price_logic_valid': price_validation['price_logic_valid'],
                            'price_logic_errors': price_validation['price_logic_errors'],
                            'manual_calculation_valid': manual_calculation['calculation_valid'],
                            'manual_calculation_error': manual_calculation['calculation_error'],
                            'consistency_check_passed': consistency_analysis['consistent'],
                            'consistency_difference': consistency_analysis['difference'],
                            'rr_valid': rr_valid,
                            'suspicious_rr': api_rr > self.rr_validation_thresholds['suspicious_rr'] or manual_rr > self.rr_validation_thresholds['suspicious_rr'],
                            'formula_used': manual_calculation['formula_used'],
                            'risk': manual_calculation['risk'],
                            'reward': manual_calculation['reward']
                        }
                        
                        rr_validation_results['detailed_analyses'].append(detailed_analysis)
                        self.rr_analyses.append(detailed_analysis)
                        
                    else:
                        logger.error(f"      ❌ {symbol} analyse échouée: HTTP {response.status_code}")
                        if response.text:
                            error_text = response.text[:300]
                            logger.error(f"         Erreur: {error_text}")
                            self.error_logs.append(f"{symbol}: {error_text}")
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} exception: {e}")
                    self.error_logs.append(f"{symbol}: {str(e)}")
                
                # Attendre entre les analyses
                if symbol != self.test_symbols[-1]:
                    logger.info(f"      ⏳ Attente 10 secondes...")
                    await asyncio.sleep(10)
            
            # Analyse finale des résultats
            success_rate = rr_validation_results['analyses_successful'] / max(rr_validation_results['analyses_attempted'], 1)
            rr_valid_rate = rr_validation_results['rr_calculations_valid'] / max(rr_validation_results['analyses_successful'], 1)
            consistency_rate = rr_validation_results['consistency_checks_passed'] / max(rr_validation_results['analyses_successful'], 1)
            
            logger.info(f"\n   📊 RÉSULTATS VALIDATION CALCULS RR:")
            logger.info(f"      Analyses tentées: {rr_validation_results['analyses_attempted']}")
            logger.info(f"      Analyses réussies: {rr_validation_results['analyses_successful']} ({success_rate:.2f})")
            logger.info(f"      Calculs RR valides: {rr_validation_results['rr_calculations_valid']} ({rr_valid_rate:.2f})")
            logger.info(f"      Logique prix valide: {rr_validation_results['price_logic_valid']}")
            logger.info(f"      Calculs manuels réussis: {rr_validation_results['manual_calculations_successful']}")
            logger.info(f"      Cohérence API/Manuel: {rr_validation_results['consistency_checks_passed']} ({consistency_rate:.2f})")
            logger.info(f"      Valeurs RR suspectes: {rr_validation_results['suspicious_rr_values']}")
            logger.info(f"      Erreurs division par zéro: {rr_validation_results['division_by_zero_errors']}")
            
            # Critères de succès
            success_criteria = [
                rr_validation_results['analyses_successful'] >= 2,  # Au moins 2 analyses réussies
                success_rate >= 0.6,  # Au moins 60% de succès
                rr_validation_results['rr_calculations_valid'] >= 1,  # Au moins 1 calcul RR valide
                rr_validation_results['division_by_zero_errors'] == 0,  # Aucune division par zéro
                rr_validation_results['suspicious_rr_values'] <= 1  # Maximum 1 valeur suspecte
            ]
            
            success_count = sum(success_criteria)
            test_success = success_count >= 4  # 4/5 critères minimum
            
            if test_success:
                self.log_test_result("Validation Calculs RR", True, 
                                   f"Validation réussie: {success_count}/5 critères. Taux succès: {success_rate:.2f}, Taux validité RR: {rr_valid_rate:.2f}, Cohérence: {consistency_rate:.2f}")
            else:
                self.log_test_result("Validation Calculs RR", False, 
                                   f"Validation échouée: {success_count}/5 critères. Problèmes détectés dans les calculs RR")
                
        except Exception as e:
            self.log_test_result("Validation Calculs RR", False, f"Exception: {str(e)}")
    
    async def test_3_backend_error_analysis(self):
        """Test 3: Analyse des erreurs backend liées aux calculs RR"""
        logger.info("\n🔍 TEST 3: Analyse Erreurs Backend RR")
        
        try:
            logger.info("   📋 Capture et analyse des logs backend...")
            
            backend_logs = await self._capture_backend_logs()
            
            error_analysis = {
                'total_log_lines': len(backend_logs),
                'rr_related_errors': 0,
                'division_by_zero_errors': 0,
                'risk_reward_calculator_errors': 0,
                'optimal_rr_setup_errors': 0,
                'pydantic_validation_errors': 0,
                'error_samples': [],
                'rr_calculation_logs': [],
                'success_logs': []
            }
            
            # Patterns d'erreurs à rechercher
            error_patterns = {
                'division_by_zero': [
                    'division by zero',
                    'float division by zero',
                    'ZeroDivisionError'
                ],
                'rr_calculator': [
                    'ERROR:risk_reward_calculator',
                    'Error in optimal_rr_setup',
                    'risk_reward_calculator.*error'
                ],
                'rr_calculation': [
                    'RR calculation',
                    'risk_reward_ratio',
                    'calculated_rr'
                ],
                'pydantic_validation': [
                    'validation error',
                    'pydantic',
                    'fibonacci_key_level_proximity'
                ]
            }
            
            for log_line in backend_logs:
                log_lower = log_line.lower()
                
                # Recherche d'erreurs de division par zéro
                if any(pattern in log_lower for pattern in error_patterns['division_by_zero']):
                    error_analysis['division_by_zero_errors'] += 1
                    error_analysis['error_samples'].append(f"DIVISION_BY_ZERO: {log_line.strip()}")
                
                # Recherche d'erreurs risk_reward_calculator
                if any(pattern in log_lower for pattern in error_patterns['rr_calculator']):
                    error_analysis['risk_reward_calculator_errors'] += 1
                    error_analysis['error_samples'].append(f"RR_CALCULATOR: {log_line.strip()}")
                
                # Recherche d'erreurs optimal_rr_setup spécifiques
                if 'optimal_rr_setup' in log_lower and 'error' in log_lower:
                    error_analysis['optimal_rr_setup_errors'] += 1
                    error_analysis['error_samples'].append(f"OPTIMAL_RR_SETUP: {log_line.strip()}")
                
                # Recherche d'erreurs Pydantic
                if any(pattern in log_lower for pattern in error_patterns['pydantic_validation']):
                    error_analysis['pydantic_validation_errors'] += 1
                    error_analysis['error_samples'].append(f"PYDANTIC: {log_line.strip()}")
                
                # Recherche de logs liés aux calculs RR (succès et échecs)
                if any(pattern in log_lower for pattern in error_patterns['rr_calculation']):
                    if 'error' in log_lower or 'failed' in log_lower:
                        error_analysis['rr_related_errors'] += 1
                        error_analysis['rr_calculation_logs'].append(f"ERROR: {log_line.strip()}")
                    else:
                        error_analysis['success_logs'].append(f"SUCCESS: {log_line.strip()}")
            
            logger.info(f"      📊 Analyse de {error_analysis['total_log_lines']} lignes de logs:")
            logger.info(f"         - Erreurs RR générales: {error_analysis['rr_related_errors']}")
            logger.info(f"         - Erreurs division par zéro: {error_analysis['division_by_zero_errors']}")
            logger.info(f"         - Erreurs risk_reward_calculator: {error_analysis['risk_reward_calculator_errors']}")
            logger.info(f"         - Erreurs optimal_rr_setup: {error_analysis['optimal_rr_setup_errors']}")
            logger.info(f"         - Erreurs Pydantic: {error_analysis['pydantic_validation_errors']}")
            logger.info(f"         - Logs succès RR: {len(error_analysis['success_logs'])}")
            
            # Afficher des échantillons d'erreurs
            if error_analysis['error_samples']:
                logger.info(f"      🚨 Échantillons d'erreurs détectées:")
                for i, error_sample in enumerate(error_analysis['error_samples'][:5]):  # Max 5 échantillons
                    logger.info(f"         {i+1}. {error_sample}")
            
            # Afficher des logs de succès RR si disponibles
            if error_analysis['success_logs']:
                logger.info(f"      ✅ Échantillons de calculs RR réussis:")
                for i, success_log in enumerate(error_analysis['success_logs'][:3]):  # Max 3 échantillons
                    logger.info(f"         {i+1}. {success_log}")
            
            # Recherche spécifique de l'erreur mentionnée dans la demande
            specific_error_found = False
            for log_line in backend_logs:
                if "ERROR:risk_reward_calculator:Error in optimal_rr_setup: float division by zero" in log_line:
                    specific_error_found = True
                    logger.error(f"      🎯 ERREUR SPÉCIFIQUE TROUVÉE: {log_line.strip()}")
                    break
            
            if specific_error_found:
                logger.error(f"      🚨 L'erreur spécifique mentionnée dans la demande a été trouvée!")
            else:
                logger.info(f"      ✅ L'erreur spécifique 'ERROR:risk_reward_calculator:Error in optimal_rr_setup: float division by zero' n'a pas été trouvée")
            
            # Évaluation du succès du test
            total_errors = (error_analysis['division_by_zero_errors'] + 
                          error_analysis['risk_reward_calculator_errors'] + 
                          error_analysis['optimal_rr_setup_errors'])
            
            success_criteria = [
                error_analysis['total_log_lines'] > 0,  # Logs capturés
                error_analysis['division_by_zero_errors'] == 0,  # Pas d'erreurs division par zéro
                error_analysis['optimal_rr_setup_errors'] == 0,  # Pas d'erreurs optimal_rr_setup
                not specific_error_found,  # Erreur spécifique pas trouvée
                len(error_analysis['success_logs']) > 0  # Au moins quelques succès RR
            ]
            
            success_count = sum(success_criteria)
            test_success = success_count >= 4  # 4/5 critères minimum
            
            if test_success:
                self.log_test_result("Analyse Erreurs Backend RR", True, 
                                   f"Analyse réussie: {success_count}/5 critères. Total erreurs RR: {total_errors}, Succès RR: {len(error_analysis['success_logs'])}")
            else:
                self.log_test_result("Analyse Erreurs Backend RR", False, 
                                   f"Problèmes détectés: {success_count}/5 critères. Erreurs division par zéro: {error_analysis['division_by_zero_errors']}, Erreurs RR: {total_errors}")
            
            # Stocker les résultats pour le rapport final
            self.error_logs.extend(error_analysis['error_samples'])
            
        except Exception as e:
            self.log_test_result("Analyse Erreurs Backend RR", False, f"Exception: {str(e)}")
    
    async def test_4_database_rr_consistency(self):
        """Test 4: Vérification de la cohérence des valeurs RR en base de données"""
        logger.info("\n🔍 TEST 4: Cohérence RR Base de Données")
        
        try:
            logger.info("   🗄️ Connexion à MongoDB et analyse des valeurs RR stockées...")
            
            # Connexion à MongoDB
            client = MongoClient(self.mongo_url)
            db = client[self.db_name]
            
            db_analysis = {
                'connection_successful': True,
                'total_analyses': 0,
                'analyses_with_rr': 0,
                'rr_field_variations': {},
                'rr_value_distribution': {},
                'suspicious_rr_count': 0,
                'zero_rr_count': 0,
                'negative_rr_count': 0,
                'default_rr_count': 0,
                'sample_analyses': []
            }
            
            # Récupérer les analyses récentes
            recent_analyses = list(db.technical_analyses.find().sort("timestamp", -1).limit(50))
            db_analysis['total_analyses'] = len(recent_analyses)
            
            logger.info(f"      📊 Analyse de {len(recent_analyses)} analyses récentes...")
            
            rr_values = []
            rr_field_names = set()
            
            for i, analysis in enumerate(recent_analyses):
                symbol = analysis.get('symbol', 'UNKNOWN')
                timestamp = analysis.get('timestamp', 'UNKNOWN')
                
                # Chercher différents champs RR possibles
                rr_fields = {
                    'calculated_rr': analysis.get('calculated_rr'),
                    'risk_reward_ratio': analysis.get('risk_reward_ratio'),
                    'rr_ratio': analysis.get('rr_ratio'),
                    'risk_reward': analysis.get('risk_reward')
                }
                
                # Identifier quels champs RR sont présents
                present_rr_fields = {k: v for k, v in rr_fields.items() if v is not None}
                
                if present_rr_fields:
                    db_analysis['analyses_with_rr'] += 1
                    
                    # Enregistrer les noms de champs utilisés
                    for field_name in present_rr_fields.keys():
                        rr_field_names.add(field_name)
                        if field_name in db_analysis['rr_field_variations']:
                            db_analysis['rr_field_variations'][field_name] += 1
                        else:
                            db_analysis['rr_field_variations'][field_name] = 1
                    
                    # Analyser les valeurs RR
                    for field_name, rr_value in present_rr_fields.items():
                        if isinstance(rr_value, (int, float)):
                            rr_values.append(rr_value)
                            
                            # Catégoriser les valeurs RR
                            if rr_value == 0.0:
                                db_analysis['zero_rr_count'] += 1
                            elif rr_value < 0:
                                db_analysis['negative_rr_count'] += 1
                            elif rr_value == 1.0:
                                db_analysis['default_rr_count'] += 1
                            elif rr_value > self.rr_validation_thresholds['suspicious_rr']:
                                db_analysis['suspicious_rr_count'] += 1
                
                # Stocker des échantillons pour inspection détaillée
                if i < 10:  # Premiers 10 échantillons
                    sample = {
                        'symbol': symbol,
                        'timestamp': str(timestamp),
                        'rr_fields': present_rr_fields,
                        'entry_price': analysis.get('entry_price'),
                        'stop_loss_price': analysis.get('stop_loss_price'),
                        'take_profit_price': analysis.get('take_profit_price'),
                        'signal': analysis.get('ia1_signal')
                    }
                    db_analysis['sample_analyses'].append(sample)
            
            # Analyse statistique des valeurs RR
            if rr_values:
                rr_stats = {
                    'count': len(rr_values),
                    'min': min(rr_values),
                    'max': max(rr_values),
                    'avg': sum(rr_values) / len(rr_values),
                    'unique_values': len(set(rr_values))
                }
                
                # Distribution des valeurs RR
                rr_ranges = {
                    '0.0': sum(1 for v in rr_values if v == 0.0),
                    '0.0-1.0': sum(1 for v in rr_values if 0.0 < v < 1.0),
                    '1.0': sum(1 for v in rr_values if v == 1.0),
                    '1.0-2.0': sum(1 for v in rr_values if 1.0 < v < 2.0),
                    '2.0-5.0': sum(1 for v in rr_values if 2.0 <= v < 5.0),
                    '5.0+': sum(1 for v in rr_values if v >= 5.0)
                }
                
                db_analysis['rr_value_distribution'] = rr_ranges
                
                logger.info(f"      📊 Statistiques RR:")
                logger.info(f"         - Nombre de valeurs: {rr_stats['count']}")
                logger.info(f"         - Min/Max: {rr_stats['min']:.3f} / {rr_stats['max']:.3f}")
                logger.info(f"         - Moyenne: {rr_stats['avg']:.3f}")
                logger.info(f"         - Valeurs uniques: {rr_stats['unique_values']}")
                
                logger.info(f"      📊 Distribution RR:")
                for range_name, count in rr_ranges.items():
                    percentage = (count / len(rr_values)) * 100
                    logger.info(f"         - {range_name}: {count} ({percentage:.1f}%)")
            
            logger.info(f"      📊 Analyse champs RR:")
            logger.info(f"         - Analyses avec RR: {db_analysis['analyses_with_rr']}/{db_analysis['total_analyses']}")
            logger.info(f"         - Champs RR utilisés: {list(rr_field_names)}")
            logger.info(f"         - Variations champs: {db_analysis['rr_field_variations']}")
            
            logger.info(f"      🚨 Valeurs problématiques:")
            logger.info(f"         - RR = 0.0: {db_analysis['zero_rr_count']}")
            logger.info(f"         - RR négatif: {db_analysis['negative_rr_count']}")
            logger.info(f"         - RR = 1.0 (défaut?): {db_analysis['default_rr_count']}")
            logger.info(f"         - RR suspect (>10): {db_analysis['suspicious_rr_count']}")
            
            # Afficher des échantillons
            if db_analysis['sample_analyses']:
                logger.info(f"      📋 Échantillons d'analyses:")
                for i, sample in enumerate(db_analysis['sample_analyses'][:5]):
                    logger.info(f"         {i+1}. {sample['symbol']}: RR={sample['rr_fields']}, Signal={sample['signal']}")
            
            client.close()
            
            # Évaluation du succès
            rr_coverage = db_analysis['analyses_with_rr'] / max(db_analysis['total_analyses'], 1)
            problematic_ratio = (db_analysis['zero_rr_count'] + db_analysis['negative_rr_count'] + db_analysis['suspicious_rr_count']) / max(len(rr_values), 1)
            
            success_criteria = [
                db_analysis['connection_successful'],
                db_analysis['total_analyses'] >= 10,  # Au moins 10 analyses
                rr_coverage >= 0.5,  # Au moins 50% ont des valeurs RR
                len(rr_field_names) > 0,  # Au moins un champ RR utilisé
                problematic_ratio <= 0.3  # Maximum 30% de valeurs problématiques
            ]
            
            success_count = sum(success_criteria)
            test_success = success_count >= 4  # 4/5 critères minimum
            
            if test_success:
                self.log_test_result("Cohérence RR Base de Données", True, 
                                   f"Cohérence validée: {success_count}/5 critères. Couverture RR: {rr_coverage:.2f}, Ratio problématique: {problematic_ratio:.2f}")
            else:
                self.log_test_result("Cohérence RR Base de Données", False, 
                                   f"Problèmes détectés: {success_count}/5 critères. Valeurs RR incohérentes ou manquantes")
            
        except Exception as e:
            self.log_test_result("Cohérence RR Base de Données", False, f"Exception: {str(e)}")
    
    async def generate_final_report(self):
        """Générer le rapport final de diagnostic RR"""
        logger.info("\n📋 RAPPORT FINAL - DIAGNOSTIC CALCULS RISK-REWARD IA1")
        
        # Résumé des tests
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        
        logger.info(f"\n🎯 RÉSUMÉ EXÉCUTIF:")
        logger.info(f"   Tests exécutés: {total_tests}")
        logger.info(f"   Tests réussis: {passed_tests}")
        logger.info(f"   Taux de succès: {(passed_tests/max(total_tests,1)):.2f}")
        
        # Détail des résultats par test
        logger.info(f"\n📊 DÉTAIL DES TESTS:")
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            logger.info(f"   {status} {result['test']}")
            if result['details']:
                logger.info(f"      {result['details']}")
        
        # Analyse des calculs RR
        if self.rr_analyses:
            logger.info(f"\n🔍 ANALYSE DÉTAILLÉE DES CALCULS RR:")
            logger.info(f"   Symboles analysés: {len(self.rr_analyses)}")
            
            valid_rr_count = sum(1 for analysis in self.rr_analyses if analysis['rr_valid'])
            consistent_count = sum(1 for analysis in self.rr_analyses if analysis['consistency_check_passed'])
            suspicious_count = sum(1 for analysis in self.rr_analyses if analysis['suspicious_rr'])
            
            logger.info(f"   Calculs RR valides: {valid_rr_count}/{len(self.rr_analyses)}")
            logger.info(f"   Cohérence API/Manuel: {consistent_count}/{len(self.rr_analyses)}")
            logger.info(f"   Valeurs suspectes: {suspicious_count}")
            
            # Exemples de calculs
            logger.info(f"\n   📋 EXEMPLES DE CALCULS:")
            for analysis in self.rr_analyses[:3]:  # Premiers 3 exemples
                logger.info(f"      {analysis['symbol']} ({analysis['signal']}):")
                logger.info(f"         Entry: {analysis['entry_price']:.6f}, SL: {analysis['stop_loss_price']:.6f}, TP: {analysis['take_profit_price']:.6f}")
                logger.info(f"         API RR: {analysis['api_rr']:.3f}, Manuel RR: {analysis['manual_rr']:.3f}")
                logger.info(f"         Formule: {analysis['formula_used']}")
                logger.info(f"         Risk: {analysis['risk']:.6f}, Reward: {analysis['reward']:.6f}")
                logger.info(f"         Valide: {analysis['rr_valid']}, Cohérent: {analysis['consistency_check_passed']}")
        
        # Erreurs détectées
        if self.error_logs:
            logger.info(f"\n🚨 ERREURS DÉTECTÉES:")
            for i, error in enumerate(self.error_logs[:5]):  # Max 5 erreurs
                logger.info(f"   {i+1}. {error}")
        
        # Recommandations
        logger.info(f"\n💡 RECOMMANDATIONS:")
        
        if passed_tests == total_tests:
            logger.info(f"   ✅ Système de calcul RR fonctionne correctement")
            logger.info(f"   ✅ Aucun problème majeur détecté")
        else:
            logger.info(f"   ⚠️ Problèmes détectés nécessitant attention:")
            
            # Recommandations spécifiques basées sur les résultats
            failed_tests = [result for result in self.test_results if not result['success']]
            for failed_test in failed_tests:
                if "Division par zéro" in failed_test['details']:
                    logger.info(f"      - Corriger les erreurs de division par zéro dans risk_reward_calculator")
                elif "Incohérence" in failed_test['details']:
                    logger.info(f"      - Vérifier la cohérence entre calculs API et formules manuelles")
                elif "Valeurs suspectes" in failed_test['details']:
                    logger.info(f"      - Investiguer les valeurs RR aberrantes (>10 ou négatives)")
        
        logger.info(f"\n🎯 CONCLUSION:")
        if passed_tests >= total_tests * 0.8:  # 80% de succès
            logger.info(f"   ✅ Le système de calcul Risk-Reward IA1 est globalement fonctionnel")
        else:
            logger.info(f"   ❌ Le système de calcul Risk-Reward IA1 présente des dysfonctionnements significatifs")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / max(total_tests, 1),
            'rr_analyses': self.rr_analyses,
            'error_logs': self.error_logs,
            'test_results': self.test_results
        }

async def main():
    """Fonction principale pour exécuter le diagnostic RR"""
    logger.info("🚀 DÉMARRAGE DIAGNOSTIC RISK-REWARD IA1")
    
    # Initialiser la suite de tests
    test_suite = RRCalculationDiagnosticSuite()
    
    try:
        # Exécuter les tests dans l'ordre
        await test_suite.test_1_get_available_symbols()
        await test_suite.test_2_rr_calculation_validation()
        await test_suite.test_3_backend_error_analysis()
        await test_suite.test_4_database_rr_consistency()
        
        # Générer le rapport final
        final_report = await test_suite.generate_final_report()
        
        logger.info("✅ DIAGNOSTIC TERMINÉ")
        return final_report
        
    except Exception as e:
        logger.error(f"❌ ERREUR CRITIQUE DIAGNOSTIC: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())
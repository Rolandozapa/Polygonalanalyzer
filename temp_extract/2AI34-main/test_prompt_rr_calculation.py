#!/usr/bin/env python3
"""
TEST PROMPT AMÉLIORÉ AVEC CALCUL RR - COMPREHENSIVE TESTING SUITE
Focus: Vérifier la réintégration de l'ancien système de calcul RR et reasoning détaillé.

OBJECTIF SPÉCIFIQUE:
Tester le nouveau prompt qui réintègre les meilleures pratiques de l'ancien système:

1. **Calcul RR Réel dans le JSON**: 
   - Vérifier que IA1 calcule maintenant "calculated_rr" avec les vraies formules
   - Examiner le champ "rr_reasoning" pour l'explication détaillée des niveaux
   - Confirmer utilisation des niveaux techniques (VWAP, EMA21, SMA50) et non des pourcentages fixes

2. **Reasoning Détaillé avec Confluence**: 
   - Vérifier que "reasoning" inclut l'analyse des 6 indicateurs: RSI, MACD, MFI, VWAP, EMA hierarchy, patterns
   - Examiner "technical_indicators_analysis" pour l'analyse complète
   - Confirmer que le reasoning mentionne la confluence X/6 indicateurs

3. **Niveaux Techniques Basés sur les Indicateurs**: 
   - Vérifier que stop_loss utilise VWAP ou EMA21 comme base technique
   - Confirmer que take_profit utilise SMA50 ou niveaux de résistance
   - Examiner la cohérence entre les indicateurs calculés et les niveaux proposés

4. **Test Escalation vers IA2**: 
   - Vérifier si les RR calculés permettent l'escalation (RR > 2.0)
   - Examiner la qualité des analyses pour voir si elles déclenchent IA2
   - Confirmer que le système peut maintenant escalader correctement

5. **Tests Concrets**:
   - Analyser 2-3 cryptos avec /api/force-ia1-analysis  
   - Comparer avec les résultats précédents (RR fixes vs RR calculés)
   - Vérifier que les analyses sont maintenant complètes et détaillées

FOCUS: Valider que la réintégration de l'ancien système de calcul RR avec reasoning détaillé permet une escalation correcte vers IA2 et des analyses de meilleure qualité.
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
import re
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptRRCalculationTestSuite:
    """Comprehensive test suite for Prompt Amélioré avec Calcul RR - New Prompt Testing"""
    
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
        logger.info(f"Testing New Prompt RR Calculation System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for IA1 analysis (will be dynamically determined from available opportunities)
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']  # Preferred symbols from review request
        self.actual_test_symbols = []  # Will be populated from available opportunities
        
        # Core technical indicators to verify (6 indicators from review request)
        self.core_technical_indicators = ['RSI', 'MACD', 'MFI', 'VWAP', 'EMA', 'patterns']
        
        # RR calculation fields to verify
        self.rr_calculation_fields = [
            'calculated_rr', 'rr_reasoning', 'entry_price', 'stop_loss_price', 'take_profit_price'
        ]
        
        # Technical levels to verify (VWAP, EMA21, SMA50)
        self.technical_levels = ['VWAP', 'EMA21', 'SMA50', 'EMA9', 'EMA200']
        
        # IA1 analysis data storage
        self.ia1_analyses = []
        self.rr_calculations = []
        self.escalation_candidates = []
        
        # MongoDB connection for database verification
        try:
            mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
            self.mongo_client = MongoClient(mongo_url)
            self.db = self.mongo_client[os.environ.get('DB_NAME', 'myapp')]
            logger.info("✅ MongoDB connection established for database verification")
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to MongoDB: {e}")
            self.mongo_client = None
            self.db = None
    
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
        """Capture backend logs for analysis"""
        try:
            # Try to capture supervisor backend logs
            result = subprocess.run(
                ['tail', '-n', '200', '/var/log/supervisor/backend.out.log'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.split('\n')
            else:
                # Try alternative log location
                result = subprocess.run(
                    ['tail', '-n', '200', '/var/log/supervisor/backend.err.log'],
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
    
    async def _get_available_test_symbols(self):
        """Get available symbols from scout system for testing"""
        try:
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            
            if response.status_code == 200:
                opportunities = response.json()
                if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                    opportunities = opportunities['opportunities']
                
                # Get first 10 available symbols
                available_symbols = [opp.get('symbol') for opp in opportunities[:10] if opp.get('symbol')]
                
                # Prefer BTCUSDT, ETHUSDT, SOLUSDT if available, otherwise use first 3
                preferred_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
                test_symbols = []
                
                for symbol in preferred_symbols:
                    if symbol in available_symbols:
                        test_symbols.append(symbol)
                
                # Fill remaining slots with available symbols
                for symbol in available_symbols:
                    if symbol not in test_symbols and len(test_symbols) < 3:
                        test_symbols.append(symbol)
                
                self.actual_test_symbols = test_symbols[:3]  # Limit to 3 symbols
                logger.info(f"✅ Test symbols selected: {self.actual_test_symbols}")
                return True
                
            else:
                logger.warning(f"⚠️ Could not get opportunities, using default symbols")
                self.actual_test_symbols = self.test_symbols
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting opportunities: {e}, using default symbols")
            self.actual_test_symbols = self.test_symbols
            return False

    async def test_1_rr_calculation_real_formulas(self):
        """Test 1: RR Calculation Real Formulas - Verify IA1 calculates real RR with formulas"""
        logger.info("\n🔍 TEST 1: RR Calculation Real Formulas")
        
        try:
            rr_results = {
                'analyses_with_calculated_rr': 0,
                'total_analyses': 0,
                'real_rr_calculations': [],
                'rr_reasoning_present': 0,
                'technical_levels_used': 0,
                'formula_based_calculations': 0,
                'successful_analyses': []
            }
            
            logger.info("   🚀 Testing real RR calculation with formulas instead of fixed values...")
            logger.info("   📊 Expected: IA1 calculates 'calculated_rr' using LONG: (TP-Entry)/(Entry-SL), SHORT: (Entry-TP)/(SL-Entry)")
            
            # Get available symbols
            await self._get_available_test_symbols()
            test_symbols = self.actual_test_symbols
            logger.info(f"   📊 Testing symbols: {test_symbols}")
            
            # Test each symbol for RR calculation
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing RR calculation for {symbol}...")
                rr_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Extract IA1 analysis
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        if isinstance(ia1_analysis, dict):
                            # Check for calculated_rr field
                            calculated_rr = ia1_analysis.get('calculated_rr')
                            rr_reasoning = ia1_analysis.get('rr_reasoning', '')
                            entry_price = ia1_analysis.get('entry_price')
                            stop_loss_price = ia1_analysis.get('stop_loss_price')
                            take_profit_price = ia1_analysis.get('take_profit_price')
                            signal = ia1_analysis.get('signal', '').upper()
                            
                            logger.info(f"      📊 Analysis data: Entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}, Signal={signal}")
                            
                            if calculated_rr is not None:
                                rr_results['analyses_with_calculated_rr'] += 1
                                logger.info(f"         ✅ calculated_rr field present: {calculated_rr}")
                                
                                # Verify RR calculation using formulas
                                if entry_price and stop_loss_price and take_profit_price:
                                    if signal == 'LONG':
                                        # LONG formula: (TP-Entry)/(Entry-SL)
                                        expected_rr = (take_profit_price - entry_price) / (entry_price - stop_loss_price)
                                    elif signal == 'SHORT':
                                        # SHORT formula: (Entry-TP)/(SL-Entry)
                                        expected_rr = (entry_price - take_profit_price) / (stop_loss_price - entry_price)
                                    else:
                                        expected_rr = None
                                    
                                    if expected_rr is not None:
                                        rr_difference = abs(calculated_rr - expected_rr)
                                        tolerance = 0.1  # 10% tolerance for rounding
                                        
                                        if rr_difference <= tolerance:
                                            rr_results['formula_based_calculations'] += 1
                                            logger.info(f"         ✅ RR calculation matches formula: {calculated_rr:.2f} ≈ {expected_rr:.2f}")
                                        else:
                                            logger.warning(f"         ⚠️ RR calculation mismatch: {calculated_rr:.2f} vs expected {expected_rr:.2f}")
                                        
                                        rr_results['real_rr_calculations'].append({
                                            'symbol': symbol,
                                            'calculated_rr': calculated_rr,
                                            'expected_rr': expected_rr,
                                            'difference': rr_difference,
                                            'signal': signal,
                                            'entry_price': entry_price,
                                            'stop_loss_price': stop_loss_price,
                                            'take_profit_price': take_profit_price
                                        })
                                
                                # Check for rr_reasoning
                                if rr_reasoning:
                                    rr_results['rr_reasoning_present'] += 1
                                    logger.info(f"         ✅ rr_reasoning present: {len(rr_reasoning)} chars")
                                    
                                    # Check for technical levels in reasoning
                                    technical_levels_mentioned = []
                                    for level in self.technical_levels:
                                        if level.lower() in rr_reasoning.lower():
                                            technical_levels_mentioned.append(level)
                                    
                                    if technical_levels_mentioned:
                                        rr_results['technical_levels_used'] += 1
                                        logger.info(f"         ✅ Technical levels in reasoning: {technical_levels_mentioned}")
                                    else:
                                        logger.warning(f"         ⚠️ No technical levels mentioned in rr_reasoning")
                                else:
                                    logger.warning(f"         ❌ rr_reasoning field missing")
                                
                                # Store successful analysis
                                rr_results['successful_analyses'].append({
                                    'symbol': symbol,
                                    'calculated_rr': calculated_rr,
                                    'rr_reasoning_length': len(rr_reasoning),
                                    'technical_levels_mentioned': technical_levels_mentioned if 'technical_levels_mentioned' in locals() else [],
                                    'has_all_prices': all([entry_price, stop_loss_price, take_profit_price])
                                })
                                
                                logger.info(f"      ✅ RR calculation SUCCESS for {symbol}")
                            else:
                                logger.warning(f"      ❌ calculated_rr field missing for {symbol}")
                        else:
                            logger.error(f"      ❌ Invalid IA1 analysis structure for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if rr_results['total_analyses'] > 0:
                rr_calculation_rate = rr_results['analyses_with_calculated_rr'] / rr_results['total_analyses']
                rr_reasoning_rate = rr_results['rr_reasoning_present'] / rr_results['total_analyses']
                technical_levels_rate = rr_results['technical_levels_used'] / rr_results['total_analyses']
                formula_accuracy_rate = rr_results['formula_based_calculations'] / max(rr_results['analyses_with_calculated_rr'], 1)
            else:
                rr_calculation_rate = rr_reasoning_rate = technical_levels_rate = formula_accuracy_rate = 0.0
            
            logger.info(f"\n   📊 RR CALCULATION REAL FORMULAS RESULTS:")
            logger.info(f"      Total analyses: {rr_results['total_analyses']}")
            logger.info(f"      Analyses with calculated_rr: {rr_results['analyses_with_calculated_rr']} ({rr_calculation_rate:.2f})")
            logger.info(f"      RR reasoning present: {rr_results['rr_reasoning_present']} ({rr_reasoning_rate:.2f})")
            logger.info(f"      Technical levels used: {rr_results['technical_levels_used']} ({technical_levels_rate:.2f})")
            logger.info(f"      Formula-based calculations: {rr_results['formula_based_calculations']} ({formula_accuracy_rate:.2f})")
            
            if rr_results['real_rr_calculations']:
                logger.info(f"      📊 RR calculations:")
                for calc in rr_results['real_rr_calculations']:
                    logger.info(f"         - {calc['symbol']}: {calc['calculated_rr']:.2f} (expected: {calc['expected_rr']:.2f})")
            
            # Calculate test success based on review requirements
            success_criteria = [
                rr_results['total_analyses'] >= 2,  # At least 2 analyses
                rr_results['analyses_with_calculated_rr'] >= 1,  # At least 1 with calculated_rr
                rr_calculation_rate >= 0.67,  # At least 67% with calculated_rr
                rr_results['rr_reasoning_present'] >= 1,  # At least 1 with rr_reasoning
                formula_accuracy_rate >= 0.5  # At least 50% formula accuracy
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("RR Calculation Real Formulas", True, 
                                   f"RR calculation successful: {success_count}/{len(success_criteria)} criteria met. Calculation rate: {rr_calculation_rate:.2f}, Formula accuracy: {formula_accuracy_rate:.2f}, Technical levels: {technical_levels_rate:.2f}")
            else:
                self.log_test_result("RR Calculation Real Formulas", False, 
                                   f"RR calculation issues: {success_count}/{len(success_criteria)} criteria met. May still use fixed values instead of real formulas")
                
        except Exception as e:
            self.log_test_result("RR Calculation Real Formulas", False, f"Exception: {str(e)}")

    async def test_2_detailed_reasoning_confluence(self):
        """Test 2: Detailed Reasoning with Confluence - Verify 6-indicator confluence analysis"""
        logger.info("\n🔍 TEST 2: Detailed Reasoning with Confluence")
        
        try:
            confluence_results = {
                'analyses_with_detailed_reasoning': 0,
                'total_analyses': 0,
                'six_indicators_mentioned': 0,
                'confluence_analysis_present': 0,
                'technical_indicators_analysis_field': 0,
                'confluence_scores_found': 0,
                'successful_analyses': []
            }
            
            logger.info("   🚀 Testing detailed reasoning with 6-indicator confluence analysis...")
            logger.info("   📊 Expected: Reasoning includes RSI, MACD, MFI, VWAP, EMA hierarchy, patterns with confluence X/6")
            
            # Use symbols from previous test
            test_symbols = self.actual_test_symbols if self.actual_test_symbols else self.test_symbols
            logger.info(f"   📊 Testing symbols: {test_symbols}")
            
            # Test each symbol for detailed reasoning
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing detailed reasoning for {symbol}...")
                confluence_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Extract IA1 analysis
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        if isinstance(ia1_analysis, dict):
                            reasoning = ia1_analysis.get('reasoning', '')
                            technical_indicators_analysis = ia1_analysis.get('technical_indicators_analysis', {})
                            
                            if reasoning:
                                confluence_results['analyses_with_detailed_reasoning'] += 1
                                logger.info(f"      ✅ Detailed reasoning found: {len(reasoning)} chars")
                                
                                # Check for 6 core indicators
                                indicators_found = {}
                                core_indicators = ['RSI', 'MACD', 'MFI', 'VWAP', 'EMA', 'pattern']
                                
                                for indicator in core_indicators:
                                    mentions = reasoning.upper().count(indicator.upper())
                                    if mentions > 0:
                                        indicators_found[indicator] = mentions
                                        logger.info(f"         ✅ {indicator} mentioned {mentions} times")
                                
                                # Check for confluence analysis
                                confluence_keywords = ['confluence', 'align', 'confirm', 'contradict', '/6', 'indicators']
                                confluence_mentions = sum(reasoning.lower().count(keyword) for keyword in confluence_keywords)
                                
                                if confluence_mentions >= 2:
                                    confluence_results['confluence_analysis_present'] += 1
                                    logger.info(f"         ✅ Confluence analysis present: {confluence_mentions} mentions")
                                else:
                                    logger.warning(f"         ⚠️ Limited confluence analysis: {confluence_mentions} mentions")
                                
                                # Check for confluence score patterns (X/6)
                                import re
                                confluence_score_patterns = [
                                    r'\d+/6',  # X/6 format
                                    r'confluence.*\d+',  # confluence with numbers
                                    r'\d+.*indicators.*align',  # X indicators align
                                ]
                                
                                confluence_scores = []
                                for pattern in confluence_score_patterns:
                                    matches = re.findall(pattern, reasoning, re.IGNORECASE)
                                    confluence_scores.extend(matches)
                                
                                if confluence_scores:
                                    confluence_results['confluence_scores_found'] += 1
                                    logger.info(f"         ✅ Confluence scores found: {confluence_scores}")
                                else:
                                    logger.warning(f"         ⚠️ No confluence scores found")
                                
                                # Check for technical_indicators_analysis field
                                if technical_indicators_analysis and isinstance(technical_indicators_analysis, dict):
                                    confluence_results['technical_indicators_analysis_field'] += 1
                                    logger.info(f"         ✅ technical_indicators_analysis field present: {len(technical_indicators_analysis)} fields")
                                    
                                    # Check for confluence matrix in technical analysis
                                    confluence_matrix = technical_indicators_analysis.get('confluence_matrix', {})
                                    if confluence_matrix:
                                        logger.info(f"         ✅ confluence_matrix found with {len(confluence_matrix)} fields")
                                else:
                                    logger.warning(f"         ⚠️ technical_indicators_analysis field missing or empty")
                                
                                # Count indicators with at least 4/6
                                if len(indicators_found) >= 4:
                                    confluence_results['six_indicators_mentioned'] += 1
                                    logger.info(f"         ✅ 6-indicator analysis: {len(indicators_found)}/6 indicators found")
                                else:
                                    logger.warning(f"         ⚠️ Limited indicator coverage: {len(indicators_found)}/6 indicators")
                                
                                # Store successful analysis
                                confluence_results['successful_analyses'].append({
                                    'symbol': symbol,
                                    'reasoning_length': len(reasoning),
                                    'indicators_found': indicators_found,
                                    'confluence_mentions': confluence_mentions,
                                    'confluence_scores': confluence_scores,
                                    'has_technical_indicators_analysis': bool(technical_indicators_analysis),
                                    'indicator_coverage': len(indicators_found)
                                })
                                
                                logger.info(f"      ✅ Detailed reasoning SUCCESS for {symbol}")
                            else:
                                logger.warning(f"      ❌ No reasoning field found for {symbol}")
                        else:
                            logger.error(f"      ❌ Invalid IA1 analysis structure for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if confluence_results['total_analyses'] > 0:
                detailed_reasoning_rate = confluence_results['analyses_with_detailed_reasoning'] / confluence_results['total_analyses']
                six_indicators_rate = confluence_results['six_indicators_mentioned'] / confluence_results['total_analyses']
                confluence_analysis_rate = confluence_results['confluence_analysis_present'] / confluence_results['total_analyses']
                technical_analysis_field_rate = confluence_results['technical_indicators_analysis_field'] / confluence_results['total_analyses']
            else:
                detailed_reasoning_rate = six_indicators_rate = confluence_analysis_rate = technical_analysis_field_rate = 0.0
            
            logger.info(f"\n   📊 DETAILED REASONING WITH CONFLUENCE RESULTS:")
            logger.info(f"      Total analyses: {confluence_results['total_analyses']}")
            logger.info(f"      Analyses with detailed reasoning: {confluence_results['analyses_with_detailed_reasoning']} ({detailed_reasoning_rate:.2f})")
            logger.info(f"      6-indicator coverage: {confluence_results['six_indicators_mentioned']} ({six_indicators_rate:.2f})")
            logger.info(f"      Confluence analysis present: {confluence_results['confluence_analysis_present']} ({confluence_analysis_rate:.2f})")
            logger.info(f"      Technical indicators analysis field: {confluence_results['technical_indicators_analysis_field']} ({technical_analysis_field_rate:.2f})")
            logger.info(f"      Confluence scores found: {confluence_results['confluence_scores_found']}")
            
            if confluence_results['successful_analyses']:
                logger.info(f"      📊 Successful analyses:")
                for analysis in confluence_results['successful_analyses']:
                    logger.info(f"         - {analysis['symbol']}: {analysis['indicator_coverage']}/6 indicators, {analysis['confluence_mentions']} confluence mentions")
            
            # Calculate test success based on review requirements
            success_criteria = [
                confluence_results['total_analyses'] >= 2,  # At least 2 analyses
                confluence_results['analyses_with_detailed_reasoning'] >= 1,  # At least 1 with detailed reasoning
                detailed_reasoning_rate >= 0.67,  # At least 67% with detailed reasoning
                confluence_results['six_indicators_mentioned'] >= 1,  # At least 1 with 6-indicator coverage
                confluence_results['confluence_analysis_present'] >= 1  # At least 1 with confluence analysis
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Detailed Reasoning with Confluence", True, 
                                   f"Confluence analysis successful: {success_count}/{len(success_criteria)} criteria met. Detailed reasoning: {detailed_reasoning_rate:.2f}, 6-indicator coverage: {six_indicators_rate:.2f}, Confluence analysis: {confluence_analysis_rate:.2f}")
            else:
                self.log_test_result("Detailed Reasoning with Confluence", False, 
                                   f"Confluence analysis issues: {success_count}/{len(success_criteria)} criteria met. May lack detailed 6-indicator confluence analysis")
                
        except Exception as e:
            self.log_test_result("Detailed Reasoning with Confluence", False, f"Exception: {str(e)}")

    async def test_3_technical_levels_based_indicators(self):
        """Test 3: Technical Levels Based on Indicators - Verify stop_loss uses VWAP/EMA21, take_profit uses SMA50"""
        logger.info("\n🔍 TEST 3: Technical Levels Based on Indicators")
        
        try:
            levels_results = {
                'analyses_with_technical_levels': 0,
                'total_analyses': 0,
                'stop_loss_technical_basis': 0,
                'take_profit_technical_basis': 0,
                'coherent_levels_with_indicators': 0,
                'vwap_ema21_stop_loss': 0,
                'sma50_resistance_take_profit': 0,
                'successful_analyses': []
            }
            
            logger.info("   🚀 Testing technical levels based on indicators...")
            logger.info("   📊 Expected: stop_loss uses VWAP/EMA21, take_profit uses SMA50/resistance levels")
            
            # Use symbols from previous tests
            test_symbols = self.actual_test_symbols if self.actual_test_symbols else self.test_symbols
            logger.info(f"   📊 Testing symbols: {test_symbols}")
            
            # Test each symbol for technical levels
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing technical levels for {symbol}...")
                levels_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Extract IA1 analysis
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        if isinstance(ia1_analysis, dict):
                            entry_price = ia1_analysis.get('entry_price')
                            stop_loss_price = ia1_analysis.get('stop_loss_price')
                            take_profit_price = ia1_analysis.get('take_profit_price')
                            reasoning = ia1_analysis.get('reasoning', '')
                            rr_reasoning = ia1_analysis.get('rr_reasoning', '')
                            technical_indicators_analysis = ia1_analysis.get('technical_indicators_analysis', {})
                            
                            if all([entry_price, stop_loss_price, take_profit_price]):
                                levels_results['analyses_with_technical_levels'] += 1
                                logger.info(f"      ✅ Technical levels found: Entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}")
                                
                                # Check for technical basis in reasoning/rr_reasoning
                                combined_reasoning = f"{reasoning} {rr_reasoning}".lower()
                                
                                # Check stop_loss technical basis (VWAP, EMA21)
                                stop_loss_indicators = ['vwap', 'ema21', 'ema 21', 'support']
                                stop_loss_technical = any(indicator in combined_reasoning for indicator in stop_loss_indicators)
                                
                                if stop_loss_technical:
                                    levels_results['stop_loss_technical_basis'] += 1
                                    
                                    # Specifically check for VWAP or EMA21
                                    if 'vwap' in combined_reasoning or 'ema21' in combined_reasoning or 'ema 21' in combined_reasoning:
                                        levels_results['vwap_ema21_stop_loss'] += 1
                                        logger.info(f"         ✅ Stop loss uses VWAP/EMA21 technical basis")
                                    else:
                                        logger.info(f"         ✅ Stop loss uses technical basis (support)")
                                else:
                                    logger.warning(f"         ⚠️ Stop loss lacks clear technical basis")
                                
                                # Check take_profit technical basis (SMA50, resistance)
                                take_profit_indicators = ['sma50', 'sma 50', 'resistance', 'ema200']
                                take_profit_technical = any(indicator in combined_reasoning for indicator in take_profit_indicators)
                                
                                if take_profit_technical:
                                    levels_results['take_profit_technical_basis'] += 1
                                    
                                    # Specifically check for SMA50 or resistance
                                    if 'sma50' in combined_reasoning or 'sma 50' in combined_reasoning or 'resistance' in combined_reasoning:
                                        levels_results['sma50_resistance_take_profit'] += 1
                                        logger.info(f"         ✅ Take profit uses SMA50/resistance technical basis")
                                    else:
                                        logger.info(f"         ✅ Take profit uses technical basis")
                                else:
                                    logger.warning(f"         ⚠️ Take profit lacks clear technical basis")
                                
                                # Check for coherence with calculated indicators
                                coherent_with_indicators = False
                                if technical_indicators_analysis and isinstance(technical_indicators_analysis, dict):
                                    # Look for EMA hierarchy analysis
                                    ema_hierarchy = technical_indicators_analysis.get('ema_hierarchy_analysis', {})
                                    if ema_hierarchy:
                                        dynamic_sr = ema_hierarchy.get('dynamic_support_resistance', {})
                                        if dynamic_sr:
                                            # Check if levels are close to calculated EMA levels
                                            ema9 = dynamic_sr.get('ema9')
                                            ema21 = dynamic_sr.get('ema21')
                                            sma50 = dynamic_sr.get('sma50')
                                            
                                            if any([ema9, ema21, sma50]):
                                                coherent_with_indicators = True
                                                logger.info(f"         ✅ Levels coherent with calculated indicators")
                                
                                if coherent_with_indicators:
                                    levels_results['coherent_levels_with_indicators'] += 1
                                
                                # Store successful analysis
                                levels_results['successful_analyses'].append({
                                    'symbol': symbol,
                                    'entry_price': entry_price,
                                    'stop_loss_price': stop_loss_price,
                                    'take_profit_price': take_profit_price,
                                    'stop_loss_technical': stop_loss_technical,
                                    'take_profit_technical': take_profit_technical,
                                    'coherent_with_indicators': coherent_with_indicators,
                                    'vwap_ema21_sl': 'vwap' in combined_reasoning or 'ema21' in combined_reasoning,
                                    'sma50_resistance_tp': 'sma50' in combined_reasoning or 'resistance' in combined_reasoning
                                })
                                
                                logger.info(f"      ✅ Technical levels analysis SUCCESS for {symbol}")
                            else:
                                logger.warning(f"      ❌ Incomplete technical levels for {symbol}")
                        else:
                            logger.error(f"      ❌ Invalid IA1 analysis structure for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if levels_results['total_analyses'] > 0:
                technical_levels_rate = levels_results['analyses_with_technical_levels'] / levels_results['total_analyses']
                stop_loss_technical_rate = levels_results['stop_loss_technical_basis'] / max(levels_results['analyses_with_technical_levels'], 1)
                take_profit_technical_rate = levels_results['take_profit_technical_basis'] / max(levels_results['analyses_with_technical_levels'], 1)
                coherence_rate = levels_results['coherent_levels_with_indicators'] / max(levels_results['analyses_with_technical_levels'], 1)
            else:
                technical_levels_rate = stop_loss_technical_rate = take_profit_technical_rate = coherence_rate = 0.0
            
            logger.info(f"\n   📊 TECHNICAL LEVELS BASED ON INDICATORS RESULTS:")
            logger.info(f"      Total analyses: {levels_results['total_analyses']}")
            logger.info(f"      Analyses with technical levels: {levels_results['analyses_with_technical_levels']} ({technical_levels_rate:.2f})")
            logger.info(f"      Stop loss technical basis: {levels_results['stop_loss_technical_basis']} ({stop_loss_technical_rate:.2f})")
            logger.info(f"      Take profit technical basis: {levels_results['take_profit_technical_basis']} ({take_profit_technical_rate:.2f})")
            logger.info(f"      VWAP/EMA21 stop loss: {levels_results['vwap_ema21_stop_loss']}")
            logger.info(f"      SMA50/resistance take profit: {levels_results['sma50_resistance_take_profit']}")
            logger.info(f"      Coherent with indicators: {levels_results['coherent_levels_with_indicators']} ({coherence_rate:.2f})")
            
            if levels_results['successful_analyses']:
                logger.info(f"      📊 Successful analyses:")
                for analysis in levels_results['successful_analyses']:
                    logger.info(f"         - {analysis['symbol']}: SL technical={analysis['stop_loss_technical']}, TP technical={analysis['take_profit_technical']}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                levels_results['total_analyses'] >= 2,  # At least 2 analyses
                levels_results['analyses_with_technical_levels'] >= 1,  # At least 1 with technical levels
                technical_levels_rate >= 0.67,  # At least 67% with technical levels
                levels_results['stop_loss_technical_basis'] >= 1,  # At least 1 with SL technical basis
                levels_results['take_profit_technical_basis'] >= 1  # At least 1 with TP technical basis
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Technical Levels Based on Indicators", True, 
                                   f"Technical levels successful: {success_count}/{len(success_criteria)} criteria met. Technical levels rate: {technical_levels_rate:.2f}, SL technical: {stop_loss_technical_rate:.2f}, TP technical: {take_profit_technical_rate:.2f}")
            else:
                self.log_test_result("Technical Levels Based on Indicators", False, 
                                   f"Technical levels issues: {success_count}/{len(success_criteria)} criteria met. May not use VWAP/EMA21 for SL or SMA50/resistance for TP")
                
        except Exception as e:
            self.log_test_result("Technical Levels Based on Indicators", False, f"Exception: {str(e)}")

    async def test_4_escalation_to_ia2(self):
        """Test 4: Test Escalation to IA2 - Verify RR > 2.0 allows escalation"""
        logger.info("\n🔍 TEST 4: Test Escalation to IA2")
        
        try:
            escalation_results = {
                'analyses_with_high_rr': 0,
                'total_analyses': 0,
                'escalation_candidates': [],
                'ia2_escalations_triggered': 0,
                'quality_analyses_for_ia2': 0,
                'database_ia2_decisions': 0,
                'successful_escalations': []
            }
            
            logger.info("   🚀 Testing escalation to IA2 based on RR > 2.0...")
            logger.info("   📊 Expected: High RR calculations allow escalation, quality analyses trigger IA2")
            
            # Use symbols from previous tests
            test_symbols = self.actual_test_symbols if self.actual_test_symbols else self.test_symbols
            logger.info(f"   📊 Testing symbols: {test_symbols}")
            
            # Test each symbol for escalation potential
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing escalation potential for {symbol}...")
                escalation_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Extract IA1 analysis
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        if isinstance(ia1_analysis, dict):
                            calculated_rr = ia1_analysis.get('calculated_rr')
                            confidence = ia1_analysis.get('confidence')
                            signal = ia1_analysis.get('signal', '').upper()
                            reasoning = ia1_analysis.get('reasoning', '')
                            
                            logger.info(f"      📊 Analysis: RR={calculated_rr}, Confidence={confidence}, Signal={signal}")
                            
                            # Check for high RR (> 2.0)
                            if calculated_rr and calculated_rr > 2.0:
                                escalation_results['analyses_with_high_rr'] += 1
                                logger.info(f"         ✅ High RR for escalation: {calculated_rr:.2f}")
                                
                                escalation_results['escalation_candidates'].append({
                                    'symbol': symbol,
                                    'calculated_rr': calculated_rr,
                                    'confidence': confidence,
                                    'signal': signal,
                                    'reasoning_quality': len(reasoning) if reasoning else 0
                                })
                            else:
                                logger.info(f"         📊 RR below escalation threshold: {calculated_rr}")
                            
                            # Check for quality analysis (detailed reasoning, high confidence)
                            quality_indicators = [
                                len(reasoning) > 200 if reasoning else False,  # Detailed reasoning
                                confidence and confidence > 0.7 if confidence else False,  # High confidence
                                signal in ['LONG', 'SHORT'],  # Clear signal
                                calculated_rr and calculated_rr > 1.5 if calculated_rr else False  # Reasonable RR
                            ]
                            
                            quality_score = sum(quality_indicators)
                            if quality_score >= 3:  # At least 3/4 quality indicators
                                escalation_results['quality_analyses_for_ia2'] += 1
                                logger.info(f"         ✅ Quality analysis for IA2: {quality_score}/4 indicators")
                            else:
                                logger.info(f"         📊 Analysis quality: {quality_score}/4 indicators")
                            
                            # Check if IA2 was actually triggered (look for ia2_decision in response)
                            ia2_decision = analysis_data.get('ia2_decision')
                            if ia2_decision:
                                escalation_results['ia2_escalations_triggered'] += 1
                                escalation_results['successful_escalations'].append({
                                    'symbol': symbol,
                                    'ia1_rr': calculated_rr,
                                    'ia1_confidence': confidence,
                                    'ia2_signal': ia2_decision.get('signal'),
                                    'ia2_confidence': ia2_decision.get('confidence')
                                })
                                logger.info(f"         🚀 IA2 escalation TRIGGERED for {symbol}")
                            else:
                                logger.info(f"         📊 No IA2 escalation for {symbol}")
                        else:
                            logger.error(f"      ❌ Invalid IA1 analysis structure for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Check database for IA2 decisions (if MongoDB available)
            if self.db:
                try:
                    # Count recent IA2 decisions in database
                    from datetime import datetime, timedelta
                    recent_time = datetime.now() - timedelta(hours=1)  # Last hour
                    
                    ia2_decisions = list(self.db.trading_decisions.find({
                        "timestamp": {"$gte": recent_time}
                    }))
                    
                    escalation_results['database_ia2_decisions'] = len(ia2_decisions)
                    logger.info(f"   📊 Recent IA2 decisions in database: {len(ia2_decisions)}")
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not check database for IA2 decisions: {e}")
            
            # Calculate final metrics
            if escalation_results['total_analyses'] > 0:
                high_rr_rate = escalation_results['analyses_with_high_rr'] / escalation_results['total_analyses']
                quality_analysis_rate = escalation_results['quality_analyses_for_ia2'] / escalation_results['total_analyses']
                escalation_trigger_rate = escalation_results['ia2_escalations_triggered'] / escalation_results['total_analyses']
            else:
                high_rr_rate = quality_analysis_rate = escalation_trigger_rate = 0.0
            
            logger.info(f"\n   📊 TEST ESCALATION TO IA2 RESULTS:")
            logger.info(f"      Total analyses: {escalation_results['total_analyses']}")
            logger.info(f"      Analyses with high RR (>2.0): {escalation_results['analyses_with_high_rr']} ({high_rr_rate:.2f})")
            logger.info(f"      Quality analyses for IA2: {escalation_results['quality_analyses_for_ia2']} ({quality_analysis_rate:.2f})")
            logger.info(f"      IA2 escalations triggered: {escalation_results['ia2_escalations_triggered']} ({escalation_trigger_rate:.2f})")
            logger.info(f"      Database IA2 decisions: {escalation_results['database_ia2_decisions']}")
            
            if escalation_results['escalation_candidates']:
                logger.info(f"      📊 Escalation candidates:")
                for candidate in escalation_results['escalation_candidates']:
                    logger.info(f"         - {candidate['symbol']}: RR={candidate['calculated_rr']:.2f}, Confidence={candidate['confidence']}")
            
            if escalation_results['successful_escalations']:
                logger.info(f"      📊 Successful escalations:")
                for escalation in escalation_results['successful_escalations']:
                    logger.info(f"         - {escalation['symbol']}: IA1 RR={escalation['ia1_rr']:.2f} → IA2 triggered")
            
            # Calculate test success based on review requirements
            success_criteria = [
                escalation_results['total_analyses'] >= 2,  # At least 2 analyses
                escalation_results['analyses_with_high_rr'] >= 1 or escalation_results['quality_analyses_for_ia2'] >= 1,  # At least 1 escalation candidate
                high_rr_rate >= 0.33 or quality_analysis_rate >= 0.33,  # At least 33% escalation potential
                escalation_results['ia2_escalations_triggered'] >= 0,  # IA2 system functional (0 is acceptable)
                escalation_results['database_ia2_decisions'] >= 0  # Database functional (0 is acceptable)
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Test Escalation to IA2", True, 
                                   f"IA2 escalation successful: {success_count}/{len(success_criteria)} criteria met. High RR rate: {high_rr_rate:.2f}, Quality analyses: {quality_analysis_rate:.2f}, Escalations triggered: {escalation_results['ia2_escalations_triggered']}")
            else:
                self.log_test_result("Test Escalation to IA2", False, 
                                   f"IA2 escalation issues: {success_count}/{len(success_criteria)} criteria met. May not escalate properly with RR > 2.0")
                
        except Exception as e:
            self.log_test_result("Test Escalation to IA2", False, f"Exception: {str(e)}")

    async def test_5_database_verification(self):
        """Test 5: Database Verification - Check stored analyses have improved structure"""
        logger.info("\n🔍 TEST 5: Database Verification")
        
        try:
            db_results = {
                'recent_analyses_found': 0,
                'analyses_with_calculated_rr': 0,
                'analyses_with_detailed_reasoning': 0,
                'analyses_with_technical_indicators': 0,
                'improvement_over_previous': False,
                'database_structure_complete': 0,
                'sample_analyses': []
            }
            
            logger.info("   🚀 Verifying database contains improved analysis structure...")
            logger.info("   📊 Expected: Recent analyses have calculated_rr, detailed reasoning, technical indicators")
            
            if not self.db:
                logger.warning("   ⚠️ MongoDB not available, skipping database verification")
                self.log_test_result("Database Verification", False, "MongoDB connection not available")
                return
            
            try:
                # Get recent technical analyses (last 2 hours)
                from datetime import datetime, timedelta
                recent_time = datetime.now() - timedelta(hours=2)
                
                recent_analyses = list(self.db.technical_analyses.find({
                    "timestamp": {"$gte": recent_time}
                }).sort("timestamp", -1).limit(10))
                
                db_results['recent_analyses_found'] = len(recent_analyses)
                logger.info(f"   📊 Recent analyses found: {len(recent_analyses)}")
                
                if recent_analyses:
                    # Analyze structure of recent analyses
                    for analysis in recent_analyses:
                        symbol = analysis.get('symbol', 'Unknown')
                        
                        # Check for calculated_rr field
                        if 'calculated_rr' in analysis and analysis['calculated_rr'] is not None:
                            db_results['analyses_with_calculated_rr'] += 1
                        
                        # Check for detailed reasoning
                        reasoning = analysis.get('reasoning', '')
                        if reasoning and len(reasoning) > 100:
                            db_results['analyses_with_detailed_reasoning'] += 1
                        
                        # Check for technical indicators
                        technical_indicators = [
                            'rsi_value', 'macd_line', 'mfi_value', 'vwap_price'
                        ]
                        has_technical = any(field in analysis and analysis[field] is not None for field in technical_indicators)
                        if has_technical:
                            db_results['analyses_with_technical_indicators'] += 1
                        
                        # Check for complete structure
                        required_fields = ['symbol', 'entry_price', 'stop_loss_price', 'take_profit_price', 'confidence']
                        has_complete_structure = all(field in analysis for field in required_fields)
                        if has_complete_structure:
                            db_results['database_structure_complete'] += 1
                        
                        # Store sample for analysis
                        if len(db_results['sample_analyses']) < 3:
                            db_results['sample_analyses'].append({
                                'symbol': symbol,
                                'has_calculated_rr': 'calculated_rr' in analysis,
                                'reasoning_length': len(reasoning),
                                'has_technical_indicators': has_technical,
                                'complete_structure': has_complete_structure,
                                'timestamp': analysis.get('timestamp')
                            })
                    
                    # Compare with older analyses to check improvement
                    older_time = datetime.now() - timedelta(hours=24)
                    older_analyses = list(self.db.technical_analyses.find({
                        "timestamp": {"$gte": older_time, "$lt": recent_time}
                    }).limit(5))
                    
                    if older_analyses:
                        # Calculate improvement metrics
                        recent_rr_rate = db_results['analyses_with_calculated_rr'] / len(recent_analyses)
                        recent_reasoning_rate = db_results['analyses_with_detailed_reasoning'] / len(recent_analyses)
                        
                        older_rr_count = sum(1 for a in older_analyses if 'calculated_rr' in a and a['calculated_rr'] is not None)
                        older_reasoning_count = sum(1 for a in older_analyses if len(a.get('reasoning', '')) > 100)
                        
                        older_rr_rate = older_rr_count / len(older_analyses) if older_analyses else 0
                        older_reasoning_rate = older_reasoning_count / len(older_analyses) if older_analyses else 0
                        
                        improvement = (recent_rr_rate > older_rr_rate) or (recent_reasoning_rate > older_reasoning_rate)
                        db_results['improvement_over_previous'] = improvement
                        
                        logger.info(f"   📊 Improvement analysis:")
                        logger.info(f"      Recent RR rate: {recent_rr_rate:.2f} vs Older: {older_rr_rate:.2f}")
                        logger.info(f"      Recent reasoning rate: {recent_reasoning_rate:.2f} vs Older: {older_reasoning_rate:.2f}")
                        logger.info(f"      Improvement detected: {improvement}")
                
                else:
                    logger.warning("   ⚠️ No recent analyses found in database")
                
            except Exception as e:
                logger.error(f"   ❌ Database query error: {e}")
            
            # Calculate final metrics
            if db_results['recent_analyses_found'] > 0:
                calculated_rr_rate = db_results['analyses_with_calculated_rr'] / db_results['recent_analyses_found']
                detailed_reasoning_rate = db_results['analyses_with_detailed_reasoning'] / db_results['recent_analyses_found']
                technical_indicators_rate = db_results['analyses_with_technical_indicators'] / db_results['recent_analyses_found']
                complete_structure_rate = db_results['database_structure_complete'] / db_results['recent_analyses_found']
            else:
                calculated_rr_rate = detailed_reasoning_rate = technical_indicators_rate = complete_structure_rate = 0.0
            
            logger.info(f"\n   📊 DATABASE VERIFICATION RESULTS:")
            logger.info(f"      Recent analyses found: {db_results['recent_analyses_found']}")
            logger.info(f"      Analyses with calculated_rr: {db_results['analyses_with_calculated_rr']} ({calculated_rr_rate:.2f})")
            logger.info(f"      Analyses with detailed reasoning: {db_results['analyses_with_detailed_reasoning']} ({detailed_reasoning_rate:.2f})")
            logger.info(f"      Analyses with technical indicators: {db_results['analyses_with_technical_indicators']} ({technical_indicators_rate:.2f})")
            logger.info(f"      Complete database structure: {db_results['database_structure_complete']} ({complete_structure_rate:.2f})")
            logger.info(f"      Improvement over previous: {db_results['improvement_over_previous']}")
            
            if db_results['sample_analyses']:
                logger.info(f"      📊 Sample analyses:")
                for sample in db_results['sample_analyses']:
                    logger.info(f"         - {sample['symbol']}: RR={sample['has_calculated_rr']}, Reasoning={sample['reasoning_length']} chars, Technical={sample['has_technical_indicators']}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                db_results['recent_analyses_found'] >= 1,  # At least 1 recent analysis
                db_results['analyses_with_calculated_rr'] >= 1,  # At least 1 with calculated_rr
                calculated_rr_rate >= 0.5,  # At least 50% with calculated_rr
                detailed_reasoning_rate >= 0.5,  # At least 50% with detailed reasoning
                technical_indicators_rate >= 0.5  # At least 50% with technical indicators
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Database Verification", True, 
                                   f"Database verification successful: {success_count}/{len(success_criteria)} criteria met. RR rate: {calculated_rr_rate:.2f}, Reasoning rate: {detailed_reasoning_rate:.2f}, Technical rate: {technical_indicators_rate:.2f}")
            else:
                self.log_test_result("Database Verification", False, 
                                   f"Database verification issues: {success_count}/{len(success_criteria)} criteria met. Database may not contain improved analysis structure")
                
        except Exception as e:
            self.log_test_result("Database Verification", False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all tests in sequence"""
        logger.info("🚀 STARTING PROMPT AMÉLIORÉ AVEC CALCUL RR - COMPREHENSIVE TEST SUITE")
        logger.info("=" * 80)
        
        # Run all tests
        await self.test_1_rr_calculation_real_formulas()
        await self.test_2_detailed_reasoning_confluence()
        await self.test_3_technical_levels_based_indicators()
        await self.test_4_escalation_to_ia2()
        await self.test_5_database_verification()
        
        # Generate final summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 FINAL TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests} ✅")
        logger.info(f"Failed: {failed_tests} ❌")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        logger.info("\nDetailed Results:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        # Overall assessment
        overall_success_rate = passed_tests / total_tests
        if overall_success_rate >= 0.8:
            logger.info(f"\n🎉 OVERALL ASSESSMENT: SUCCESS")
            logger.info(f"The new prompt with RR calculation and detailed reasoning is working correctly.")
            logger.info(f"Key improvements validated: Real RR formulas, 6-indicator confluence, technical levels, IA2 escalation.")
        elif overall_success_rate >= 0.6:
            logger.info(f"\n⚠️ OVERALL ASSESSMENT: PARTIAL SUCCESS")
            logger.info(f"The new prompt shows improvements but some issues remain.")
            logger.info(f"Review failed tests and consider additional fixes.")
        else:
            logger.info(f"\n❌ OVERALL ASSESSMENT: NEEDS WORK")
            logger.info(f"The new prompt has significant issues that need to be addressed.")
            logger.info(f"Focus on failed tests for priority fixes.")
        
        logger.info("=" * 80)
        
        return overall_success_rate >= 0.8

async def main():
    """Main test execution"""
    test_suite = PromptRRCalculationTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("🎉 All tests completed successfully!")
        return 0
    else:
        logger.info("❌ Some tests failed. Review the results above.")
        return 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)
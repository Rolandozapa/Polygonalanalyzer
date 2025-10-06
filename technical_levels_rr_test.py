#!/usr/bin/env python3
"""
TECHNICAL LEVELS & RISK-REWARD CALCULATION VERIFICATION TEST SUITE
Focus: Vérifier Calcul des Niveaux Techniques et RR - Analyse détaillée des niveaux support/résistance et calcul Risk-Reward.

OBJECTIF SPÉCIFIQUE:
Examiner en détail si IA1 calcule et utilise correctement les niveaux techniques nécessaires pour un RR précis:

1. **Niveaux Techniques Calculés**: 
   - Vérifier si IA1 calcule des niveaux de support et résistance basés sur l'analyse technique
   - Examiner si les prix d'entrée, stop-loss et take-profit sont définis de manière réaliste
   - Confirmer que les niveaux utilisent les données OHLCV historiques et non des pourcentages arbitraires

2. **Calcul Risk-Reward (RR)**:
   - Analyser les valeurs RR dans les réponses IA1 
   - Vérifier la formule utilisée: LONG RR = (TP-Entry)/(Entry-SL), SHORT RR = (Entry-SL)/(TP-Entry)
   - Confirmer que les niveaux sont cohérents avec l'analyse technique (VWAP, EMA, supports/résistances historiques)

3. **Qualité des Niveaux**:
   - Examiner si les niveaux correspondent aux indicateurs techniques (VWAP, EMA21, SMA50, Fibonacci, etc.)
   - Vérifier si IA1 utilise les swing highs/lows historiques
   - Confirmer réalisme des niveaux par rapport à la volatilité actuelle

4. **Tests Concrets**:
   - Analyser 2-3 cryptos avec /api/force-ia1-analysis
   - Examiner les champs: entry_price, stop_loss_price, take_profit_price, risk_reward_ratio
   - Comparer avec les indicateurs techniques calculés (VWAP, EMA21, etc.)

5. **Validation Logique**:
   - Vérifier cohérence entre le signal (LONG/SHORT) et les niveaux proposés
   - Confirmer que les niveaux respectent la hiérarchie technique (support < entry < résistance pour LONG)
   - Examiner si le RR calculé correspond aux critères d'escalation vers IA2 (RR > 2.0)

FOCUS: Déterminer si IA1 utilise une analyse technique rigoureuse pour définir les niveaux ou s'il utilise des pourcentages fixes peu précis.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests
import subprocess
import re
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalLevelsRRTestSuite:
    """Comprehensive test suite for Technical Levels & Risk-Reward Calculation Verification"""
    
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
        logger.info(f"Testing Technical Levels & RR System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for IA1 analysis (will be dynamically determined from available opportunities)
        self.preferred_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']  # From review request
        self.actual_test_symbols = []  # Will be populated from available opportunities
        
        # Technical indicators to verify in levels calculation
        self.technical_indicators = ['VWAP', 'EMA21', 'SMA50', 'RSI', 'MACD', 'MFI', 'Fibonacci']
        
        # RR calculation formulas to verify
        self.rr_formulas = {
            'LONG': '(TP-Entry)/(Entry-SL)',
            'SHORT': '(Entry-SL)/(TP-Entry)'
        }
        
        # Expected fields in IA1 analysis for technical levels
        self.technical_level_fields = [
            'entry_price', 'stop_loss_price', 'take_profit_price', 'risk_reward_ratio',
            'support', 'resistance', 'current_price'
        ]
        
        # IA1 analysis data storage
        self.ia1_analyses = []
        self.technical_levels_data = []
        
        # MongoDB connection for database verification
        try:
            with open('/app/backend/.env', 'r') as f:
                for line in f:
                    if line.startswith('MONGO_URL='):
                        mongo_url = line.split('=')[1].strip().strip('"')
                        break
                else:
                    mongo_url = "mongodb://localhost:27017"
            
            self.mongo_client = MongoClient(mongo_url)
            self.db = self.mongo_client['myapp']
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
    
    async def _get_available_symbols(self) -> List[str]:
        """Get available symbols from scout system"""
        try:
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            
            if response.status_code == 200:
                opportunities = response.json()
                if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                    opportunities = opportunities['opportunities']
                
                # Get available symbols
                available_symbols = [opp.get('symbol') for opp in opportunities if opp.get('symbol')]
                
                # Prefer BTCUSDT, ETHUSDT, SOLUSDT if available
                test_symbols = []
                for symbol in self.preferred_symbols:
                    if symbol in available_symbols:
                        test_symbols.append(symbol)
                
                # Fill remaining slots with available symbols
                for symbol in available_symbols:
                    if symbol not in test_symbols and len(test_symbols) < 3:
                        test_symbols.append(symbol)
                
                self.actual_test_symbols = test_symbols[:3]  # Limit to 3 symbols
                logger.info(f"✅ Test symbols selected: {self.actual_test_symbols}")
                return self.actual_test_symbols
                
            else:
                logger.warning(f"⚠️ Could not get opportunities, using default symbols")
                self.actual_test_symbols = self.preferred_symbols
                return self.actual_test_symbols
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting opportunities: {e}, using default symbols")
            self.actual_test_symbols = self.preferred_symbols
            return self.actual_test_symbols
    
    def _validate_rr_calculation(self, analysis_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Validate Risk-Reward calculation using the correct formulas"""
        validation_result = {
            'rr_present': False,
            'rr_value': None,
            'rr_calculated_correctly': False,
            'formula_used': None,
            'expected_rr': None,
            'levels_present': False,
            'levels_realistic': False,
            'levels_data': {}
        }
        
        try:
            # Extract IA1 analysis
            ia1_analysis = analysis_data.get('ia1_analysis', {})
            if not ia1_analysis:
                return validation_result
            
            # Check for RR value
            rr_value = ia1_analysis.get('risk_reward_ratio')
            if rr_value is not None:
                validation_result['rr_present'] = True
                validation_result['rr_value'] = rr_value
                logger.info(f"      📊 RR value found: {rr_value}")
            
            # Extract price levels
            entry_price = ia1_analysis.get('entry_price')
            stop_loss = ia1_analysis.get('stop_loss_price')
            take_profit = ia1_analysis.get('take_profit_price')
            signal = ia1_analysis.get('signal', '').upper()
            
            if all([entry_price, stop_loss, take_profit, signal]):
                validation_result['levels_present'] = True
                validation_result['levels_data'] = {
                    'entry_price': entry_price,
                    'stop_loss_price': stop_loss,
                    'take_profit_price': take_profit,
                    'signal': signal
                }
                
                logger.info(f"      📊 Levels found - Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}, Signal: {signal}")
                
                # Validate RR calculation using correct formula
                if signal in ['LONG', 'BUY']:
                    # LONG RR = (TP-Entry)/(Entry-SL)
                    if entry_price > stop_loss and take_profit > entry_price:
                        expected_rr = (take_profit - entry_price) / (entry_price - stop_loss)
                        validation_result['formula_used'] = 'LONG: (TP-Entry)/(Entry-SL)'
                        validation_result['expected_rr'] = expected_rr
                        
                        # Check if calculated RR matches expected (within 5% tolerance)
                        if rr_value and abs(rr_value - expected_rr) / expected_rr <= 0.05:
                            validation_result['rr_calculated_correctly'] = True
                            logger.info(f"         ✅ RR calculation correct: {rr_value:.2f} ≈ {expected_rr:.2f}")
                        else:
                            logger.warning(f"         ❌ RR calculation incorrect: {rr_value} vs expected {expected_rr:.2f}")
                        
                        # Check level hierarchy for LONG
                        if stop_loss < entry_price < take_profit:
                            validation_result['levels_realistic'] = True
                            logger.info(f"         ✅ LONG level hierarchy correct: SL < Entry < TP")
                        else:
                            logger.warning(f"         ❌ LONG level hierarchy incorrect")
                
                elif signal in ['SHORT', 'SELL']:
                    # SHORT RR = (Entry-TP)/(SL-Entry)
                    if stop_loss > entry_price and take_profit < entry_price:
                        expected_rr = (entry_price - take_profit) / (stop_loss - entry_price)
                        validation_result['formula_used'] = 'SHORT: (Entry-TP)/(SL-Entry)'
                        validation_result['expected_rr'] = expected_rr
                        
                        # Check if calculated RR matches expected (within 5% tolerance)
                        if rr_value and abs(rr_value - expected_rr) / expected_rr <= 0.05:
                            validation_result['rr_calculated_correctly'] = True
                            logger.info(f"         ✅ RR calculation correct: {rr_value:.2f} ≈ {expected_rr:.2f}")
                        else:
                            logger.warning(f"         ❌ RR calculation incorrect: {rr_value} vs expected {expected_rr:.2f}")
                        
                        # Check level hierarchy for SHORT
                        if take_profit < entry_price < stop_loss:
                            validation_result['levels_realistic'] = True
                            logger.info(f"         ✅ SHORT level hierarchy correct: TP < Entry < SL")
                        else:
                            logger.warning(f"         ❌ SHORT level hierarchy incorrect")
            
        except Exception as e:
            logger.error(f"      ❌ Error validating RR calculation: {e}")
        
        return validation_result
    
    def _analyze_technical_indicators_usage(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if technical indicators are used in level calculation"""
        indicator_usage = {
            'indicators_mentioned': [],
            'specific_levels_referenced': [],
            'vwap_usage': False,
            'ema_usage': False,
            'fibonacci_usage': False,
            'support_resistance_analysis': False,
            'technical_basis_score': 0.0
        }
        
        try:
            # Extract reasoning and analysis text
            ia1_analysis = analysis_data.get('ia1_analysis', {})
            reasoning = ia1_analysis.get('reasoning', '')
            
            if reasoning:
                reasoning_lower = reasoning.lower()
                
                # Check for technical indicator mentions
                for indicator in self.technical_indicators:
                    if indicator.lower() in reasoning_lower:
                        indicator_usage['indicators_mentioned'].append(indicator)
                        
                        # Look for specific level references
                        level_patterns = [
                            rf'{indicator.lower()}\s*[:\s]*[\d\.\$]+',
                            rf'{indicator.lower()}\s+level\s*[:\s]*[\d\.\$]+',
                            rf'{indicator.lower()}\s+support\s*[:\s]*[\d\.\$]+',
                            rf'{indicator.lower()}\s+resistance\s*[:\s]*[\d\.\$]+'
                        ]
                        
                        for pattern in level_patterns:
                            matches = re.findall(pattern, reasoning, re.IGNORECASE)
                            if matches:
                                indicator_usage['specific_levels_referenced'].extend(matches)
                
                # Check for specific indicator usage
                indicator_usage['vwap_usage'] = 'vwap' in reasoning_lower
                indicator_usage['ema_usage'] = any(term in reasoning_lower for term in ['ema', 'ema21', 'ema50'])
                indicator_usage['fibonacci_usage'] = any(term in reasoning_lower for term in ['fibonacci', 'fib', 'retracement'])
                
                # Check for support/resistance analysis
                sr_keywords = ['support', 'resistance', 'level', 'zone', 'confluence']
                indicator_usage['support_resistance_analysis'] = any(keyword in reasoning_lower for keyword in sr_keywords)
                
                # Calculate technical basis score
                score_factors = [
                    len(indicator_usage['indicators_mentioned']) > 0,
                    len(indicator_usage['specific_levels_referenced']) > 0,
                    indicator_usage['vwap_usage'],
                    indicator_usage['ema_usage'],
                    indicator_usage['support_resistance_analysis']
                ]
                indicator_usage['technical_basis_score'] = sum(score_factors) / len(score_factors)
                
                logger.info(f"      📊 Technical indicators mentioned: {indicator_usage['indicators_mentioned']}")
                logger.info(f"      📊 Specific levels referenced: {len(indicator_usage['specific_levels_referenced'])}")
                logger.info(f"      📊 Technical basis score: {indicator_usage['technical_basis_score']:.2f}")
        
        except Exception as e:
            logger.error(f"      ❌ Error analyzing technical indicators usage: {e}")
        
        return indicator_usage
    
    async def _get_database_technical_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get technical analysis data from database for comparison"""
        if not self.db:
            return None
        
        try:
            # Get latest technical analysis for symbol
            latest_analysis = self.db.technical_analyses.find_one(
                {'symbol': symbol},
                sort=[('timestamp', -1)]
            )
            
            if latest_analysis:
                return {
                    'vwap_price': latest_analysis.get('vwap_price'),
                    'ema21': latest_analysis.get('ema21'),
                    'sma50': latest_analysis.get('sma50'),
                    'rsi': latest_analysis.get('rsi'),
                    'macd_line': latest_analysis.get('macd_line'),
                    'mfi_value': latest_analysis.get('mfi_value'),
                    'current_price': latest_analysis.get('current_price'),
                    'timestamp': latest_analysis.get('timestamp')
                }
        except Exception as e:
            logger.warning(f"      ⚠️ Could not get database technical data: {e}")
        
        return None
    
    async def test_1_technical_levels_calculation_verification(self):
        """Test 1: Technical Levels Calculation Verification - Verify IA1 calculates realistic technical levels"""
        logger.info("\n🔍 TEST 1: Technical Levels Calculation Verification")
        
        try:
            levels_results = {
                'total_analyses': 0,
                'analyses_with_levels': 0,
                'realistic_levels_count': 0,
                'technical_basis_count': 0,
                'ohlcv_based_levels': 0,
                'percentage_based_levels': 0,
                'level_quality_scores': [],
                'successful_analyses': []
            }
            
            logger.info("   🚀 Testing technical levels calculation in IA1 analyses...")
            logger.info("   📊 Expected: IA1 calculates realistic support/resistance levels based on technical analysis")
            
            # Get available symbols
            await self._get_available_symbols()
            
            # Test each symbol for technical levels calculation
            for symbol in self.actual_test_symbols:
                logger.info(f"\n   📞 Testing technical levels calculation for {symbol}...")
                levels_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Validate RR calculation and levels
                        rr_validation = self._validate_rr_calculation(analysis_data, symbol)
                        
                        if rr_validation['levels_present']:
                            levels_results['analyses_with_levels'] += 1
                            logger.info(f"      ✅ Technical levels found for {symbol}")
                            
                            # Check if levels are realistic (proper hierarchy)
                            if rr_validation['levels_realistic']:
                                levels_results['realistic_levels_count'] += 1
                                logger.info(f"         ✅ Levels hierarchy is realistic")
                            
                            # Analyze technical indicators usage
                            indicator_analysis = self._analyze_technical_indicators_usage(analysis_data)
                            
                            if indicator_analysis['technical_basis_score'] > 0.5:
                                levels_results['technical_basis_count'] += 1
                                logger.info(f"         ✅ Strong technical basis for levels")
                            
                            # Get database technical data for comparison
                            db_technical_data = await self._get_database_technical_data(symbol)
                            
                            # Calculate level quality score
                            quality_factors = [
                                rr_validation['levels_realistic'],
                                rr_validation['rr_present'],
                                indicator_analysis['technical_basis_score'] > 0.3,
                                len(indicator_analysis['indicators_mentioned']) >= 2,
                                indicator_analysis['support_resistance_analysis']
                            ]
                            quality_score = sum(quality_factors) / len(quality_factors)
                            levels_results['level_quality_scores'].append(quality_score)
                            
                            # Store successful analysis
                            levels_results['successful_analyses'].append({
                                'symbol': symbol,
                                'levels_data': rr_validation['levels_data'],
                                'rr_validation': rr_validation,
                                'indicator_analysis': indicator_analysis,
                                'quality_score': quality_score,
                                'db_technical_data': db_technical_data
                            })
                            
                            logger.info(f"      📊 Level quality score: {quality_score:.2f}")
                        else:
                            logger.warning(f"      ❌ No technical levels found for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if levels_results['total_analyses'] > 0:
                levels_present_rate = levels_results['analyses_with_levels'] / levels_results['total_analyses']
                realistic_levels_rate = levels_results['realistic_levels_count'] / levels_results['total_analyses']
                technical_basis_rate = levels_results['technical_basis_count'] / levels_results['total_analyses']
                avg_quality_score = sum(levels_results['level_quality_scores']) / len(levels_results['level_quality_scores']) if levels_results['level_quality_scores'] else 0.0
            else:
                levels_present_rate = realistic_levels_rate = technical_basis_rate = avg_quality_score = 0.0
            
            logger.info(f"\n   📊 TECHNICAL LEVELS CALCULATION VERIFICATION RESULTS:")
            logger.info(f"      Total analyses: {levels_results['total_analyses']}")
            logger.info(f"      Analyses with levels: {levels_results['analyses_with_levels']} ({levels_present_rate:.2f})")
            logger.info(f"      Realistic levels: {levels_results['realistic_levels_count']} ({realistic_levels_rate:.2f})")
            logger.info(f"      Technical basis: {levels_results['technical_basis_count']} ({technical_basis_rate:.2f})")
            logger.info(f"      Average quality score: {avg_quality_score:.2f}")
            
            # Calculate test success
            success_criteria = [
                levels_results['total_analyses'] >= 2,  # At least 2 analyses
                levels_present_rate >= 0.67,  # At least 67% with levels
                realistic_levels_rate >= 0.67,  # At least 67% realistic
                technical_basis_rate >= 0.5,  # At least 50% with technical basis
                avg_quality_score >= 0.6  # Average quality >= 60%
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Technical Levels Calculation Verification", True, 
                                   f"Technical levels calculation successful: {success_count}/{len(success_criteria)} criteria met. Levels present: {levels_present_rate:.2f}, Realistic: {realistic_levels_rate:.2f}, Technical basis: {technical_basis_rate:.2f}, Quality: {avg_quality_score:.2f}")
            else:
                self.log_test_result("Technical Levels Calculation Verification", False, 
                                   f"Technical levels calculation issues: {success_count}/{len(success_criteria)} criteria met. May lack realistic levels or technical basis")
                
        except Exception as e:
            self.log_test_result("Technical Levels Calculation Verification", False, f"Exception: {str(e)}")
    
    async def test_2_risk_reward_calculation_accuracy(self):
        """Test 2: Risk-Reward Calculation Accuracy - Verify RR formulas are used correctly"""
        logger.info("\n🔍 TEST 2: Risk-Reward Calculation Accuracy")
        
        try:
            rr_results = {
                'total_analyses': 0,
                'analyses_with_rr': 0,
                'correct_rr_calculations': 0,
                'rr_above_2_count': 0,
                'formula_usage_correct': 0,
                'rr_values': [],
                'formula_validations': []
            }
            
            logger.info("   🚀 Testing Risk-Reward calculation accuracy...")
            logger.info("   📊 Expected: RR calculated using correct formulas - LONG: (TP-Entry)/(Entry-SL), SHORT: (Entry-TP)/(SL-Entry)")
            
            # Use symbols from previous test
            test_symbols = self.actual_test_symbols if self.actual_test_symbols else await self._get_available_symbols()
            
            # Test each symbol for RR calculation accuracy
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing RR calculation accuracy for {symbol}...")
                rr_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Validate RR calculation
                        rr_validation = self._validate_rr_calculation(analysis_data, symbol)
                        
                        if rr_validation['rr_present']:
                            rr_results['analyses_with_rr'] += 1
                            rr_value = rr_validation['rr_value']
                            rr_results['rr_values'].append(rr_value)
                            
                            logger.info(f"      ✅ RR value found: {rr_value}")
                            
                            # Check if RR calculation is correct
                            if rr_validation['rr_calculated_correctly']:
                                rr_results['correct_rr_calculations'] += 1
                                logger.info(f"         ✅ RR calculation is correct")
                            
                            # Check if formula usage is correct
                            if rr_validation['formula_used']:
                                rr_results['formula_usage_correct'] += 1
                                logger.info(f"         ✅ Formula used: {rr_validation['formula_used']}")
                            
                            # Check if RR is above 2.0 (IA2 escalation criteria)
                            if rr_value and rr_value >= 2.0:
                                rr_results['rr_above_2_count'] += 1
                                logger.info(f"         ✅ RR ≥ 2.0 (IA2 escalation criteria met)")
                            
                            # Store validation details
                            rr_results['formula_validations'].append({
                                'symbol': symbol,
                                'rr_value': rr_value,
                                'expected_rr': rr_validation['expected_rr'],
                                'formula_used': rr_validation['formula_used'],
                                'calculation_correct': rr_validation['rr_calculated_correctly'],
                                'levels_data': rr_validation['levels_data']
                            })
                        else:
                            logger.warning(f"      ❌ No RR value found for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if rr_results['total_analyses'] > 0:
                rr_present_rate = rr_results['analyses_with_rr'] / rr_results['total_analyses']
                rr_accuracy_rate = rr_results['correct_rr_calculations'] / max(rr_results['analyses_with_rr'], 1)
                formula_correct_rate = rr_results['formula_usage_correct'] / max(rr_results['analyses_with_rr'], 1)
                rr_above_2_rate = rr_results['rr_above_2_count'] / max(rr_results['analyses_with_rr'], 1)
                avg_rr = sum(rr_results['rr_values']) / len(rr_results['rr_values']) if rr_results['rr_values'] else 0.0
            else:
                rr_present_rate = rr_accuracy_rate = formula_correct_rate = rr_above_2_rate = avg_rr = 0.0
            
            logger.info(f"\n   📊 RISK-REWARD CALCULATION ACCURACY RESULTS:")
            logger.info(f"      Total analyses: {rr_results['total_analyses']}")
            logger.info(f"      Analyses with RR: {rr_results['analyses_with_rr']} ({rr_present_rate:.2f})")
            logger.info(f"      Correct RR calculations: {rr_results['correct_rr_calculations']} ({rr_accuracy_rate:.2f})")
            logger.info(f"      Formula usage correct: {rr_results['formula_usage_correct']} ({formula_correct_rate:.2f})")
            logger.info(f"      RR ≥ 2.0 (IA2 criteria): {rr_results['rr_above_2_count']} ({rr_above_2_rate:.2f})")
            logger.info(f"      Average RR: {avg_rr:.2f}")
            
            if rr_results['formula_validations']:
                logger.info(f"      📊 Formula validation details:")
                for validation in rr_results['formula_validations']:
                    logger.info(f"         - {validation['symbol']}: RR={validation['rr_value']:.2f}, Expected={validation['expected_rr']:.2f if validation['expected_rr'] else 'N/A'}, Correct={validation['calculation_correct']}")
            
            # Calculate test success
            success_criteria = [
                rr_results['total_analyses'] >= 2,  # At least 2 analyses
                rr_present_rate >= 0.67,  # At least 67% with RR
                rr_accuracy_rate >= 0.8,  # At least 80% accurate calculations
                formula_correct_rate >= 0.8,  # At least 80% correct formula usage
                avg_rr >= 1.5  # Average RR should be reasonable
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Risk-Reward Calculation Accuracy", True, 
                                   f"RR calculation accuracy successful: {success_count}/{len(success_criteria)} criteria met. RR present: {rr_present_rate:.2f}, Accuracy: {rr_accuracy_rate:.2f}, Formula correct: {formula_correct_rate:.2f}, Avg RR: {avg_rr:.2f}")
            else:
                self.log_test_result("Risk-Reward Calculation Accuracy", False, 
                                   f"RR calculation accuracy issues: {success_count}/{len(success_criteria)} criteria met. May have incorrect formulas or unrealistic RR values")
                
        except Exception as e:
            self.log_test_result("Risk-Reward Calculation Accuracy", False, f"Exception: {str(e)}")
    
    async def test_3_technical_indicators_levels_coherence(self):
        """Test 3: Technical Indicators & Levels Coherence - Verify levels correspond to technical indicators"""
        logger.info("\n🔍 TEST 3: Technical Indicators & Levels Coherence")
        
        try:
            coherence_results = {
                'total_analyses': 0,
                'analyses_with_both': 0,
                'vwap_coherence_count': 0,
                'ema_coherence_count': 0,
                'fibonacci_coherence_count': 0,
                'overall_coherence_count': 0,
                'coherence_scores': [],
                'detailed_analyses': []
            }
            
            logger.info("   🚀 Testing coherence between technical indicators and price levels...")
            logger.info("   📊 Expected: Levels correspond to VWAP, EMA21, SMA50, Fibonacci, etc.")
            
            # Use symbols from previous tests
            test_symbols = self.actual_test_symbols if self.actual_test_symbols else await self._get_available_symbols()
            
            # Test each symbol for technical indicators & levels coherence
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing technical indicators & levels coherence for {symbol}...")
                coherence_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Get RR validation and indicator analysis
                        rr_validation = self._validate_rr_calculation(analysis_data, symbol)
                        indicator_analysis = self._analyze_technical_indicators_usage(analysis_data)
                        
                        # Get database technical data for comparison
                        db_technical_data = await self._get_database_technical_data(symbol)
                        
                        if rr_validation['levels_present'] and len(indicator_analysis['indicators_mentioned']) > 0:
                            coherence_results['analyses_with_both'] += 1
                            logger.info(f"      ✅ Both levels and technical indicators found")
                            
                            levels_data = rr_validation['levels_data']
                            entry_price = levels_data['entry_price']
                            stop_loss = levels_data['stop_loss_price']
                            take_profit = levels_data['take_profit_price']
                            
                            coherence_factors = []
                            
                            # Check VWAP coherence
                            if indicator_analysis['vwap_usage'] and db_technical_data and db_technical_data.get('vwap_price'):
                                vwap_price = db_technical_data['vwap_price']
                                # Check if levels are reasonably close to VWAP (within 10%)
                                vwap_tolerance = vwap_price * 0.1
                                
                                vwap_coherent = any([
                                    abs(entry_price - vwap_price) <= vwap_tolerance,
                                    abs(stop_loss - vwap_price) <= vwap_tolerance,
                                    abs(take_profit - vwap_price) <= vwap_tolerance
                                ])
                                
                                if vwap_coherent:
                                    coherence_results['vwap_coherence_count'] += 1
                                    coherence_factors.append(True)
                                    logger.info(f"         ✅ VWAP coherence: levels near VWAP {vwap_price}")
                                else:
                                    coherence_factors.append(False)
                                    logger.info(f"         ⚠️ VWAP coherence: levels distant from VWAP {vwap_price}")
                            
                            # Check EMA coherence
                            if indicator_analysis['ema_usage'] and db_technical_data:
                                ema21 = db_technical_data.get('ema21')
                                if ema21:
                                    ema_tolerance = ema21 * 0.05  # 5% tolerance
                                    
                                    ema_coherent = any([
                                        abs(entry_price - ema21) <= ema_tolerance,
                                        abs(stop_loss - ema21) <= ema_tolerance,
                                        abs(take_profit - ema21) <= ema_tolerance
                                    ])
                                    
                                    if ema_coherent:
                                        coherence_results['ema_coherence_count'] += 1
                                        coherence_factors.append(True)
                                        logger.info(f"         ✅ EMA coherence: levels near EMA21 {ema21}")
                                    else:
                                        coherence_factors.append(False)
                                        logger.info(f"         ⚠️ EMA coherence: levels distant from EMA21 {ema21}")
                            
                            # Check Fibonacci coherence
                            if indicator_analysis['fibonacci_usage']:
                                # Fibonacci levels are typically at 23.6%, 38.2%, 50%, 61.8%, 78.6%
                                # Check if levels align with common Fibonacci ratios
                                if db_technical_data and db_technical_data.get('current_price'):
                                    current_price = db_technical_data['current_price']
                                    
                                    # Simple Fibonacci level check (this is a basic approximation)
                                    fib_levels = [
                                        current_price * 0.764,  # 23.6% retracement
                                        current_price * 0.618,  # 38.2% retracement
                                        current_price * 0.5,    # 50% retracement
                                        current_price * 0.382,  # 61.8% retracement
                                        current_price * 0.236   # 78.6% retracement
                                    ]
                                    
                                    fib_tolerance = current_price * 0.02  # 2% tolerance
                                    
                                    fib_coherent = any([
                                        any(abs(level - fib_level) <= fib_tolerance for fib_level in fib_levels)
                                        for level in [entry_price, stop_loss, take_profit]
                                    ])
                                    
                                    if fib_coherent:
                                        coherence_results['fibonacci_coherence_count'] += 1
                                        coherence_factors.append(True)
                                        logger.info(f"         ✅ Fibonacci coherence: levels align with Fib ratios")
                                    else:
                                        coherence_factors.append(False)
                                        logger.info(f"         ⚠️ Fibonacci coherence: levels don't align with Fib ratios")
                            
                            # Calculate overall coherence score
                            coherence_score = sum(coherence_factors) / len(coherence_factors) if coherence_factors else 0.0
                            coherence_results['coherence_scores'].append(coherence_score)
                            
                            if coherence_score >= 0.5:
                                coherence_results['overall_coherence_count'] += 1
                                logger.info(f"      ✅ Overall coherence: {coherence_score:.2f}")
                            else:
                                logger.info(f"      ⚠️ Overall coherence: {coherence_score:.2f}")
                            
                            # Store detailed analysis
                            coherence_results['detailed_analyses'].append({
                                'symbol': symbol,
                                'levels_data': levels_data,
                                'indicators_mentioned': indicator_analysis['indicators_mentioned'],
                                'coherence_score': coherence_score,
                                'db_technical_data': db_technical_data,
                                'coherence_factors': coherence_factors
                            })
                        else:
                            logger.warning(f"      ❌ Missing levels or technical indicators for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if coherence_results['total_analyses'] > 0:
                both_present_rate = coherence_results['analyses_with_both'] / coherence_results['total_analyses']
                vwap_coherence_rate = coherence_results['vwap_coherence_count'] / max(coherence_results['analyses_with_both'], 1)
                ema_coherence_rate = coherence_results['ema_coherence_count'] / max(coherence_results['analyses_with_both'], 1)
                fibonacci_coherence_rate = coherence_results['fibonacci_coherence_count'] / max(coherence_results['analyses_with_both'], 1)
                overall_coherence_rate = coherence_results['overall_coherence_count'] / max(coherence_results['analyses_with_both'], 1)
                avg_coherence_score = sum(coherence_results['coherence_scores']) / len(coherence_results['coherence_scores']) if coherence_results['coherence_scores'] else 0.0
            else:
                both_present_rate = vwap_coherence_rate = ema_coherence_rate = fibonacci_coherence_rate = overall_coherence_rate = avg_coherence_score = 0.0
            
            logger.info(f"\n   📊 TECHNICAL INDICATORS & LEVELS COHERENCE RESULTS:")
            logger.info(f"      Total analyses: {coherence_results['total_analyses']}")
            logger.info(f"      Analyses with both levels & indicators: {coherence_results['analyses_with_both']} ({both_present_rate:.2f})")
            logger.info(f"      VWAP coherence: {coherence_results['vwap_coherence_count']} ({vwap_coherence_rate:.2f})")
            logger.info(f"      EMA coherence: {coherence_results['ema_coherence_count']} ({ema_coherence_rate:.2f})")
            logger.info(f"      Fibonacci coherence: {coherence_results['fibonacci_coherence_count']} ({fibonacci_coherence_rate:.2f})")
            logger.info(f"      Overall coherence: {coherence_results['overall_coherence_count']} ({overall_coherence_rate:.2f})")
            logger.info(f"      Average coherence score: {avg_coherence_score:.2f}")
            
            # Calculate test success
            success_criteria = [
                coherence_results['total_analyses'] >= 2,  # At least 2 analyses
                both_present_rate >= 0.67,  # At least 67% with both levels & indicators
                overall_coherence_rate >= 0.5,  # At least 50% overall coherence
                avg_coherence_score >= 0.4,  # Average coherence >= 40%
                coherence_results['analyses_with_both'] >= 1  # At least 1 analysis with both
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Technical Indicators & Levels Coherence", True, 
                                   f"Technical coherence successful: {success_count}/{len(success_criteria)} criteria met. Both present: {both_present_rate:.2f}, Overall coherence: {overall_coherence_rate:.2f}, Avg score: {avg_coherence_score:.2f}")
            else:
                self.log_test_result("Technical Indicators & Levels Coherence", False, 
                                   f"Technical coherence issues: {success_count}/{len(success_criteria)} criteria met. May lack coherence between indicators and levels")
                
        except Exception as e:
            self.log_test_result("Technical Indicators & Levels Coherence", False, f"Exception: {str(e)}")
    
    async def test_4_ia2_escalation_criteria_validation(self):
        """Test 4: IA2 Escalation Criteria Validation - Verify RR > 2.0 triggers IA2 escalation"""
        logger.info("\n🔍 TEST 4: IA2 Escalation Criteria Validation")
        
        try:
            escalation_results = {
                'total_analyses': 0,
                'rr_above_2_count': 0,
                'confidence_above_95_count': 0,
                'escalation_criteria_met': 0,
                'ia2_escalations_found': 0,
                'escalation_validations': []
            }
            
            logger.info("   🚀 Testing IA2 escalation criteria validation...")
            logger.info("   📊 Expected: RR > 2.0 OR confidence > 95% triggers IA2 escalation")
            
            # Use symbols from previous tests
            test_symbols = self.actual_test_symbols if self.actual_test_symbols else await self._get_available_symbols()
            
            # Test each symbol for IA2 escalation criteria
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing IA2 escalation criteria for {symbol}...")
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
                        
                        # Check escalation criteria
                        rr_value = ia1_analysis.get('risk_reward_ratio')
                        confidence = ia1_analysis.get('confidence')
                        
                        meets_rr_criteria = False
                        meets_confidence_criteria = False
                        
                        if rr_value and rr_value >= 2.0:
                            escalation_results['rr_above_2_count'] += 1
                            meets_rr_criteria = True
                            logger.info(f"      ✅ RR escalation criteria met: {rr_value:.2f} ≥ 2.0")
                        
                        if confidence and confidence >= 0.95:
                            escalation_results['confidence_above_95_count'] += 1
                            meets_confidence_criteria = True
                            logger.info(f"      ✅ Confidence escalation criteria met: {confidence:.2f} ≥ 0.95")
                        
                        if meets_rr_criteria or meets_confidence_criteria:
                            escalation_results['escalation_criteria_met'] += 1
                            logger.info(f"      ✅ IA2 escalation criteria met for {symbol}")
                            
                            # Check if IA2 analysis is present (indicating escalation occurred)
                            ia2_analysis = analysis_data.get('ia2_analysis')
                            if ia2_analysis:
                                escalation_results['ia2_escalations_found'] += 1
                                logger.info(f"         ✅ IA2 analysis found - escalation occurred")
                            else:
                                logger.info(f"         ⚠️ IA2 analysis not found - escalation may not have occurred")
                        
                        # Store validation details
                        escalation_results['escalation_validations'].append({
                            'symbol': symbol,
                            'rr_value': rr_value,
                            'confidence': confidence,
                            'meets_rr_criteria': meets_rr_criteria,
                            'meets_confidence_criteria': meets_confidence_criteria,
                            'escalation_criteria_met': meets_rr_criteria or meets_confidence_criteria,
                            'ia2_analysis_present': ia2_analysis is not None
                        })
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate final metrics
            if escalation_results['total_analyses'] > 0:
                rr_criteria_rate = escalation_results['rr_above_2_count'] / escalation_results['total_analyses']
                confidence_criteria_rate = escalation_results['confidence_above_95_count'] / escalation_results['total_analyses']
                escalation_criteria_rate = escalation_results['escalation_criteria_met'] / escalation_results['total_analyses']
                escalation_success_rate = escalation_results['ia2_escalations_found'] / max(escalation_results['escalation_criteria_met'], 1)
            else:
                rr_criteria_rate = confidence_criteria_rate = escalation_criteria_rate = escalation_success_rate = 0.0
            
            logger.info(f"\n   📊 IA2 ESCALATION CRITERIA VALIDATION RESULTS:")
            logger.info(f"      Total analyses: {escalation_results['total_analyses']}")
            logger.info(f"      RR ≥ 2.0 criteria met: {escalation_results['rr_above_2_count']} ({rr_criteria_rate:.2f})")
            logger.info(f"      Confidence ≥ 95% criteria met: {escalation_results['confidence_above_95_count']} ({confidence_criteria_rate:.2f})")
            logger.info(f"      Escalation criteria met: {escalation_results['escalation_criteria_met']} ({escalation_criteria_rate:.2f})")
            logger.info(f"      IA2 escalations found: {escalation_results['ia2_escalations_found']} ({escalation_success_rate:.2f})")
            
            if escalation_results['escalation_validations']:
                logger.info(f"      📊 Escalation validation details:")
                for validation in escalation_results['escalation_validations']:
                    logger.info(f"         - {validation['symbol']}: RR={validation['rr_value']}, Conf={validation['confidence']}, Criteria met={validation['escalation_criteria_met']}, IA2 present={validation['ia2_analysis_present']}")
            
            # Calculate test success
            success_criteria = [
                escalation_results['total_analyses'] >= 2,  # At least 2 analyses
                escalation_results['escalation_criteria_met'] >= 1,  # At least 1 meeting criteria
                escalation_criteria_rate >= 0.33,  # At least 33% meeting escalation criteria
                len(escalation_results['escalation_validations']) >= 2  # At least 2 validations
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("IA2 Escalation Criteria Validation", True, 
                                   f"IA2 escalation criteria validation successful: {success_count}/{len(success_criteria)} criteria met. RR criteria: {rr_criteria_rate:.2f}, Confidence criteria: {confidence_criteria_rate:.2f}, Overall: {escalation_criteria_rate:.2f}")
            else:
                self.log_test_result("IA2 Escalation Criteria Validation", False, 
                                   f"IA2 escalation criteria validation issues: {success_count}/{len(success_criteria)} criteria met. May not meet RR > 2.0 or confidence > 95% criteria")
                
        except Exception as e:
            self.log_test_result("IA2 Escalation Criteria Validation", False, f"Exception: {str(e)}")
    
    async def run_all_tests(self):
        """Run all technical levels and RR calculation tests"""
        logger.info("🚀 STARTING TECHNICAL LEVELS & RISK-REWARD CALCULATION VERIFICATION TEST SUITE")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests
        await self.test_1_technical_levels_calculation_verification()
        await self.test_2_risk_reward_calculation_accuracy()
        await self.test_3_technical_indicators_levels_coherence()
        await self.test_4_ia2_escalation_criteria_validation()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 TECHNICAL LEVELS & RR CALCULATION TEST SUITE SUMMARY")
        logger.info("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests:.2%}")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        
        logger.info("\n📋 DETAILED TEST RESULTS:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        # Close MongoDB connection
        if self.mongo_client:
            self.mongo_client.close()
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests/total_tests,
            'total_time': total_time,
            'test_results': self.test_results
        }

async def main():
    """Main function to run the test suite"""
    test_suite = TechnicalLevelsRRTestSuite()
    results = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    if results['failed_tests'] == 0:
        logger.info("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        logger.error(f"❌ {results['failed_tests']} TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
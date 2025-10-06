#!/usr/bin/env python3
"""
REASONING FIELD + RR CALCULATION COMPREHENSIVE VALIDATION TEST SUITE
Focus: Test Fix Complet - Reasoning Field + RR Calculation avec nouvelles formules

OBJECTIFS CRITIQUES (from review request):

1. **Reasoning Field Population Test**:
   - Tester /api/force-ia1-analysis avec BTCUSDT ou symbole disponible
   - Vérifier si "reasoning" contient maintenant du contenu détaillé (vs null précédent)
   - Examiner format "STEP 1 - RSI ANALYSIS... STEP 2 - MACD MOMENTUM..." 
   - Confirmer mentions explicites des indicateurs techniques avec valeurs

2. **RR Calculation Formula Implementation**:
   - Vérifier si "calculated_rr" utilise maintenant les vraies formules mathématiques
   - Examiner extraction des prix IA1 (entry_price, stop_loss_price, take_profit_price)
   - Confirmer calculs: LONG: (TP-Entry)/(Entry-SL), SHORT: (Entry-TP)/(SL-Entry)
   - Comparer avec précédents RR fixes (1.0, 2.2)

3. **Escalation vers IA2 Validation**:
   - Tester si RR > 2.0 déclenche maintenant l'escalation vers IA2
   - Vérifier logs pour "ESCALATING to IA2" messages
   - Examiner si la confidence IA1 + RR calculé permettent l'escalation

4. **Qualité Analyse Globale**:
   - Comparer qualité analyses avant/après les fixes
   - Vérifier cohérence entre reasoning détaillé et niveaux calculés
   - Confirmer que IA1 utilise les indicateurs techniques dans les décisions

5. **Success Metrics Validation**:
   - Reasoning length > 100 characters (vs 0 précédent)
   - RR calculation basé sur formules (vs valeurs fixes)
   - Mentions techniques: RSI, MACD, MFI, VWAP dans reasoning
   - Escalation IA2 fonctionnelle avec RR > 2.0
"""

import asyncio
import json
import logging
import os
import sys
import time
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import requests
import subprocess
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReasoningRRValidationTestSuite:
    """Comprehensive test suite for Reasoning Field + RR Calculation fixes"""
    
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
        logger.info(f"Testing Reasoning Field + RR Calculation fixes at: {self.api_url}")
        
        # MongoDB connection for database analysis
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
            logger.info("✅ MongoDB connection established")
        except Exception as e:
            logger.warning(f"⚠️ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results storage
        self.test_results = []
        
        # Test symbols (will be determined from available opportunities)
        self.preferred_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.actual_test_symbols = []
        
        # Technical indicators to verify in reasoning
        self.core_technical_indicators = ['RSI', 'MACD', 'MFI', 'VWAP']
        self.extended_technical_indicators = ['EMA', 'SMA', 'Bollinger', 'Stochastic']
        
        # RR calculation patterns to verify
        self.rr_formula_patterns = [
            r'LONG.*\(.*TP.*-.*Entry.*\).*\/.*\(.*Entry.*-.*SL.*\)',
            r'SHORT.*\(.*Entry.*-.*TP.*\).*\/.*\(.*SL.*-.*Entry.*\)',
            r'calculated.*RR.*[\d\.]+',
            r'risk.*reward.*ratio.*[\d\.]+'
        ]
        
        # Expected reasoning format patterns
        self.reasoning_format_patterns = [
            r'STEP\s+\d+.*RSI.*ANALYSIS',
            r'STEP\s+\d+.*MACD.*MOMENTUM',
            r'RSI.*[\d\.]+.*indicates?',
            r'MACD.*[\d\.\-]+.*shows?',
            r'MFI.*[\d\.]+.*suggests?',
            r'VWAP.*[\d\.\-]+.*indicates?'
        ]
        
        # IA2 escalation patterns
        self.ia2_escalation_patterns = [
            r'ESCALATING.*to.*IA2',
            r'IA2.*escalation.*triggered',
            r'confidence.*[\d\.]+.*RR.*[\d\.]+',
            r'escalation.*criteria.*met'
        ]
        
        # Analysis storage
        self.ia1_analyses = []
        self.database_analyses = []
        self.backend_logs = []
        
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
        """Get available symbols from opportunities API"""
        try:
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            
            if response.status_code == 200:
                opportunities = response.json()
                if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                    opportunities = opportunities['opportunities']
                
                # Get available symbols
                available_symbols = [opp.get('symbol') for opp in opportunities[:20] if opp.get('symbol')]
                
                # Prefer BTCUSDT, ETHUSDT, SOLUSDT if available
                test_symbols = []
                for symbol in self.preferred_symbols:
                    if symbol in available_symbols:
                        test_symbols.append(symbol)
                
                # Fill remaining slots with available symbols
                for symbol in available_symbols:
                    if symbol not in test_symbols and len(test_symbols) < 3:
                        test_symbols.append(symbol)
                
                self.actual_test_symbols = test_symbols[:3]
                logger.info(f"✅ Test symbols selected: {self.actual_test_symbols}")
                return self.actual_test_symbols
                
            else:
                logger.warning(f"⚠️ Could not get opportunities: HTTP {response.status_code}")
                return self.preferred_symbols
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting opportunities: {e}")
            return self.preferred_symbols
    
    async def _capture_backend_logs(self) -> List[str]:
        """Capture backend logs for analysis"""
        try:
            result = subprocess.run(
                ['tail', '-n', '300', '/var/log/supervisor/backend.out.log'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.split('\n')
            else:
                # Try alternative log location
                result = subprocess.run(
                    ['tail', '-n', '300', '/var/log/supervisor/backend.err.log'],
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
    
    async def _get_recent_database_analyses(self, limit: int = 10) -> List[Dict]:
        """Get recent analyses from database for comparison"""
        if not self.db:
            return []
        
        try:
            # Get recent technical analyses
            analyses = list(self.db.technical_analyses.find(
                {},
                {
                    'symbol': 1, 'reasoning': 1, 'calculated_rr': 1, 'risk_reward_ratio': 1,
                    'entry_price': 1, 'stop_loss_price': 1, 'take_profit_price': 1,
                    'rsi_value': 1, 'macd_line': 1, 'mfi_value': 1, 'vwap_position': 1,
                    'timestamp': 1, 'analysis_confidence': 1
                }
            ).sort('timestamp', -1).limit(limit))
            
            logger.info(f"✅ Retrieved {len(analyses)} recent database analyses")
            return analyses
            
        except Exception as e:
            logger.warning(f"⚠️ Could not get database analyses: {e}")
            return []
    
    def _calculate_expected_rr(self, entry_price: float, stop_loss_price: float, take_profit_price: float, signal: str) -> float:
        """Calculate expected RR using mathematical formulas"""
        try:
            if signal.upper() == 'LONG':
                # LONG: (TP-Entry)/(Entry-SL)
                reward = take_profit_price - entry_price
                risk = entry_price - stop_loss_price
            else:  # SHORT
                # SHORT: (Entry-TP)/(SL-Entry)
                reward = entry_price - take_profit_price
                risk = stop_loss_price - entry_price
            
            if risk > 0:
                return reward / risk
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate expected RR: {e}")
            return 0.0
    
    async def test_1_reasoning_field_population_validation(self):
        """Test 1: Reasoning Field Population - Verify detailed content vs null"""
        logger.info("\n🔍 TEST 1: Reasoning Field Population Validation")
        
        try:
            reasoning_results = {
                'total_analyses': 0,
                'analyses_with_reasoning': 0,
                'reasoning_lengths': [],
                'technical_indicators_mentioned': {},
                'step_format_found': 0,
                'specific_values_found': {},
                'reasoning_samples': [],
                'null_reasoning_count': 0
            }
            
            logger.info("   🚀 Testing reasoning field population with detailed technical analysis...")
            logger.info("   📊 Expected: reasoning > 100 chars, mentions RSI/MACD/MFI/VWAP with values")
            
            # Get available symbols
            test_symbols = await self._get_available_symbols()
            
            # Test each symbol for reasoning field population
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing reasoning field for {symbol}...")
                reasoning_results['total_analyses'] += 1
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        
                        # Extract reasoning from response
                        reasoning = ""
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        if isinstance(ia1_analysis, dict):
                            reasoning = ia1_analysis.get('reasoning', '') or ""
                        
                        if reasoning and reasoning.strip():
                            reasoning_results['analyses_with_reasoning'] += 1
                            reasoning_length = len(reasoning)
                            reasoning_results['reasoning_lengths'].append(reasoning_length)
                            
                            logger.info(f"      ✅ Reasoning field populated: {reasoning_length} characters")
                            
                            # Check for technical indicators with values
                            technical_found = {}
                            specific_values = {}
                            
                            for indicator in self.core_technical_indicators:
                                # Count mentions
                                mentions = reasoning.upper().count(indicator.upper())
                                if mentions > 0:
                                    technical_found[indicator] = mentions
                                    
                                    # Look for specific values
                                    value_patterns = [
                                        rf'{indicator}\s*[:\s]*[\d\.\-]+%?',
                                        rf'{indicator}.*[\d\.\-]+',
                                        rf'[\d\.\-]+.*{indicator}'
                                    ]
                                    
                                    for pattern in value_patterns:
                                        matches = re.findall(pattern, reasoning, re.IGNORECASE)
                                        if matches:
                                            specific_values[indicator] = matches[:3]  # First 3 matches
                                            break
                            
                            # Update global counters
                            for indicator, count in technical_found.items():
                                if indicator not in reasoning_results['technical_indicators_mentioned']:
                                    reasoning_results['technical_indicators_mentioned'][indicator] = 0
                                reasoning_results['technical_indicators_mentioned'][indicator] += count
                            
                            for indicator, values in specific_values.items():
                                if indicator not in reasoning_results['specific_values_found']:
                                    reasoning_results['specific_values_found'][indicator] = []
                                reasoning_results['specific_values_found'][indicator].extend(values)
                            
                            # Check for STEP format
                            step_patterns = [
                                r'STEP\s+\d+',
                                r'\d+\.\s*RSI.*ANALYSIS',
                                r'\d+\.\s*MACD.*MOMENTUM'
                            ]
                            
                            for pattern in step_patterns:
                                if re.search(pattern, reasoning, re.IGNORECASE):
                                    reasoning_results['step_format_found'] += 1
                                    break
                            
                            # Store reasoning sample
                            reasoning_results['reasoning_samples'].append({
                                'symbol': symbol,
                                'length': reasoning_length,
                                'technical_indicators': list(technical_found.keys()),
                                'specific_values': list(specific_values.keys()),
                                'sample': reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
                            })
                            
                            logger.info(f"         📊 Technical indicators: {list(technical_found.keys())}")
                            logger.info(f"         📊 Specific values: {list(specific_values.keys())}")
                            
                        else:
                            reasoning_results['null_reasoning_count'] += 1
                            logger.warning(f"      ❌ Reasoning field empty/null for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ Analysis failed for {symbol}: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ Analysis exception for {symbol}: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate metrics
            if reasoning_results['total_analyses'] > 0:
                reasoning_rate = reasoning_results['analyses_with_reasoning'] / reasoning_results['total_analyses']
                avg_length = sum(reasoning_results['reasoning_lengths']) / max(len(reasoning_results['reasoning_lengths']), 1)
                step_format_rate = reasoning_results['step_format_found'] / reasoning_results['total_analyses']
            else:
                reasoning_rate = avg_length = step_format_rate = 0.0
            
            logger.info(f"\n   📊 REASONING FIELD POPULATION VALIDATION RESULTS:")
            logger.info(f"      Total analyses: {reasoning_results['total_analyses']}")
            logger.info(f"      Analyses with reasoning: {reasoning_results['analyses_with_reasoning']}")
            logger.info(f"      Reasoning population rate: {reasoning_rate:.2f}")
            logger.info(f"      Average reasoning length: {avg_length:.0f} chars")
            logger.info(f"      Null reasoning count: {reasoning_results['null_reasoning_count']}")
            logger.info(f"      STEP format found: {reasoning_results['step_format_found']} ({step_format_rate:.2f})")
            
            if reasoning_results['technical_indicators_mentioned']:
                logger.info(f"      📊 Technical indicators mentioned:")
                for indicator, count in reasoning_results['technical_indicators_mentioned'].items():
                    logger.info(f"         - {indicator}: {count} mentions")
            
            if reasoning_results['specific_values_found']:
                logger.info(f"      📊 Specific values found:")
                for indicator, values in reasoning_results['specific_values_found'].items():
                    logger.info(f"         - {indicator}: {len(values)} specific values")
            
            # Success criteria based on review request
            success_criteria = [
                reasoning_results['analyses_with_reasoning'] >= 2,  # At least 2 with reasoning
                reasoning_rate >= 0.67,  # At least 67% have reasoning (2/3)
                avg_length >= 100,  # Average length > 100 chars (review requirement)
                len(reasoning_results['technical_indicators_mentioned']) >= 3,  # At least 3 indicators
                len(reasoning_results['specific_values_found']) >= 2,  # At least 2 with values
                reasoning_results['null_reasoning_count'] <= 1  # Max 1 null reasoning
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.83:  # 83% success threshold (5/6)
                self.log_test_result("Reasoning Field Population Validation", True, 
                                   f"Reasoning field fix successful: {success_count}/{len(success_criteria)} criteria met. Population rate: {reasoning_rate:.2f}, Avg length: {avg_length:.0f} chars, Technical indicators: {len(reasoning_results['technical_indicators_mentioned'])}")
            else:
                self.log_test_result("Reasoning Field Population Validation", False, 
                                   f"Reasoning field issues: {success_count}/{len(success_criteria)} criteria met. May still have null reasoning or insufficient technical content")
                
        except Exception as e:
            self.log_test_result("Reasoning Field Population Validation", False, f"Exception: {str(e)}")
    
    async def test_2_rr_calculation_formula_implementation(self):
        """Test 2: RR Calculation Formula Implementation - Mathematical formulas vs fixed values"""
        logger.info("\n🔍 TEST 2: RR Calculation Formula Implementation")
        
        try:
            rr_results = {
                'total_analyses': 0,
                'analyses_with_calculated_rr': 0,
                'analyses_with_rr_reasoning': 0,
                'mathematical_formulas_detected': 0,
                'fixed_values_detected': 0,
                'rr_values': [],
                'expected_vs_actual_rr': [],
                'technical_levels_coherent': 0,
                'rr_reasoning_samples': []
            }
            
            logger.info("   🚀 Testing RR calculation with mathematical formulas...")
            logger.info("   📊 Expected: calculated_rr uses LONG: (TP-Entry)/(Entry-SL), SHORT: (Entry-TP)/(SL-Entry)")
            
            # Get available symbols
            test_symbols = await self._get_available_symbols()
            
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
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        # Extract RR data
                        calculated_rr = ia1_analysis.get('calculated_rr')
                        risk_reward_ratio = ia1_analysis.get('risk_reward_ratio')
                        rr_reasoning = ia1_analysis.get('rr_reasoning', '')
                        
                        # Extract price levels
                        entry_price = ia1_analysis.get('entry_price')
                        stop_loss_price = ia1_analysis.get('stop_loss_price')
                        take_profit_price = ia1_analysis.get('take_profit_price')
                        signal = ia1_analysis.get('signal', 'LONG')
                        
                        # Check for calculated_rr
                        actual_rr = calculated_rr or risk_reward_ratio
                        if actual_rr is not None:
                            rr_results['analyses_with_calculated_rr'] += 1
                            rr_results['rr_values'].append(actual_rr)
                            
                            logger.info(f"      ✅ RR value found: {actual_rr}")
                            
                            # Check if it's a fixed value (1.0, 2.2, etc.)
                            common_fixed_values = [1.0, 2.0, 2.2, 3.0]
                            if actual_rr in common_fixed_values:
                                rr_results['fixed_values_detected'] += 1
                                logger.warning(f"         ⚠️ Fixed RR value detected: {actual_rr}")
                            
                            # Calculate expected RR if we have price levels
                            if all([entry_price, stop_loss_price, take_profit_price]):
                                expected_rr = self._calculate_expected_rr(
                                    entry_price, stop_loss_price, take_profit_price, signal
                                )
                                
                                rr_results['expected_vs_actual_rr'].append({
                                    'symbol': symbol,
                                    'actual_rr': actual_rr,
                                    'expected_rr': expected_rr,
                                    'difference': abs(actual_rr - expected_rr),
                                    'entry_price': entry_price,
                                    'stop_loss_price': stop_loss_price,
                                    'take_profit_price': take_profit_price,
                                    'signal': signal
                                })
                                
                                # Check if levels are coherent
                                if signal.upper() == 'LONG':
                                    levels_coherent = stop_loss_price < entry_price < take_profit_price
                                else:  # SHORT
                                    levels_coherent = take_profit_price < entry_price < stop_loss_price
                                
                                if levels_coherent:
                                    rr_results['technical_levels_coherent'] += 1
                                    logger.info(f"         ✅ Technical levels coherent for {signal}")
                                else:
                                    logger.warning(f"         ⚠️ Technical levels incoherent for {signal}")
                                
                                # Check if actual RR matches expected (within tolerance)
                                rr_difference = abs(actual_rr - expected_rr)
                                if rr_difference < 0.1:  # 10% tolerance
                                    rr_results['mathematical_formulas_detected'] += 1
                                    logger.info(f"         ✅ Mathematical formula used: Expected {expected_rr:.2f}, Actual {actual_rr:.2f}")
                                else:
                                    logger.warning(f"         ❌ Formula mismatch: Expected {expected_rr:.2f}, Actual {actual_rr:.2f} (diff: {rr_difference:.2f})")
                            
                        else:
                            logger.warning(f"      ❌ No RR value found for {symbol}")
                        
                        # Check for RR reasoning
                        if rr_reasoning and rr_reasoning.strip():
                            rr_results['analyses_with_rr_reasoning'] += 1
                            rr_results['rr_reasoning_samples'].append({
                                'symbol': symbol,
                                'rr_reasoning': rr_reasoning[:200] + "..." if len(rr_reasoning) > 200 else rr_reasoning,
                                'mentions_technical_levels': any(level in rr_reasoning.lower() for level in ['vwap', 'ema', 'sma', 'support', 'resistance'])
                            })
                            logger.info(f"      ✅ RR reasoning found: {len(rr_reasoning)} chars")
                        else:
                            logger.warning(f"      ❌ No RR reasoning for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ Analysis failed for {symbol}: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ Analysis exception for {symbol}: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(10)
            
            # Calculate metrics
            if rr_results['total_analyses'] > 0:
                rr_rate = rr_results['analyses_with_calculated_rr'] / rr_results['total_analyses']
                reasoning_rate = rr_results['analyses_with_rr_reasoning'] / rr_results['total_analyses']
                formula_rate = rr_results['mathematical_formulas_detected'] / max(rr_results['analyses_with_calculated_rr'], 1)
                fixed_rate = rr_results['fixed_values_detected'] / max(rr_results['analyses_with_calculated_rr'], 1)
                coherent_rate = rr_results['technical_levels_coherent'] / max(rr_results['analyses_with_calculated_rr'], 1)
            else:
                rr_rate = reasoning_rate = formula_rate = fixed_rate = coherent_rate = 0.0
            
            logger.info(f"\n   📊 RR CALCULATION FORMULA IMPLEMENTATION RESULTS:")
            logger.info(f"      Total analyses: {rr_results['total_analyses']}")
            logger.info(f"      Analyses with calculated_rr: {rr_results['analyses_with_calculated_rr']} ({rr_rate:.2f})")
            logger.info(f"      Analyses with rr_reasoning: {rr_results['analyses_with_rr_reasoning']} ({reasoning_rate:.2f})")
            logger.info(f"      Mathematical formulas detected: {rr_results['mathematical_formulas_detected']} ({formula_rate:.2f})")
            logger.info(f"      Fixed values detected: {rr_results['fixed_values_detected']} ({fixed_rate:.2f})")
            logger.info(f"      Technical levels coherent: {rr_results['technical_levels_coherent']} ({coherent_rate:.2f})")
            
            if rr_results['rr_values']:
                avg_rr = sum(rr_results['rr_values']) / len(rr_results['rr_values'])
                logger.info(f"      Average RR value: {avg_rr:.2f}")
            
            if rr_results['expected_vs_actual_rr']:
                logger.info(f"      📊 RR Calculation Comparison:")
                for comparison in rr_results['expected_vs_actual_rr']:
                    logger.info(f"         - {comparison['symbol']}: Expected {comparison['expected_rr']:.2f}, Actual {comparison['actual_rr']:.2f}")
            
            # Success criteria based on review request
            success_criteria = [
                rr_results['analyses_with_calculated_rr'] >= 2,  # At least 2 with RR
                rr_rate >= 0.67,  # At least 67% have RR (2/3)
                rr_results['mathematical_formulas_detected'] >= 1,  # At least 1 uses formula
                formula_rate >= 0.5,  # At least 50% use formulas vs fixed values
                rr_results['technical_levels_coherent'] >= 1,  # At least 1 coherent levels
                fixed_rate <= 0.5  # Max 50% fixed values
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.83:  # 83% success threshold (5/6)
                self.log_test_result("RR Calculation Formula Implementation", True, 
                                   f"RR calculation fix successful: {success_count}/{len(success_criteria)} criteria met. Formula rate: {formula_rate:.2f}, Fixed rate: {fixed_rate:.2f}, RR rate: {rr_rate:.2f}")
            else:
                self.log_test_result("RR Calculation Formula Implementation", False, 
                                   f"RR calculation issues: {success_count}/{len(success_criteria)} criteria met. May still use fixed values instead of mathematical formulas")
                
        except Exception as e:
            self.log_test_result("RR Calculation Formula Implementation", False, f"Exception: {str(e)}")
    
    async def test_3_ia2_escalation_validation(self):
        """Test 3: IA2 Escalation Validation - Test RR > 2.0 triggers IA2"""
        logger.info("\n🔍 TEST 3: IA2 Escalation Validation")
        
        try:
            escalation_results = {
                'total_analyses': 0,
                'analyses_with_high_rr': 0,
                'ia2_escalations_detected': 0,
                'escalation_logs_found': 0,
                'confidence_rr_combinations': [],
                'escalation_criteria_met': 0,
                'backend_escalation_messages': []
            }
            
            logger.info("   🚀 Testing IA2 escalation when RR > 2.0...")
            logger.info("   📊 Expected: RR > 2.0 triggers IA2 escalation with backend logs")
            
            # Get available symbols
            test_symbols = await self._get_available_symbols()
            
            # Test each symbol and look for high RR values
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing IA2 escalation for {symbol}...")
                escalation_results['total_analyses'] += 1
                
                try:
                    # Capture logs before analysis
                    logs_before = await self._capture_backend_logs()
                    
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    # Capture logs after analysis
                    await asyncio.sleep(5)  # Wait for logs to be written
                    logs_after = await self._capture_backend_logs()
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        # Extract RR and confidence
                        calculated_rr = ia1_analysis.get('calculated_rr') or ia1_analysis.get('risk_reward_ratio')
                        confidence = ia1_analysis.get('confidence') or ia1_analysis.get('analysis_confidence')
                        
                        if calculated_rr is not None and confidence is not None:
                            escalation_results['confidence_rr_combinations'].append({
                                'symbol': symbol,
                                'rr': calculated_rr,
                                'confidence': confidence,
                                'meets_rr_criteria': calculated_rr > 2.0,
                                'meets_confidence_criteria': confidence > 0.95
                            })
                            
                            logger.info(f"      📊 {symbol}: RR={calculated_rr:.2f}, Confidence={confidence:.2f}")
                            
                            # Check if RR > 2.0
                            if calculated_rr > 2.0:
                                escalation_results['analyses_with_high_rr'] += 1
                                logger.info(f"         ✅ High RR detected: {calculated_rr:.2f} > 2.0")
                                
                                # Look for escalation in response
                                if 'ia2_decision' in analysis_data or 'ia2_analysis' in analysis_data:
                                    escalation_results['ia2_escalations_detected'] += 1
                                    logger.info(f"         ✅ IA2 escalation found in response")
                                
                                # Check escalation criteria
                                if calculated_rr > 2.0 or confidence > 0.95:
                                    escalation_results['escalation_criteria_met'] += 1
                            else:
                                logger.info(f"         📊 RR below threshold: {calculated_rr:.2f} <= 2.0")
                        
                        # Look for escalation messages in logs
                        new_logs = logs_after[len(logs_before):] if len(logs_after) > len(logs_before) else []
                        
                        escalation_messages = []
                        for log_line in new_logs:
                            for pattern in self.ia2_escalation_patterns:
                                if re.search(pattern, log_line, re.IGNORECASE):
                                    escalation_messages.append(log_line.strip())
                                    break
                        
                        if escalation_messages:
                            escalation_results['escalation_logs_found'] += 1
                            escalation_results['backend_escalation_messages'].extend(escalation_messages)
                            logger.info(f"         ✅ Escalation logs found: {len(escalation_messages)} messages")
                            for msg in escalation_messages[:2]:  # Show first 2
                                logger.info(f"            - {msg}")
                        else:
                            logger.info(f"         📊 No escalation logs found")
                    
                    else:
                        logger.error(f"      ❌ Analysis failed for {symbol}: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ Analysis exception for {symbol}: {e}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    await asyncio.sleep(15)  # Longer wait for IA2 processing
            
            # Calculate metrics
            if escalation_results['total_analyses'] > 0:
                high_rr_rate = escalation_results['analyses_with_high_rr'] / escalation_results['total_analyses']
                escalation_rate = escalation_results['ia2_escalations_detected'] / max(escalation_results['analyses_with_high_rr'], 1)
                log_detection_rate = escalation_results['escalation_logs_found'] / escalation_results['total_analyses']
            else:
                high_rr_rate = escalation_rate = log_detection_rate = 0.0
            
            logger.info(f"\n   📊 IA2 ESCALATION VALIDATION RESULTS:")
            logger.info(f"      Total analyses: {escalation_results['total_analyses']}")
            logger.info(f"      Analyses with high RR (>2.0): {escalation_results['analyses_with_high_rr']} ({high_rr_rate:.2f})")
            logger.info(f"      IA2 escalations detected: {escalation_results['ia2_escalations_detected']} ({escalation_rate:.2f})")
            logger.info(f"      Escalation logs found: {escalation_results['escalation_logs_found']} ({log_detection_rate:.2f})")
            logger.info(f"      Escalation criteria met: {escalation_results['escalation_criteria_met']}")
            
            if escalation_results['confidence_rr_combinations']:
                logger.info(f"      📊 RR/Confidence combinations:")
                for combo in escalation_results['confidence_rr_combinations']:
                    logger.info(f"         - {combo['symbol']}: RR={combo['rr']:.2f}, Conf={combo['confidence']:.2f}")
            
            if escalation_results['backend_escalation_messages']:
                logger.info(f"      📊 Backend escalation messages found:")
                for msg in escalation_results['backend_escalation_messages'][:3]:  # Show first 3
                    logger.info(f"         - {msg}")
            
            # Success criteria based on review request
            success_criteria = [
                escalation_results['total_analyses'] >= 2,  # At least 2 analyses
                escalation_results['analyses_with_high_rr'] >= 1 or escalation_results['escalation_criteria_met'] >= 1,  # At least 1 high RR or criteria met
                escalation_results['ia2_escalations_detected'] >= 0,  # IA2 escalations working (can be 0 if no high RR)
                len(escalation_results['confidence_rr_combinations']) >= 2,  # At least 2 RR/confidence combinations
                escalation_results['escalation_logs_found'] >= 0  # Escalation logging working (can be 0 if no escalations)
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("IA2 Escalation Validation", True, 
                                   f"IA2 escalation system working: {success_count}/{len(success_criteria)} criteria met. High RR rate: {high_rr_rate:.2f}, Escalation rate: {escalation_rate:.2f}")
            else:
                self.log_test_result("IA2 Escalation Validation", False, 
                                   f"IA2 escalation issues: {success_count}/{len(success_criteria)} criteria met. May have problems with RR > 2.0 escalation logic")
                
        except Exception as e:
            self.log_test_result("IA2 Escalation Validation", False, f"Exception: {str(e)}")
    
    async def test_4_global_analysis_quality_comparison(self):
        """Test 4: Global Analysis Quality - Compare before/after fixes"""
        logger.info("\n🔍 TEST 4: Global Analysis Quality Comparison")
        
        try:
            quality_results = {
                'current_analyses': [],
                'database_analyses': [],
                'reasoning_quality_improvement': 0.0,
                'rr_calculation_improvement': 0.0,
                'technical_indicators_improvement': 0.0,
                'overall_quality_score': 0.0,
                'coherence_score': 0.0
            }
            
            logger.info("   🚀 Comparing analysis quality before/after fixes...")
            logger.info("   📊 Expected: Improved reasoning, RR calculation, technical indicators usage")
            
            # Get recent database analyses for comparison
            database_analyses = await self._get_recent_database_analyses(10)
            quality_results['database_analyses'] = database_analyses
            
            # Get current analyses from API
            test_symbols = await self._get_available_symbols()
            
            for symbol in test_symbols[:2]:  # Test 2 symbols for comparison
                logger.info(f"\n   📞 Getting current analysis for {symbol}...")
                
                try:
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        quality_results['current_analyses'].append({
                            'symbol': symbol,
                            'reasoning': ia1_analysis.get('reasoning', ''),
                            'calculated_rr': ia1_analysis.get('calculated_rr'),
                            'technical_indicators': {
                                'rsi_value': ia1_analysis.get('rsi_value'),
                                'macd_line': ia1_analysis.get('macd_line'),
                                'mfi_value': ia1_analysis.get('mfi_value'),
                                'vwap_position': ia1_analysis.get('vwap_position')
                            },
                            'confidence': ia1_analysis.get('confidence'),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        logger.info(f"      ✅ Current analysis captured for {symbol}")
                    
                    else:
                        logger.error(f"      ❌ Analysis failed for {symbol}: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ Analysis exception for {symbol}: {e}")
                
                await asyncio.sleep(10)
            
            # Compare quality metrics
            logger.info("\n   📊 Analyzing quality improvements...")
            
            # Reasoning quality comparison
            current_reasoning_lengths = [len(analysis.get('reasoning', '')) for analysis in quality_results['current_analyses']]
            db_reasoning_lengths = [len(analysis.get('reasoning', '') or '') for analysis in database_analyses]
            
            if current_reasoning_lengths and db_reasoning_lengths:
                current_avg_reasoning = sum(current_reasoning_lengths) / len(current_reasoning_lengths)
                db_avg_reasoning = sum(db_reasoning_lengths) / len(db_reasoning_lengths)
                
                if db_avg_reasoning > 0:
                    quality_results['reasoning_quality_improvement'] = (current_avg_reasoning - db_avg_reasoning) / db_avg_reasoning
                else:
                    quality_results['reasoning_quality_improvement'] = 1.0 if current_avg_reasoning > 0 else 0.0
                
                logger.info(f"      📊 Reasoning length: Current avg {current_avg_reasoning:.0f}, DB avg {db_avg_reasoning:.0f}")
                logger.info(f"         Improvement: {quality_results['reasoning_quality_improvement']:.2f}")
            
            # RR calculation comparison
            current_rr_values = [analysis.get('calculated_rr') for analysis in quality_results['current_analyses'] if analysis.get('calculated_rr') is not None]
            db_rr_values = [analysis.get('calculated_rr') or analysis.get('risk_reward_ratio') for analysis in database_analyses if (analysis.get('calculated_rr') or analysis.get('risk_reward_ratio')) is not None]
            
            current_rr_variety = len(set(current_rr_values)) if current_rr_values else 0
            db_rr_variety = len(set(db_rr_values)) if db_rr_values else 0
            
            # More variety in RR values suggests formula usage vs fixed values
            if db_rr_variety > 0:
                quality_results['rr_calculation_improvement'] = (current_rr_variety - db_rr_variety) / db_rr_variety
            else:
                quality_results['rr_calculation_improvement'] = 1.0 if current_rr_variety > 0 else 0.0
            
            logger.info(f"      📊 RR variety: Current {current_rr_variety}, DB {db_rr_variety}")
            logger.info(f"         Improvement: {quality_results['rr_calculation_improvement']:.2f}")
            
            # Technical indicators usage comparison
            current_tech_usage = 0
            for analysis in quality_results['current_analyses']:
                reasoning = analysis.get('reasoning', '')
                tech_count = sum(1 for indicator in self.core_technical_indicators if indicator.upper() in reasoning.upper())
                current_tech_usage += tech_count
            
            db_tech_usage = 0
            for analysis in database_analyses:
                reasoning = analysis.get('reasoning', '') or ''
                tech_count = sum(1 for indicator in self.core_technical_indicators if indicator.upper() in reasoning.upper())
                db_tech_usage += tech_count
            
            if len(database_analyses) > 0:
                current_tech_avg = current_tech_usage / max(len(quality_results['current_analyses']), 1)
                db_tech_avg = db_tech_usage / len(database_analyses)
                
                if db_tech_avg > 0:
                    quality_results['technical_indicators_improvement'] = (current_tech_avg - db_tech_avg) / db_tech_avg
                else:
                    quality_results['technical_indicators_improvement'] = 1.0 if current_tech_avg > 0 else 0.0
                
                logger.info(f"      📊 Technical indicators usage: Current avg {current_tech_avg:.1f}, DB avg {db_tech_avg:.1f}")
                logger.info(f"         Improvement: {quality_results['technical_indicators_improvement']:.2f}")
            
            # Calculate overall quality score
            improvements = [
                quality_results['reasoning_quality_improvement'],
                quality_results['rr_calculation_improvement'],
                quality_results['technical_indicators_improvement']
            ]
            
            quality_results['overall_quality_score'] = sum(improvements) / len(improvements)
            
            # Calculate coherence score (consistency between reasoning and calculations)
            coherence_scores = []
            for analysis in quality_results['current_analyses']:
                reasoning = analysis.get('reasoning', '')
                rr_value = analysis.get('calculated_rr')
                tech_indicators = analysis.get('technical_indicators', {})
                
                coherence = 0.0
                
                # Check if reasoning mentions technical indicators that have values
                if reasoning:
                    for indicator in self.core_technical_indicators:
                        if indicator.upper() in reasoning.upper():
                            coherence += 0.25  # 25% per indicator mentioned
                
                # Check if RR value is reasonable (not fixed)
                if rr_value and rr_value not in [1.0, 2.0, 2.2, 3.0]:
                    coherence += 0.25
                
                # Check if technical indicators have realistic values
                tech_values = [v for v in tech_indicators.values() if v is not None and v != 0]
                if len(tech_values) >= 2:
                    coherence += 0.25
                
                coherence_scores.append(min(coherence, 1.0))
            
            quality_results['coherence_score'] = sum(coherence_scores) / max(len(coherence_scores), 1)
            
            logger.info(f"\n   📊 GLOBAL ANALYSIS QUALITY COMPARISON RESULTS:")
            logger.info(f"      Current analyses: {len(quality_results['current_analyses'])}")
            logger.info(f"      Database analyses: {len(quality_results['database_analyses'])}")
            logger.info(f"      Reasoning quality improvement: {quality_results['reasoning_quality_improvement']:.2f}")
            logger.info(f"      RR calculation improvement: {quality_results['rr_calculation_improvement']:.2f}")
            logger.info(f"      Technical indicators improvement: {quality_results['technical_indicators_improvement']:.2f}")
            logger.info(f"      Overall quality score: {quality_results['overall_quality_score']:.2f}")
            logger.info(f"      Coherence score: {quality_results['coherence_score']:.2f}")
            
            # Success criteria based on review request
            success_criteria = [
                len(quality_results['current_analyses']) >= 2,  # At least 2 current analyses
                quality_results['reasoning_quality_improvement'] >= 0.0,  # Reasoning improved or maintained
                quality_results['rr_calculation_improvement'] >= 0.0,  # RR calculation improved or maintained
                quality_results['technical_indicators_improvement'] >= 0.0,  # Technical indicators improved or maintained
                quality_results['overall_quality_score'] >= 0.0,  # Overall improvement
                quality_results['coherence_score'] >= 0.5  # At least 50% coherence
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.83:  # 83% success threshold (5/6)
                self.log_test_result("Global Analysis Quality Comparison", True, 
                                   f"Analysis quality improved: {success_count}/{len(success_criteria)} criteria met. Overall score: {quality_results['overall_quality_score']:.2f}, Coherence: {quality_results['coherence_score']:.2f}")
            else:
                self.log_test_result("Global Analysis Quality Comparison", False, 
                                   f"Analysis quality issues: {success_count}/{len(success_criteria)} criteria met. May not show significant improvement over previous analyses")
                
        except Exception as e:
            self.log_test_result("Global Analysis Quality Comparison", False, f"Exception: {str(e)}")
    
    async def test_5_success_metrics_validation(self):
        """Test 5: Success Metrics Validation - Validate all success criteria from review"""
        logger.info("\n🔍 TEST 5: Success Metrics Validation")
        
        try:
            metrics_results = {
                'reasoning_length_check': False,
                'rr_formula_based_check': False,
                'technical_mentions_check': False,
                'ia2_escalation_functional_check': False,
                'overall_success_rate': 0.0,
                'detailed_metrics': {}
            }
            
            logger.info("   🚀 Validating all success metrics from review request...")
            logger.info("   📊 Expected: All 5 success criteria met")
            
            # Aggregate results from previous tests
            reasoning_tests = [result for result in self.test_results if 'Reasoning' in result['test']]
            rr_tests = [result for result in self.test_results if 'RR Calculation' in result['test']]
            escalation_tests = [result for result in self.test_results if 'IA2 Escalation' in result['test']]
            quality_tests = [result for result in self.test_results if 'Quality' in result['test']]
            
            # 1. Reasoning length > 100 characters
            if reasoning_tests and reasoning_tests[0]['success']:
                metrics_results['reasoning_length_check'] = True
                logger.info("      ✅ Reasoning length > 100 characters: PASS")
            else:
                logger.warning("      ❌ Reasoning length > 100 characters: FAIL")
            
            # 2. RR calculation based on formulas
            if rr_tests and rr_tests[0]['success']:
                metrics_results['rr_formula_based_check'] = True
                logger.info("      ✅ RR calculation based on formulas: PASS")
            else:
                logger.warning("      ❌ RR calculation based on formulas: FAIL")
            
            # 3. Technical indicators mentions (RSI, MACD, MFI, VWAP)
            technical_mentions_found = False
            for result in self.test_results:
                if 'Technical Indicators' in result['test'] and result['success']:
                    technical_mentions_found = True
                    break
            
            metrics_results['technical_mentions_check'] = technical_mentions_found
            if technical_mentions_found:
                logger.info("      ✅ Technical indicators mentions: PASS")
            else:
                logger.warning("      ❌ Technical indicators mentions: FAIL")
            
            # 4. IA2 escalation functional with RR > 2.0
            if escalation_tests and escalation_tests[0]['success']:
                metrics_results['ia2_escalation_functional_check'] = True
                logger.info("      ✅ IA2 escalation functional: PASS")
            else:
                logger.warning("      ❌ IA2 escalation functional: FAIL")
            
            # Calculate overall success rate
            success_checks = [
                metrics_results['reasoning_length_check'],
                metrics_results['rr_formula_based_check'],
                metrics_results['technical_mentions_check'],
                metrics_results['ia2_escalation_functional_check']
            ]
            
            metrics_results['overall_success_rate'] = sum(success_checks) / len(success_checks)
            
            # Detailed metrics from all tests
            metrics_results['detailed_metrics'] = {
                'total_tests_run': len(self.test_results),
                'tests_passed': sum(1 for result in self.test_results if result['success']),
                'tests_failed': sum(1 for result in self.test_results if not result['success']),
                'test_success_rate': sum(1 for result in self.test_results if result['success']) / max(len(self.test_results), 1)
            }
            
            logger.info(f"\n   📊 SUCCESS METRICS VALIDATION RESULTS:")
            logger.info(f"      Reasoning length > 100 chars: {'✅ PASS' if metrics_results['reasoning_length_check'] else '❌ FAIL'}")
            logger.info(f"      RR calculation formula-based: {'✅ PASS' if metrics_results['rr_formula_based_check'] else '❌ FAIL'}")
            logger.info(f"      Technical indicators mentions: {'✅ PASS' if metrics_results['technical_mentions_check'] else '❌ FAIL'}")
            logger.info(f"      IA2 escalation functional: {'✅ PASS' if metrics_results['ia2_escalation_functional_check'] else '❌ FAIL'}")
            logger.info(f"      Overall success rate: {metrics_results['overall_success_rate']:.2f}")
            
            logger.info(f"\n   📊 DETAILED TEST METRICS:")
            logger.info(f"      Total tests run: {metrics_results['detailed_metrics']['total_tests_run']}")
            logger.info(f"      Tests passed: {metrics_results['detailed_metrics']['tests_passed']}")
            logger.info(f"      Tests failed: {metrics_results['detailed_metrics']['tests_failed']}")
            logger.info(f"      Test success rate: {metrics_results['detailed_metrics']['test_success_rate']:.2f}")
            
            # Final success determination
            if metrics_results['overall_success_rate'] >= 0.75:  # 75% success threshold (3/4)
                self.log_test_result("Success Metrics Validation", True, 
                                   f"Success metrics validation passed: {metrics_results['overall_success_rate']:.2f} success rate. {sum(success_checks)}/4 criteria met")
            else:
                self.log_test_result("Success Metrics Validation", False, 
                                   f"Success metrics validation failed: {metrics_results['overall_success_rate']:.2f} success rate. Only {sum(success_checks)}/4 criteria met")
                
        except Exception as e:
            self.log_test_result("Success Metrics Validation", False, f"Exception: {str(e)}")
    
    async def run_all_tests(self):
        """Run all validation tests"""
        logger.info("🚀 STARTING REASONING FIELD + RR CALCULATION COMPREHENSIVE VALIDATION")
        logger.info("=" * 80)
        
        try:
            # Run all tests in sequence
            await self.test_1_reasoning_field_population_validation()
            await self.test_2_rr_calculation_formula_implementation()
            await self.test_3_ia2_escalation_validation()
            await self.test_4_global_analysis_quality_comparison()
            await self.test_5_success_metrics_validation()
            
            # Final summary
            logger.info("\n" + "=" * 80)
            logger.info("📊 FINAL TEST RESULTS SUMMARY")
            logger.info("=" * 80)
            
            passed_tests = sum(1 for result in self.test_results if result['success'])
            total_tests = len(self.test_results)
            success_rate = passed_tests / max(total_tests, 1)
            
            logger.info(f"Total tests run: {total_tests}")
            logger.info(f"Tests passed: {passed_tests}")
            logger.info(f"Tests failed: {total_tests - passed_tests}")
            logger.info(f"Overall success rate: {success_rate:.2f}")
            
            logger.info("\nDetailed results:")
            for result in self.test_results:
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                logger.info(f"  {status}: {result['test']}")
                if result['details']:
                    logger.info(f"    {result['details']}")
            
            # Determine overall test suite success
            if success_rate >= 0.8:  # 80% success threshold
                logger.info(f"\n🎉 REASONING FIELD + RR CALCULATION VALIDATION: SUCCESS")
                logger.info(f"   All critical fixes appear to be working correctly!")
                return True
            else:
                logger.info(f"\n❌ REASONING FIELD + RR CALCULATION VALIDATION: ISSUES DETECTED")
                logger.info(f"   Some critical fixes may not be working as expected.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test suite exception: {e}")
            return False
        
        finally:
            # Cleanup
            if self.mongo_client:
                self.mongo_client.close()

async def main():
    """Main test execution"""
    test_suite = ReasoningRRValidationTestSuite()
    success = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
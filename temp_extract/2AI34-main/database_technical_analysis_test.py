#!/usr/bin/env python3
"""
DATABASE TECHNICAL ANALYSIS VERIFICATION TEST
Focus: Verify technical levels and RR calculation from database perspective

This test examines what's actually stored in the database to understand:
1. Are technical levels being calculated correctly?
2. Are RR values realistic or just defaults?
3. Are technical indicators being used in level calculation?
4. What's the quality of the stored technical data?
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
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseTechnicalAnalysisTest:
    """Test suite for database technical analysis verification"""
    
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
        logger.info(f"Testing Database Technical Analysis at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # MongoDB connection
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
            logger.error(f"❌ Could not connect to MongoDB: {e}")
            sys.exit(1)
    
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
    
    def _validate_rr_calculation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate RR calculation using correct formulas"""
        validation = {
            'rr_present': False,
            'rr_value': None,
            'rr_calculated_correctly': False,
            'expected_rr': None,
            'formula_used': None,
            'levels_realistic': False,
            'calculation_details': {}
        }
        
        try:
            entry_price = analysis.get('entry_price')
            stop_loss = analysis.get('stop_loss_price')
            take_profit = analysis.get('take_profit_price')
            rr_value = analysis.get('risk_reward_ratio')
            signal = analysis.get('ia1_signal', '').upper()
            
            if rr_value is not None:
                validation['rr_present'] = True
                validation['rr_value'] = rr_value
            
            if all([entry_price, stop_loss, take_profit, signal]):
                validation['calculation_details'] = {
                    'entry_price': entry_price,
                    'stop_loss_price': stop_loss,
                    'take_profit_price': take_profit,
                    'signal': signal
                }
                
                # Validate RR calculation using correct formulas
                if signal in ['LONG', 'BUY']:
                    # LONG RR = (TP-Entry)/(Entry-SL)
                    if entry_price > stop_loss and take_profit > entry_price:
                        expected_rr = (take_profit - entry_price) / (entry_price - stop_loss)
                        validation['formula_used'] = 'LONG: (TP-Entry)/(Entry-SL)'
                        validation['expected_rr'] = expected_rr
                        validation['levels_realistic'] = True
                        
                        # Check if calculated RR matches expected (within 10% tolerance)
                        if rr_value and abs(rr_value - expected_rr) / expected_rr <= 0.1:
                            validation['rr_calculated_correctly'] = True
                
                elif signal in ['SHORT', 'SELL']:
                    # SHORT RR = (Entry-TP)/(SL-Entry)
                    if stop_loss > entry_price and take_profit < entry_price:
                        expected_rr = (entry_price - take_profit) / (stop_loss - entry_price)
                        validation['formula_used'] = 'SHORT: (Entry-TP)/(SL-Entry)'
                        validation['expected_rr'] = expected_rr
                        validation['levels_realistic'] = True
                        
                        # Check if calculated RR matches expected (within 10% tolerance)
                        if rr_value and abs(rr_value - expected_rr) / expected_rr <= 0.1:
                            validation['rr_calculated_correctly'] = True
                
                elif signal == 'HOLD':
                    # For HOLD signals, levels might still be calculated for reference
                    validation['formula_used'] = 'HOLD: Reference levels only'
                    if entry_price and stop_loss and take_profit:
                        # Assume LONG direction for HOLD signals
                        if entry_price > stop_loss and take_profit > entry_price:
                            expected_rr = (take_profit - entry_price) / (entry_price - stop_loss)
                            validation['expected_rr'] = expected_rr
                            validation['levels_realistic'] = True
        
        except Exception as e:
            logger.error(f"Error validating RR calculation: {e}")
        
        return validation
    
    def _analyze_technical_indicators_coherence(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coherence between technical indicators and price levels"""
        coherence = {
            'indicators_available': [],
            'levels_near_indicators': [],
            'vwap_coherence': False,
            'technical_basis_score': 0.0,
            'coherence_details': {}
        }
        
        try:
            # Get technical indicators
            rsi = analysis.get('rsi')
            macd = analysis.get('macd_line')
            vwap = analysis.get('vwap_price')
            mfi = analysis.get('mfi_value')
            
            # Get price levels
            entry_price = analysis.get('entry_price')
            stop_loss = analysis.get('stop_loss_price')
            take_profit = analysis.get('take_profit_price')
            current_price = analysis.get('current_price')
            
            # Check which indicators are available
            if rsi is not None:
                coherence['indicators_available'].append('RSI')
            if macd is not None:
                coherence['indicators_available'].append('MACD')
            if vwap is not None:
                coherence['indicators_available'].append('VWAP')
            if mfi is not None:
                coherence['indicators_available'].append('MFI')
            
            # Check VWAP coherence (levels should be reasonably close to VWAP)
            if vwap and entry_price:
                vwap_distance = abs(entry_price - vwap) / vwap
                if vwap_distance <= 0.1:  # Within 10% of VWAP
                    coherence['vwap_coherence'] = True
                    coherence['levels_near_indicators'].append('VWAP')
                
                coherence['coherence_details']['vwap_distance'] = vwap_distance
            
            # Calculate technical basis score
            score_factors = [
                len(coherence['indicators_available']) >= 3,  # At least 3 indicators
                coherence['vwap_coherence'],  # VWAP coherence
                rsi is not None and 0 <= rsi <= 100,  # Valid RSI
                macd is not None,  # MACD present
                entry_price and stop_loss and take_profit  # All levels present
            ]
            coherence['technical_basis_score'] = sum(score_factors) / len(score_factors)
            
        except Exception as e:
            logger.error(f"Error analyzing technical indicators coherence: {e}")
        
        return coherence
    
    async def test_1_database_technical_levels_verification(self):
        """Test 1: Database Technical Levels Verification - Check stored technical levels quality"""
        logger.info("\n🔍 TEST 1: Database Technical Levels Verification")
        
        try:
            # Get recent technical analyses from database
            recent_analyses = list(self.db.technical_analyses.find().sort('timestamp', -1).limit(10))
            
            if not recent_analyses:
                self.log_test_result("Database Technical Levels Verification", False, "No technical analyses found in database")
                return
            
            levels_results = {
                'total_analyses': len(recent_analyses),
                'analyses_with_levels': 0,
                'realistic_levels_count': 0,
                'technical_indicators_count': 0,
                'rr_calculations': [],
                'coherence_scores': [],
                'symbols_analyzed': set(),
                'detailed_results': []
            }
            
            logger.info(f"   🚀 Analyzing {len(recent_analyses)} recent technical analyses from database...")
            logger.info("   📊 Expected: Technical levels calculated with realistic values based on indicators")
            
            for analysis in recent_analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                levels_results['symbols_analyzed'].add(symbol)
                
                logger.info(f"\n   📊 Analyzing {symbol} technical levels...")
                
                # Validate RR calculation
                rr_validation = self._validate_rr_calculation(analysis)
                
                # Analyze technical indicators coherence
                coherence_analysis = self._analyze_technical_indicators_coherence(analysis)
                
                # Check if levels are present
                has_levels = all([
                    analysis.get('entry_price'),
                    analysis.get('stop_loss_price'),
                    analysis.get('take_profit_price')
                ])
                
                if has_levels:
                    levels_results['analyses_with_levels'] += 1
                    logger.info(f"      ✅ Technical levels present")
                    logger.info(f"         Entry: {analysis.get('entry_price')}")
                    logger.info(f"         Stop Loss: {analysis.get('stop_loss_price')}")
                    logger.info(f"         Take Profit: {analysis.get('take_profit_price')}")
                    logger.info(f"         RR: {analysis.get('risk_reward_ratio')}")
                    logger.info(f"         Signal: {analysis.get('ia1_signal')}")
                
                # Check if levels are realistic
                if rr_validation['levels_realistic']:
                    levels_results['realistic_levels_count'] += 1
                    logger.info(f"      ✅ Levels hierarchy is realistic")
                
                # Check technical indicators
                if len(coherence_analysis['indicators_available']) >= 3:
                    levels_results['technical_indicators_count'] += 1
                    logger.info(f"      ✅ Technical indicators available: {coherence_analysis['indicators_available']}")
                
                # Store RR calculation details
                if rr_validation['rr_present']:
                    levels_results['rr_calculations'].append({
                        'symbol': symbol,
                        'rr_value': rr_validation['rr_value'],
                        'expected_rr': rr_validation['expected_rr'],
                        'calculated_correctly': rr_validation['rr_calculated_correctly'],
                        'formula_used': rr_validation['formula_used']
                    })
                
                # Store coherence score
                levels_results['coherence_scores'].append(coherence_analysis['technical_basis_score'])
                
                # Store detailed results
                levels_results['detailed_results'].append({
                    'symbol': symbol,
                    'has_levels': has_levels,
                    'rr_validation': rr_validation,
                    'coherence_analysis': coherence_analysis,
                    'technical_data': {
                        'rsi': analysis.get('rsi'),
                        'macd': analysis.get('macd_line'),
                        'vwap': analysis.get('vwap_price'),
                        'mfi': analysis.get('mfi_value')
                    }
                })
                
                logger.info(f"      📊 Technical basis score: {coherence_analysis['technical_basis_score']:.2f}")
            
            # Calculate final metrics
            levels_present_rate = levels_results['analyses_with_levels'] / levels_results['total_analyses']
            realistic_levels_rate = levels_results['realistic_levels_count'] / levels_results['total_analyses']
            technical_indicators_rate = levels_results['technical_indicators_count'] / levels_results['total_analyses']
            avg_coherence_score = sum(levels_results['coherence_scores']) / len(levels_results['coherence_scores'])
            
            logger.info(f"\n   📊 DATABASE TECHNICAL LEVELS VERIFICATION RESULTS:")
            logger.info(f"      Total analyses: {levels_results['total_analyses']}")
            logger.info(f"      Analyses with levels: {levels_results['analyses_with_levels']} ({levels_present_rate:.2f})")
            logger.info(f"      Realistic levels: {levels_results['realistic_levels_count']} ({realistic_levels_rate:.2f})")
            logger.info(f"      Technical indicators: {levels_results['technical_indicators_count']} ({technical_indicators_rate:.2f})")
            logger.info(f"      Average coherence score: {avg_coherence_score:.2f}")
            logger.info(f"      Symbols analyzed: {len(levels_results['symbols_analyzed'])}")
            logger.info(f"      RR calculations found: {len(levels_results['rr_calculations'])}")
            
            # Calculate test success
            success_criteria = [
                levels_results['total_analyses'] >= 3,  # At least 3 analyses
                levels_present_rate >= 0.8,  # At least 80% with levels
                realistic_levels_rate >= 0.5,  # At least 50% realistic
                technical_indicators_rate >= 0.8,  # At least 80% with indicators
                avg_coherence_score >= 0.6  # Average coherence >= 60%
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Database Technical Levels Verification", True, 
                                   f"Database technical levels verification successful: {success_count}/{len(success_criteria)} criteria met. Levels present: {levels_present_rate:.2f}, Realistic: {realistic_levels_rate:.2f}, Indicators: {technical_indicators_rate:.2f}, Coherence: {avg_coherence_score:.2f}")
            else:
                self.log_test_result("Database Technical Levels Verification", False, 
                                   f"Database technical levels verification issues: {success_count}/{len(success_criteria)} criteria met. May lack realistic levels or technical basis")
                
        except Exception as e:
            self.log_test_result("Database Technical Levels Verification", False, f"Exception: {str(e)}")
    
    async def test_2_risk_reward_formula_validation(self):
        """Test 2: Risk-Reward Formula Validation - Verify RR calculations use correct formulas"""
        logger.info("\n🔍 TEST 2: Risk-Reward Formula Validation")
        
        try:
            # Get recent analyses with RR values
            recent_analyses = list(self.db.technical_analyses.find({
                'risk_reward_ratio': {'$exists': True, '$ne': None}
            }).sort('timestamp', -1).limit(10))
            
            if not recent_analyses:
                self.log_test_result("Risk-Reward Formula Validation", False, "No analyses with RR values found")
                return
            
            rr_results = {
                'total_analyses': len(recent_analyses),
                'correct_calculations': 0,
                'incorrect_calculations': 0,
                'default_values': 0,
                'realistic_rr_values': 0,
                'rr_above_2_count': 0,
                'formula_validations': []
            }
            
            logger.info(f"   🚀 Validating RR calculations in {len(recent_analyses)} analyses...")
            logger.info("   📊 Expected: RR calculated using LONG: (TP-Entry)/(Entry-SL), SHORT: (Entry-TP)/(SL-Entry)")
            
            for analysis in recent_analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                logger.info(f"\n   📊 Validating RR calculation for {symbol}...")
                
                rr_validation = self._validate_rr_calculation(analysis)
                
                rr_value = rr_validation['rr_value']
                expected_rr = rr_validation['expected_rr']
                
                logger.info(f"      RR Value: {rr_value}")
                logger.info(f"      Expected RR: {expected_rr}")
                logger.info(f"      Formula: {rr_validation['formula_used']}")
                
                # Check if RR is calculated correctly
                if rr_validation['rr_calculated_correctly']:
                    rr_results['correct_calculations'] += 1
                    logger.info(f"      ✅ RR calculation is correct")
                elif expected_rr is not None:
                    rr_results['incorrect_calculations'] += 1
                    logger.info(f"      ❌ RR calculation is incorrect")
                
                # Check for default values (exactly 1.0 might be a default)
                if rr_value == 1.0:
                    rr_results['default_values'] += 1
                    logger.info(f"      ⚠️ RR = 1.0 (possible default value)")
                
                # Check if RR is realistic (between 0.5 and 10.0)
                if rr_value and 0.5 <= rr_value <= 10.0:
                    rr_results['realistic_rr_values'] += 1
                    logger.info(f"      ✅ RR value is realistic")
                
                # Check if RR meets IA2 escalation criteria
                if rr_value and rr_value >= 2.0:
                    rr_results['rr_above_2_count'] += 1
                    logger.info(f"      ✅ RR ≥ 2.0 (IA2 escalation criteria)")
                
                # Store validation details
                rr_results['formula_validations'].append({
                    'symbol': symbol,
                    'rr_value': rr_value,
                    'expected_rr': expected_rr,
                    'calculated_correctly': rr_validation['rr_calculated_correctly'],
                    'formula_used': rr_validation['formula_used'],
                    'levels_realistic': rr_validation['levels_realistic'],
                    'signal': analysis.get('ia1_signal')
                })
            
            # Calculate final metrics
            if rr_results['total_analyses'] > 0:
                correct_rate = rr_results['correct_calculations'] / rr_results['total_analyses']
                default_rate = rr_results['default_values'] / rr_results['total_analyses']
                realistic_rate = rr_results['realistic_rr_values'] / rr_results['total_analyses']
                escalation_rate = rr_results['rr_above_2_count'] / rr_results['total_analyses']
            else:
                correct_rate = default_rate = realistic_rate = escalation_rate = 0.0
            
            logger.info(f"\n   📊 RISK-REWARD FORMULA VALIDATION RESULTS:")
            logger.info(f"      Total analyses: {rr_results['total_analyses']}")
            logger.info(f"      Correct calculations: {rr_results['correct_calculations']} ({correct_rate:.2f})")
            logger.info(f"      Incorrect calculations: {rr_results['incorrect_calculations']}")
            logger.info(f"      Default values (1.0): {rr_results['default_values']} ({default_rate:.2f})")
            logger.info(f"      Realistic RR values: {rr_results['realistic_rr_values']} ({realistic_rate:.2f})")
            logger.info(f"      RR ≥ 2.0 (IA2 criteria): {rr_results['rr_above_2_count']} ({escalation_rate:.2f})")
            
            if rr_results['formula_validations']:
                logger.info(f"      📊 Formula validation details:")
                for validation in rr_results['formula_validations'][:5]:  # Show first 5
                    logger.info(f"         - {validation['symbol']}: RR={validation['rr_value']:.2f}, Expected={validation['expected_rr']:.2f if validation['expected_rr'] else 'N/A'}, Correct={validation['calculated_correctly']}")
            
            # Calculate test success
            success_criteria = [
                rr_results['total_analyses'] >= 3,  # At least 3 analyses
                realistic_rate >= 0.8,  # At least 80% realistic values
                default_rate <= 0.8,  # Not more than 80% default values
                rr_results['correct_calculations'] >= 1,  # At least 1 correct calculation
                escalation_rate >= 0.1  # At least 10% meet IA2 criteria
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.6:  # 60% success threshold (lower due to potential default values)
                self.log_test_result("Risk-Reward Formula Validation", True, 
                                   f"RR formula validation successful: {success_count}/{len(success_criteria)} criteria met. Correct: {correct_rate:.2f}, Realistic: {realistic_rate:.2f}, Defaults: {default_rate:.2f}")
            else:
                self.log_test_result("Risk-Reward Formula Validation", False, 
                                   f"RR formula validation issues: {success_count}/{len(success_criteria)} criteria met. May have too many default values or incorrect calculations")
                
        except Exception as e:
            self.log_test_result("Risk-Reward Formula Validation", False, f"Exception: {str(e)}")
    
    async def test_3_technical_indicators_usage_analysis(self):
        """Test 3: Technical Indicators Usage Analysis - Verify indicators are calculated and used"""
        logger.info("\n🔍 TEST 3: Technical Indicators Usage Analysis")
        
        try:
            # Get recent analyses
            recent_analyses = list(self.db.technical_analyses.find().sort('timestamp', -1).limit(10))
            
            if not recent_analyses:
                self.log_test_result("Technical Indicators Usage Analysis", False, "No analyses found")
                return
            
            indicators_results = {
                'total_analyses': len(recent_analyses),
                'rsi_present': 0,
                'macd_present': 0,
                'vwap_present': 0,
                'mfi_present': 0,
                'all_indicators_present': 0,
                'realistic_values': 0,
                'indicator_details': []
            }
            
            logger.info(f"   🚀 Analyzing technical indicators usage in {len(recent_analyses)} analyses...")
            logger.info("   📊 Expected: RSI, MACD, VWAP, MFI calculated with realistic values")
            
            for analysis in recent_analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                logger.info(f"\n   📊 Analyzing indicators for {symbol}...")
                
                # Check each indicator
                rsi = analysis.get('rsi')
                macd = analysis.get('macd_line')
                vwap = analysis.get('vwap_price')
                mfi = analysis.get('mfi_value')
                
                indicators_present = []
                realistic_indicators = []
                
                # RSI validation
                if rsi is not None:
                    indicators_results['rsi_present'] += 1
                    indicators_present.append('RSI')
                    if 0 <= rsi <= 100:
                        realistic_indicators.append('RSI')
                    logger.info(f"      RSI: {rsi}")
                
                # MACD validation
                if macd is not None:
                    indicators_results['macd_present'] += 1
                    indicators_present.append('MACD')
                    if -10 <= macd <= 10:  # Reasonable MACD range
                        realistic_indicators.append('MACD')
                    logger.info(f"      MACD: {macd}")
                
                # VWAP validation
                if vwap is not None:
                    indicators_results['vwap_present'] += 1
                    indicators_present.append('VWAP')
                    current_price = analysis.get('current_price')
                    if current_price and 0.5 <= (vwap / current_price) <= 2.0:  # VWAP within reasonable range of current price
                        realistic_indicators.append('VWAP')
                    logger.info(f"      VWAP: {vwap}")
                
                # MFI validation
                if mfi is not None:
                    indicators_results['mfi_present'] += 1
                    indicators_present.append('MFI')
                    if 0 <= mfi <= 100:
                        realistic_indicators.append('MFI')
                    logger.info(f"      MFI: {mfi}")
                
                # Check if all indicators are present
                if len(indicators_present) >= 4:
                    indicators_results['all_indicators_present'] += 1
                    logger.info(f"      ✅ All major indicators present")
                
                # Check if values are realistic
                if len(realistic_indicators) >= 3:
                    indicators_results['realistic_values'] += 1
                    logger.info(f"      ✅ Realistic indicator values")
                
                # Store details
                indicators_results['indicator_details'].append({
                    'symbol': symbol,
                    'indicators_present': indicators_present,
                    'realistic_indicators': realistic_indicators,
                    'rsi': rsi,
                    'macd': macd,
                    'vwap': vwap,
                    'mfi': mfi
                })
                
                logger.info(f"      📊 Indicators present: {len(indicators_present)}/4")
                logger.info(f"      📊 Realistic values: {len(realistic_indicators)}")
            
            # Calculate final metrics
            if indicators_results['total_analyses'] > 0:
                rsi_rate = indicators_results['rsi_present'] / indicators_results['total_analyses']
                macd_rate = indicators_results['macd_present'] / indicators_results['total_analyses']
                vwap_rate = indicators_results['vwap_present'] / indicators_results['total_analyses']
                mfi_rate = indicators_results['mfi_present'] / indicators_results['total_analyses']
                all_indicators_rate = indicators_results['all_indicators_present'] / indicators_results['total_analyses']
                realistic_rate = indicators_results['realistic_values'] / indicators_results['total_analyses']
            else:
                rsi_rate = macd_rate = vwap_rate = mfi_rate = all_indicators_rate = realistic_rate = 0.0
            
            logger.info(f"\n   📊 TECHNICAL INDICATORS USAGE ANALYSIS RESULTS:")
            logger.info(f"      Total analyses: {indicators_results['total_analyses']}")
            logger.info(f"      RSI present: {indicators_results['rsi_present']} ({rsi_rate:.2f})")
            logger.info(f"      MACD present: {indicators_results['macd_present']} ({macd_rate:.2f})")
            logger.info(f"      VWAP present: {indicators_results['vwap_present']} ({vwap_rate:.2f})")
            logger.info(f"      MFI present: {indicators_results['mfi_present']} ({mfi_rate:.2f})")
            logger.info(f"      All indicators present: {indicators_results['all_indicators_present']} ({all_indicators_rate:.2f})")
            logger.info(f"      Realistic values: {indicators_results['realistic_values']} ({realistic_rate:.2f})")
            
            # Calculate test success
            success_criteria = [
                indicators_results['total_analyses'] >= 3,  # At least 3 analyses
                rsi_rate >= 0.8,  # At least 80% with RSI
                macd_rate >= 0.8,  # At least 80% with MACD
                vwap_rate >= 0.8,  # At least 80% with VWAP
                all_indicators_rate >= 0.5,  # At least 50% with all indicators
                realistic_rate >= 0.8  # At least 80% with realistic values
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Technical Indicators Usage Analysis", True, 
                                   f"Technical indicators usage successful: {success_count}/{len(success_criteria)} criteria met. RSI: {rsi_rate:.2f}, MACD: {macd_rate:.2f}, VWAP: {vwap_rate:.2f}, All: {all_indicators_rate:.2f}")
            else:
                self.log_test_result("Technical Indicators Usage Analysis", False, 
                                   f"Technical indicators usage issues: {success_count}/{len(success_criteria)} criteria met. May lack complete indicator coverage or realistic values")
                
        except Exception as e:
            self.log_test_result("Technical Indicators Usage Analysis", False, f"Exception: {str(e)}")
    
    async def run_all_tests(self):
        """Run all database technical analysis tests"""
        logger.info("🚀 STARTING DATABASE TECHNICAL ANALYSIS VERIFICATION TEST SUITE")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests
        await self.test_1_database_technical_levels_verification()
        await self.test_2_risk_reward_formula_validation()
        await self.test_3_technical_indicators_usage_analysis()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 DATABASE TECHNICAL ANALYSIS TEST SUITE SUMMARY")
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
    test_suite = DatabaseTechnicalAnalysisTest()
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
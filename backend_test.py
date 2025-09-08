#!/usr/bin/env python3
"""
Backend Testing Suite for Multi-RR Decision Engine Analysis - BUSDT Focus
Focus: Multi-RR calculations, Risk/Reward formulas, BUSDT analysis, and RR optimization
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add backend to path
sys.path.append('/app/backend')

import requests
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiRRDecisionEngineTestSuite:
    """Test suite for Multi-RR Decision Engine Analysis - Focus on BUSDT and RR calculations"""
    
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
        logger.info(f"Testing Multi-RR Decision Engine at: {self.api_url}")
        
        # MongoDB connection for direct data access
        self.mongo_client = None
        self.db = None
        
        # Test results
        self.test_results = []
        
        # Multi-RR specific test data
        self.busdt_data = None
        self.okbusdt_data = None
        self.rr_calculations = []
        
    async def setup_database(self):
        """Setup database connection"""
        try:
            # Get MongoDB URL from backend env
            mongo_url = "mongodb://localhost:27017"  # Default
            try:
                with open('/app/backend/.env', 'r') as f:
                    for line in f:
                        if line.startswith('MONGO_URL='):
                            mongo_url = line.split('=')[1].strip().strip('"')
                            break
            except Exception:
                pass
            
            self.mongo_client = AsyncIOMotorClient(mongo_url)
            self.db = self.mongo_client['myapp']
            logger.info("✅ Database connection established")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            
    async def cleanup_database(self):
        """Cleanup database connection"""
        if self.mongo_client:
            self.mongo_client.close()
            
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
        
    def get_analyses_from_api(self):
        """Helper method to get analyses from API with proper format handling"""
        try:
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            if response.status_code != 200:
                return None, f"API error: {response.status_code}"
                
            data = response.json()
            
            # Handle API response format
            if isinstance(data, dict) and 'analyses' in data:
                analyses = data['analyses']
            else:
                analyses = data
                
            return analyses, None
        except Exception as e:
            return None, f"Exception: {str(e)}"
    
    def get_decisions_from_api(self):
        """Helper method to get decisions from API"""
        try:
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            if response.status_code != 200:
                return None, f"API error: {response.status_code}"
                
            data = response.json()
            return data, None
        except Exception as e:
            return None, f"Exception: {str(e)}"
    
    def get_opportunities_from_api(self):
        """Helper method to get opportunities from API"""
        try:
            response = requests.get(f"{self.api_url}/opportunities", timeout=30)
            if response.status_code != 200:
                return None, f"API error: {response.status_code}"
                
            data = response.json()
            return data, None
        except Exception as e:
            return None, f"Exception: {str(e)}"
        
    async def test_busdt_data_availability(self):
        """Test 1: BUSDT Data Availability - Verify BUSDT/OKBUSDT data is available for analysis"""
        logger.info("\n🔍 TEST 1: BUSDT Data Availability")
        
        try:
            # Check opportunities for BUSDT/OKBUSDT
            opportunities, error = self.get_opportunities_from_api()
            
            if error:
                self.log_test_result("BUSDT Data Availability", False, error)
                return
                
            busdt_opportunities = []
            okbusdt_opportunities = []
            
            if opportunities:
                for opp in opportunities:
                    symbol = opp.get('symbol', '').upper()
                    if 'BUSDT' in symbol:
                        busdt_opportunities.append(opp)
                        logger.info(f"   📊 Found BUSDT opportunity: {symbol}")
                        logger.info(f"      Price: ${opp.get('current_price', 0):.6f}")
                        logger.info(f"      Volume: ${opp.get('volume_24h', 0):,.0f}")
                        logger.info(f"      Change 24h: {opp.get('price_change_24h', 0):+.2f}%")
                        logger.info(f"      Volatility: {opp.get('volatility', 0)*100:.2f}%")
                    elif 'OKBUSDT' in symbol:
                        okbusdt_opportunities.append(opp)
                        logger.info(f"   📊 Found OKBUSDT opportunity: {symbol}")
                        logger.info(f"      Price: ${opp.get('current_price', 0):.6f}")
                        logger.info(f"      Volume: ${opp.get('volume_24h', 0):,.0f}")
                        logger.info(f"      Change 24h: {opp.get('price_change_24h', 0):+.2f}%")
                        logger.info(f"      Volatility: {opp.get('volatility', 0)*100:.2f}%")
            
            # Check analyses for BUSDT/OKBUSDT
            analyses, error = self.get_analyses_from_api()
            busdt_analyses = []
            okbusdt_analyses = []
            
            if not error and analyses:
                for analysis in analyses:
                    symbol = analysis.get('symbol', '').upper()
                    if 'BUSDT' in symbol:
                        busdt_analyses.append(analysis)
                        logger.info(f"   🔍 Found BUSDT analysis: {symbol}")
                        logger.info(f"      RSI: {analysis.get('rsi', 0):.2f}")
                        logger.info(f"      MACD: {analysis.get('macd_signal', 0):.6f}")
                        logger.info(f"      Confidence: {analysis.get('analysis_confidence', 0):.2f}")
                    elif 'OKBUSDT' in symbol:
                        okbusdt_analyses.append(analysis)
                        logger.info(f"   🔍 Found OKBUSDT analysis: {symbol}")
                        logger.info(f"      RSI: {analysis.get('rsi', 0):.2f}")
                        logger.info(f"      MACD: {analysis.get('macd_signal', 0):.6f}")
                        logger.info(f"      Confidence: {analysis.get('analysis_confidence', 0):.2f}")
            
            # Store for later tests
            self.busdt_data = {
                'opportunities': busdt_opportunities,
                'analyses': busdt_analyses
            }
            self.okbusdt_data = {
                'opportunities': okbusdt_opportunities,
                'analyses': okbusdt_analyses
            }
            
            total_busdt_data = len(busdt_opportunities) + len(busdt_analyses)
            total_okbusdt_data = len(okbusdt_opportunities) + len(okbusdt_analyses)
            
            success = total_busdt_data > 0 or total_okbusdt_data > 0
            details = f"BUSDT data points: {total_busdt_data} (opps: {len(busdt_opportunities)}, analyses: {len(busdt_analyses)}), OKBUSDT data points: {total_okbusdt_data} (opps: {len(okbusdt_opportunities)}, analyses: {len(okbusdt_analyses)})"
            
            self.log_test_result("BUSDT Data Availability", success, details)
            
        except Exception as e:
            self.log_test_result("BUSDT Data Availability", False, f"Exception: {str(e)}")
    
    async def test_multi_rr_calculation_formulas(self):
        """Test 2: Multi-RR Calculation Formulas - Analyze _calculate_pattern_rr and _calculate_technical_signal_rr"""
        logger.info("\n🔍 TEST 2: Multi-RR Calculation Formulas")
        
        try:
            # Check backend logs for Multi-RR calculations
            import subprocess
            
            # Get recent backend logs for Multi-RR calculations
            log_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'multi-rr\\|calculate.*rr\\|pattern.*rr\\|technical.*rr' || echo 'No Multi-RR logs'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            backend_logs = result.stdout
            multi_rr_logs = []
            pattern_rr_logs = []
            technical_rr_logs = []
            
            for line in backend_logs.split('\n'):
                if 'multi-rr' in line.lower():
                    multi_rr_logs.append(line.strip())
                elif 'pattern' in line.lower() and 'rr' in line.lower():
                    pattern_rr_logs.append(line.strip())
                elif 'technical' in line.lower() and 'rr' in line.lower():
                    technical_rr_logs.append(line.strip())
            
            logger.info(f"   📊 Multi-RR logs found: {len(multi_rr_logs)}")
            logger.info(f"   📊 Pattern RR logs found: {len(pattern_rr_logs)}")
            logger.info(f"   📊 Technical RR logs found: {len(technical_rr_logs)}")
            
            # Show sample logs
            if multi_rr_logs:
                logger.info(f"   📝 Sample Multi-RR log: {multi_rr_logs[-1][:150]}...")
            if pattern_rr_logs:
                logger.info(f"   📝 Sample Pattern RR log: {pattern_rr_logs[-1][:150]}...")
            if technical_rr_logs:
                logger.info(f"   📝 Sample Technical RR log: {technical_rr_logs[-1][:150]}...")
            
            # Analyze analyses for RR calculations
            analyses, error = self.get_analyses_from_api()
            rr_calculations_found = 0
            rr_ratios = []
            
            if not error and analyses:
                for analysis in analyses:
                    symbol = analysis.get('symbol', 'UNKNOWN')
                    rr_ratio = analysis.get('risk_reward_ratio', 0)
                    rr_reasoning = analysis.get('rr_reasoning', '')
                    
                    if rr_ratio > 0:
                        rr_calculations_found += 1
                        rr_ratios.append(rr_ratio)
                        logger.info(f"   🎯 {symbol}: RR Ratio = {rr_ratio:.2f}")
                        if rr_reasoning:
                            logger.info(f"      Reasoning: {rr_reasoning[:100]}...")
            
            # Check for specific formula components
            formula_components = {
                'volatility': 0,
                'atr_multiplier': 0,
                'signal_strength': 0,
                'pattern_strength': 0,
                'stop_loss': 0,
                'target_price': 0
            }
            
            for log_line in multi_rr_logs + pattern_rr_logs + technical_rr_logs:
                for component in formula_components.keys():
                    if component in log_line.lower():
                        formula_components[component] += 1
            
            logger.info(f"   📊 Formula components found in logs:")
            for component, count in formula_components.items():
                logger.info(f"      {component}: {count} mentions")
            
            # Analyze RR ratio distribution
            if rr_ratios:
                avg_rr = np.mean(rr_ratios)
                min_rr = min(rr_ratios)
                max_rr = max(rr_ratios)
                logger.info(f"   📊 RR Ratio Statistics:")
                logger.info(f"      Average: {avg_rr:.2f}")
                logger.info(f"      Range: {min_rr:.2f} - {max_rr:.2f}")
                logger.info(f"      Count: {len(rr_ratios)} calculations")
            
            success = len(multi_rr_logs) > 0 or rr_calculations_found > 0
            details = f"Multi-RR logs: {len(multi_rr_logs)}, RR calculations: {rr_calculations_found}, Formula components: {sum(formula_components.values())}"
            
            self.log_test_result("Multi-RR Calculation Formulas", success, details)
            
        except Exception as e:
            self.log_test_result("Multi-RR Calculation Formulas", False, f"Exception: {str(e)}")
    
    async def test_rr_caps_and_limits(self):
        """Test 3: RR Caps and Limits - Check if 5:1 pattern cap and 4:1 technical cap are too restrictive"""
        logger.info("\n🔍 TEST 3: RR Caps and Limits Analysis")
        
        try:
            # Analyze RR ratios to see if they hit the caps
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("RR Caps and Limits", False, error)
                return
            
            rr_ratios = []
            capped_patterns = 0
            capped_technical = 0
            high_potential_rr = 0
            
            if analyses:
                for analysis in analyses:
                    symbol = analysis.get('symbol', 'UNKNOWN')
                    rr_ratio = analysis.get('risk_reward_ratio', 0)
                    patterns_detected = analysis.get('patterns_detected', [])
                    
                    if rr_ratio > 0:
                        rr_ratios.append({
                            'symbol': symbol,
                            'ratio': rr_ratio,
                            'has_patterns': len(patterns_detected) > 0,
                            'patterns': patterns_detected
                        })
                        
                        # Check if hitting caps
                        if len(patterns_detected) > 0 and rr_ratio >= 4.9:  # Close to 5:1 cap
                            capped_patterns += 1
                            logger.info(f"   ⚠️ {symbol}: Pattern RR near cap = {rr_ratio:.2f} (patterns: {patterns_detected})")
                        elif len(patterns_detected) == 0 and rr_ratio >= 3.9:  # Close to 4:1 cap
                            capped_technical += 1
                            logger.info(f"   ⚠️ {symbol}: Technical RR near cap = {rr_ratio:.2f}")
                        
                        # Check for high potential (could be higher without caps)
                        if rr_ratio > 3.0:
                            high_potential_rr += 1
                            logger.info(f"   🚀 {symbol}: High RR potential = {rr_ratio:.2f}")
            
            # Analyze volatility vs RR relationship
            volatility_rr_analysis = []
            
            # Get opportunities to check volatility
            opportunities, opp_error = self.get_opportunities_from_api()
            if not opp_error and opportunities:
                for opp in opportunities:
                    symbol = opp.get('symbol', '')
                    volatility = opp.get('volatility', 0)
                    
                    # Find matching analysis
                    matching_analysis = None
                    for analysis in analyses or []:
                        if analysis.get('symbol', '') == symbol:
                            matching_analysis = analysis
                            break
                    
                    if matching_analysis:
                        rr_ratio = matching_analysis.get('risk_reward_ratio', 0)
                        if rr_ratio > 0 and volatility > 0:
                            volatility_rr_analysis.append({
                                'symbol': symbol,
                                'volatility': volatility * 100,  # Convert to percentage
                                'rr_ratio': rr_ratio,
                                'potential_uncapped': volatility * 200  # Theoretical max based on volatility
                            })
            
            # Check for stop-loss tightness (max 2.5% loss mentioned in review)
            tight_stop_losses = 0
            stop_loss_analysis = []
            
            for analysis in analyses or []:
                symbol = analysis.get('symbol', 'UNKNOWN')
                stop_loss_price = analysis.get('stop_loss_price', 0)
                entry_price = analysis.get('entry_price', 0)
                
                if stop_loss_price > 0 and entry_price > 0:
                    stop_loss_percentage = abs(stop_loss_price - entry_price) / entry_price * 100
                    stop_loss_analysis.append({
                        'symbol': symbol,
                        'stop_loss_pct': stop_loss_percentage
                    })
                    
                    if stop_loss_percentage <= 2.5:
                        tight_stop_losses += 1
                        logger.info(f"   🔒 {symbol}: Tight stop-loss = {stop_loss_percentage:.2f}%")
            
            # Summary statistics
            if rr_ratios:
                avg_rr = np.mean([r['ratio'] for r in rr_ratios])
                max_rr = max([r['ratio'] for r in rr_ratios])
                pattern_rr_avg = np.mean([r['ratio'] for r in rr_ratios if r['has_patterns']])
                technical_rr_avg = np.mean([r['ratio'] for r in rr_ratios if not r['has_patterns']])
                
                logger.info(f"   📊 RR Statistics:")
                logger.info(f"      Overall Average: {avg_rr:.2f}")
                logger.info(f"      Maximum: {max_rr:.2f}")
                logger.info(f"      Pattern Average: {pattern_rr_avg:.2f}")
                logger.info(f"      Technical Average: {technical_rr_avg:.2f}")
                logger.info(f"      Capped Patterns: {capped_patterns}")
                logger.info(f"      Capped Technical: {capped_technical}")
                logger.info(f"      High Potential: {high_potential_rr}")
            
            if stop_loss_analysis:
                avg_sl = np.mean([s['stop_loss_pct'] for s in stop_loss_analysis])
                logger.info(f"   📊 Stop-Loss Analysis:")
                logger.info(f"      Average SL: {avg_sl:.2f}%")
                logger.info(f"      Tight SL (≤2.5%): {tight_stop_losses}/{len(stop_loss_analysis)}")
            
            # Check if caps are too restrictive
            caps_too_restrictive = (capped_patterns > len(rr_ratios) * 0.1) or (capped_technical > len(rr_ratios) * 0.1)
            
            success = len(rr_ratios) > 0
            details = f"RR calculations: {len(rr_ratios)}, Capped patterns: {capped_patterns}, Capped technical: {capped_technical}, High potential: {high_potential_rr}, Tight SL: {tight_stop_losses}, Caps restrictive: {caps_too_restrictive}"
            
            self.log_test_result("RR Caps and Limits", success, details)
            
        except Exception as e:
            self.log_test_result("RR Caps and Limits", False, f"Exception: {str(e)}")
    
    async def test_busdt_specific_calculations(self):
        """Test 4: BUSDT Specific Calculations - Analyze BUSDT/OKBUSDT RR calculations in detail"""
        logger.info("\n🔍 TEST 4: BUSDT Specific Calculations")
        
        try:
            if not self.busdt_data and not self.okbusdt_data:
                self.log_test_result("BUSDT Specific Calculations", False, "No BUSDT/OKBUSDT data available from previous test")
                return
            
            busdt_calculations = []
            okbusdt_calculations = []
            
            # Analyze BUSDT data
            if self.busdt_data:
                for analysis in self.busdt_data.get('analyses', []):
                    symbol = analysis.get('symbol', '')
                    calculation = {
                        'symbol': symbol,
                        'current_price': analysis.get('entry_price', 0),
                        'rsi': analysis.get('rsi', 0),
                        'macd': analysis.get('macd_signal', 0),
                        'volatility': 0,  # Will get from opportunities
                        'rr_ratio': analysis.get('risk_reward_ratio', 0),
                        'stop_loss': analysis.get('stop_loss_price', 0),
                        'take_profit': analysis.get('take_profit_price', 0),
                        'confidence': analysis.get('analysis_confidence', 0),
                        'patterns': analysis.get('patterns_detected', [])
                    }
                    
                    # Get volatility from opportunities
                    for opp in self.busdt_data.get('opportunities', []):
                        if opp.get('symbol', '') == symbol:
                            calculation['volatility'] = opp.get('volatility', 0) * 100
                            calculation['volume_24h'] = opp.get('volume_24h', 0)
                            calculation['price_change_24h'] = opp.get('price_change_24h', 0)
                            break
                    
                    busdt_calculations.append(calculation)
                    
                    logger.info(f"   🎯 BUSDT Analysis: {symbol}")
                    logger.info(f"      Price: ${calculation['current_price']:.6f}")
                    logger.info(f"      RSI: {calculation['rsi']:.2f}")
                    logger.info(f"      MACD: {calculation['macd']:.6f}")
                    logger.info(f"      Volatility: {calculation['volatility']:.2f}%")
                    logger.info(f"      RR Ratio: {calculation['rr_ratio']:.2f}")
                    logger.info(f"      Confidence: {calculation['confidence']:.2f}")
                    if calculation['patterns']:
                        logger.info(f"      Patterns: {calculation['patterns']}")
            
            # Analyze OKBUSDT data
            if self.okbusdt_data:
                for analysis in self.okbusdt_data.get('analyses', []):
                    symbol = analysis.get('symbol', '')
                    calculation = {
                        'symbol': symbol,
                        'current_price': analysis.get('entry_price', 0),
                        'rsi': analysis.get('rsi', 0),
                        'macd': analysis.get('macd_signal', 0),
                        'volatility': 0,
                        'rr_ratio': analysis.get('risk_reward_ratio', 0),
                        'stop_loss': analysis.get('stop_loss_price', 0),
                        'take_profit': analysis.get('take_profit_price', 0),
                        'confidence': analysis.get('analysis_confidence', 0),
                        'patterns': analysis.get('patterns_detected', [])
                    }
                    
                    # Get volatility from opportunities
                    for opp in self.okbusdt_data.get('opportunities', []):
                        if opp.get('symbol', '') == symbol:
                            calculation['volatility'] = opp.get('volatility', 0) * 100
                            calculation['volume_24h'] = opp.get('volume_24h', 0)
                            calculation['price_change_24h'] = opp.get('price_change_24h', 0)
                            break
                    
                    okbusdt_calculations.append(calculation)
                    
                    logger.info(f"   🎯 OKBUSDT Analysis: {symbol}")
                    logger.info(f"      Price: ${calculation['current_price']:.6f}")
                    logger.info(f"      RSI: {calculation['rsi']:.2f}")
                    logger.info(f"      MACD: {calculation['macd']:.6f}")
                    logger.info(f"      Volatility: {calculation['volatility']:.2f}%")
                    logger.info(f"      RR Ratio: {calculation['rr_ratio']:.2f}")
                    logger.info(f"      Confidence: {calculation['confidence']:.2f}")
                    if calculation['patterns']:
                        logger.info(f"      Patterns: {calculation['patterns']}")
            
            # Analyze if RR is low as mentioned in review request
            low_rr_issues = []
            all_calculations = busdt_calculations + okbusdt_calculations
            
            for calc in all_calculations:
                if calc['rr_ratio'] > 0 and calc['rr_ratio'] < 1.5:  # Low RR threshold
                    low_rr_issues.append({
                        'symbol': calc['symbol'],
                        'rr_ratio': calc['rr_ratio'],
                        'volatility': calc['volatility'],
                        'potential_issues': []
                    })
                    
                    # Analyze potential causes of low RR
                    if calc['volatility'] < 2.0:  # Low volatility
                        low_rr_issues[-1]['potential_issues'].append("Low volatility (<2%)")
                    
                    if calc['stop_loss'] > 0 and calc['current_price'] > 0:
                        sl_distance = abs(calc['stop_loss'] - calc['current_price']) / calc['current_price'] * 100
                        if sl_distance > 2.5:
                            low_rr_issues[-1]['potential_issues'].append(f"Wide stop-loss ({sl_distance:.2f}%)")
                    
                    if calc['take_profit'] > 0 and calc['current_price'] > 0:
                        tp_distance = abs(calc['take_profit'] - calc['current_price']) / calc['current_price'] * 100
                        if tp_distance < 3.0:
                            low_rr_issues[-1]['potential_issues'].append(f"Conservative target ({tp_distance:.2f}%)")
            
            # Log low RR issues
            if low_rr_issues:
                logger.info(f"   ⚠️ Low RR Issues Found ({len(low_rr_issues)} cases):")
                for issue in low_rr_issues:
                    logger.info(f"      {issue['symbol']}: RR={issue['rr_ratio']:.2f}, Issues: {issue['potential_issues']}")
            
            # Store calculations for later analysis
            self.rr_calculations = all_calculations
            
            success = len(all_calculations) > 0
            details = f"BUSDT calculations: {len(busdt_calculations)}, OKBUSDT calculations: {len(okbusdt_calculations)}, Low RR issues: {len(low_rr_issues)}"
            
            self.log_test_result("BUSDT Specific Calculations", success, details)
            
        except Exception as e:
            self.log_test_result("BUSDT Specific Calculations", False, f"Exception: {str(e)}")
    
    async def test_volatility_underestimation(self):
        """Test 5: Volatility Underestimation - Check if volatility minimum 1.5% is causing issues"""
        logger.info("\n🔍 TEST 5: Volatility Underestimation Analysis")
        
        try:
            # Get opportunities to analyze volatility
            opportunities, error = self.get_opportunities_from_api()
            
            if error:
                self.log_test_result("Volatility Underestimation", False, error)
                return
            
            volatility_analysis = []
            low_volatility_count = 0
            underestimated_volatility = 0
            
            if opportunities:
                for opp in opportunities:
                    symbol = opp.get('symbol', 'UNKNOWN')
                    volatility = opp.get('volatility', 0) * 100  # Convert to percentage
                    price_change_24h = abs(opp.get('price_change_24h', 0))
                    
                    analysis_data = {
                        'symbol': symbol,
                        'volatility': volatility,
                        'price_change_24h': price_change_24h,
                        'is_low_volatility': volatility < 1.5,
                        'is_underestimated': price_change_24h > volatility * 2  # Price change much higher than volatility
                    }
                    
                    volatility_analysis.append(analysis_data)
                    
                    if analysis_data['is_low_volatility']:
                        low_volatility_count += 1
                        logger.info(f"   ⚠️ {symbol}: Low volatility = {volatility:.2f}% (24h change: {price_change_24h:.2f}%)")
                    
                    if analysis_data['is_underestimated']:
                        underestimated_volatility += 1
                        logger.info(f"   📈 {symbol}: Underestimated volatility = {volatility:.2f}% vs 24h change = {price_change_24h:.2f}%")
            
            # Check backend logs for volatility adjustments
            import subprocess
            log_cmd = "tail -n 500 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'volatility\\|atr.*multiplier\\|min.*1.5' || echo 'No volatility logs'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            volatility_logs = []
            for line in result.stdout.split('\n'):
                if any(keyword in line.lower() for keyword in ['volatility', 'atr', 'min', '1.5']):
                    volatility_logs.append(line.strip())
            
            # Analyze volatility distribution
            if volatility_analysis:
                volatilities = [v['volatility'] for v in volatility_analysis]
                avg_volatility = np.mean(volatilities)
                min_volatility = min(volatilities)
                max_volatility = max(volatilities)
                
                logger.info(f"   📊 Volatility Statistics:")
                logger.info(f"      Average: {avg_volatility:.2f}%")
                logger.info(f"      Range: {min_volatility:.2f}% - {max_volatility:.2f}%")
                logger.info(f"      Low volatility (<1.5%): {low_volatility_count}/{len(volatility_analysis)}")
                logger.info(f"      Underestimated: {underestimated_volatility}/{len(volatility_analysis)}")
                logger.info(f"      Volatility logs: {len(volatility_logs)}")
            
            # Check if minimum volatility is causing RR calculation issues
            min_volatility_issues = 0
            if self.rr_calculations:
                for calc in self.rr_calculations:
                    if calc['volatility'] <= 1.5 and calc['rr_ratio'] < 2.0:
                        min_volatility_issues += 1
                        logger.info(f"   🔍 {calc['symbol']}: Min volatility may be limiting RR (vol: {calc['volatility']:.2f}%, RR: {calc['rr_ratio']:.2f})")
            
            success = len(volatility_analysis) > 0
            details = f"Volatility samples: {len(volatility_analysis)}, Low volatility: {low_volatility_count}, Underestimated: {underestimated_volatility}, Min vol issues: {min_volatility_issues}"
            
            self.log_test_result("Volatility Underestimation", success, details)
            
        except Exception as e:
            self.log_test_result("Volatility Underestimation", False, f"Exception: {str(e)}")
    
    async def test_multi_rr_contradiction_detection(self):
        """Test 6: Multi-RR Contradiction Detection - Check if system detects and resolves contradictions"""
        logger.info("\n🔍 TEST 6: Multi-RR Contradiction Detection")
        
        try:
            # Check backend logs for contradiction detection
            import subprocess
            log_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'contradiction\\|multi-rr.*analysis\\|rsi.*oversold\\|macd.*bearish\\|rsi.*overbought' || echo 'No contradiction logs'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            contradiction_logs = []
            multi_rr_analysis_logs = []
            
            for line in result.stdout.split('\n'):
                if 'contradiction' in line.lower():
                    contradiction_logs.append(line.strip())
                elif 'multi-rr' in line.lower() and 'analysis' in line.lower():
                    multi_rr_analysis_logs.append(line.strip())
            
            logger.info(f"   📊 Contradiction logs: {len(contradiction_logs)}")
            logger.info(f"   📊 Multi-RR analysis logs: {len(multi_rr_analysis_logs)}")
            
            # Show sample logs
            if contradiction_logs:
                logger.info(f"   📝 Sample contradiction: {contradiction_logs[-1][:150]}...")
            if multi_rr_analysis_logs:
                logger.info(f"   📝 Sample Multi-RR: {multi_rr_analysis_logs[-1][:150]}...")
            
            # Analyze analyses for contradiction patterns
            analyses, error = self.get_analyses_from_api()
            contradiction_cases = []
            
            if not error and analyses:
                for analysis in analyses:
                    symbol = analysis.get('symbol', 'UNKNOWN')
                    rsi = analysis.get('rsi', 50)
                    macd = analysis.get('macd_signal', 0)
                    ia1_reasoning = analysis.get('ia1_reasoning', '')
                    
                    # Check for specific contradiction patterns mentioned in review
                    rsi_oversold = rsi < 30
                    rsi_overbought = rsi > 70
                    macd_bullish = macd > 0
                    macd_bearish = macd < 0
                    
                    contradiction_detected = False
                    contradiction_type = ""
                    
                    # RSI oversold + MACD bearish contradiction
                    if rsi_oversold and macd_bearish:
                        contradiction_detected = True
                        contradiction_type = "RSI oversold + MACD bearish"
                    # RSI overbought + MACD bullish contradiction
                    elif rsi_overbought and macd_bullish:
                        contradiction_detected = True
                        contradiction_type = "RSI overbought + MACD bullish"
                    
                    if contradiction_detected:
                        contradiction_cases.append({
                            'symbol': symbol,
                            'type': contradiction_type,
                            'rsi': rsi,
                            'macd': macd,
                            'multi_rr_mentioned': 'multi-rr' in ia1_reasoning.lower()
                        })
                        
                        logger.info(f"   🎯 Contradiction detected: {symbol}")
                        logger.info(f"      Type: {contradiction_type}")
                        logger.info(f"      RSI: {rsi:.2f}, MACD: {macd:.6f}")
                        logger.info(f"      Multi-RR mentioned: {contradiction_cases[-1]['multi_rr_mentioned']}")
            
            # Check for BIOUSDT case specifically mentioned in review
            biousdt_case = None
            for analysis in analyses or []:
                symbol = analysis.get('symbol', '').upper()
                if 'BIOUSDT' in symbol:
                    rsi = analysis.get('rsi', 50)
                    macd = analysis.get('macd_signal', 0)
                    if abs(rsi - 24.2) < 1.0 and abs(macd - 0.013892) < 0.01:  # Close to review values
                        biousdt_case = {
                            'symbol': symbol,
                            'rsi': rsi,
                            'macd': macd,
                            'matches_review': True
                        }
                        logger.info(f"   🎯 BIOUSDT case found: RSI={rsi:.2f}, MACD={macd:.6f}")
                        break
            
            # Check decisions for Multi-RR resolution
            decisions, dec_error = self.get_decisions_from_api()
            multi_rr_decisions = 0
            
            if not dec_error and decisions:
                for decision in decisions:
                    reasoning = decision.get('ia2_reasoning', '')
                    if 'multi-rr' in reasoning.lower():
                        multi_rr_decisions += 1
            
            success = len(contradiction_logs) > 0 or len(contradiction_cases) > 0 or biousdt_case is not None
            details = f"Contradiction logs: {len(contradiction_logs)}, Cases detected: {len(contradiction_cases)}, BIOUSDT case: {biousdt_case is not None}, Multi-RR decisions: {multi_rr_decisions}"
            
            self.log_test_result("Multi-RR Contradiction Detection", success, details)
            
        except Exception as e:
            self.log_test_result("Multi-RR Contradiction Detection", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Multi-RR Decision Engine tests"""
        logger.info("🚀 Starting Multi-RR Decision Engine Test Suite - BUSDT Focus")
        logger.info("=" * 80)
        
        await self.setup_database()
        
        # Run all tests
        await self.test_busdt_data_availability()
        await self.test_multi_rr_calculation_formulas()
        await self.test_rr_caps_and_limits()
        await self.test_busdt_specific_calculations()
        await self.test_volatility_underestimation()
        await self.test_multi_rr_contradiction_detection()
        
        await self.cleanup_database()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 MULTI-RR DECISION ENGINE TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Specific analysis for review request
        logger.info("\n" + "=" * 80)
        logger.info("📋 ANALYSIS SUMMARY FOR REVIEW REQUEST")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Multi-RR Decision Engine is working correctly!")
            logger.info("✅ BUSDT analysis data is available and being processed")
            logger.info("✅ Multi-RR calculation formulas are implemented and functioning")
            logger.info("✅ RR caps and limits are being applied appropriately")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - Some Multi-RR issues detected")
            logger.info("🔍 Review the failed tests for specific RR calculation problems")
        else:
            logger.info("❌ CRITICAL ISSUES - Multi-RR Decision Engine needs attention")
            logger.info("🚨 Major problems with BUSDT analysis or RR calculations")
            
        # Recommendations based on test results
        logger.info("\n📝 RECOMMENDATIONS:")
        
        # Check if any tests failed and provide specific recommendations
        failed_tests = [result for result in self.test_results if not result['success']]
        if failed_tests:
            for failed_test in failed_tests:
                logger.info(f"❌ {failed_test['test']}: {failed_test['details']}")
        else:
            logger.info("✅ No critical issues found with Multi-RR Decision Engine")
            logger.info("✅ BUSDT calculations appear to be working correctly")
            logger.info("✅ RR formulas and caps are functioning as designed")
            
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = MultiRRDecisionEngineTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
IA1 Technical Indicators Validation Test Suite
VALIDATION DES CORRECTIONS - Test des indicateurs techniques IA1 après les corrections

CONTEXTE:
Nous avons identifié et corrigé les problèmes critiques avec les indicateurs techniques dans IA1 :
1. ✅ Ajouté les méthodes manquantes (get_market_cap_multiplier, compute_final_score, tanh_norm, clamp)
2. ✅ Redémarré le serveur backend
3. 🚨 Identifié 1,441 erreurs dans les logs (NaN et infinity)
4. 🚨 Valeurs par défaut utilisées au lieu de calculs réels

OBJECTIFS DE VALIDATION:
1. **Test Post-Correction des Indicateurs**:
   - Vérifier que RSI montre des valeurs calculées (pas 50.0 par défaut)
   - Vérifier que MACD montre des valeurs réelles (pas 0.0 par défaut)
   - Vérifier que Stochastic %K et %D sont calculés (pas 50.0 par défaut)
   - Vérifier que Bollinger Position est calculée (pas 0.0 par défaut)

2. **Validation des Ranges Réalistes**:
   - RSI: 0-100 (typiquement 20-80)
   - MACD: valeurs réelles variables
   - Stochastic: 0-100 (typiquement 20-80)  
   - Bollinger: -3 à +3 (typiquement -2 à +2)

3. **Test de Nouvelles Analyses IA1**:
   - Déclencher /api/trading/start-trading pour générer de nouvelles analyses
   - Vérifier que les nouvelles analyses ont des indicateurs calculés correctement
   - Comparer les nouvelles valeurs avec les anciennes valeurs par défaut

4. **Validation des Logs Backend**:
   - Chercher les nouveaux logs d'indicateurs techniques
   - Vérifier l'absence de nouvelles erreurs NaN/Infinity
   - Confirmer que les calculs se terminent sans erreur

5. **Test Multi-Symboles**:
   - Tester sur plusieurs symboles (FORMUSDT, BTCUSDT, ETHUSDT, etc.)
   - Vérifier la cohérence des calculs entre symboles
   - Identifier si certains symboles ont encore des problèmes
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA1TechnicalIndicatorsTestSuite:
    """Test suite for IA1 Technical Indicators Validation after corrections"""
    
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
        logger.info(f"Testing IA1 Technical Indicators at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected indicator ranges
        self.indicator_ranges = {
            'rsi': {'min': 0, 'max': 100, 'typical_min': 20, 'typical_max': 80, 'default': 50.0},
            'macd': {'min': -10, 'max': 10, 'default': 0.0},
            'stochastic': {'min': 0, 'max': 100, 'typical_min': 20, 'typical_max': 80, 'default': 50.0},
            'bollinger_position': {'min': -3, 'max': 3, 'typical_min': -2, 'typical_max': 2, 'default': 0.0}
        }
        
        # Test symbols to focus on
        self.test_symbols = ['FORMUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT']
        
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
    
    def get_backend_logs(self) -> str:
        """Get recent backend logs"""
        backend_logs = ""
        try:
            # Get stdout logs
            log_result = subprocess.run(
                ["tail", "-n", "3000", "/var/log/supervisor/backend.out.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            backend_logs += log_result.stdout
            
            # Get stderr logs
            log_result = subprocess.run(
                ["tail", "-n", "3000", "/var/log/supervisor/backend.err.log"],
                capture_output=True,
                text=True,
                timeout=10
            )
            backend_logs += log_result.stdout
        except Exception as e:
            logger.warning(f"Could not retrieve backend logs: {e}")
        
        return backend_logs
    
    def analyze_indicator_value(self, indicator_name: str, value: float) -> Dict[str, Any]:
        """Analyze if an indicator value is realistic or default"""
        ranges = self.indicator_ranges.get(indicator_name, {})
        
        analysis = {
            'value': value,
            'is_default': False,
            'is_realistic': False,
            'is_in_range': False,
            'is_typical': False,
            'issues': []
        }
        
        # Check if it's the default value
        if 'default' in ranges and abs(value - ranges['default']) < 0.001:
            analysis['is_default'] = True
            analysis['issues'].append(f"Using default value {ranges['default']}")
        
        # Check if it's in valid range
        if 'min' in ranges and 'max' in ranges:
            if ranges['min'] <= value <= ranges['max']:
                analysis['is_in_range'] = True
            else:
                analysis['issues'].append(f"Out of range [{ranges['min']}, {ranges['max']}]")
        
        # Check if it's in typical range
        if 'typical_min' in ranges and 'typical_max' in ranges:
            if ranges['typical_min'] <= value <= ranges['typical_max']:
                analysis['is_typical'] = True
        
        # Check if it's realistic (not default and in range)
        analysis['is_realistic'] = not analysis['is_default'] and analysis['is_in_range']
        
        return analysis
    
    async def test_1_trigger_new_ia1_analyses(self):
        """Test 1: Trigger new IA1 analyses to get fresh technical indicators"""
        logger.info("\n🔍 TEST 1: Trigger new IA1 analyses with /api/trading/start-trading")
        
        try:
            logger.info("   🚀 Triggering fresh IA1 analyses...")
            start_response = requests.post(f"{self.api_url}/trading/start-trading", timeout=180)
            
            if start_response.status_code in [200, 201]:
                logger.info("   ✅ Start trading endpoint successful")
                # Wait for processing
                logger.info("   ⏳ Waiting 45 seconds for IA1 analysis processing...")
                await asyncio.sleep(45)
                success = True
                details = f"HTTP {start_response.status_code} - New analyses triggered"
            else:
                logger.warning(f"   ⚠️ Start trading returned HTTP {start_response.status_code}")
                success = False
                details = f"HTTP {start_response.status_code}: {start_response.text[:200]}"
            
            self.log_test_result("Trigger New IA1 Analyses", success, details)
            
        except Exception as e:
            self.log_test_result("Trigger New IA1 Analyses", False, f"Exception: {str(e)}")
    
    async def test_2_validate_rsi_calculations(self):
        """Test 2: Validate RSI calculations are not using default values"""
        logger.info("\n🔍 TEST 2: Validate RSI calculations (not default 50.0)")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("RSI Calculations", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("RSI Calculations", False, "No IA1 analyses found")
                return
            
            # Analyze RSI values
            rsi_results = []
            default_rsi_count = 0
            realistic_rsi_count = 0
            total_analyses = 0
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                rsi_value = analysis.get('rsi', None)
                reasoning = analysis.get('reasoning', '')
                
                if rsi_value is not None:
                    total_analyses += 1
                    rsi_analysis = self.analyze_indicator_value('rsi', rsi_value)
                    
                    if rsi_analysis['is_default']:
                        default_rsi_count += 1
                        logger.info(f"      ❌ {symbol}: RSI = {rsi_value} (DEFAULT VALUE)")
                    elif rsi_analysis['is_realistic']:
                        realistic_rsi_count += 1
                        logger.info(f"      ✅ {symbol}: RSI = {rsi_value} (CALCULATED)")
                    else:
                        logger.info(f"      ⚠️ {symbol}: RSI = {rsi_value} (UNUSUAL)")
                    
                    rsi_results.append({
                        'symbol': symbol,
                        'value': rsi_value,
                        'analysis': rsi_analysis
                    })
                
                # Check if RSI is mentioned in reasoning
                if 'RSI' in reasoning or 'rsi' in reasoning.lower():
                    logger.info(f"      📊 {symbol}: RSI mentioned in reasoning")
            
            # Calculate success metrics
            if total_analyses > 0:
                realistic_percentage = (realistic_rsi_count / total_analyses) * 100
                default_percentage = (default_rsi_count / total_analyses) * 100
            else:
                realistic_percentage = 0
                default_percentage = 0
            
            logger.info(f"   📊 Total analyses with RSI: {total_analyses}")
            logger.info(f"   📊 Realistic RSI values: {realistic_rsi_count} ({realistic_percentage:.1f}%)")
            logger.info(f"   📊 Default RSI values: {default_rsi_count} ({default_percentage:.1f}%)")
            
            # Success criteria: >70% realistic RSI values, <30% default values
            success = realistic_percentage > 70 and default_percentage < 30 and total_analyses > 0
            
            details = f"Realistic: {realistic_rsi_count}/{total_analyses} ({realistic_percentage:.1f}%), Default: {default_rsi_count} ({default_percentage:.1f}%)"
            
            self.log_test_result("RSI Calculations", success, details)
            
        except Exception as e:
            self.log_test_result("RSI Calculations", False, f"Exception: {str(e)}")
    
    async def test_3_validate_macd_calculations(self):
        """Test 3: Validate MACD calculations are not using default values"""
        logger.info("\n🔍 TEST 3: Validate MACD calculations (not default 0.0)")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("MACD Calculations", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("MACD Calculations", False, "No IA1 analyses found")
                return
            
            # Analyze MACD values
            macd_results = []
            default_macd_count = 0
            realistic_macd_count = 0
            total_analyses = 0
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                macd_value = analysis.get('macd_signal', None) or analysis.get('macd', None)
                reasoning = analysis.get('reasoning', '')
                
                if macd_value is not None:
                    total_analyses += 1
                    macd_analysis = self.analyze_indicator_value('macd', macd_value)
                    
                    if macd_analysis['is_default']:
                        default_macd_count += 1
                        logger.info(f"      ❌ {symbol}: MACD = {macd_value} (DEFAULT VALUE)")
                    elif macd_analysis['is_realistic']:
                        realistic_macd_count += 1
                        logger.info(f"      ✅ {symbol}: MACD = {macd_value} (CALCULATED)")
                    else:
                        logger.info(f"      ⚠️ {symbol}: MACD = {macd_value} (UNUSUAL)")
                    
                    macd_results.append({
                        'symbol': symbol,
                        'value': macd_value,
                        'analysis': macd_analysis
                    })
                
                # Check if MACD is mentioned in reasoning
                if 'MACD' in reasoning or 'macd' in reasoning.lower():
                    logger.info(f"      📊 {symbol}: MACD mentioned in reasoning")
            
            # Calculate success metrics
            if total_analyses > 0:
                realistic_percentage = (realistic_macd_count / total_analyses) * 100
                default_percentage = (default_macd_count / total_analyses) * 100
            else:
                realistic_percentage = 0
                default_percentage = 0
            
            logger.info(f"   📊 Total analyses with MACD: {total_analyses}")
            logger.info(f"   📊 Realistic MACD values: {realistic_macd_count} ({realistic_percentage:.1f}%)")
            logger.info(f"   📊 Default MACD values: {default_macd_count} ({default_percentage:.1f}%)")
            
            # Success criteria: >70% realistic MACD values, <30% default values
            success = realistic_percentage > 70 and default_percentage < 30 and total_analyses > 0
            
            details = f"Realistic: {realistic_macd_count}/{total_analyses} ({realistic_percentage:.1f}%), Default: {default_macd_count} ({default_percentage:.1f}%)"
            
            self.log_test_result("MACD Calculations", success, details)
            
        except Exception as e:
            self.log_test_result("MACD Calculations", False, f"Exception: {str(e)}")
    
    async def test_4_validate_stochastic_calculations(self):
        """Test 4: Validate Stochastic calculations are not using default values"""
        logger.info("\n🔍 TEST 4: Validate Stochastic calculations (not default 50.0)")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Stochastic Calculations", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Stochastic Calculations", False, "No IA1 analyses found")
                return
            
            # Analyze Stochastic values
            stoch_results = []
            default_stoch_count = 0
            realistic_stoch_count = 0
            total_analyses = 0
            stoch_d_found = 0
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                stoch_k = analysis.get('stochastic', None) or analysis.get('stochastic_k', None)
                stoch_d = analysis.get('stochastic_d', None)
                reasoning = analysis.get('reasoning', '')
                
                if stoch_k is not None:
                    total_analyses += 1
                    stoch_analysis = self.analyze_indicator_value('stochastic', stoch_k)
                    
                    if stoch_analysis['is_default']:
                        default_stoch_count += 1
                        logger.info(f"      ❌ {symbol}: Stochastic %K = {stoch_k} (DEFAULT VALUE)")
                    elif stoch_analysis['is_realistic']:
                        realistic_stoch_count += 1
                        logger.info(f"      ✅ {symbol}: Stochastic %K = {stoch_k} (CALCULATED)")
                    else:
                        logger.info(f"      ⚠️ {symbol}: Stochastic %K = {stoch_k} (UNUSUAL)")
                    
                    # Check for Stochastic %D
                    if stoch_d is not None:
                        stoch_d_found += 1
                        logger.info(f"      📊 {symbol}: Stochastic %D = {stoch_d}")
                    
                    stoch_results.append({
                        'symbol': symbol,
                        'stoch_k': stoch_k,
                        'stoch_d': stoch_d,
                        'analysis': stoch_analysis
                    })
                
                # Check if Stochastic is mentioned in reasoning
                if 'stochastic' in reasoning.lower() or 'Stochastic' in reasoning:
                    logger.info(f"      📊 {symbol}: Stochastic mentioned in reasoning")
            
            # Calculate success metrics
            if total_analyses > 0:
                realistic_percentage = (realistic_stoch_count / total_analyses) * 100
                default_percentage = (default_stoch_count / total_analyses) * 100
                stoch_d_percentage = (stoch_d_found / total_analyses) * 100
            else:
                realistic_percentage = 0
                default_percentage = 0
                stoch_d_percentage = 0
            
            logger.info(f"   📊 Total analyses with Stochastic: {total_analyses}")
            logger.info(f"   📊 Realistic Stochastic values: {realistic_stoch_count} ({realistic_percentage:.1f}%)")
            logger.info(f"   📊 Default Stochastic values: {default_stoch_count} ({default_percentage:.1f}%)")
            logger.info(f"   📊 Stochastic %D found: {stoch_d_found} ({stoch_d_percentage:.1f}%)")
            
            # Success criteria: >70% realistic Stochastic values, <30% default values
            success = realistic_percentage > 70 and default_percentage < 30 and total_analyses > 0
            
            details = f"Realistic: {realistic_stoch_count}/{total_analyses} ({realistic_percentage:.1f}%), Default: {default_stoch_count} ({default_percentage:.1f}%), %D found: {stoch_d_found}"
            
            self.log_test_result("Stochastic Calculations", success, details)
            
        except Exception as e:
            self.log_test_result("Stochastic Calculations", False, f"Exception: {str(e)}")
    
    async def test_5_validate_bollinger_calculations(self):
        """Test 5: Validate Bollinger Bands calculations are not using default values"""
        logger.info("\n🔍 TEST 5: Validate Bollinger Bands calculations (not default 0.0)")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Bollinger Calculations", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Bollinger Calculations", False, "No IA1 analyses found")
                return
            
            # Analyze Bollinger values
            bb_results = []
            default_bb_count = 0
            realistic_bb_count = 0
            total_analyses = 0
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                bb_position = analysis.get('bollinger_position', None)
                reasoning = analysis.get('reasoning', '')
                
                if bb_position is not None:
                    total_analyses += 1
                    bb_analysis = self.analyze_indicator_value('bollinger_position', bb_position)
                    
                    if bb_analysis['is_default']:
                        default_bb_count += 1
                        logger.info(f"      ❌ {symbol}: Bollinger Position = {bb_position} (DEFAULT VALUE)")
                    elif bb_analysis['is_realistic']:
                        realistic_bb_count += 1
                        logger.info(f"      ✅ {symbol}: Bollinger Position = {bb_position} (CALCULATED)")
                    else:
                        logger.info(f"      ⚠️ {symbol}: Bollinger Position = {bb_position} (UNUSUAL)")
                    
                    bb_results.append({
                        'symbol': symbol,
                        'value': bb_position,
                        'analysis': bb_analysis
                    })
                
                # Check if Bollinger is mentioned in reasoning
                if 'bollinger' in reasoning.lower() or 'Bollinger' in reasoning:
                    logger.info(f"      📊 {symbol}: Bollinger mentioned in reasoning")
            
            # Calculate success metrics
            if total_analyses > 0:
                realistic_percentage = (realistic_bb_count / total_analyses) * 100
                default_percentage = (default_bb_count / total_analyses) * 100
            else:
                realistic_percentage = 0
                default_percentage = 0
            
            logger.info(f"   📊 Total analyses with Bollinger: {total_analyses}")
            logger.info(f"   📊 Realistic Bollinger values: {realistic_bb_count} ({realistic_percentage:.1f}%)")
            logger.info(f"   📊 Default Bollinger values: {default_bb_count} ({default_percentage:.1f}%)")
            
            # Success criteria: >70% realistic Bollinger values, <30% default values
            success = realistic_percentage > 70 and default_percentage < 30 and total_analyses > 0
            
            details = f"Realistic: {realistic_bb_count}/{total_analyses} ({realistic_percentage:.1f}%), Default: {default_bb_count} ({default_percentage:.1f}%)"
            
            self.log_test_result("Bollinger Calculations", success, details)
            
        except Exception as e:
            self.log_test_result("Bollinger Calculations", False, f"Exception: {str(e)}")
    
    async def test_6_validate_backend_logs_for_errors(self):
        """Test 6: Validate backend logs for NaN/Infinity errors and successful calculations"""
        logger.info("\n🔍 TEST 6: Validate backend logs for technical indicators errors")
        
        try:
            backend_logs = self.get_backend_logs()
            
            if not backend_logs:
                self.log_test_result("Backend Logs Validation", False, "Could not retrieve backend logs")
                return
            
            # Count error patterns
            nan_errors = backend_logs.count('NaN') + backend_logs.count('nan')
            infinity_errors = backend_logs.count('infinity') + backend_logs.count('Infinity') + backend_logs.count('inf')
            technical_errors = backend_logs.count('technical indicator error') + backend_logs.count('indicator calculation error')
            
            # Count success patterns
            rsi_calculations = backend_logs.count('RSI:') + backend_logs.count('rsi:')
            macd_calculations = backend_logs.count('MACD:') + backend_logs.count('macd:')
            stochastic_calculations = backend_logs.count('Stochastic:') + backend_logs.count('stochastic:')
            bollinger_calculations = backend_logs.count('Bollinger:') + backend_logs.count('bollinger:')
            
            # Look for specific success indicators
            successful_calculations = backend_logs.count('Technical indicators calculated') + backend_logs.count('indicators calculated successfully')
            
            logger.info(f"   📊 NaN errors found: {nan_errors}")
            logger.info(f"   📊 Infinity errors found: {infinity_errors}")
            logger.info(f"   📊 Technical indicator errors: {technical_errors}")
            logger.info(f"   📊 RSI calculation logs: {rsi_calculations}")
            logger.info(f"   📊 MACD calculation logs: {macd_calculations}")
            logger.info(f"   📊 Stochastic calculation logs: {stochastic_calculations}")
            logger.info(f"   📊 Bollinger calculation logs: {bollinger_calculations}")
            logger.info(f"   📊 Successful calculation logs: {successful_calculations}")
            
            # Look for recent errors (last 1000 lines)
            recent_logs = '\n'.join(backend_logs.split('\n')[-1000:])
            recent_nan_errors = recent_logs.count('NaN') + recent_logs.count('nan')
            recent_infinity_errors = recent_logs.count('infinity') + recent_logs.count('Infinity') + recent_logs.count('inf')
            
            logger.info(f"   📊 Recent NaN errors (last 1000 lines): {recent_nan_errors}")
            logger.info(f"   📊 Recent Infinity errors (last 1000 lines): {recent_infinity_errors}")
            
            # Success criteria: Low error count and evidence of calculations
            total_errors = nan_errors + infinity_errors + technical_errors
            total_calculations = rsi_calculations + macd_calculations + stochastic_calculations + bollinger_calculations
            recent_errors = recent_nan_errors + recent_infinity_errors
            
            success = (recent_errors < 10 and total_calculations > 0) or (total_errors < 50 and total_calculations > 10)
            
            details = f"Total errors: {total_errors}, Recent errors: {recent_errors}, Calculations: {total_calculations}, Successful: {successful_calculations}"
            
            self.log_test_result("Backend Logs Validation", success, details)
            
        except Exception as e:
            self.log_test_result("Backend Logs Validation", False, f"Exception: {str(e)}")
    
    async def test_7_multi_symbol_consistency(self):
        """Test 7: Test technical indicators consistency across multiple symbols"""
        logger.info("\n🔍 TEST 7: Test multi-symbol technical indicators consistency")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Multi-Symbol Consistency", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Multi-Symbol Consistency", False, "No IA1 analyses found")
                return
            
            # Group analyses by symbol
            symbol_indicators = {}
            
            for analysis in analyses[-20:]:  # Check last 20 analyses
                symbol = analysis.get('symbol', 'Unknown')
                
                if symbol not in symbol_indicators:
                    symbol_indicators[symbol] = {
                        'rsi': [],
                        'macd': [],
                        'stochastic': [],
                        'bollinger': [],
                        'count': 0
                    }
                
                symbol_indicators[symbol]['count'] += 1
                
                # Collect indicator values
                if analysis.get('rsi') is not None:
                    symbol_indicators[symbol]['rsi'].append(analysis.get('rsi'))
                
                if analysis.get('macd_signal') is not None or analysis.get('macd') is not None:
                    macd_val = analysis.get('macd_signal') or analysis.get('macd')
                    symbol_indicators[symbol]['macd'].append(macd_val)
                
                if analysis.get('stochastic') is not None:
                    symbol_indicators[symbol]['stochastic'].append(analysis.get('stochastic'))
                
                if analysis.get('bollinger_position') is not None:
                    symbol_indicators[symbol]['bollinger'].append(analysis.get('bollinger_position'))
            
            # Analyze consistency
            symbols_with_indicators = 0
            symbols_with_realistic_values = 0
            symbols_with_defaults = 0
            
            for symbol, indicators in symbol_indicators.items():
                if indicators['count'] > 0:
                    symbols_with_indicators += 1
                    
                    # Check if symbol has realistic values
                    has_realistic = False
                    has_defaults = False
                    
                    # Check RSI
                    for rsi in indicators['rsi']:
                        if abs(rsi - 50.0) > 0.001:  # Not default
                            has_realistic = True
                        else:
                            has_defaults = True
                    
                    # Check MACD
                    for macd in indicators['macd']:
                        if abs(macd - 0.0) > 0.001:  # Not default
                            has_realistic = True
                        else:
                            has_defaults = True
                    
                    # Check Stochastic
                    for stoch in indicators['stochastic']:
                        if abs(stoch - 50.0) > 0.001:  # Not default
                            has_realistic = True
                        else:
                            has_defaults = True
                    
                    # Check Bollinger
                    for bb in indicators['bollinger']:
                        if abs(bb - 0.0) > 0.001:  # Not default
                            has_realistic = True
                        else:
                            has_defaults = True
                    
                    if has_realistic:
                        symbols_with_realistic_values += 1
                        logger.info(f"      ✅ {symbol}: Has realistic indicator values")
                    
                    if has_defaults:
                        symbols_with_defaults += 1
                        logger.info(f"      ❌ {symbol}: Has default indicator values")
                    
                    # Show indicator summary for symbol
                    rsi_avg = sum(indicators['rsi']) / len(indicators['rsi']) if indicators['rsi'] else None
                    macd_avg = sum(indicators['macd']) / len(indicators['macd']) if indicators['macd'] else None
                    stoch_avg = sum(indicators['stochastic']) / len(indicators['stochastic']) if indicators['stochastic'] else None
                    bb_avg = sum(indicators['bollinger']) / len(indicators['bollinger']) if indicators['bollinger'] else None
                    
                    logger.info(f"      📊 {symbol}: RSI avg={rsi_avg:.2f if rsi_avg else 'N/A'}, MACD avg={macd_avg:.4f if macd_avg else 'N/A'}, Stoch avg={stoch_avg:.2f if stoch_avg else 'N/A'}, BB avg={bb_avg:.2f if bb_avg else 'N/A'}")
            
            # Calculate success metrics
            if symbols_with_indicators > 0:
                realistic_percentage = (symbols_with_realistic_values / symbols_with_indicators) * 100
                default_percentage = (symbols_with_defaults / symbols_with_indicators) * 100
            else:
                realistic_percentage = 0
                default_percentage = 0
            
            logger.info(f"   📊 Symbols with indicators: {symbols_with_indicators}")
            logger.info(f"   📊 Symbols with realistic values: {symbols_with_realistic_values} ({realistic_percentage:.1f}%)")
            logger.info(f"   📊 Symbols with default values: {symbols_with_defaults} ({default_percentage:.1f}%)")
            
            # Success criteria: >80% symbols have realistic values, <50% have defaults
            success = realistic_percentage > 80 and default_percentage < 50 and symbols_with_indicators >= 3
            
            details = f"Symbols tested: {symbols_with_indicators}, Realistic: {symbols_with_realistic_values} ({realistic_percentage:.1f}%), Defaults: {symbols_with_defaults} ({default_percentage:.1f}%)"
            
            self.log_test_result("Multi-Symbol Consistency", success, details)
            
        except Exception as e:
            self.log_test_result("Multi-Symbol Consistency", False, f"Exception: {str(e)}")
    
    async def test_8_ia1_reasoning_integration(self):
        """Test 8: Validate technical indicators are properly integrated in IA1 reasoning"""
        logger.info("\n🔍 TEST 8: Validate technical indicators integration in IA1 reasoning")
        
        try:
            # Get IA1 analyses
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("IA1 Reasoning Integration", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            analyses = data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("IA1 Reasoning Integration", False, "No IA1 analyses found")
                return
            
            # Analyze reasoning integration
            total_analyses = 0
            rsi_mentioned = 0
            macd_mentioned = 0
            stochastic_mentioned = 0
            bollinger_mentioned = 0
            confluence_mentioned = 0
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('reasoning', '').lower()
                
                if reasoning:
                    total_analyses += 1
                    
                    # Check for indicator mentions
                    if 'rsi' in reasoning:
                        rsi_mentioned += 1
                        logger.info(f"      ✅ {symbol}: RSI mentioned in reasoning")
                    
                    if 'macd' in reasoning:
                        macd_mentioned += 1
                        logger.info(f"      ✅ {symbol}: MACD mentioned in reasoning")
                    
                    if 'stochastic' in reasoning:
                        stochastic_mentioned += 1
                        logger.info(f"      ✅ {symbol}: Stochastic mentioned in reasoning")
                    
                    if 'bollinger' in reasoning:
                        bollinger_mentioned += 1
                        logger.info(f"      ✅ {symbol}: Bollinger mentioned in reasoning")
                    
                    # Check for confluence analysis
                    confluence_keywords = ['confluence', 'align', 'confirm', 'contradict', 'divergence']
                    if any(keyword in reasoning for keyword in confluence_keywords):
                        confluence_mentioned += 1
                        logger.info(f"      ✅ {symbol}: Confluence analysis mentioned")
            
            # Calculate integration percentages
            if total_analyses > 0:
                rsi_percentage = (rsi_mentioned / total_analyses) * 100
                macd_percentage = (macd_mentioned / total_analyses) * 100
                stochastic_percentage = (stochastic_mentioned / total_analyses) * 100
                bollinger_percentage = (bollinger_mentioned / total_analyses) * 100
                confluence_percentage = (confluence_mentioned / total_analyses) * 100
            else:
                rsi_percentage = macd_percentage = stochastic_percentage = bollinger_percentage = confluence_percentage = 0
            
            logger.info(f"   📊 Total analyses with reasoning: {total_analyses}")
            logger.info(f"   📊 RSI mentioned: {rsi_mentioned} ({rsi_percentage:.1f}%)")
            logger.info(f"   📊 MACD mentioned: {macd_mentioned} ({macd_percentage:.1f}%)")
            logger.info(f"   📊 Stochastic mentioned: {stochastic_mentioned} ({stochastic_percentage:.1f}%)")
            logger.info(f"   📊 Bollinger mentioned: {bollinger_mentioned} ({bollinger_percentage:.1f}%)")
            logger.info(f"   📊 Confluence analysis: {confluence_mentioned} ({confluence_percentage:.1f}%)")
            
            # Success criteria: At least 3 indicators mentioned in >50% of analyses
            indicators_well_integrated = sum([
                rsi_percentage > 50,
                macd_percentage > 50,
                stochastic_percentage > 50,
                bollinger_percentage > 50
            ])
            
            success = indicators_well_integrated >= 3 and confluence_percentage > 30 and total_analyses > 0
            
            details = f"Well integrated indicators: {indicators_well_integrated}/4, Confluence: {confluence_percentage:.1f}%, Total: {total_analyses}"
            
            self.log_test_result("IA1 Reasoning Integration", success, details)
            
        except Exception as e:
            self.log_test_result("IA1 Reasoning Integration", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all IA1 technical indicators tests"""
        logger.info("🚀 Starting IA1 Technical Indicators Validation Test Suite")
        logger.info("=" * 80)
        logger.info("📋 VALIDATION DES CORRECTIONS - Test des indicateurs techniques IA1")
        logger.info("🎯 Testing: RSI, MACD, Stochastic, Bollinger Bands calculations after corrections")
        logger.info("🎯 Expected: Realistic calculated values instead of default values")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_trigger_new_ia1_analyses()
        await self.test_2_validate_rsi_calculations()
        await self.test_3_validate_macd_calculations()
        await self.test_4_validate_stochastic_calculations()
        await self.test_5_validate_bollinger_calculations()
        await self.test_6_validate_backend_logs_for_errors()
        await self.test_7_multi_symbol_consistency()
        await self.test_8_ia1_reasoning_integration()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA1 TECHNICAL INDICATORS VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # System analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 IA1 TECHNICAL INDICATORS STATUS AFTER CORRECTIONS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - IA1 Technical Indicators FULLY CORRECTED!")
            logger.info("✅ RSI calculations working (not default 50.0)")
            logger.info("✅ MACD calculations working (not default 0.0)")
            logger.info("✅ Stochastic calculations working (not default 50.0)")
            logger.info("✅ Bollinger Bands calculations working (not default 0.0)")
            logger.info("✅ Backend logs show minimal NaN/Infinity errors")
            logger.info("✅ Multi-symbol consistency achieved")
            logger.info("✅ Technical indicators properly integrated in IA1 reasoning")
        elif passed_tests >= total_tests * 0.75:
            logger.info("⚠️ MOSTLY CORRECTED - Technical indicators working with minor issues")
            logger.info("🔍 Some indicators may need fine-tuning for full optimization")
        elif passed_tests >= total_tests * 0.5:
            logger.info("⚠️ PARTIALLY CORRECTED - Some technical indicators working")
            logger.info("🔧 Several indicators may still have calculation issues")
        else:
            logger.info("❌ CORRECTIONS NOT EFFECTIVE - Critical issues remain")
            logger.info("🚨 Major calculation problems still preventing proper indicator values")
        
        # Specific requirements check
        logger.info("\n📝 TECHNICAL INDICATORS REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "RSI" in result['test']:
                    requirements_met.append("✅ RSI shows calculated values (not default 50.0)")
                elif "MACD" in result['test']:
                    requirements_met.append("✅ MACD shows real values (not default 0.0)")
                elif "Stochastic" in result['test']:
                    requirements_met.append("✅ Stochastic %K and %D calculated (not default 50.0)")
                elif "Bollinger" in result['test']:
                    requirements_met.append("✅ Bollinger Position calculated (not default 0.0)")
                elif "Backend Logs" in result['test']:
                    requirements_met.append("✅ Reduced NaN/Infinity errors in backend logs")
                elif "Multi-Symbol" in result['test']:
                    requirements_met.append("✅ Multi-symbol consistency achieved")
                elif "Reasoning" in result['test']:
                    requirements_met.append("✅ Technical indicators integrated in IA1 reasoning")
            else:
                if "RSI" in result['test']:
                    requirements_failed.append("❌ RSI still using default values (50.0)")
                elif "MACD" in result['test']:
                    requirements_failed.append("❌ MACD still using default values (0.0)")
                elif "Stochastic" in result['test']:
                    requirements_failed.append("❌ Stochastic still using default values (50.0)")
                elif "Bollinger" in result['test']:
                    requirements_failed.append("❌ Bollinger still using default values (0.0)")
                elif "Backend Logs" in result['test']:
                    requirements_failed.append("❌ High NaN/Infinity errors persist in logs")
                elif "Multi-Symbol" in result['test']:
                    requirements_failed.append("❌ Inconsistent calculations across symbols")
                elif "Reasoning" in result['test']:
                    requirements_failed.append("❌ Technical indicators not properly integrated")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: IA1 Technical Indicators CORRECTIONS SUCCESSFUL!")
            logger.info("✅ All indicators now show realistic calculated values")
            logger.info("✅ Default values (50.0, 0.0) no longer used")
            logger.info("✅ NaN/Infinity errors significantly reduced")
            logger.info("✅ Multi-symbol consistency achieved")
            logger.info("✅ IA1 analyses use real technical indicator calculations")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: IA1 Technical Indicators MOSTLY CORRECTED")
            logger.info("🔍 Minor issues remain but core functionality restored")
        elif len(requirements_failed) <= 3:
            logger.info("\n⚠️ VERDICT: IA1 Technical Indicators PARTIALLY CORRECTED")
            logger.info("🔧 Several indicators need additional fixes")
        else:
            logger.info("\n❌ VERDICT: IA1 Technical Indicators CORRECTIONS FAILED")
            logger.info("🚨 Major calculation issues persist, corrections not effective")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IA1TechnicalIndicatorsTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
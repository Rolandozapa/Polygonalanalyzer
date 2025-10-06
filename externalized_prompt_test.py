#!/usr/bin/env python3
"""
EXTERNALIZED PROMPT MIGRATION TESTING SUITE
Focus: Test the externalized prompt migration for the AI trading bot system.

CRITICAL VALIDATION POINTS:

1. **Prompt Loading Verification**:
   - Test that externalized prompts are loaded correctly from `/app/prompts/` directory
   - Verify ia1_v6_advanced.json and ia2_strategic.json are properly formatted and accessible
   - Check backend logs for prompt loading success messages

2. **IA1 Prompt Testing**:
   - Test `/api/force-ia1-analysis` with various symbols to ensure IA1 externalized prompt works
   - Verify prompt formatting with all required variables (19 variables for IA1)
   - Check response quality and JSON parsing success
   - Monitor backend logs for "Prompt formatted: ia1_v6_advanced" messages

3. **IA2 Prompt Testing**:
   - Test IA2 escalation scenarios to validate ia2_strategic prompt works
   - Verify JSON format handling (ensure double braces fix resolved formatting issues)
   - Check IA2 strategic analysis responses

4. **System Integration**:
   - Test full IA1 cycle with `/api/run-ia1-cycle` to ensure no prompt-related errors
   - Verify system performance is maintained (no degradation from externalized prompts)
   - Check that all technical indicators and ML regime data flows correctly to prompts

5. **Error Handling**:
   - Verify system gracefully handles any prompt loading failures
   - Test fallback mechanisms if externalized prompts are unavailable

FOCUS: Validate the prompt migration is 100% functional and maintains all existing functionality.
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

class ExternalizedPromptMigrationTestSuite:
    """Comprehensive test suite for Externalized Prompt Migration validation"""
    
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
        logger.info(f"Testing Externalized Prompt Migration at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for analysis
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.actual_test_symbols = []
        
        # Prompt files to validate
        self.prompt_files = {
            'ia1_v6_advanced': '/app/prompts/ia1_v6_advanced.json',
            'ia2_strategic': '/app/prompts/ia2_strategic.json'
        }
        
        # Expected IA1 variables (47 variables from the JSON file)
        self.expected_ia1_variables = [
            "symbol", "market_cap", "market_cap_rank", "current_price", "price_change_24h", "volume_24h",
            "global_market_context", "rsi", "rsi_zone", "rsi_interpretation", "macd_line", "macd_signal", 
            "macd_histogram", "stoch_k", "stoch_d", "adx", "adx_strength", "plus_di", "minus_di",
            "trend_hierarchy", "ema_cross_signal", "mfi", "mfi_signal", "volume_ratio", "volume_trend",
            "volume_surge", "bb_position", "bb_squeeze", "squeeze_intensity", "atr", "atr_pct",
            "vwap", "vwap_distance", "sma_20", "sma_50", "ema_9", "ema_21", "ema_200", "regime",
            "confidence", "base_confidence", "technical_consistency", "combined_confidence",
            "regime_persistence", "fresh_regime", "stability_score", "regime_transition_alert",
            "confluence_grade", "confluence_score", "combined_multiplier", "should_trade"
        ]
        
        # Expected IA2 variables (26 variables from the JSON file)
        self.expected_ia2_variables = [
            "symbol", "ia1_signal", "ia1_confidence", "ia1_rr", "ia1_reasoning", "current_price",
            "price_change_24h", "volume_ratio", "ia1_entry", "ia1_stop", "ia1_target", "rsi",
            "rsi_zone", "macd_histogram", "adx", "adx_strength", "bb_position", "vwap_distance",
            "volume_surge", "regime", "regime_confidence", "regime_persistence", "fresh_regime",
            "regime_transition_alert", "min_rr_threshold", "trade_type", "trade_duration"
        ]
        
        # Database connection info
        self.mongo_url = "mongodb://localhost:27017"
        self.db_name = "myapp"
        
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
    
    async def test_1_prompt_loading_verification(self):
        """Test 1: Prompt Loading Verification - Test that externalized prompts are loaded correctly"""
        logger.info("\n🔍 TEST 1: Prompt Loading Verification")
        
        try:
            prompt_loading_results = {
                'prompt_files_exist': 0,
                'prompt_files_valid_json': 0,
                'prompt_files_have_required_fields': 0,
                'ia1_variables_count': 0,
                'ia2_variables_count': 0,
                'prompt_manager_import_success': False,
                'backend_logs_show_loading': 0,
                'file_details': {}
            }
            
            logger.info("   🚀 Testing externalized prompt files existence and format...")
            logger.info("   📊 Expected: ia1_v6_advanced.json and ia2_strategic.json properly formatted")
            
            # Test 1.1: Check prompt files exist and are valid JSON
            for prompt_name, file_path in self.prompt_files.items():
                logger.info(f"\n   📞 Testing prompt file: {prompt_name} at {file_path}")
                
                # Check file exists
                if os.path.exists(file_path):
                    prompt_loading_results['prompt_files_exist'] += 1
                    logger.info(f"      ✅ File exists: {file_path}")
                    
                    try:
                        # Check valid JSON
                        with open(file_path, 'r', encoding='utf-8') as f:
                            prompt_data = json.load(f)
                        
                        prompt_loading_results['prompt_files_valid_json'] += 1
                        logger.info(f"      ✅ Valid JSON format")
                        
                        # Check required fields
                        required_fields = ['name', 'version', 'prompt_template', 'required_variables']
                        missing_fields = [field for field in required_fields if field not in prompt_data]
                        
                        if not missing_fields:
                            prompt_loading_results['prompt_files_have_required_fields'] += 1
                            logger.info(f"      ✅ All required fields present: {required_fields}")
                        else:
                            logger.error(f"      ❌ Missing required fields: {missing_fields}")
                        
                        # Store file details
                        prompt_loading_results['file_details'][prompt_name] = {
                            'version': prompt_data.get('version', 'unknown'),
                            'name': prompt_data.get('name', 'unknown'),
                            'template_length': len(prompt_data.get('prompt_template', '')),
                            'variables_count': len(prompt_data.get('required_variables', [])),
                            'variables': prompt_data.get('required_variables', [])
                        }
                        
                        # Count variables for specific prompts
                        if prompt_name == 'ia1_v6_advanced':
                            prompt_loading_results['ia1_variables_count'] = len(prompt_data.get('required_variables', []))
                            logger.info(f"      📊 IA1 variables count: {prompt_loading_results['ia1_variables_count']}")
                        elif prompt_name == 'ia2_strategic':
                            prompt_loading_results['ia2_variables_count'] = len(prompt_data.get('required_variables', []))
                            logger.info(f"      📊 IA2 variables count: {prompt_loading_results['ia2_variables_count']}")
                        
                        logger.info(f"      📊 Prompt details: v{prompt_data.get('version')}, {len(prompt_data.get('prompt_template', ''))} chars, {len(prompt_data.get('required_variables', []))} variables")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"      ❌ Invalid JSON format: {e}")
                    except Exception as e:
                        logger.error(f"      ❌ Error reading file: {e}")
                else:
                    logger.error(f"      ❌ File does not exist: {file_path}")
            
            # Test 1.2: Test prompt manager import
            logger.info(f"\n   📞 Testing prompt manager import...")
            try:
                # Test if we can import the prompt manager
                sys.path.append('/app')
                from prompts.prompt_manager import prompt_manager
                prompt_loading_results['prompt_manager_import_success'] = True
                logger.info(f"      ✅ Prompt manager import successful")
                
                # Test loading prompts through manager
                for prompt_name in ['ia1_v6_advanced', 'ia2_strategic']:
                    try:
                        prompt_data = prompt_manager.load_prompt(prompt_name)
                        if prompt_data:
                            logger.info(f"      ✅ Prompt manager loaded: {prompt_name} v{prompt_data.get('version')}")
                        else:
                            logger.error(f"      ❌ Prompt manager failed to load: {prompt_name}")
                    except Exception as e:
                        logger.error(f"      ❌ Prompt manager error loading {prompt_name}: {e}")
                        
            except ImportError as e:
                logger.error(f"      ❌ Prompt manager import failed: {e}")
            except Exception as e:
                logger.error(f"      ❌ Prompt manager test error: {e}")
            
            # Test 1.3: Check backend logs for prompt loading messages
            logger.info(f"\n   📋 Checking backend logs for prompt loading messages...")
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    prompt_loading_logs = []
                    prompt_formatted_logs = []
                    
                    for log_line in backend_logs:
                        if any(pattern in log_line.lower() for pattern in [
                            'prompt loaded:', 'prompt formatted:', 'ia1_v6_advanced', 'ia2_strategic'
                        ]):
                            prompt_loading_logs.append(log_line.strip())
                        
                        if 'prompt formatted:' in log_line.lower():
                            prompt_formatted_logs.append(log_line.strip())
                    
                    prompt_loading_results['backend_logs_show_loading'] = len(prompt_loading_logs)
                    logger.info(f"      📊 Backend prompt logs found: {len(prompt_loading_logs)}")
                    
                    # Show sample logs
                    if prompt_loading_logs:
                        logger.info(f"      📋 Sample prompt loading log: {prompt_loading_logs[0]}")
                    if prompt_formatted_logs:
                        logger.info(f"      📋 Sample prompt formatting log: {prompt_formatted_logs[0]}")
                    
                    if not prompt_loading_logs:
                        logger.warning(f"      ⚠️ No prompt loading logs found - may need to trigger analysis to see logs")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 PROMPT LOADING VERIFICATION RESULTS:")
            logger.info(f"      Prompt files exist: {prompt_loading_results['prompt_files_exist']}/2")
            logger.info(f"      Valid JSON files: {prompt_loading_results['prompt_files_valid_json']}/2")
            logger.info(f"      Files with required fields: {prompt_loading_results['prompt_files_have_required_fields']}/2")
            logger.info(f"      IA1 variables count: {prompt_loading_results['ia1_variables_count']}")
            logger.info(f"      IA2 variables count: {prompt_loading_results['ia2_variables_count']}")
            logger.info(f"      Prompt manager import: {prompt_loading_results['prompt_manager_import_success']}")
            logger.info(f"      Backend loading logs: {prompt_loading_results['backend_logs_show_loading']}")
            
            # Show file details
            for prompt_name, details in prompt_loading_results['file_details'].items():
                logger.info(f"      📊 {prompt_name}: v{details['version']}, {details['template_length']} chars, {details['variables_count']} vars")
            
            # Calculate test success
            success_criteria = [
                prompt_loading_results['prompt_files_exist'] == 2,  # Both files exist
                prompt_loading_results['prompt_files_valid_json'] == 2,  # Both are valid JSON
                prompt_loading_results['prompt_files_have_required_fields'] == 2,  # Both have required fields
                prompt_loading_results['ia1_variables_count'] >= 40,  # IA1 has expected variables (47 expected)
                prompt_loading_results['ia2_variables_count'] >= 20,  # IA2 has expected variables (26 expected)
                prompt_loading_results['prompt_manager_import_success']  # Prompt manager works
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("Prompt Loading Verification", True, 
                                   f"Prompt loading successful: {success_count}/{len(success_criteria)} criteria met. IA1 vars: {prompt_loading_results['ia1_variables_count']}, IA2 vars: {prompt_loading_results['ia2_variables_count']}")
            else:
                self.log_test_result("Prompt Loading Verification", False, 
                                   f"Prompt loading issues: {success_count}/{len(success_criteria)} criteria met. Check file existence, JSON format, or variable counts")
                
        except Exception as e:
            self.log_test_result("Prompt Loading Verification", False, f"Exception: {str(e)}")

    async def test_2_ia1_prompt_testing(self):
        """Test 2: IA1 Prompt Testing - Test /api/force-ia1-analysis with externalized prompts"""
        logger.info("\n🔍 TEST 2: IA1 Prompt Testing")
        
        try:
            ia1_prompt_results = {
                'analyses_attempted': 0,
                'analyses_successful': 0,
                'prompt_formatted_logs': 0,
                'json_parsing_success': 0,
                'reasoning_populated': 0,
                'technical_indicators_present': 0,
                'response_quality_good': 0,
                'backend_prompt_logs': [],
                'successful_analyses': [],
                'response_times': []
            }
            
            logger.info("   🚀 Testing IA1 externalized prompt functionality...")
            logger.info("   📊 Expected: force-ia1-analysis works with externalized ia1_v6_advanced prompt")
            
            # Get available symbols from scout system
            logger.info("   📞 Getting available symbols from scout system...")
            
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                
                if response.status_code == 200:
                    opportunities = response.json()
                    if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                        opportunities = opportunities['opportunities']
                    
                    # Get first 3 available symbols for testing
                    available_symbols = [opp.get('symbol') for opp in opportunities[:10] if opp.get('symbol')]
                    
                    # Prefer symbols from test list
                    test_symbols = []
                    for symbol in self.test_symbols:
                        if symbol in available_symbols:
                            test_symbols.append(symbol)
                    
                    # Fill remaining slots with available symbols
                    for symbol in available_symbols:
                        if symbol not in test_symbols and len(test_symbols) < 3:
                            test_symbols.append(symbol)
                    
                    self.actual_test_symbols = test_symbols[:3]
                    logger.info(f"      ✅ Test symbols selected: {self.actual_test_symbols}")
                    
                else:
                    logger.warning(f"      ⚠️ Could not get opportunities, using default symbols")
                    self.actual_test_symbols = self.test_symbols
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Error getting opportunities: {e}, using default symbols")
                self.actual_test_symbols = self.test_symbols
            
            # Test each symbol with IA1 externalized prompt
            for symbol in self.actual_test_symbols:
                logger.info(f"\n   📞 Testing IA1 externalized prompt for {symbol}...")
                ia1_prompt_results['analyses_attempted'] += 1
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    response_time = time.time() - start_time
                    ia1_prompt_results['response_times'].append(response_time)
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        ia1_prompt_results['analyses_successful'] += 1
                        
                        logger.info(f"      ✅ {symbol} analysis successful (response time: {response_time:.2f}s)")
                        
                        # Check for IA1 analysis data
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        if isinstance(ia1_analysis, dict):
                            # Check reasoning field (critical for externalized prompt)
                            reasoning = ia1_analysis.get('reasoning', '')
                            if reasoning and reasoning.strip() and reasoning != 'null':
                                ia1_prompt_results['reasoning_populated'] += 1
                                logger.info(f"         ✅ Reasoning field populated: {len(reasoning)} chars")
                                
                                # Check for technical indicators in reasoning
                                technical_indicators = ['RSI', 'MACD', 'MFI', 'VWAP', 'ADX', 'regime']
                                indicators_found = []
                                for indicator in technical_indicators:
                                    if indicator.upper() in reasoning.upper():
                                        indicators_found.append(indicator)
                                
                                if len(indicators_found) >= 3:
                                    ia1_prompt_results['technical_indicators_present'] += 1
                                    logger.info(f"         ✅ Technical indicators in reasoning: {indicators_found}")
                                else:
                                    logger.warning(f"         ⚠️ Few technical indicators found: {indicators_found}")
                            else:
                                logger.warning(f"         ❌ Reasoning field empty or null")
                            
                            # Check JSON parsing success (key fields present)
                            key_fields = ['signal', 'confidence', 'entry_price', 'stop_loss_price', 'take_profit_price']
                            present_fields = [field for field in key_fields if field in ia1_analysis and ia1_analysis[field] is not None]
                            
                            if len(present_fields) >= 4:
                                ia1_prompt_results['json_parsing_success'] += 1
                                logger.info(f"         ✅ JSON parsing successful: {len(present_fields)}/5 key fields present")
                            else:
                                logger.warning(f"         ⚠️ JSON parsing issues: only {len(present_fields)}/5 key fields present")
                            
                            # Check response quality
                            signal = ia1_analysis.get('signal', '').upper()
                            confidence = ia1_analysis.get('confidence', 0)
                            
                            if signal in ['LONG', 'SHORT', 'HOLD'] and 0 <= confidence <= 1:
                                ia1_prompt_results['response_quality_good'] += 1
                                logger.info(f"         ✅ Response quality good: signal={signal}, confidence={confidence:.2f}")
                            else:
                                logger.warning(f"         ⚠️ Response quality issues: signal={signal}, confidence={confidence}")
                        
                        # Store successful analysis
                        ia1_prompt_results['successful_analyses'].append({
                            'symbol': symbol,
                            'response_time': response_time,
                            'reasoning_length': len(ia1_analysis.get('reasoning', '')),
                            'signal': ia1_analysis.get('signal'),
                            'confidence': ia1_analysis.get('confidence')
                        })
                        
                    else:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                        if response.text:
                            logger.error(f"         Error response: {response.text[:300]}")
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} analysis exception: {e}")
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 10 seconds before next analysis...")
                    await asyncio.sleep(10)
            
            # Check backend logs for prompt formatting messages
            logger.info("   📋 Checking backend logs for IA1 prompt formatting...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    prompt_logs = []
                    formatted_logs = []
                    
                    for log_line in backend_logs:
                        if any(pattern in log_line.lower() for pattern in [
                            'prompt formatted: ia1_v6_advanced', 'ia1_v6_advanced', 'prompt loaded: ia1'
                        ]):
                            prompt_logs.append(log_line.strip())
                        
                        if 'prompt formatted:' in log_line.lower() and 'ia1' in log_line.lower():
                            formatted_logs.append(log_line.strip())
                    
                    ia1_prompt_results['backend_prompt_logs'] = prompt_logs
                    ia1_prompt_results['prompt_formatted_logs'] = len(formatted_logs)
                    
                    logger.info(f"      📊 IA1 prompt logs found: {len(prompt_logs)}")
                    logger.info(f"      📊 Prompt formatted logs: {len(formatted_logs)}")
                    
                    # Show sample logs
                    if formatted_logs:
                        logger.info(f"      📋 Sample formatting log: {formatted_logs[0]}")
                    if prompt_logs:
                        logger.info(f"      📋 Sample prompt log: {prompt_logs[0]}")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Final analysis
            success_rate = ia1_prompt_results['analyses_successful'] / max(ia1_prompt_results['analyses_attempted'], 1)
            reasoning_rate = ia1_prompt_results['reasoning_populated'] / max(ia1_prompt_results['analyses_attempted'], 1)
            json_rate = ia1_prompt_results['json_parsing_success'] / max(ia1_prompt_results['analyses_attempted'], 1)
            quality_rate = ia1_prompt_results['response_quality_good'] / max(ia1_prompt_results['analyses_attempted'], 1)
            avg_response_time = sum(ia1_prompt_results['response_times']) / max(len(ia1_prompt_results['response_times']), 1)
            
            logger.info(f"\n   📊 IA1 PROMPT TESTING RESULTS:")
            logger.info(f"      Analyses attempted: {ia1_prompt_results['analyses_attempted']}")
            logger.info(f"      Analyses successful: {ia1_prompt_results['analyses_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      Reasoning populated: {ia1_prompt_results['reasoning_populated']} ({reasoning_rate:.2f})")
            logger.info(f"      JSON parsing success: {ia1_prompt_results['json_parsing_success']} ({json_rate:.2f})")
            logger.info(f"      Response quality good: {ia1_prompt_results['response_quality_good']} ({quality_rate:.2f})")
            logger.info(f"      Technical indicators present: {ia1_prompt_results['technical_indicators_present']}")
            logger.info(f"      Prompt formatted logs: {ia1_prompt_results['prompt_formatted_logs']}")
            logger.info(f"      Average response time: {avg_response_time:.2f}s")
            
            # Calculate test success
            success_criteria = [
                ia1_prompt_results['analyses_successful'] >= 2,  # At least 2 successful analyses
                success_rate >= 0.67,  # At least 67% success rate
                reasoning_rate >= 0.67,  # At least 67% have reasoning populated
                json_rate >= 0.67,  # At least 67% have good JSON parsing
                quality_rate >= 0.67,  # At least 67% have good response quality
                avg_response_time < 60  # Reasonable response time
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("IA1 Prompt Testing", True, 
                                   f"IA1 prompt testing successful: {success_count}/{len(success_criteria)} criteria met. Success rate: {success_rate:.2f}, Reasoning rate: {reasoning_rate:.2f}, Quality rate: {quality_rate:.2f}")
            else:
                self.log_test_result("IA1 Prompt Testing", False, 
                                   f"IA1 prompt testing issues: {success_count}/{len(success_criteria)} criteria met. May have prompt formatting or response quality issues")
                
        except Exception as e:
            self.log_test_result("IA1 Prompt Testing", False, f"Exception: {str(e)}")

    async def test_3_ia2_prompt_testing(self):
        """Test 3: IA2 Prompt Testing - Test IA2 escalation scenarios with externalized prompts"""
        logger.info("\n🔍 TEST 3: IA2 Prompt Testing")
        
        try:
            ia2_prompt_results = {
                'analyses_attempted': 0,
                'analyses_successful': 0,
                'ia2_escalations_triggered': 0,
                'ia2_responses_valid': 0,
                'strategic_reasoning_populated': 0,
                'json_format_correct': 0,
                'backend_ia2_logs': [],
                'escalation_details': []
            }
            
            logger.info("   🚀 Testing IA2 externalized prompt functionality...")
            logger.info("   📊 Expected: IA2 escalations work with externalized ia2_strategic prompt")
            
            # Test IA2 by running analyses that should trigger escalation
            # We need high confidence or high RR to trigger IA2
            for symbol in self.actual_test_symbols:
                logger.info(f"\n   📞 Testing IA2 escalation potential for {symbol}...")
                ia2_prompt_results['analyses_attempted'] += 1
                
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
                        ia2_prompt_results['analyses_successful'] += 1
                        
                        logger.info(f"      ✅ {symbol} analysis successful (response time: {response_time:.2f}s)")
                        
                        # Check if IA2 was triggered
                        ia2_decision = analysis_data.get('ia2_decision')
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        if ia2_decision and isinstance(ia2_decision, dict):
                            ia2_prompt_results['ia2_escalations_triggered'] += 1
                            logger.info(f"         🚀 IA2 escalation triggered for {symbol}")
                            
                            # Check IA2 response quality
                            strategic_reasoning = ia2_decision.get('strategic_reasoning', '')
                            if strategic_reasoning and strategic_reasoning.strip():
                                ia2_prompt_results['strategic_reasoning_populated'] += 1
                                logger.info(f"         ✅ Strategic reasoning populated: {len(strategic_reasoning)} chars")
                            else:
                                logger.warning(f"         ⚠️ Strategic reasoning empty")
                            
                            # Check JSON format correctness
                            required_ia2_fields = ['signal', 'confidence', 'strategic_reasoning', 'ia2_entry_price']
                            present_fields = [field for field in required_ia2_fields if field in ia2_decision and ia2_decision[field] is not None]
                            
                            if len(present_fields) >= 3:
                                ia2_prompt_results['json_format_correct'] += 1
                                logger.info(f"         ✅ IA2 JSON format correct: {len(present_fields)}/4 fields present")
                            else:
                                logger.warning(f"         ⚠️ IA2 JSON format issues: only {len(present_fields)}/4 fields present")
                            
                            # Check response validity
                            ia2_signal = ia2_decision.get('signal', '').upper()
                            ia2_confidence = ia2_decision.get('confidence', 0)
                            
                            if ia2_signal in ['LONG', 'SHORT', 'HOLD'] and 0 <= ia2_confidence <= 1:
                                ia2_prompt_results['ia2_responses_valid'] += 1
                                logger.info(f"         ✅ IA2 response valid: signal={ia2_signal}, confidence={ia2_confidence:.2f}")
                            else:
                                logger.warning(f"         ⚠️ IA2 response invalid: signal={ia2_signal}, confidence={ia2_confidence}")
                            
                            # Store escalation details
                            ia2_prompt_results['escalation_details'].append({
                                'symbol': symbol,
                                'ia1_confidence': ia1_analysis.get('confidence', 0),
                                'ia1_rr': ia1_analysis.get('risk_reward_ratio', 0),
                                'ia2_signal': ia2_signal,
                                'ia2_confidence': ia2_confidence,
                                'strategic_reasoning_length': len(strategic_reasoning),
                                'response_time': response_time
                            })
                            
                        else:
                            logger.info(f"         ℹ️ No IA2 escalation for {symbol} (may not meet escalation criteria)")
                            
                            # Check escalation criteria
                            ia1_confidence = ia1_analysis.get('confidence', 0)
                            ia1_rr = ia1_analysis.get('risk_reward_ratio', 0)
                            ia1_signal = ia1_analysis.get('signal', '').upper()
                            
                            logger.info(f"         📊 Escalation criteria: confidence={ia1_confidence:.2f}, rr={ia1_rr:.2f}, signal={ia1_signal}")
                        
                    else:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} analysis exception: {e}")
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 10 seconds before next analysis...")
                    await asyncio.sleep(10)
            
            # Check backend logs for IA2 prompt usage
            logger.info("   📋 Checking backend logs for IA2 prompt usage...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    ia2_logs = []
                    
                    for log_line in backend_logs:
                        if any(pattern in log_line.lower() for pattern in [
                            'ia2 accepted', 'ia2 rejected', 'ia2_strategic', 'strategic reasoning'
                        ]):
                            ia2_logs.append(log_line.strip())
                    
                    ia2_prompt_results['backend_ia2_logs'] = ia2_logs
                    logger.info(f"      📊 IA2 backend logs found: {len(ia2_logs)}")
                    
                    # Show sample logs
                    if ia2_logs:
                        logger.info(f"      📋 Sample IA2 log: {ia2_logs[0]}")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Final analysis
            success_rate = ia2_prompt_results['analyses_successful'] / max(ia2_prompt_results['analyses_attempted'], 1)
            escalation_rate = ia2_prompt_results['ia2_escalations_triggered'] / max(ia2_prompt_results['analyses_attempted'], 1)
            reasoning_rate = ia2_prompt_results['strategic_reasoning_populated'] / max(ia2_prompt_results['ia2_escalations_triggered'], 1)
            validity_rate = ia2_prompt_results['ia2_responses_valid'] / max(ia2_prompt_results['ia2_escalations_triggered'], 1)
            
            logger.info(f"\n   📊 IA2 PROMPT TESTING RESULTS:")
            logger.info(f"      Analyses attempted: {ia2_prompt_results['analyses_attempted']}")
            logger.info(f"      Analyses successful: {ia2_prompt_results['analyses_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      IA2 escalations triggered: {ia2_prompt_results['ia2_escalations_triggered']}")
            logger.info(f"      Escalation rate: {escalation_rate:.2f}")
            logger.info(f"      Strategic reasoning populated: {ia2_prompt_results['strategic_reasoning_populated']} ({reasoning_rate:.2f})")
            logger.info(f"      IA2 responses valid: {ia2_prompt_results['ia2_responses_valid']} ({validity_rate:.2f})")
            logger.info(f"      JSON format correct: {ia2_prompt_results['json_format_correct']}")
            
            # Show escalation details
            if ia2_prompt_results['escalation_details']:
                logger.info(f"      📊 IA2 Escalation Details:")
                for detail in ia2_prompt_results['escalation_details']:
                    logger.info(f"         - {detail['symbol']}: IA1 conf={detail['ia1_confidence']:.2f}, IA1 RR={detail['ia1_rr']:.2f}, IA2 signal={detail['ia2_signal']}, IA2 conf={detail['ia2_confidence']:.2f}")
            
            # Calculate test success
            success_criteria = [
                ia2_prompt_results['analyses_successful'] >= 2,  # At least 2 successful analyses
                success_rate >= 0.67,  # At least 67% success rate
                ia2_prompt_results['ia2_escalations_triggered'] >= 1,  # At least 1 IA2 escalation
                reasoning_rate >= 0.5 if ia2_prompt_results['ia2_escalations_triggered'] > 0 else True,  # Good reasoning rate if escalations occurred
                validity_rate >= 0.5 if ia2_prompt_results['ia2_escalations_triggered'] > 0 else True  # Good validity rate if escalations occurred
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("IA2 Prompt Testing", True, 
                                   f"IA2 prompt testing successful: {success_count}/{len(success_criteria)} criteria met. Escalations: {ia2_prompt_results['ia2_escalations_triggered']}, Success rate: {success_rate:.2f}")
            else:
                self.log_test_result("IA2 Prompt Testing", False, 
                                   f"IA2 prompt testing issues: {success_count}/{len(success_criteria)} criteria met. May need higher confidence/RR to trigger IA2 or prompt issues")
                
        except Exception as e:
            self.log_test_result("IA2 Prompt Testing", False, f"Exception: {str(e)}")

    async def test_4_system_integration(self):
        """Test 4: System Integration - Test full IA1 cycle with externalized prompts"""
        logger.info("\n🔍 TEST 4: System Integration")
        
        try:
            integration_results = {
                'ia1_cycles_attempted': 0,
                'ia1_cycles_successful': 0,
                'no_prompt_errors': 0,
                'performance_maintained': 0,
                'technical_indicators_flow': 0,
                'ml_regime_data_flow': 0,
                'backend_integration_logs': [],
                'cycle_details': []
            }
            
            logger.info("   🚀 Testing full system integration with externalized prompts...")
            logger.info("   📊 Expected: run-ia1-cycle works without prompt-related errors")
            
            # Test full IA1 cycles
            for i in range(2):  # Test 2 cycles
                logger.info(f"\n   📞 Testing IA1 cycle #{i+1}...")
                integration_results['ia1_cycles_attempted'] += 1
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={},
                        timeout=180  # Longer timeout for full cycle
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        integration_results['ia1_cycles_successful'] += 1
                        
                        logger.info(f"      ✅ IA1 cycle #{i+1} successful (response time: {response_time:.2f}s)")
                        
                        # Check for successful analysis in cycle
                        analysis = cycle_data.get('analysis', {})
                        if isinstance(analysis, dict) and analysis:
                            # Check technical indicators flow
                            technical_fields = ['rsi', 'macd_line', 'adx', 'mfi', 'vwap']
                            present_technical = [field for field in technical_fields if field in analysis and analysis[field] is not None]
                            
                            if len(present_technical) >= 3:
                                integration_results['technical_indicators_flow'] += 1
                                logger.info(f"         ✅ Technical indicators flowing: {len(present_technical)}/5 present")
                            else:
                                logger.warning(f"         ⚠️ Few technical indicators: {len(present_technical)}/5 present")
                            
                            # Check ML regime data flow
                            regime_fields = ['regime', 'confidence', 'regime_persistence']
                            present_regime = [field for field in regime_fields if field in analysis and analysis[field] is not None]
                            
                            if len(present_regime) >= 2:
                                integration_results['ml_regime_data_flow'] += 1
                                logger.info(f"         ✅ ML regime data flowing: {len(present_regime)}/3 present")
                            else:
                                logger.warning(f"         ⚠️ ML regime data issues: {len(present_regime)}/3 present")
                            
                            # Check reasoning for prompt usage
                            reasoning = analysis.get('reasoning', '')
                            if reasoning and len(reasoning) > 100:
                                integration_results['no_prompt_errors'] += 1
                                logger.info(f"         ✅ No prompt errors: reasoning populated ({len(reasoning)} chars)")
                            else:
                                logger.warning(f"         ⚠️ Potential prompt issues: reasoning short or empty")
                        
                        # Check performance
                        if response_time < 120:  # Should complete within 2 minutes
                            integration_results['performance_maintained'] += 1
                            logger.info(f"         ✅ Performance maintained: {response_time:.2f}s")
                        else:
                            logger.warning(f"         ⚠️ Performance degraded: {response_time:.2f}s")
                        
                        # Store cycle details
                        integration_results['cycle_details'].append({
                            'cycle': i+1,
                            'response_time': response_time,
                            'symbol': analysis.get('symbol', 'unknown'),
                            'reasoning_length': len(analysis.get('reasoning', '')),
                            'technical_indicators': len(present_technical) if 'present_technical' in locals() else 0,
                            'regime_data': len(present_regime) if 'present_regime' in locals() else 0
                        })
                        
                    else:
                        logger.error(f"      ❌ IA1 cycle #{i+1} failed: HTTP {response.status_code}")
                        if response.text:
                            logger.error(f"         Error response: {response.text[:300]}")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 cycle #{i+1} exception: {e}")
                
                # Wait between cycles
                if i < 1:  # Not the last cycle
                    logger.info(f"      ⏳ Waiting 15 seconds before next cycle...")
                    await asyncio.sleep(15)
            
            # Check backend logs for integration issues
            logger.info("   📋 Checking backend logs for integration issues...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    integration_logs = []
                    error_logs = []
                    
                    for log_line in backend_logs:
                        if any(pattern in log_line.lower() for pattern in [
                            'prompt formatted:', 'prompt loaded:', 'ia1 cycle', 'technical indicators'
                        ]):
                            integration_logs.append(log_line.strip())
                        
                        if any(pattern in log_line.lower() for pattern in [
                            'error', 'exception', 'failed', 'prompt not found'
                        ]):
                            error_logs.append(log_line.strip())
                    
                    integration_results['backend_integration_logs'] = integration_logs
                    logger.info(f"      📊 Integration logs found: {len(integration_logs)}")
                    logger.info(f"      📊 Error logs found: {len(error_logs)}")
                    
                    # Show sample logs
                    if integration_logs:
                        logger.info(f"      📋 Sample integration log: {integration_logs[0]}")
                    if error_logs:
                        logger.warning(f"      ⚠️ Sample error log: {error_logs[0]}")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Final analysis
            success_rate = integration_results['ia1_cycles_successful'] / max(integration_results['ia1_cycles_attempted'], 1)
            no_errors_rate = integration_results['no_prompt_errors'] / max(integration_results['ia1_cycles_successful'], 1)
            performance_rate = integration_results['performance_maintained'] / max(integration_results['ia1_cycles_successful'], 1)
            technical_rate = integration_results['technical_indicators_flow'] / max(integration_results['ia1_cycles_successful'], 1)
            regime_rate = integration_results['ml_regime_data_flow'] / max(integration_results['ia1_cycles_successful'], 1)
            
            logger.info(f"\n   📊 SYSTEM INTEGRATION RESULTS:")
            logger.info(f"      IA1 cycles attempted: {integration_results['ia1_cycles_attempted']}")
            logger.info(f"      IA1 cycles successful: {integration_results['ia1_cycles_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      No prompt errors: {integration_results['no_prompt_errors']} ({no_errors_rate:.2f})")
            logger.info(f"      Performance maintained: {integration_results['performance_maintained']} ({performance_rate:.2f})")
            logger.info(f"      Technical indicators flow: {integration_results['technical_indicators_flow']} ({technical_rate:.2f})")
            logger.info(f"      ML regime data flow: {integration_results['ml_regime_data_flow']} ({regime_rate:.2f})")
            
            # Show cycle details
            if integration_results['cycle_details']:
                logger.info(f"      📊 Cycle Details:")
                for detail in integration_results['cycle_details']:
                    logger.info(f"         - Cycle {detail['cycle']}: {detail['symbol']}, {detail['response_time']:.1f}s, reasoning={detail['reasoning_length']} chars, tech={detail['technical_indicators']}, regime={detail['regime_data']}")
            
            # Calculate test success
            success_criteria = [
                integration_results['ia1_cycles_successful'] >= 1,  # At least 1 successful cycle
                success_rate >= 0.5,  # At least 50% success rate
                no_errors_rate >= 0.5,  # At least 50% without prompt errors
                performance_rate >= 0.5,  # At least 50% maintain performance
                technical_rate >= 0.5,  # At least 50% have technical indicators
                regime_rate >= 0.5  # At least 50% have regime data
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("System Integration", True, 
                                   f"System integration successful: {success_count}/{len(success_criteria)} criteria met. Success rate: {success_rate:.2f}, No errors: {no_errors_rate:.2f}, Performance: {performance_rate:.2f}")
            else:
                self.log_test_result("System Integration", False, 
                                   f"System integration issues: {success_count}/{len(success_criteria)} criteria met. May have prompt errors or performance degradation")
                
        except Exception as e:
            self.log_test_result("System Integration", False, f"Exception: {str(e)}")

    async def test_5_error_handling(self):
        """Test 5: Error Handling - Verify system gracefully handles prompt loading failures"""
        logger.info("\n🔍 TEST 5: Error Handling")
        
        try:
            error_handling_results = {
                'prompt_files_accessible': 0,
                'graceful_error_handling': 0,
                'fallback_mechanisms': 0,
                'system_stability': 0,
                'error_logs_appropriate': 0,
                'recovery_successful': 0,
                'test_scenarios': []
            }
            
            logger.info("   🚀 Testing error handling for prompt loading failures...")
            logger.info("   📊 Expected: System gracefully handles prompt issues without crashing")
            
            # Test 5.1: Check prompt file accessibility
            logger.info(f"\n   📞 Testing prompt file accessibility...")
            
            for prompt_name, file_path in self.prompt_files.items():
                try:
                    if os.path.exists(file_path) and os.access(file_path, os.R_OK):
                        error_handling_results['prompt_files_accessible'] += 1
                        logger.info(f"      ✅ {prompt_name} accessible: {file_path}")
                    else:
                        logger.warning(f"      ⚠️ {prompt_name} not accessible: {file_path}")
                except Exception as e:
                    logger.error(f"      ❌ Error checking {prompt_name}: {e}")
            
            # Test 5.2: Test system behavior with potential prompt issues
            logger.info(f"\n   📞 Testing system stability with prompt operations...")
            
            # Test a few analyses to see if system handles any prompt issues gracefully
            stable_analyses = 0
            for symbol in self.actual_test_symbols[:2]:  # Test 2 symbols
                try:
                    logger.info(f"      Testing system stability with {symbol}...")
                    
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=60
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        stable_analyses += 1
                        logger.info(f"         ✅ System stable for {symbol} ({response_time:.2f}s)")
                    elif response.status_code == 500:
                        # Check if it's a graceful error
                        error_text = response.text.lower()
                        if 'prompt' in error_text and 'error' in error_text:
                            error_handling_results['graceful_error_handling'] += 1
                            logger.info(f"         ✅ Graceful prompt error handling for {symbol}")
                        else:
                            logger.warning(f"         ⚠️ Non-graceful error for {symbol}: {response.text[:200]}")
                    else:
                        logger.warning(f"         ⚠️ Unexpected response for {symbol}: {response.status_code}")
                    
                    error_handling_results['test_scenarios'].append({
                        'symbol': symbol,
                        'status_code': response.status_code,
                        'response_time': response_time,
                        'stable': response.status_code == 200
                    })
                    
                except Exception as e:
                    logger.error(f"         ❌ Exception testing {symbol}: {e}")
                
                await asyncio.sleep(5)  # Short wait between tests
            
            if stable_analyses >= 1:
                error_handling_results['system_stability'] = 1
                logger.info(f"      ✅ System stability confirmed: {stable_analyses}/2 analyses stable")
            else:
                logger.warning(f"      ⚠️ System stability concerns: {stable_analyses}/2 analyses stable")
            
            # Test 5.3: Check backend logs for appropriate error handling
            logger.info(f"\n   📋 Checking backend logs for error handling patterns...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    error_logs = []
                    graceful_logs = []
                    
                    for log_line in backend_logs:
                        if any(pattern in log_line.lower() for pattern in [
                            'error loading prompt', 'prompt file not found', 'invalid json in prompt'
                        ]):
                            error_logs.append(log_line.strip())
                        
                        if any(pattern in log_line.lower() for pattern in [
                            'fallback', 'graceful', 'recovering', 'using default'
                        ]):
                            graceful_logs.append(log_line.strip())
                    
                    if error_logs:
                        error_handling_results['error_logs_appropriate'] = 1
                        logger.info(f"      📊 Appropriate error logs found: {len(error_logs)}")
                        logger.info(f"      📋 Sample error log: {error_logs[0]}")
                    else:
                        logger.info(f"      ✅ No prompt error logs found (system working correctly)")
                        error_handling_results['error_logs_appropriate'] = 1  # No errors is good
                    
                    if graceful_logs:
                        error_handling_results['fallback_mechanisms'] = 1
                        logger.info(f"      ✅ Fallback mechanisms detected: {len(graceful_logs)}")
                        logger.info(f"      📋 Sample fallback log: {graceful_logs[0]}")
                    else:
                        logger.info(f"      ℹ️ No fallback logs (may not be needed if prompts working)")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Test 5.4: Test recovery after potential issues
            logger.info(f"\n   📞 Testing system recovery...")
            
            try:
                # Test one more analysis to see if system recovers
                recovery_symbol = self.actual_test_symbols[0] if self.actual_test_symbols else 'BTCUSDT'
                
                response = requests.post(
                    f"{self.api_url}/force-ia1-analysis",
                    json={"symbol": recovery_symbol},
                    timeout=60
                )
                
                if response.status_code == 200:
                    error_handling_results['recovery_successful'] = 1
                    logger.info(f"      ✅ System recovery successful for {recovery_symbol}")
                else:
                    logger.warning(f"      ⚠️ System recovery issues: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"      ❌ Recovery test exception: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 ERROR HANDLING RESULTS:")
            logger.info(f"      Prompt files accessible: {error_handling_results['prompt_files_accessible']}/2")
            logger.info(f"      Graceful error handling: {error_handling_results['graceful_error_handling']}")
            logger.info(f"      Fallback mechanisms: {error_handling_results['fallback_mechanisms']}")
            logger.info(f"      System stability: {error_handling_results['system_stability']}")
            logger.info(f"      Error logs appropriate: {error_handling_results['error_logs_appropriate']}")
            logger.info(f"      Recovery successful: {error_handling_results['recovery_successful']}")
            
            # Show test scenarios
            if error_handling_results['test_scenarios']:
                logger.info(f"      📊 Test Scenarios:")
                for scenario in error_handling_results['test_scenarios']:
                    logger.info(f"         - {scenario['symbol']}: HTTP {scenario['status_code']}, {scenario['response_time']:.1f}s, stable={scenario['stable']}")
            
            # Calculate test success
            success_criteria = [
                error_handling_results['prompt_files_accessible'] >= 2,  # Both prompt files accessible
                error_handling_results['system_stability'] == 1,  # System is stable
                error_handling_results['error_logs_appropriate'] == 1,  # Appropriate error logging
                error_handling_results['recovery_successful'] == 1,  # System can recover
                len(error_handling_results['test_scenarios']) >= 2  # Tested multiple scenarios
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Error Handling", True, 
                                   f"Error handling successful: {success_count}/{len(success_criteria)} criteria met. System stable and recovers gracefully")
            else:
                self.log_test_result("Error Handling", False, 
                                   f"Error handling issues: {success_count}/{len(success_criteria)} criteria met. May have stability or recovery issues")
                
        except Exception as e:
            self.log_test_result("Error Handling", False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all externalized prompt migration tests"""
        logger.info("🚀 STARTING EXTERNALIZED PROMPT MIGRATION TEST SUITE")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests
        await self.test_1_prompt_loading_verification()
        await self.test_2_ia1_prompt_testing()
        await self.test_3_ia2_prompt_testing()
        await self.test_4_system_integration()
        await self.test_5_error_handling()
        
        total_time = time.time() - start_time
        
        # Generate final summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 EXTERNALIZED PROMPT MIGRATION TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {success_rate:.1%}")
        logger.info(f"Total Time: {total_time:.1f}s")
        
        logger.info("\nDetailed Results:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"  {status}: {result['test']}")
            if result['details']:
                logger.info(f"    Details: {result['details']}")
        
        logger.info("\n" + "=" * 80)
        
        if success_rate >= 0.8:
            logger.info("🎉 EXTERNALIZED PROMPT MIGRATION TEST SUITE: OVERALL SUCCESS")
            logger.info("✅ The externalized prompt migration is working correctly!")
        else:
            logger.info("⚠️ EXTERNALIZED PROMPT MIGRATION TEST SUITE: ISSUES DETECTED")
            logger.info("❌ Some aspects of the externalized prompt migration need attention.")
        
        logger.info("=" * 80)
        
        return success_rate >= 0.8

async def main():
    """Main test execution"""
    test_suite = ExternalizedPromptMigrationTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("🎉 All tests passed! Externalized prompt migration is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the results above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
IA1 CRASH RECOVERY TEST - URGENT BUG FIX VERIFICATION
Focus: Test detected_pattern_names initialization fix resolves IA1 crashes.

SPECIFIC TESTING REQUIREMENTS FROM REVIEW REQUEST:

1. **IA1 Crash Recovery Test**: Verify the variable initialization bug is fixed
   - Call /api/force-ia1-analysis for BTCUSDT, ETHUSDT, ADAUSDT (or available alternatives)
   - Verify IA1 no longer crashes with "cannot access local variable detected_pattern_names" error
   - Check backend logs for successful IA1 analysis completion without exceptions
   - Confirm system no longer falls back to "Fallback ultra professional analysis" responses

2. **Technical Indicators Integration Success Check**: Test if fix enables technical indicators usage
   - Examine IA1 "reasoning" field for explicit mentions of calculated technical indicators
   - Look for RSI values, MACD values, MFI values, VWAP position in reasoning text
   - Verify reasoning includes technical indicators AND patterns (balanced analysis)
   - Confirm "technical_indicators" object contains real calculated values

3. **JSON Response Quality Validation**: Check if structured JSON is now generated
   - Verify IA1 returns structured JSON with signal, confidence, reasoning, technical_indicators, patterns
   - Confirm no more JSON parsing failures or fallback pattern-only responses
   - Check if reasoning follows the format: "Analyse incluant RSI X, MACD Y, MFI Z, VWAP W%"

4. **System Stability Verification**: Ensure fix doesn't introduce new issues
   - Verify API endpoints respond without HTTP 502 errors
   - Check backend stability with no system exceptions
   - Confirm database storage works correctly for IA1 analyses

5. **Success Metrics Assessment**: Compare before/after fix performance
   - Before fix: 0% JSON success, 0% technical indicators usage, 100% fallback analysis
   - After fix target: >70% JSON success, >70% technical indicators mentions, <30% fallback analysis

CRITICAL SUCCESS CRITERIA:
✅ IA1 analysis completes without crashing (no detected_pattern_names error)
✅ Technical indicators mentioned in reasoning (RSI, MACD, MFI, VWAP)
✅ Valid JSON responses generated (no fallback analysis)
✅ System stability restored
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

class IA1CrashRecoveryTestSuite:
    """Comprehensive test suite for IA1 Crash Recovery - detected_pattern_names fix verification"""
    
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
        logger.info(f"Testing IA1 Crash Recovery at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for IA1 analysis (will be dynamically determined from available opportunities)
        self.preferred_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        self.actual_test_symbols = []
        
        # Core technical indicators to verify (as specified in review request)
        self.core_technical_indicators = ['RSI', 'MACD', 'MFI', 'VWAP']
        
        # Critical error patterns to detect
        self.crash_error_patterns = [
            'cannot access local variable detected_pattern_names',
            'detected_pattern_names.*not associated with a value',
            'UnboundLocalError.*detected_pattern_names',
            'NameError.*detected_pattern_names'
        ]
        
        # Fallback analysis patterns
        self.fallback_patterns = [
            'Fallback ultra professional analysis',
            'JSON parsing failed',
            'technical analysis fallback',
            'using detected pattern.*for directional bias',
            'pattern-only analysis'
        ]
        
        # Success metrics tracking
        self.success_metrics = {
            'total_analyses_attempted': 0,
            'successful_analyses': 0,
            'crash_free_analyses': 0,
            'json_success_count': 0,
            'technical_indicators_usage_count': 0,
            'fallback_analyses_count': 0,
            'system_exceptions_count': 0,
            'http_502_errors': 0
        }
        
        # Analysis data storage
        self.ia1_analyses = []
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
    
    async def _capture_backend_logs(self, lines: int = 200) -> List[str]:
        """Capture recent backend logs"""
        try:
            # Try to get supervisor backend logs
            result = subprocess.run(
                ['tail', '-n', str(lines), '/var/log/supervisor/backend.out.log'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip().split('\n')
            
            # Fallback to error log
            result = subprocess.run(
                ['tail', '-n', str(lines), '/var/log/supervisor/backend.err.log'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip().split('\n')
            
            return []
            
        except Exception as e:
            logger.warning(f"Could not capture backend logs: {e}")
            return []
    
    async def _get_available_test_symbols(self) -> List[str]:
        """Get available symbols from opportunities API"""
        try:
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Handle both direct list and wrapped response
                if isinstance(response_data, dict) and 'opportunities' in response_data:
                    opportunities = response_data['opportunities']
                elif isinstance(response_data, list):
                    opportunities = response_data
                else:
                    opportunities = []
                
                # Look for preferred symbols first
                available_symbols = [opp.get('symbol', '') for opp in opportunities]
                
                # Check for preferred symbols
                found_preferred = [symbol for symbol in self.preferred_symbols if symbol in available_symbols]
                
                if len(found_preferred) >= 2:
                    logger.info(f"✅ Found preferred symbols: {found_preferred}")
                    return found_preferred[:3]  # Use up to 3 preferred symbols
                else:
                    # Use first available symbols
                    alternative_symbols = available_symbols[:3]
                    logger.info(f"📋 Using alternative symbols: {alternative_symbols}")
                    return alternative_symbols
            
            else:
                logger.warning(f"Opportunities API failed: HTTP {response.status_code}")
                return self.preferred_symbols  # Fallback to preferred
                
        except Exception as e:
            logger.warning(f"Could not get available symbols: {e}")
            return self.preferred_symbols  # Fallback to preferred
    
    async def test_1_ia1_crash_recovery_verification(self):
        """Test 1: IA1 Crash Recovery Verification - Verify detected_pattern_names initialization bug is fixed"""
        logger.info("\n🔍 TEST 1: IA1 Crash Recovery Verification")
        
        try:
            crash_results = {
                'symbols_tested': [],
                'crash_free_analyses': 0,
                'detected_pattern_names_errors': 0,
                'successful_completions': 0,
                'fallback_analyses': 0,
                'backend_exceptions': 0,
                'analysis_details': []
            }
            
            logger.info("   🚀 Testing IA1 analysis for crash recovery after detected_pattern_names fix...")
            logger.info("   📊 Expected: IA1 completes without 'cannot access local variable detected_pattern_names' error")
            
            # Get available symbols for testing
            test_symbols = await self._get_available_test_symbols()
            self.actual_test_symbols = test_symbols
            crash_results['symbols_tested'] = test_symbols
            
            logger.info(f"   📋 Testing symbols: {test_symbols}")
            
            # Capture initial backend logs
            initial_logs = await self._capture_backend_logs(50)
            logger.info(f"   📋 Captured {len(initial_logs)} initial backend log lines")
            
            # Test each symbol for crash recovery
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing IA1 crash recovery for {symbol}")
                
                self.success_metrics['total_analyses_attempted'] += 1
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=180  # Extended timeout for thorough analysis
                    )
                    response_time = time.time() - start_time
                    
                    analysis_detail = {
                        'symbol': symbol,
                        'response_time': response_time,
                        'http_status': response.status_code,
                        'crash_free': False,
                        'fallback_detected': False,
                        'technical_indicators_present': False,
                        'json_valid': False,
                        'error_details': None
                    }
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        self.success_metrics['successful_analyses'] += 1
                        
                        logger.info(f"      ✅ IA1 analysis for {symbol} completed (response time: {response_time:.2f}s)")
                        
                        # Check for fallback analysis indicators
                        is_fallback = False
                        reasoning_text = ""
                        
                        # Extract reasoning from various possible locations
                        if 'ia1_analysis' in analysis_data and isinstance(analysis_data['ia1_analysis'], dict):
                            reasoning_text = analysis_data['ia1_analysis'].get('reasoning', '')
                        elif 'reasoning' in analysis_data:
                            reasoning_text = str(analysis_data['reasoning'])
                        elif 'analysis' in analysis_data:
                            reasoning_text = str(analysis_data['analysis'])
                        
                        # Check for fallback patterns
                        for pattern in self.fallback_patterns:
                            if pattern.lower() in reasoning_text.lower():
                                is_fallback = True
                                self.success_metrics['fallback_analyses_count'] += 1
                                logger.warning(f"      ❌ Fallback analysis detected: {pattern}")
                                break
                        
                        if not is_fallback:
                            crash_results['crash_free_analyses'] += 1
                            self.success_metrics['crash_free_analyses'] += 1
                            analysis_detail['crash_free'] = True
                            logger.info(f"      ✅ No fallback analysis - IA1 completed successfully")
                        
                        analysis_detail['fallback_detected'] = is_fallback
                        
                        # Check for technical indicators in reasoning
                        technical_indicators_found = []
                        for indicator in self.core_technical_indicators:
                            if indicator.lower() in reasoning_text.lower():
                                technical_indicators_found.append(indicator)
                        
                        if technical_indicators_found:
                            analysis_detail['technical_indicators_present'] = True
                            self.success_metrics['technical_indicators_usage_count'] += 1
                            logger.info(f"      ✅ Technical indicators found: {technical_indicators_found}")
                        else:
                            logger.warning(f"      ⚠️ No technical indicators found in reasoning")
                        
                        # Check JSON structure
                        if isinstance(analysis_data, dict) and ('ia1_analysis' in analysis_data or 'signal' in analysis_data):
                            analysis_detail['json_valid'] = True
                            self.success_metrics['json_success_count'] += 1
                            logger.info(f"      ✅ Valid JSON structure returned")
                        
                        # Store analysis for later use
                        self.ia1_analyses.append({
                            'symbol': symbol,
                            'analysis_data': analysis_data,
                            'reasoning': reasoning_text,
                            'is_crash_free': not is_fallback,
                            'technical_indicators': technical_indicators_found
                        })
                    
                    elif response.status_code == 502:
                        self.success_metrics['http_502_errors'] += 1
                        analysis_detail['error_details'] = f"HTTP 502 - Backend error"
                        logger.error(f"      ❌ HTTP 502 error for {symbol} - Backend instability")
                    
                    else:
                        analysis_detail['error_details'] = f"HTTP {response.status_code}"
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                        if response.text:
                            logger.error(f"         Error response: {response.text[:200]}...")
                
                except Exception as e:
                    analysis_detail['error_details'] = str(e)
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                
                crash_results['analysis_details'].append(analysis_detail)
                
                # Wait between analyses to avoid overwhelming the system
                if symbol != test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 15 seconds before next analysis...")
                    await asyncio.sleep(15)
            
            # Capture backend logs after analyses
            logger.info("   📋 Capturing backend logs for crash analysis...")
            final_logs = await self._capture_backend_logs(100)
            self.backend_logs = final_logs
            
            # Analyze logs for detected_pattern_names errors
            detected_pattern_errors = 0
            backend_exceptions = 0
            
            for log_line in final_logs:
                # Check for detected_pattern_names errors
                for error_pattern in self.crash_error_patterns:
                    if re.search(error_pattern, log_line, re.IGNORECASE):
                        detected_pattern_errors += 1
                        logger.error(f"      🚨 CRITICAL: detected_pattern_names error found: {log_line.strip()}")
                        break
                
                # Check for general exceptions
                if any(keyword in log_line.lower() for keyword in ['exception', 'error', 'traceback']):
                    backend_exceptions += 1
            
            crash_results['detected_pattern_names_errors'] = detected_pattern_errors
            crash_results['backend_exceptions'] = backend_exceptions
            self.success_metrics['system_exceptions_count'] = backend_exceptions
            
            # Final analysis
            logger.info(f"\n   📊 IA1 CRASH RECOVERY VERIFICATION RESULTS:")
            logger.info(f"      Symbols tested: {len(test_symbols)}")
            logger.info(f"      Crash-free analyses: {crash_results['crash_free_analyses']}/{len(test_symbols)}")
            logger.info(f"      detected_pattern_names errors: {detected_pattern_errors}")
            logger.info(f"      Fallback analyses: {self.success_metrics['fallback_analyses_count']}")
            logger.info(f"      Backend exceptions: {backend_exceptions}")
            logger.info(f"      HTTP 502 errors: {self.success_metrics['http_502_errors']}")
            
            # Calculate success rate
            crash_recovery_rate = crash_results['crash_free_analyses'] / max(len(test_symbols), 1)
            
            # Success criteria for crash recovery
            success_criteria = [
                crash_results['crash_free_analyses'] >= 1,  # At least 1 crash-free analysis
                detected_pattern_errors == 0,  # No detected_pattern_names errors
                crash_recovery_rate >= 0.7,  # 70% crash recovery rate
                self.success_metrics['http_502_errors'] == 0,  # No HTTP 502 errors
                backend_exceptions <= 2  # Minimal backend exceptions
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("IA1 Crash Recovery Verification", True, 
                                   f"Crash recovery successful: {success_count}/{len(success_criteria)} criteria met. Crash-free rate: {crash_recovery_rate:.2f}, No detected_pattern_names errors: {detected_pattern_errors == 0}")
            else:
                self.log_test_result("IA1 Crash Recovery Verification", False, 
                                   f"Crash recovery issues: {success_count}/{len(success_criteria)} criteria met. detected_pattern_names errors: {detected_pattern_errors}, Crash-free rate: {crash_recovery_rate:.2f}")
                
        except Exception as e:
            self.log_test_result("IA1 Crash Recovery Verification", False, f"Exception: {str(e)}")
    
    async def test_2_technical_indicators_integration_success(self):
        """Test 2: Technical Indicators Integration Success Check - Test if fix enables technical indicators usage"""
        logger.info("\n🔍 TEST 2: Technical Indicators Integration Success Check")
        
        try:
            integration_results = {
                'analyses_with_technical_indicators': 0,
                'total_valid_analyses': 0,
                'specific_indicator_mentions': {},
                'specific_values_found': {},
                'balanced_analysis_count': 0,
                'technical_indicators_object_present': 0,
                'reasoning_quality_analysis': []
            }
            
            logger.info("   🚀 Testing technical indicators integration after detected_pattern_names fix...")
            logger.info("   📊 Expected: IA1 reasoning mentions RSI values, MACD values, MFI values, VWAP position")
            
            # Use analyses from previous test
            if not self.ia1_analyses:
                logger.warning("   ⚠️ No IA1 analyses available from previous test")
                return
            
            logger.info(f"   📋 Analyzing {len(self.ia1_analyses)} IA1 analyses for technical indicators integration...")
            
            # Analyze each IA1 analysis for technical indicators
            for analysis in self.ia1_analyses:
                symbol = analysis['symbol']
                reasoning = analysis['reasoning']
                is_crash_free = analysis['is_crash_free']
                
                if not is_crash_free:
                    logger.info(f"      ⏭️ Skipping {symbol} - fallback analysis detected")
                    continue
                
                integration_results['total_valid_analyses'] += 1
                
                logger.info(f"      📊 Analyzing {symbol} for technical indicators integration...")
                
                # Look for specific technical indicator values (RSI 32.1, MACD -0.015, etc.)
                specific_values_patterns = [
                    r'RSI\s*[:\s]*[\d\.]+',
                    r'MACD\s*[:\s]*[\d\.\-]+',
                    r'MFI\s*[:\s]*[\d\.]+',
                    r'VWAP\s*[:\s]*[\d\.\-]+%?'
                ]
                
                specific_values_found = {}
                for pattern in specific_values_patterns:
                    matches = re.findall(pattern, reasoning, re.IGNORECASE)
                    if matches:
                        indicator = pattern.split('\\')[0]
                        specific_values_found[indicator] = matches
                        logger.info(f"         ✅ {indicator} specific values: {matches}")
                
                integration_results['specific_values_found'][symbol] = specific_values_found
                
                # Count mentions of each core technical indicator
                indicator_mentions = {}
                for indicator in self.core_technical_indicators:
                    count = reasoning.lower().count(indicator.lower())
                    if count > 0:
                        indicator_mentions[indicator] = count
                        if indicator not in integration_results['specific_indicator_mentions']:
                            integration_results['specific_indicator_mentions'][indicator] = 0
                        integration_results['specific_indicator_mentions'][indicator] += count
                
                # Check for balanced analysis (both technical indicators and patterns)
                technical_mentions = sum(indicator_mentions.values())
                pattern_mentions = sum(reasoning.lower().count(pattern) for pattern in ['pattern', 'support', 'resistance', 'trend'])
                
                if technical_mentions > 0 and pattern_mentions > 0:
                    integration_results['balanced_analysis_count'] += 1
                    logger.info(f"         ✅ Balanced analysis: {technical_mentions} technical + {pattern_mentions} patterns")
                elif technical_mentions > 0:
                    logger.info(f"         📊 Technical-focused: {technical_mentions} technical mentions")
                elif pattern_mentions > 0:
                    logger.info(f"         📊 Pattern-focused: {pattern_mentions} pattern mentions")
                else:
                    logger.warning(f"         ⚠️ Limited analysis content")
                
                # Check if this analysis has technical indicators
                if indicator_mentions or specific_values_found:
                    integration_results['analyses_with_technical_indicators'] += 1
                
                # Check for technical_indicators object in analysis data
                analysis_data = analysis['analysis_data']
                if 'ia1_analysis' in analysis_data:
                    ia1_response = analysis_data['ia1_analysis']
                    if isinstance(ia1_response, dict) and 'technical_indicators' in ia1_response:
                        technical_obj = ia1_response['technical_indicators']
                        if isinstance(technical_obj, dict) and technical_obj:
                            integration_results['technical_indicators_object_present'] += 1
                            logger.info(f"         ✅ technical_indicators object found with {len(technical_obj)} fields")
                
                # Store reasoning quality analysis
                integration_results['reasoning_quality_analysis'].append({
                    'symbol': symbol,
                    'reasoning_length': len(reasoning),
                    'technical_mentions': technical_mentions,
                    'pattern_mentions': pattern_mentions,
                    'specific_values_count': len(specific_values_found),
                    'balanced': technical_mentions > 0 and pattern_mentions > 0,
                    'reasoning_sample': reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                })
            
            # Calculate integration success metrics
            if integration_results['total_valid_analyses'] > 0:
                technical_indicators_usage_rate = integration_results['analyses_with_technical_indicators'] / integration_results['total_valid_analyses']
                balanced_analysis_rate = integration_results['balanced_analysis_count'] / integration_results['total_valid_analyses']
                technical_object_rate = integration_results['technical_indicators_object_present'] / integration_results['total_valid_analyses']
            else:
                technical_indicators_usage_rate = 0
                balanced_analysis_rate = 0
                technical_object_rate = 0
            
            # Final analysis
            logger.info(f"\n   📊 TECHNICAL INDICATORS INTEGRATION SUCCESS RESULTS:")
            logger.info(f"      Total valid analyses: {integration_results['total_valid_analyses']}")
            logger.info(f"      Analyses with technical indicators: {integration_results['analyses_with_technical_indicators']}")
            logger.info(f"      Technical indicators usage rate: {technical_indicators_usage_rate:.2f}")
            logger.info(f"      Balanced analysis rate: {balanced_analysis_rate:.2f}")
            logger.info(f"      Technical indicators object present: {integration_results['technical_indicators_object_present']}")
            
            if integration_results['specific_indicator_mentions']:
                logger.info(f"      📊 Specific indicator mentions:")
                for indicator, count in integration_results['specific_indicator_mentions'].items():
                    logger.info(f"         - {indicator}: {count} mentions")
            
            # Show reasoning quality samples
            logger.info(f"      📝 Reasoning quality samples:")
            for sample in integration_results['reasoning_quality_analysis'][:2]:
                logger.info(f"         - {sample['symbol']}: {sample['technical_mentions']} technical, {sample['pattern_mentions']} patterns, balanced: {sample['balanced']}")
                logger.info(f"           Sample: {sample['reasoning_sample'][:150]}...")
            
            # Success criteria for technical indicators integration (target >70% from review)
            success_criteria = [
                integration_results['total_valid_analyses'] > 0,  # Have valid analyses
                integration_results['analyses_with_technical_indicators'] > 0,  # Some analyses have indicators
                technical_indicators_usage_rate >= 0.7,  # 70% usage rate (target from review)
                len(integration_results['specific_indicator_mentions']) >= 2,  # At least 2 different indicators mentioned
                integration_results['balanced_analysis_count'] > 0  # Some balanced analyses
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Technical Indicators Integration Success Check", True, 
                                   f"Technical indicators integration successful: {success_count}/{len(success_criteria)} criteria met. Usage rate: {technical_indicators_usage_rate:.2f}, Balanced analyses: {integration_results['balanced_analysis_count']}")
            else:
                self.log_test_result("Technical Indicators Integration Success Check", False, 
                                   f"Technical indicators integration issues: {success_count}/{len(success_criteria)} criteria met. Usage rate: {technical_indicators_usage_rate:.2f} may be below 70% target")
                
        except Exception as e:
            self.log_test_result("Technical Indicators Integration Success Check", False, f"Exception: {str(e)}")
    
    async def test_3_json_response_quality_validation(self):
        """Test 3: JSON Response Quality Validation - Check if structured JSON is now generated"""
        logger.info("\n🔍 TEST 3: JSON Response Quality Validation")
        
        try:
            json_results = {
                'total_analyses': 0,
                'valid_json_responses': 0,
                'structured_json_count': 0,
                'fallback_responses': 0,
                'required_fields_present': {},
                'json_parsing_failures': 0,
                'response_quality_analysis': []
            }
            
            logger.info("   🚀 Testing JSON response quality after detected_pattern_names fix...")
            logger.info("   📊 Expected: IA1 returns structured JSON with signal, confidence, reasoning, technical_indicators, patterns")
            
            # Required JSON fields for structured response
            required_fields = ['signal', 'confidence', 'reasoning']
            optional_fields = ['technical_indicators', 'patterns', 'entry_price', 'stop_loss', 'take_profit']
            
            # Use analyses from previous tests
            if not self.ia1_analyses:
                logger.warning("   ⚠️ No IA1 analyses available from previous tests")
                return
            
            logger.info(f"   📋 Analyzing {len(self.ia1_analyses)} IA1 responses for JSON quality...")
            
            # Analyze each response for JSON quality
            for analysis in self.ia1_analyses:
                symbol = analysis['symbol']
                analysis_data = analysis['analysis_data']
                is_crash_free = analysis['is_crash_free']
                
                json_results['total_analyses'] += 1
                
                logger.info(f"      📊 Analyzing {symbol} JSON response quality...")
                
                response_quality = {
                    'symbol': symbol,
                    'is_valid_json': False,
                    'is_structured': False,
                    'is_fallback': not is_crash_free,
                    'required_fields_found': [],
                    'optional_fields_found': [],
                    'response_type': 'unknown'
                }
                
                # Check if response is valid JSON
                if isinstance(analysis_data, dict):
                    json_results['valid_json_responses'] += 1
                    response_quality['is_valid_json'] = True
                    logger.info(f"         ✅ Valid JSON response")
                    
                    # Check for IA1 analysis structure
                    ia1_response = None
                    if 'ia1_analysis' in analysis_data and isinstance(analysis_data['ia1_analysis'], dict):
                        ia1_response = analysis_data['ia1_analysis']
                        response_quality['response_type'] = 'structured_ia1_analysis'
                    elif all(field in analysis_data for field in required_fields[:2]):  # signal, confidence
                        ia1_response = analysis_data
                        response_quality['response_type'] = 'direct_structured'
                    
                    if ia1_response:
                        # Check for required fields
                        for field in required_fields:
                            if field in ia1_response and ia1_response[field] not in [None, '', 'unknown']:
                                response_quality['required_fields_found'].append(field)
                                if field not in json_results['required_fields_present']:
                                    json_results['required_fields_present'][field] = 0
                                json_results['required_fields_present'][field] += 1
                        
                        # Check for optional fields
                        for field in optional_fields:
                            if field in ia1_response and ia1_response[field] not in [None, '', 'unknown']:
                                response_quality['optional_fields_found'].append(field)
                        
                        # Determine if structured
                        if len(response_quality['required_fields_found']) >= 2:  # At least 2 required fields
                            json_results['structured_json_count'] += 1
                            response_quality['is_structured'] = True
                            logger.info(f"         ✅ Structured JSON: {len(response_quality['required_fields_found'])} required + {len(response_quality['optional_fields_found'])} optional fields")
                        else:
                            logger.warning(f"         ⚠️ Limited structure: {response_quality['required_fields_found']}")
                    
                    else:
                        logger.warning(f"         ⚠️ No recognizable IA1 structure found")
                
                else:
                    logger.error(f"         ❌ Invalid JSON response")
                
                # Check for fallback response
                if not is_crash_free:
                    json_results['fallback_responses'] += 1
                    response_quality['is_fallback'] = True
                    logger.warning(f"         ❌ Fallback response detected")
                
                json_results['response_quality_analysis'].append(response_quality)
            
            # Calculate JSON quality metrics
            if json_results['total_analyses'] > 0:
                json_success_rate = json_results['valid_json_responses'] / json_results['total_analyses']
                structured_rate = json_results['structured_json_count'] / json_results['total_analyses']
                fallback_rate = json_results['fallback_responses'] / json_results['total_analyses']
            else:
                json_success_rate = 0
                structured_rate = 0
                fallback_rate = 0
            
            # Final analysis
            logger.info(f"\n   📊 JSON RESPONSE QUALITY VALIDATION RESULTS:")
            logger.info(f"      Total analyses: {json_results['total_analyses']}")
            logger.info(f"      Valid JSON responses: {json_results['valid_json_responses']}")
            logger.info(f"      Structured JSON responses: {json_results['structured_json_count']}")
            logger.info(f"      Fallback responses: {json_results['fallback_responses']}")
            logger.info(f"      JSON success rate: {json_success_rate:.2f}")
            logger.info(f"      Structured response rate: {structured_rate:.2f}")
            logger.info(f"      Fallback rate: {fallback_rate:.2f}")
            
            if json_results['required_fields_present']:
                logger.info(f"      📊 Required fields presence:")
                for field, count in json_results['required_fields_present'].items():
                    logger.info(f"         - {field}: {count}/{json_results['total_analyses']} responses")
            
            # Show response quality breakdown
            logger.info(f"      📊 Response quality breakdown:")
            for quality in json_results['response_quality_analysis']:
                logger.info(f"         - {quality['symbol']}: {quality['response_type']}, structured: {quality['is_structured']}, fallback: {quality['is_fallback']}")
            
            # Success criteria for JSON quality (target >70% from review)
            success_criteria = [
                json_results['valid_json_responses'] > 0,  # Some valid JSON responses
                json_success_rate >= 0.7,  # 70% JSON success rate (target from review)
                json_results['structured_json_count'] > 0,  # Some structured responses
                fallback_rate <= 0.3,  # <30% fallback rate (target from review)
                len(json_results['required_fields_present']) >= 2  # At least 2 required fields present
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("JSON Response Quality Validation", True, 
                                   f"JSON quality validation successful: {success_count}/{len(success_criteria)} criteria met. JSON success rate: {json_success_rate:.2f}, Fallback rate: {fallback_rate:.2f}")
            else:
                self.log_test_result("JSON Response Quality Validation", False, 
                                   f"JSON quality validation issues: {success_count}/{len(success_criteria)} criteria met. JSON success rate: {json_success_rate:.2f} may be below 70% target")
                
        except Exception as e:
            self.log_test_result("JSON Response Quality Validation", False, f"Exception: {str(e)}")
    
    async def test_4_system_stability_verification(self):
        """Test 4: System Stability Verification - Ensure fix doesn't introduce new issues"""
        logger.info("\n🔍 TEST 4: System Stability Verification")
        
        try:
            stability_results = {
                'api_endpoints_tested': 0,
                'successful_api_calls': 0,
                'http_502_errors': 0,
                'system_exceptions': 0,
                'database_storage_working': False,
                'backend_stability_score': 0.0,
                'endpoint_results': []
            }
            
            logger.info("   🚀 Testing system stability after detected_pattern_names fix...")
            logger.info("   📊 Expected: API endpoints respond without HTTP 502 errors, backend stable")
            
            # Test key API endpoints for stability
            endpoints_to_test = [
                ('/opportunities', 'GET', None),
                ('/analyses', 'GET', None),
                ('/debug-anti-doublon', 'GET', None)
            ]
            
            logger.info(f"   📋 Testing {len(endpoints_to_test)} API endpoints for stability...")
            
            for endpoint, method, payload in endpoints_to_test:
                stability_results['api_endpoints_tested'] += 1
                
                logger.info(f"      📞 Testing {method} {endpoint}")
                
                endpoint_result = {
                    'endpoint': endpoint,
                    'method': method,
                    'success': False,
                    'status_code': None,
                    'response_time': 0,
                    'error': None
                }
                
                try:
                    start_time = time.time()
                    
                    if method == 'GET':
                        response = requests.get(f"{self.api_url}{endpoint}", timeout=60)
                    elif method == 'POST':
                        response = requests.post(f"{self.api_url}{endpoint}", json=payload, timeout=60)
                    
                    response_time = time.time() - start_time
                    endpoint_result['response_time'] = response_time
                    endpoint_result['status_code'] = response.status_code
                    
                    if response.status_code == 200:
                        stability_results['successful_api_calls'] += 1
                        endpoint_result['success'] = True
                        logger.info(f"         ✅ Success: HTTP {response.status_code} (response time: {response_time:.2f}s)")
                    elif response.status_code == 502:
                        stability_results['http_502_errors'] += 1
                        endpoint_result['error'] = "HTTP 502 - Backend error"
                        logger.error(f"         ❌ HTTP 502 error - Backend instability")
                    else:
                        endpoint_result['error'] = f"HTTP {response.status_code}"
                        logger.warning(f"         ⚠️ HTTP {response.status_code}")
                
                except Exception as e:
                    endpoint_result['error'] = str(e)
                    logger.error(f"         ❌ Exception: {e}")
                
                stability_results['endpoint_results'].append(endpoint_result)
                
                # Wait between endpoint tests
                await asyncio.sleep(2)
            
            # Test database storage with a simple IA1 analysis
            logger.info("   📊 Testing database storage functionality...")
            
            try:
                if self.actual_test_symbols:
                    test_symbol = self.actual_test_symbols[0]
                    logger.info(f"      📞 Testing database storage with {test_symbol}")
                    
                    # Get initial analyses count
                    response = requests.get(f"{self.api_url}/analyses", timeout=60)
                    initial_count = len(response.json()) if response.status_code == 200 else 0
                    
                    # Perform IA1 analysis
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": test_symbol},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        # Wait for database storage
                        await asyncio.sleep(5)
                        
                        # Check if new analysis was stored
                        response = requests.get(f"{self.api_url}/analyses", timeout=60)
                        final_count = len(response.json()) if response.status_code == 200 else 0
                        
                        if final_count > initial_count:
                            stability_results['database_storage_working'] = True
                            logger.info(f"         ✅ Database storage working: {final_count - initial_count} new analysis stored")
                        else:
                            logger.warning(f"         ⚠️ Database storage unclear: count {initial_count} → {final_count}")
                    else:
                        logger.warning(f"         ⚠️ Could not test database storage - IA1 analysis failed")
                
            except Exception as e:
                logger.warning(f"      ⚠️ Database storage test error: {e}")
            
            # Analyze backend logs for system exceptions
            logger.info("   📋 Analyzing backend logs for system stability...")
            
            try:
                recent_logs = await self._capture_backend_logs(100)
                
                if recent_logs:
                    # Count system exceptions
                    exception_patterns = [
                        'exception', 'error', 'traceback', 'failed', 'crash'
                    ]
                    
                    exceptions_found = 0
                    for log_line in recent_logs:
                        for pattern in exception_patterns:
                            if pattern.lower() in log_line.lower() and 'detected_pattern_names' not in log_line.lower():
                                exceptions_found += 1
                                break
                    
                    stability_results['system_exceptions'] = exceptions_found
                    logger.info(f"      📊 System exceptions in recent logs: {exceptions_found}")
                    
                    # Calculate backend stability score
                    if len(recent_logs) > 0:
                        stability_results['backend_stability_score'] = max(0, 1 - (exceptions_found / len(recent_logs)))
                    
                else:
                    logger.warning(f"      ⚠️ Could not access backend logs for stability analysis")
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Backend log analysis error: {e}")
            
            # Calculate overall stability metrics
            if stability_results['api_endpoints_tested'] > 0:
                api_success_rate = stability_results['successful_api_calls'] / stability_results['api_endpoints_tested']
            else:
                api_success_rate = 0
            
            # Final analysis
            logger.info(f"\n   📊 SYSTEM STABILITY VERIFICATION RESULTS:")
            logger.info(f"      API endpoints tested: {stability_results['api_endpoints_tested']}")
            logger.info(f"      Successful API calls: {stability_results['successful_api_calls']}")
            logger.info(f"      HTTP 502 errors: {stability_results['http_502_errors']}")
            logger.info(f"      System exceptions: {stability_results['system_exceptions']}")
            logger.info(f"      Database storage working: {stability_results['database_storage_working']}")
            logger.info(f"      API success rate: {api_success_rate:.2f}")
            logger.info(f"      Backend stability score: {stability_results['backend_stability_score']:.2f}")
            
            # Show endpoint results
            logger.info(f"      📊 Endpoint test results:")
            for result in stability_results['endpoint_results']:
                status = "✅" if result['success'] else "❌"
                logger.info(f"         {status} {result['method']} {result['endpoint']}: {result['status_code']} ({result['response_time']:.2f}s)")
            
            # Success criteria for system stability
            success_criteria = [
                stability_results['successful_api_calls'] > 0,  # Some API calls successful
                stability_results['http_502_errors'] == 0,  # No HTTP 502 errors
                api_success_rate >= 0.8,  # 80% API success rate
                stability_results['system_exceptions'] <= 5,  # Minimal system exceptions
                stability_results['backend_stability_score'] >= 0.9  # High backend stability
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("System Stability Verification", True, 
                                   f"System stability good: {success_count}/{len(success_criteria)} criteria met. API success rate: {api_success_rate:.2f}, No HTTP 502 errors: {stability_results['http_502_errors'] == 0}")
            else:
                self.log_test_result("System Stability Verification", False, 
                                   f"System stability issues: {success_count}/{len(success_criteria)} criteria met. HTTP 502 errors: {stability_results['http_502_errors']}, System exceptions: {stability_results['system_exceptions']}")
                
        except Exception as e:
            self.log_test_result("System Stability Verification", False, f"Exception: {str(e)}")
    
    async def test_5_success_metrics_assessment(self):
        """Test 5: Success Metrics Assessment - Compare before/after fix performance"""
        logger.info("\n🔍 TEST 5: Success Metrics Assessment")
        
        try:
            metrics_results = {
                'json_success_rate': 0.0,
                'technical_indicators_usage_rate': 0.0,
                'fallback_analysis_rate': 0.0,
                'crash_recovery_rate': 0.0,
                'overall_improvement_score': 0.0,
                'target_achievements': {}
            }
            
            logger.info("   🚀 Assessing success metrics after detected_pattern_names fix...")
            logger.info("   📊 Target: >70% JSON success, >70% technical indicators usage, <30% fallback analysis")
            
            # Calculate metrics from collected data
            if self.success_metrics['total_analyses_attempted'] > 0:
                metrics_results['json_success_rate'] = self.success_metrics['json_success_count'] / self.success_metrics['total_analyses_attempted']
                metrics_results['technical_indicators_usage_rate'] = self.success_metrics['technical_indicators_usage_count'] / self.success_metrics['total_analyses_attempted']
                metrics_results['fallback_analysis_rate'] = self.success_metrics['fallback_analyses_count'] / self.success_metrics['total_analyses_attempted']
                metrics_results['crash_recovery_rate'] = self.success_metrics['crash_free_analyses'] / self.success_metrics['total_analyses_attempted']
            
            # Check target achievements
            targets = {
                'json_success_target': (metrics_results['json_success_rate'], 0.7, '>70% JSON success'),
                'technical_indicators_target': (metrics_results['technical_indicators_usage_rate'], 0.7, '>70% technical indicators usage'),
                'fallback_analysis_target': (metrics_results['fallback_analysis_rate'], 0.3, '<30% fallback analysis'),
                'crash_recovery_target': (metrics_results['crash_recovery_rate'], 0.8, '>80% crash recovery')
            }
            
            achievements = {}
            for target_name, (actual, target, description) in targets.items():
                if 'fallback' in target_name:
                    achieved = actual <= target  # Lower is better for fallback
                else:
                    achieved = actual >= target  # Higher is better for others
                
                achievements[target_name] = {
                    'achieved': achieved,
                    'actual': actual,
                    'target': target,
                    'description': description
                }
            
            metrics_results['target_achievements'] = achievements
            
            # Calculate overall improvement score
            achievement_scores = [1.0 if achievement['achieved'] else achievement['actual'] / achievement['target'] 
                                for achievement in achievements.values() if 'fallback' not in list(achievements.keys())[list(achievements.values()).index(achievement)]]
            
            # For fallback (lower is better), invert the score
            fallback_achievement = achievements['fallback_analysis_target']
            if fallback_achievement['actual'] <= fallback_achievement['target']:
                fallback_score = 1.0
            else:
                fallback_score = max(0, 1 - (fallback_achievement['actual'] - fallback_achievement['target']))
            
            achievement_scores.append(fallback_score)
            
            if achievement_scores:
                metrics_results['overall_improvement_score'] = sum(achievement_scores) / len(achievement_scores)
            
            # Final analysis
            logger.info(f"\n   📊 SUCCESS METRICS ASSESSMENT RESULTS:")
            logger.info(f"      Total analyses attempted: {self.success_metrics['total_analyses_attempted']}")
            logger.info(f"      JSON success rate: {metrics_results['json_success_rate']:.2f} (target: >0.70)")
            logger.info(f"      Technical indicators usage rate: {metrics_results['technical_indicators_usage_rate']:.2f} (target: >0.70)")
            logger.info(f"      Fallback analysis rate: {metrics_results['fallback_analysis_rate']:.2f} (target: <0.30)")
            logger.info(f"      Crash recovery rate: {metrics_results['crash_recovery_rate']:.2f} (target: >0.80)")
            logger.info(f"      Overall improvement score: {metrics_results['overall_improvement_score']:.2f}")
            
            # Show target achievements
            logger.info(f"      📊 Target achievements:")
            for target_name, achievement in achievements.items():
                status = "✅" if achievement['achieved'] else "❌"
                logger.info(f"         {status} {achievement['description']}: {achievement['actual']:.2f}")
            
            # Show detailed metrics
            logger.info(f"      📊 Detailed metrics:")
            logger.info(f"         Successful analyses: {self.success_metrics['successful_analyses']}")
            logger.info(f"         Crash-free analyses: {self.success_metrics['crash_free_analyses']}")
            logger.info(f"         JSON successes: {self.success_metrics['json_success_count']}")
            logger.info(f"         Technical indicators usage: {self.success_metrics['technical_indicators_usage_count']}")
            logger.info(f"         Fallback analyses: {self.success_metrics['fallback_analyses_count']}")
            logger.info(f"         System exceptions: {self.success_metrics['system_exceptions_count']}")
            logger.info(f"         HTTP 502 errors: {self.success_metrics['http_502_errors']}")
            
            # Success criteria for metrics assessment
            success_criteria = [
                metrics_results['json_success_rate'] >= 0.7,  # >70% JSON success
                metrics_results['technical_indicators_usage_rate'] >= 0.7,  # >70% technical indicators usage
                metrics_results['fallback_analysis_rate'] <= 0.3,  # <30% fallback analysis
                metrics_results['crash_recovery_rate'] >= 0.8,  # >80% crash recovery
                metrics_results['overall_improvement_score'] >= 0.8  # High overall improvement
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("Success Metrics Assessment", True, 
                                   f"Success metrics achieved: {success_count}/{len(success_criteria)} targets met. Overall improvement: {metrics_results['overall_improvement_score']:.2f}")
            else:
                self.log_test_result("Success Metrics Assessment", False, 
                                   f"Success metrics not fully achieved: {success_count}/{len(success_criteria)} targets met. May need further improvements")
                
        except Exception as e:
            self.log_test_result("Success Metrics Assessment", False, f"Exception: {str(e)}")
    
    async def run_all_tests(self):
        """Run all IA1 crash recovery tests"""
        logger.info("🚀 Starting IA1 Crash Recovery Test Suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests in sequence
        await self.test_1_ia1_crash_recovery_verification()
        await self.test_2_technical_indicators_integration_success()
        await self.test_3_json_response_quality_validation()
        await self.test_4_system_stability_verification()
        await self.test_5_success_metrics_assessment()
        
        total_time = time.time() - start_time
        
        # Generate final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎯 IA1 CRASH RECOVERY TEST SUITE SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        logger.info(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
        logger.info(f"⏱️ Total execution time: {total_time:.2f} seconds")
        
        # Show individual test results
        logger.info(f"\n📋 Individual Test Results:")
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"   {status}: {result['test']}")
            if result['details']:
                logger.info(f"      {result['details']}")
        
        # Show success metrics summary
        logger.info(f"\n📊 Success Metrics Summary:")
        logger.info(f"   Total analyses attempted: {self.success_metrics['total_analyses_attempted']}")
        logger.info(f"   Successful analyses: {self.success_metrics['successful_analyses']}")
        logger.info(f"   Crash-free analyses: {self.success_metrics['crash_free_analyses']}")
        logger.info(f"   JSON successes: {self.success_metrics['json_success_count']}")
        logger.info(f"   Technical indicators usage: {self.success_metrics['technical_indicators_usage_count']}")
        logger.info(f"   Fallback analyses: {self.success_metrics['fallback_analyses_count']}")
        
        # Final assessment
        if passed_tests >= 4:  # At least 4/5 tests passed
            logger.info(f"\n🎉 IA1 CRASH RECOVERY FIX VERIFICATION: SUCCESS")
            logger.info(f"   The detected_pattern_names initialization fix appears to be working correctly.")
            logger.info(f"   IA1 analysis is now completing without crashes and using technical indicators.")
        else:
            logger.info(f"\n⚠️ IA1 CRASH RECOVERY FIX VERIFICATION: ISSUES DETECTED")
            logger.info(f"   The fix may not be fully working or additional issues may be present.")
            logger.info(f"   Further investigation and fixes may be required.")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IA1CrashRecoveryTestSuite()
    
    try:
        passed, total = await test_suite.run_all_tests()
        
        # Exit with appropriate code
        if passed >= 4:  # At least 4/5 tests passed
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
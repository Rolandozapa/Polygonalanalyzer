#!/usr/bin/env python3
"""
IA2 RESPONSE FORMAT AND NEW TECHNICAL LEVELS DEBUG TEST SUITE

SPECIFIC OBJECTIVES FROM REVIEW REQUEST:
1. **Trigger IA2 Execution**: Force IA1→IA2 escalation with high confidence symbol
2. **Capture IA2 Full Response**: Monitor exact JSON response from Claude
3. **Verify New Fields**: Check if Claude generates ia2_entry_price, ia2_stop_loss, ia2_take_profit_1, etc.
4. **Debug JSON Parsing**: Identify why fallback values (Entry: $100.0, SL: $97.0) are used

FOCUS AREAS:
- IA2 prompt format and Claude's understanding
- JSON response completeness
- Field naming mismatches (position_size vs position_size_recommendation)
- Why Claude might be ignoring the new response format

The goal is to understand why IA2 uses fallback technical levels instead of generating its own strategic levels based on its analysis.
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

class IA2ResponseFormatDebugTestSuite:
    """Debug test suite specifically for IA2 response format and new technical levels fields"""
    
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
        logger.info(f"Testing IA2 Response Format Debug at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for IA2 response analysis")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # High confidence symbols likely to trigger IA2
        self.high_confidence_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
        
        # Expected new technical levels fields
        self.expected_ia2_fields = [
            "ia2_entry_price",
            "ia2_stop_loss", 
            "ia2_take_profit_1",
            "ia2_take_profit_2",
            "ia2_take_profit_3",
            "position_size",
            "trade_execution_ready",
            "calculated_rr",
            "rr_reasoning"
        ]
        
        # IA2 response capture
        self.ia2_responses = []
        
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
    
    async def test_1_force_ia2_execution_high_confidence(self):
        """Test 1: Force IA2 Execution with High Confidence Symbols"""
        logger.info("\n🔍 TEST 1: Force IA2 Execution with High Confidence Symbols")
        
        try:
            ia2_executions = []
            
            for symbol in self.high_confidence_symbols:
                try:
                    logger.info(f"   🚀 Forcing IA1→IA2 escalation for {symbol}")
                    
                    # Force IA1 analysis
                    response = requests.post(f"{self.api_url}/force-ia1-analysis", 
                                           json={"symbol": symbol}, 
                                           timeout=120)
                    
                    if response.status_code in [200, 201]:
                        result = response.json()
                        
                        execution_data = {
                            'symbol': symbol,
                            'ia1_success': result.get('success', False),
                            'ia1_confidence': result.get('confidence', 0),
                            'ia1_rr': result.get('risk_reward_ratio', 0),
                            'ia1_signal': result.get('signal', 'unknown'),
                            'ia2_triggered': False,
                            'voie_used': None,
                            'analysis_id': result.get('analysis_id'),
                            'decision_id': result.get('decision_id')
                        }
                        
                        # Check for IA2 escalation indicators
                        if result.get('success', False):
                            confidence = result.get('confidence', 0)
                            rr = result.get('risk_reward_ratio', 0)
                            signal = result.get('signal', 'hold').lower()
                            
                            # Determine VOIE escalation path
                            if confidence >= 0.95 and signal in ['long', 'short']:
                                execution_data['voie_used'] = 'VOIE 3 (≥95% confidence override)'
                                execution_data['ia2_triggered'] = True
                            elif confidence >= 0.70 and signal in ['long', 'short']:
                                execution_data['voie_used'] = 'VOIE 1 (≥70% confidence + LONG/SHORT)'
                                execution_data['ia2_triggered'] = True
                            elif rr >= 2.0:
                                execution_data['voie_used'] = 'VOIE 2 (≥2.0 RR ratio)'
                                execution_data['ia2_triggered'] = True
                            
                            logger.info(f"      ✅ {symbol}: IA1 success (Conf: {confidence:.1%}, RR: {rr:.1f}, Signal: {signal}) - {execution_data['voie_used'] or 'No escalation'}")
                        else:
                            logger.info(f"      ⚠️ {symbol}: IA1 analysis failed")
                        
                        ia2_executions.append(execution_data)
                    else:
                        logger.warning(f"      ❌ {symbol}: HTTP {response.status_code}")
                        
                    # Small delay between requests
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    logger.warning(f"      ❌ {symbol}: Exception - {str(e)}")
            
            # Wait for IA2 processing
            logger.info("   ⏳ Waiting 90 seconds for IA2 processing to complete...")
            await asyncio.sleep(90)
            
            # Analyze results
            successful_ia1 = len([e for e in ia2_executions if e['ia1_success']])
            expected_ia2 = len([e for e in ia2_executions if e['ia2_triggered']])
            
            logger.info(f"   📊 IA2 Execution Analysis:")
            logger.info(f"      Successful IA1 analyses: {successful_ia1}/{len(self.high_confidence_symbols)}")
            logger.info(f"      Expected IA2 escalations: {expected_ia2}")
            
            # VOIE distribution
            voie_distribution = {}
            for execution in ia2_executions:
                voie = execution['voie_used']
                if voie:
                    voie_key = voie.split('(')[0].strip()
                    voie_distribution[voie_key] = voie_distribution.get(voie_key, 0) + 1
            
            logger.info(f"      VOIE distribution: {voie_distribution}")
            
            # Store executions for next test
            self.ia2_executions = ia2_executions
            
            # Determine test result
            if successful_ia1 >= 3 and expected_ia2 > 0:
                self.log_test_result("Force IA2 Execution with High Confidence", True, 
                                   f"IA2 execution triggered: {successful_ia1} IA1 analyses, {expected_ia2} expected escalations")
            elif successful_ia1 >= 2:
                self.log_test_result("Force IA2 Execution with High Confidence", False, 
                                   f"Partial IA2 execution: {successful_ia1} IA1 analyses, {expected_ia2} escalations")
            else:
                self.log_test_result("Force IA2 Execution with High Confidence", False, 
                                   f"IA2 execution failed: {successful_ia1} IA1 analyses, {expected_ia2} escalations")
                
        except Exception as e:
            self.log_test_result("Force IA2 Execution with High Confidence", False, f"Exception: {str(e)}")
    
    async def test_2_capture_ia2_full_response_debug(self):
        """Test 2: Capture IA2 Full Response and Debug JSON Parsing"""
        logger.info("\n🔍 TEST 2: Capture IA2 Full Response and Debug JSON Parsing")
        
        try:
            if self.db is None:
                self.log_test_result("Capture IA2 Full Response Debug", False, 
                                   "MongoDB connection not available")
                return
            
            # Get recent IA2 decisions
            recent_decisions = list(self.db.trading_decisions.find({}).sort("timestamp", -1).limit(10))
            
            logger.info(f"   📊 Analyzing {len(recent_decisions)} recent IA2 decisions for response format")
            
            if len(recent_decisions) == 0:
                self.log_test_result("Capture IA2 Full Response Debug", False, 
                                   "No recent IA2 decisions found for response analysis")
                return
            
            # Analyze IA2 response structure
            response_analysis = {
                'total_decisions': len(recent_decisions),
                'has_new_technical_fields': 0,
                'has_fallback_values': 0,
                'field_coverage': {field: 0 for field in self.expected_ia2_fields},
                'response_completeness': [],
                'json_parsing_issues': [],
                'claude_response_quality': 0
            }
            
            # Check backend logs for IA2 debugging info
            ia2_debug_logs = await self._capture_ia2_debug_logs()
            
            for decision in recent_decisions:
                decision_analysis = {
                    'symbol': decision.get('symbol', 'unknown'),
                    'decision_id': decision.get('id', 'unknown'),
                    'timestamp': decision.get('timestamp', 'unknown'),
                    'fields_present': [],
                    'fields_missing': [],
                    'has_fallback_indicators': False,
                    'response_quality': 'unknown'
                }
                
                # Check for expected new technical fields
                for field in self.expected_ia2_fields:
                    if field in decision:
                        response_analysis['field_coverage'][field] += 1
                        decision_analysis['fields_present'].append(field)
                    else:
                        decision_analysis['fields_missing'].append(field)
                
                # Check for fallback value indicators
                fallback_indicators = [
                    "Entry: $100.0",
                    "SL: $97.0", 
                    "fallback",
                    "default_value",
                    "position_size_recommendation"  # vs position_size
                ]
                
                reasoning = decision.get('reasoning', '').lower()
                for indicator in fallback_indicators:
                    if indicator.lower() in reasoning:
                        decision_analysis['has_fallback_indicators'] = True
                        response_analysis['has_fallback_values'] += 1
                        break
                
                # Check response completeness
                required_fields = ['signal', 'confidence', 'reasoning']
                completeness = sum(1 for field in required_fields if field in decision and decision[field]) / len(required_fields)
                decision_analysis['response_quality'] = 'complete' if completeness >= 0.8 else 'incomplete'
                
                if completeness >= 0.8:
                    response_analysis['claude_response_quality'] += 1
                
                response_analysis['response_completeness'].append(decision_analysis)
            
            # Calculate field coverage percentages
            total = response_analysis['total_decisions']
            field_coverage_avg = sum(response_analysis['field_coverage'].values()) / (len(self.expected_ia2_fields) * total) if total > 0 else 0
            
            # Log detailed analysis
            logger.info(f"   📊 IA2 Response Format Analysis:")
            logger.info(f"      Total decisions analyzed: {total}")
            logger.info(f"      Average field coverage: {field_coverage_avg:.1%}")
            
            logger.info(f"   📊 Expected New Technical Fields Coverage:")
            for field, count in response_analysis['field_coverage'].items():
                coverage = count / total if total > 0 else 0
                status = "✅" if coverage > 0 else "❌"
                logger.info(f"      {status} {field}: {count}/{total} ({coverage:.1%})")
            
            logger.info(f"   📊 Response Quality Analysis:")
            logger.info(f"      Decisions with fallback values: {response_analysis['has_fallback_values']}/{total}")
            logger.info(f"      Complete Claude responses: {response_analysis['claude_response_quality']}/{total}")
            
            # Log IA2 debug information from backend logs
            if ia2_debug_logs['debug_entries'] > 0:
                logger.info(f"   📊 IA2 Debug Log Analysis:")
                logger.info(f"      Debug entries found: {ia2_debug_logs['debug_entries']}")
                logger.info(f"      Full IA2 responses captured: {ia2_debug_logs['full_responses']}")
                logger.info(f"      JSON parsing errors: {ia2_debug_logs['json_errors']}")
                
                if ia2_debug_logs['sample_responses']:
                    logger.info(f"   📊 Sample IA2 Response Patterns:")
                    for i, response in enumerate(ia2_debug_logs['sample_responses'][:3]):
                        logger.info(f"      Response {i+1}: {response[:200]}...")
            
            # Store response analysis for summary
            self.ia2_response_analysis = response_analysis
            
            # Determine test result
            if field_coverage_avg >= 0.5 and response_analysis['has_fallback_values'] < total * 0.3:
                self.log_test_result("Capture IA2 Full Response Debug", True, 
                                   f"IA2 response format working: {field_coverage_avg:.1%} field coverage, {response_analysis['has_fallback_values']} fallback values")
            elif field_coverage_avg >= 0.3:
                self.log_test_result("Capture IA2 Full Response Debug", False, 
                                   f"Partial IA2 response format: {field_coverage_avg:.1%} field coverage, {response_analysis['has_fallback_values']} fallback values")
            else:
                self.log_test_result("Capture IA2 Full Response Debug", False, 
                                   f"IA2 response format issues: {field_coverage_avg:.1%} field coverage, {response_analysis['has_fallback_values']} fallback values")
                
        except Exception as e:
            self.log_test_result("Capture IA2 Full Response Debug", False, f"Exception: {str(e)}")
    
    async def _capture_ia2_debug_logs(self):
        """Capture IA2 debug logs from backend"""
        try:
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            debug_analysis = {
                'debug_entries': 0,
                'full_responses': 0,
                'json_errors': 0,
                'sample_responses': [],
                'parsing_issues': []
            }
            
            # Patterns to look for IA2 debugging
            debug_patterns = [
                r'🔍 DEBUGGING: Full IA2 response',
                r'IA2.*Claude.*response',
                r'ia2_entry_price',
                r'ia2_stop_loss',
                r'ia2_take_profit',
                r'position_size.*recommendation',
                r'trade_execution_ready'
            ]
            
            error_patterns = [
                r'JSON.*parse.*error',
                r'KeyError.*ia2_',
                r'position_size.*not found',
                r'fallback.*values'
            ]
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '2000', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            lines = log_content.split('\n')
                            
                            # Look for debug patterns
                            for line in lines:
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in debug_patterns):
                                    debug_analysis['debug_entries'] += 1
                                    
                                    if '🔍 DEBUGGING: Full IA2 response' in line:
                                        debug_analysis['full_responses'] += 1
                                        if len(debug_analysis['sample_responses']) < 5:
                                            debug_analysis['sample_responses'].append(line.strip())
                                
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
                                    debug_analysis['json_errors'] += 1
                                    if len(debug_analysis['parsing_issues']) < 5:
                                        debug_analysis['parsing_issues'].append(line.strip())
                            
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
            
            return debug_analysis
            
        except Exception as e:
            logger.warning(f"IA2 debug log capture failed: {e}")
            return {
                'debug_entries': 0,
                'full_responses': 0,
                'json_errors': 0,
                'sample_responses': [],
                'parsing_issues': []
            }
    
    async def test_3_verify_new_technical_levels_fields(self):
        """Test 3: Verify New Technical Levels Fields Generation"""
        logger.info("\n🔍 TEST 3: Verify New Technical Levels Fields Generation")
        
        try:
            if self.db is None:
                self.log_test_result("Verify New Technical Levels Fields", False, 
                                   "MongoDB connection not available")
                return
            
            # Get recent IA2 decisions
            recent_decisions = list(self.db.trading_decisions.find({}).sort("timestamp", -1).limit(15))
            
            logger.info(f"   📊 Analyzing {len(recent_decisions)} recent IA2 decisions for new technical levels fields")
            
            if len(recent_decisions) == 0:
                self.log_test_result("Verify New Technical Levels Fields", False, 
                                   "No recent IA2 decisions found")
                return
            
            # Analyze new technical levels fields
            technical_levels_analysis = {
                'total_decisions': len(recent_decisions),
                'has_ia2_entry_price': 0,
                'has_ia2_stop_loss': 0,
                'has_ia2_take_profit_1': 0,
                'has_ia2_take_profit_2': 0,
                'has_ia2_take_profit_3': 0,
                'has_position_size': 0,
                'has_position_size_recommendation': 0,  # Check for naming mismatch
                'has_trade_execution_ready': 0,
                'has_calculated_rr': 0,
                'has_rr_reasoning': 0,
                'field_naming_issues': [],
                'sample_decisions': []
            }
            
            for decision in recent_decisions:
                decision_sample = {
                    'symbol': decision.get('symbol', 'unknown'),
                    'signal': decision.get('signal', 'unknown'),
                    'confidence': decision.get('confidence', 0),
                    'fields_found': [],
                    'potential_naming_issues': []
                }
                
                # Check for exact field names
                if 'ia2_entry_price' in decision:
                    technical_levels_analysis['has_ia2_entry_price'] += 1
                    decision_sample['fields_found'].append('ia2_entry_price')
                
                if 'ia2_stop_loss' in decision:
                    technical_levels_analysis['has_ia2_stop_loss'] += 1
                    decision_sample['fields_found'].append('ia2_stop_loss')
                
                if 'ia2_take_profit_1' in decision:
                    technical_levels_analysis['has_ia2_take_profit_1'] += 1
                    decision_sample['fields_found'].append('ia2_take_profit_1')
                
                if 'ia2_take_profit_2' in decision:
                    technical_levels_analysis['has_ia2_take_profit_2'] += 1
                    decision_sample['fields_found'].append('ia2_take_profit_2')
                
                if 'ia2_take_profit_3' in decision:
                    technical_levels_analysis['has_ia2_take_profit_3'] += 1
                    decision_sample['fields_found'].append('ia2_take_profit_3')
                
                if 'position_size' in decision:
                    technical_levels_analysis['has_position_size'] += 1
                    decision_sample['fields_found'].append('position_size')
                
                if 'position_size_recommendation' in decision:
                    technical_levels_analysis['has_position_size_recommendation'] += 1
                    decision_sample['potential_naming_issues'].append('position_size_recommendation (should be position_size)')
                
                if 'trade_execution_ready' in decision:
                    technical_levels_analysis['has_trade_execution_ready'] += 1
                    decision_sample['fields_found'].append('trade_execution_ready')
                
                if 'calculated_rr' in decision:
                    technical_levels_analysis['has_calculated_rr'] += 1
                    decision_sample['fields_found'].append('calculated_rr')
                
                if 'rr_reasoning' in decision:
                    technical_levels_analysis['has_rr_reasoning'] += 1
                    decision_sample['fields_found'].append('rr_reasoning')
                
                # Check for potential field naming issues
                all_fields = list(decision.keys())
                for field in all_fields:
                    if 'entry' in field.lower() and field != 'ia2_entry_price':
                        decision_sample['potential_naming_issues'].append(f'{field} (should be ia2_entry_price?)')
                    if 'stop' in field.lower() and field != 'ia2_stop_loss':
                        decision_sample['potential_naming_issues'].append(f'{field} (should be ia2_stop_loss?)')
                    if 'take_profit' in field.lower() and not field.startswith('ia2_take_profit'):
                        decision_sample['potential_naming_issues'].append(f'{field} (should be ia2_take_profit_X?)')
                
                if decision_sample['potential_naming_issues']:
                    technical_levels_analysis['field_naming_issues'].extend(decision_sample['potential_naming_issues'])
                
                technical_levels_analysis['sample_decisions'].append(decision_sample)
            
            # Calculate coverage percentages
            total = technical_levels_analysis['total_decisions']
            
            # Log detailed analysis
            logger.info(f"   📊 New Technical Levels Fields Analysis:")
            logger.info(f"      Total decisions analyzed: {total}")
            
            logger.info(f"   📊 Expected IA2 Technical Fields Coverage:")
            logger.info(f"      ✅ ia2_entry_price: {technical_levels_analysis['has_ia2_entry_price']}/{total} ({technical_levels_analysis['has_ia2_entry_price']/total:.1%} if total > 0 else 0)")
            logger.info(f"      ✅ ia2_stop_loss: {technical_levels_analysis['has_ia2_stop_loss']}/{total} ({technical_levels_analysis['has_ia2_stop_loss']/total:.1%} if total > 0 else 0)")
            logger.info(f"      ✅ ia2_take_profit_1: {technical_levels_analysis['has_ia2_take_profit_1']}/{total} ({technical_levels_analysis['has_ia2_take_profit_1']/total:.1%} if total > 0 else 0)")
            logger.info(f"      ✅ ia2_take_profit_2: {technical_levels_analysis['has_ia2_take_profit_2']}/{total} ({technical_levels_analysis['has_ia2_take_profit_2']/total:.1%} if total > 0 else 0)")
            logger.info(f"      ✅ ia2_take_profit_3: {technical_levels_analysis['has_ia2_take_profit_3']}/{total} ({technical_levels_analysis['has_ia2_take_profit_3']/total:.1%} if total > 0 else 0)")
            logger.info(f"      ✅ position_size: {technical_levels_analysis['has_position_size']}/{total} ({technical_levels_analysis['has_position_size']/total:.1%} if total > 0 else 0)")
            logger.info(f"      ✅ trade_execution_ready: {technical_levels_analysis['has_trade_execution_ready']}/{total} ({technical_levels_analysis['has_trade_execution_ready']/total:.1%} if total > 0 else 0)")
            logger.info(f"      ✅ calculated_rr: {technical_levels_analysis['has_calculated_rr']}/{total} ({technical_levels_analysis['has_calculated_rr']/total:.1%} if total > 0 else 0)")
            logger.info(f"      ✅ rr_reasoning: {technical_levels_analysis['has_rr_reasoning']}/{total} ({technical_levels_analysis['has_rr_reasoning']/total:.1%} if total > 0 else 0)")
            
            # Check for field naming mismatches
            if technical_levels_analysis['has_position_size_recommendation'] > 0:
                logger.info(f"   ⚠️ FIELD NAMING MISMATCH DETECTED:")
                logger.info(f"      position_size_recommendation found: {technical_levels_analysis['has_position_size_recommendation']}/{total}")
                logger.info(f"      This should be 'position_size' instead!")
            
            if technical_levels_analysis['field_naming_issues']:
                logger.info(f"   ⚠️ POTENTIAL FIELD NAMING ISSUES:")
                for issue in set(technical_levels_analysis['field_naming_issues'][:10]):  # Show unique issues, max 10
                    logger.info(f"      {issue}")
            
            # Show sample decisions with fields found
            logger.info(f"   📊 Sample Decisions Analysis:")
            for i, sample in enumerate(technical_levels_analysis['sample_decisions'][:5]):
                logger.info(f"      Decision {i+1} ({sample['symbol']}):")
                logger.info(f"        Signal: {sample['signal']}, Confidence: {sample['confidence']:.1%}")
                logger.info(f"        Fields found: {sample['fields_found'] if sample['fields_found'] else 'None'}")
                if sample['potential_naming_issues']:
                    logger.info(f"        Naming issues: {sample['potential_naming_issues']}")
            
            # Calculate overall coverage
            core_fields_coverage = (
                technical_levels_analysis['has_ia2_entry_price'] +
                technical_levels_analysis['has_ia2_stop_loss'] +
                technical_levels_analysis['has_ia2_take_profit_1'] +
                technical_levels_analysis['has_position_size'] +
                technical_levels_analysis['has_calculated_rr']
            ) / (5 * total) if total > 0 else 0
            
            # Store analysis for summary
            self.technical_levels_analysis = technical_levels_analysis
            
            # Determine test result
            if core_fields_coverage >= 0.7:
                self.log_test_result("Verify New Technical Levels Fields", True, 
                                   f"New technical levels fields working: {core_fields_coverage:.1%} core field coverage")
            elif core_fields_coverage >= 0.3:
                self.log_test_result("Verify New Technical Levels Fields", False, 
                                   f"Partial technical levels fields: {core_fields_coverage:.1%} core field coverage")
            else:
                self.log_test_result("Verify New Technical Levels Fields", False, 
                                   f"New technical levels fields missing: {core_fields_coverage:.1%} core field coverage")
                
        except Exception as e:
            self.log_test_result("Verify New Technical Levels Fields", False, f"Exception: {str(e)}")
    
    async def test_4_debug_json_parsing_fallback_values(self):
        """Test 4: Debug JSON Parsing and Identify Fallback Values Usage"""
        logger.info("\n🔍 TEST 4: Debug JSON Parsing and Identify Fallback Values Usage")
        
        try:
            # Check backend logs for JSON parsing and fallback patterns
            json_debug_analysis = await self._analyze_json_parsing_logs()
            
            # Check database for fallback value patterns
            fallback_analysis = await self._analyze_fallback_value_patterns()
            
            # Log detailed analysis
            logger.info(f"   📊 JSON Parsing Debug Analysis:")
            logger.info(f"      JSON parsing attempts: {json_debug_analysis['parsing_attempts']}")
            logger.info(f"      JSON parsing errors: {json_debug_analysis['parsing_errors']}")
            logger.info(f"      Fallback value triggers: {json_debug_analysis['fallback_triggers']}")
            logger.info(f"      Claude response issues: {json_debug_analysis['claude_issues']}")
            
            if json_debug_analysis['sample_errors']:
                logger.info(f"   📊 Sample JSON Parsing Errors:")
                for i, error in enumerate(json_debug_analysis['sample_errors'][:3]):
                    logger.info(f"      Error {i+1}: {error}")
            
            logger.info(f"   📊 Fallback Values Analysis:")
            logger.info(f"      Decisions with Entry: $100.0: {fallback_analysis['entry_100_count']}")
            logger.info(f"      Decisions with SL: $97.0: {fallback_analysis['sl_97_count']}")
            logger.info(f"      Decisions with fallback indicators: {fallback_analysis['fallback_indicators_count']}")
            logger.info(f"      Total decisions analyzed: {fallback_analysis['total_decisions']}")
            
            if fallback_analysis['sample_fallbacks']:
                logger.info(f"   📊 Sample Fallback Value Patterns:")
                for i, sample in enumerate(fallback_analysis['sample_fallbacks'][:3]):
                    logger.info(f"      Sample {i+1}: {sample}")
            
            # Determine root cause
            root_cause_analysis = {
                'json_parsing_issues': json_debug_analysis['parsing_errors'] > 0,
                'claude_response_incomplete': json_debug_analysis['claude_issues'] > json_debug_analysis['parsing_attempts'] * 0.3,
                'field_naming_mismatch': fallback_analysis['field_naming_issues'] > 0,
                'fallback_system_active': fallback_analysis['fallback_indicators_count'] > 0
            }
            
            logger.info(f"   📊 Root Cause Analysis:")
            for issue, detected in root_cause_analysis.items():
                status = "⚠️ DETECTED" if detected else "✅ OK"
                logger.info(f"      {status} {issue.replace('_', ' ').title()}")
            
            # Store analysis for summary
            self.json_debug_analysis = json_debug_analysis
            self.fallback_analysis = fallback_analysis
            self.root_cause_analysis = root_cause_analysis
            
            # Determine test result
            issues_detected = sum(1 for detected in root_cause_analysis.values() if detected)
            
            if issues_detected == 0:
                self.log_test_result("Debug JSON Parsing and Fallback Values", True, 
                                   "No JSON parsing or fallback value issues detected")
            elif issues_detected <= 2:
                self.log_test_result("Debug JSON Parsing and Fallback Values", False, 
                                   f"Some issues detected: {issues_detected}/4 potential problems")
            else:
                self.log_test_result("Debug JSON Parsing and Fallback Values", False, 
                                   f"Multiple issues detected: {issues_detected}/4 problems found")
                
        except Exception as e:
            self.log_test_result("Debug JSON Parsing and Fallback Values", False, f"Exception: {str(e)}")
    
    async def _analyze_json_parsing_logs(self):
        """Analyze backend logs for JSON parsing issues"""
        try:
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            analysis = {
                'parsing_attempts': 0,
                'parsing_errors': 0,
                'fallback_triggers': 0,
                'claude_issues': 0,
                'sample_errors': []
            }
            
            # Patterns for JSON parsing analysis
            parsing_patterns = [
                r'JSON.*parse',
                r'json\.loads',
                r'IA2.*response.*parse'
            ]
            
            error_patterns = [
                r'JSON.*error',
                r'KeyError.*ia2_',
                r'parsing.*failed',
                r'invalid.*json'
            ]
            
            fallback_patterns = [
                r'fallback.*value',
                r'using.*default',
                r'Entry.*\$100\.0',
                r'SL.*\$97\.0'
            ]
            
            claude_patterns = [
                r'Claude.*error',
                r'anthropic.*error',
                r'IA2.*timeout',
                r'response.*incomplete'
            ]
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '2000', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            lines = log_content.split('\n')
                            
                            for line in lines:
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in parsing_patterns):
                                    analysis['parsing_attempts'] += 1
                                
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
                                    analysis['parsing_errors'] += 1
                                    if len(analysis['sample_errors']) < 5:
                                        analysis['sample_errors'].append(line.strip())
                                
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in fallback_patterns):
                                    analysis['fallback_triggers'] += 1
                                
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in claude_patterns):
                                    analysis['claude_issues'] += 1
                            
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
            
            return analysis
            
        except Exception as e:
            logger.warning(f"JSON parsing log analysis failed: {e}")
            return {
                'parsing_attempts': 0,
                'parsing_errors': 0,
                'fallback_triggers': 0,
                'claude_issues': 0,
                'sample_errors': []
            }
    
    async def _analyze_fallback_value_patterns(self):
        """Analyze database for fallback value patterns"""
        try:
            if self.db is None:
                return {
                    'entry_100_count': 0,
                    'sl_97_count': 0,
                    'fallback_indicators_count': 0,
                    'total_decisions': 0,
                    'field_naming_issues': 0,
                    'sample_fallbacks': []
                }
            
            # Get recent IA2 decisions
            recent_decisions = list(self.db.trading_decisions.find({}).sort("timestamp", -1).limit(20))
            
            analysis = {
                'entry_100_count': 0,
                'sl_97_count': 0,
                'fallback_indicators_count': 0,
                'total_decisions': len(recent_decisions),
                'field_naming_issues': 0,
                'sample_fallbacks': []
            }
            
            fallback_indicators = [
                "entry: $100.0",
                "sl: $97.0",
                "fallback",
                "default_value",
                "position_size_recommendation"
            ]
            
            for decision in recent_decisions:
                reasoning = decision.get('reasoning', '').lower()
                
                # Check for specific fallback values
                if "entry: $100.0" in reasoning or "100.0" in str(decision.get('entry_price', '')):
                    analysis['entry_100_count'] += 1
                
                if "sl: $97.0" in reasoning or "97.0" in str(decision.get('stop_loss', '')):
                    analysis['sl_97_count'] += 1
                
                # Check for fallback indicators
                if any(indicator in reasoning for indicator in fallback_indicators):
                    analysis['fallback_indicators_count'] += 1
                    if len(analysis['sample_fallbacks']) < 5:
                        analysis['sample_fallbacks'].append(f"{decision.get('symbol', 'unknown')}: {reasoning[:100]}...")
                
                # Check for field naming issues
                if 'position_size_recommendation' in decision:
                    analysis['field_naming_issues'] += 1
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Fallback value analysis failed: {e}")
            return {
                'entry_100_count': 0,
                'sl_97_count': 0,
                'fallback_indicators_count': 0,
                'total_decisions': 0,
                'field_naming_issues': 0,
                'sample_fallbacks': []
            }
    
    async def run_ia2_debug_comprehensive_test(self):
        """Run comprehensive IA2 response format and technical levels debug test"""
        logger.info("🚀 Starting IA2 Response Format and New Technical Levels Debug Test")
        logger.info("=" * 80)
        logger.info("📋 IA2 RESPONSE FORMAT AND NEW TECHNICAL LEVELS DEBUG TEST SUITE")
        logger.info("🎯 Objective: Debug IA2 response format and new technical levels fields")
        logger.info("🎯 Focus: Identify why fallback values (Entry: $100.0, SL: $97.0) are used")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_force_ia2_execution_high_confidence()
        await self.test_2_capture_ia2_full_response_debug()
        await self.test_3_verify_new_technical_levels_fields()
        await self.test_4_debug_json_parsing_fallback_values()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA2 RESPONSE FORMAT DEBUG COMPREHENSIVE TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Critical findings summary
        logger.info("\n" + "=" * 80)
        logger.info("📋 CRITICAL FINDINGS SUMMARY")
        logger.info("=" * 80)
        
        # IA2 Execution Status
        if hasattr(self, 'ia2_executions'):
            successful_ia1 = len([e for e in self.ia2_executions if e['ia1_success']])
            expected_ia2 = len([e for e in self.ia2_executions if e['ia2_triggered']])
            logger.info(f"🚀 IA2 EXECUTION STATUS:")
            logger.info(f"   Successful IA1 analyses: {successful_ia1}/5")
            logger.info(f"   Expected IA2 escalations: {expected_ia2}")
        
        # Response Format Status
        if hasattr(self, 'ia2_response_analysis'):
            field_coverage = sum(self.ia2_response_analysis['field_coverage'].values()) / (len(self.expected_ia2_fields) * self.ia2_response_analysis['total_decisions']) if self.ia2_response_analysis['total_decisions'] > 0 else 0
            logger.info(f"📊 RESPONSE FORMAT STATUS:")
            logger.info(f"   Average field coverage: {field_coverage:.1%}")
            logger.info(f"   Decisions with fallback values: {self.ia2_response_analysis['has_fallback_values']}")
        
        # Technical Levels Status
        if hasattr(self, 'technical_levels_analysis'):
            core_coverage = (
                self.technical_levels_analysis['has_ia2_entry_price'] +
                self.technical_levels_analysis['has_ia2_stop_loss'] +
                self.technical_levels_analysis['has_ia2_take_profit_1'] +
                self.technical_levels_analysis['has_position_size'] +
                self.technical_levels_analysis['has_calculated_rr']
            ) / (5 * self.technical_levels_analysis['total_decisions']) if self.technical_levels_analysis['total_decisions'] > 0 else 0
            
            logger.info(f"🎯 TECHNICAL LEVELS STATUS:")
            logger.info(f"   Core technical fields coverage: {core_coverage:.1%}")
            logger.info(f"   Field naming issues detected: {len(self.technical_levels_analysis['field_naming_issues'])}")
        
        # Root Cause Analysis
        if hasattr(self, 'root_cause_analysis'):
            logger.info(f"🔍 ROOT CAUSE ANALYSIS:")
            for issue, detected in self.root_cause_analysis.items():
                status = "⚠️ ISSUE" if detected else "✅ OK"
                logger.info(f"   {status} {issue.replace('_', ' ').title()}")
        
        # Final verdict
        logger.info(f"\n🏆 DEBUG TEST RESULT: {passed_tests}/{total_tests}")
        
        if passed_tests >= total_tests * 0.75:
            logger.info("\n🎉 VERDICT: IA2 RESPONSE FORMAT DEBUG MOSTLY SUCCESSFUL!")
            logger.info("✅ IA2 execution can be triggered")
            logger.info("✅ Response format analysis completed")
            logger.info("✅ Technical levels fields analysis completed")
            logger.info("✅ JSON parsing debug completed")
            logger.info("🔍 Check detailed findings above for specific issues")
        elif passed_tests >= total_tests * 0.5:
            logger.info("\n⚠️ VERDICT: IA2 RESPONSE FORMAT DEBUG PARTIALLY SUCCESSFUL")
            logger.info("🔍 Some critical issues identified that need attention")
            logger.info("🔧 Review detailed findings for specific problems")
        else:
            logger.info("\n❌ VERDICT: IA2 RESPONSE FORMAT DEBUG REVEALED MAJOR ISSUES")
            logger.info("🚨 Critical problems preventing proper IA2 response format")
            logger.info("🚨 System needs significant debugging and fixes")
            logger.info("🔍 Focus on root cause analysis findings above")
        
        return passed_tests, total_tests

async def main():
    """Main function to run the IA2 response format debug test"""
    test_suite = IA2ResponseFormatDebugTestSuite()
    passed_tests, total_tests = await test_suite.run_ia2_debug_comprehensive_test()
    
    # Exit with appropriate code
    if passed_tests >= total_tests * 0.75:
        sys.exit(0)  # Mostly successful
    elif passed_tests >= total_tests * 0.5:
        sys.exit(1)  # Partially successful
    else:
        sys.exit(2)  # Major issues

if __name__ == "__main__":
    asyncio.run(main())
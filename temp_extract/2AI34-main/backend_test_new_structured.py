#!/usr/bin/env python3
"""
NEW IA1 STRUCTURED PROMPT APPROACH VERIFICATION TEST SUITE
Focus: Test NEW IA1 Structured Prompt Approach - Verify structured JSON format works and includes technical indicators.

SPECIFIC TESTING REQUIREMENTS FROM REVIEW REQUEST:

1. **JSON Parsing Success Verification**: Check if new structured approach fixes parsing issues
2. **Technical Indicators Integration Check**: Verify new structured format promotes technical indicators usage  
3. **Structured Response Validation**: Verify new JSON format is followed
4. **System Stability Check**: Ensure new approach doesn't break IA1 system
5. **Comparison with Previous Approach**: Assess improvement over complex mandatory format

SUCCESS CRITERIA:
✅ IA1 returns valid JSON responses without parsing failures
✅ IA1 mentions technical indicators (RSI, MACD, MFI, VWAP) in structured fields
✅ New JSON format fields are properly populated
✅ System stability improved with no datetime/parsing errors
✅ Technical indicators usage increased compared to previous approach
✅ Database storage works correctly with new structured format
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewIA1StructuredPromptTestSuite:
    """Comprehensive test suite for NEW IA1 Structured Prompt Approach Verification"""
    
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
        logger.info(f"Testing NEW IA1 Structured Prompt Approach at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for IA1 analysis (as specified in review request)
        self.test_symbols = ['BTCUSDT', 'ETHUSDT']
        
        # Core technical indicators to verify (as specified in review request)
        self.core_technical_indicators = ['RSI', 'MACD', 'MFI', 'VWAP']
        
        # New structured JSON format fields to verify
        self.structured_json_fields = [
            'decision_trading', 'strategie', 'analyse_detaillee', 
            'indicateurs_techniques_utilises', 'figures_chartistes_detectees',
            'ratio_risque_rendement', 'justification_succincte'
        ]
        
        # IA1 analysis data storage
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
    
    async def test_1_json_parsing_success_verification(self):
        """Test 1: JSON Parsing Success Verification - Check if new structured approach fixes parsing issues"""
        logger.info("\n🔍 TEST 1: JSON Parsing Success Verification")
        
        try:
            parsing_results = {
                'successful_analyses': 0,
                'total_analyses_attempted': len(self.test_symbols),
                'json_parsing_success': 0,
                'json_parsing_failures': 0,
                'fallback_analyses': 0,
                'valid_json_responses': [],
                'parsing_error_logs': [],
                'datetime_errors': 0,
                'system_stability_issues': 0
            }
            
            logger.info("   🚀 Testing NEW IA1 Structured Prompt Approach for JSON parsing success...")
            logger.info(f"   📊 Testing symbols: {self.test_symbols}")
            logger.info("   📊 Expected: IA1 returns valid JSON without parsing failures, no fallback to pattern-only analysis")
            
            # Test each symbol
            for symbol in self.test_symbols:
                logger.info(f"\n   📞 Testing IA1 structured JSON analysis for {symbol}")
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120  # Longer timeout for IA1 analysis
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        parsing_results['successful_analyses'] += 1
                        
                        logger.info(f"      ✅ IA1 analysis for {symbol} successful (response time: {response_time:.2f}s)")
                        
                        # Check for JSON parsing success indicators
                        is_valid_json = True
                        is_fallback_analysis = False
                        has_structured_fields = False
                        
                        # Look for fallback analysis indicators
                        if 'reasoning' in analysis_data:
                            reasoning_text = str(analysis_data['reasoning']).lower()
                            if any(fallback_indicator in reasoning_text for fallback_indicator in [
                                'fallback analysis', 'json parsing failed', 'pattern-only analysis',
                                'technical analysis fallback', 'using detected pattern'
                            ]):
                                is_fallback_analysis = True
                                parsing_results['fallback_analyses'] += 1
                                logger.warning(f"      ❌ Fallback analysis detected for {symbol}")
                        
                        # Check for structured JSON fields in IA1 response
                        ia1_response = analysis_data.get('ia1_analysis', {})
                        if isinstance(ia1_response, dict):
                            structured_fields_found = []
                            for field in self.structured_json_fields:
                                if field in ia1_response:
                                    structured_fields_found.append(field)
                            
                            if len(structured_fields_found) >= 3:  # At least 3 structured fields
                                has_structured_fields = True
                                logger.info(f"      ✅ Structured JSON fields found: {structured_fields_found}")
                            else:
                                logger.warning(f"      ⚠️ Limited structured fields: {structured_fields_found}")
                        
                        # Determine JSON parsing success
                        if not is_fallback_analysis and has_structured_fields:
                            parsing_results['json_parsing_success'] += 1
                            parsing_results['valid_json_responses'].append({
                                'symbol': symbol,
                                'response_time': response_time,
                                'structured_fields': structured_fields_found,
                                'analysis_data': analysis_data
                            })
                            logger.info(f"      ✅ JSON parsing SUCCESS for {symbol}")
                        else:
                            parsing_results['json_parsing_failures'] += 1
                            logger.warning(f"      ❌ JSON parsing FAILURE for {symbol} (fallback: {is_fallback_analysis}, structured: {has_structured_fields})")
                        
                        # Store full analysis for later use
                        self.ia1_analyses.append({
                            'symbol': symbol,
                            'analysis_data': analysis_data,
                            'is_valid_json': is_valid_json and not is_fallback_analysis,
                            'has_structured_fields': has_structured_fields
                        })
                    
                    else:
                        logger.error(f"      ❌ IA1 analysis for {symbol} failed: HTTP {response.status_code}")
                        if response.text:
                            logger.error(f"         Error response: {response.text[:200]}...")
                
                except Exception as e:
                    logger.error(f"      ❌ IA1 analysis for {symbol} exception: {e}")
                    # Check for datetime errors specifically
                    if 'datetime' in str(e).lower() or 'timezone' in str(e).lower():
                        parsing_results['datetime_errors'] += 1
                
                # Wait between analyses to avoid overwhelming the system
                if symbol != self.test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 10 seconds before next analysis...")
                    await asyncio.sleep(10)
            
            # Capture backend logs for JSON parsing analysis
            logger.info("   📋 Capturing backend logs for JSON parsing analysis...")
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    # Look for JSON parsing errors
                    json_error_patterns = [
                        'json parsing failed', 'fallback analysis', 'json decode error',
                        'invalid json', 'parsing error', 'malformed json'
                    ]
                    
                    json_errors_found = 0
                    for log_line in backend_logs:
                        for pattern in json_error_patterns:
                            if pattern.lower() in log_line.lower():
                                json_errors_found += 1
                                parsing_results['parsing_error_logs'].append(log_line.strip())
                                break
                    
                    # Look for datetime errors
                    datetime_error_patterns = [
                        'datetime', 'timezone', 'offset-naive', 'offset-aware',
                        'can\'t subtract', 'timezone error'
                    ]
                    
                    datetime_errors_found = 0
                    for log_line in backend_logs:
                        for pattern in datetime_error_patterns:
                            if pattern.lower() in log_line.lower() and 'error' in log_line.lower():
                                datetime_errors_found += 1
                                break
                    
                    parsing_results['datetime_errors'] += datetime_errors_found
                    
                    logger.info(f"      📊 Backend logs analysis: {len(backend_logs)} lines, {json_errors_found} JSON errors, {datetime_errors_found} datetime errors")
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Final analysis
            logger.info(f"\n   📊 JSON PARSING SUCCESS VERIFICATION RESULTS:")
            logger.info(f"      Successful analyses: {parsing_results['successful_analyses']}/{parsing_results['total_analyses_attempted']}")
            logger.info(f"      JSON parsing successes: {parsing_results['json_parsing_success']}")
            logger.info(f"      JSON parsing failures: {parsing_results['json_parsing_failures']}")
            logger.info(f"      Fallback analyses: {parsing_results['fallback_analyses']}")
            logger.info(f"      Datetime errors: {parsing_results['datetime_errors']}")
            
            # Calculate JSON parsing success rate
            json_success_rate = parsing_results['json_parsing_success'] / max(parsing_results['successful_analyses'], 1)
            fallback_rate = parsing_results['fallback_analyses'] / max(parsing_results['successful_analyses'], 1)
            
            logger.info(f"      📊 Performance metrics:")
            logger.info(f"         JSON parsing success rate: {json_success_rate:.2f} ({parsing_results['json_parsing_success']}/{parsing_results['successful_analyses']})")
            logger.info(f"         Fallback analysis rate: {fallback_rate:.2f}")
            logger.info(f"         System stability: {'Good' if parsing_results['datetime_errors'] == 0 else 'Issues detected'}")
            
            # Show valid JSON responses
            if parsing_results['valid_json_responses']:
                logger.info(f"      ✅ Valid JSON responses:")
                for response in parsing_results['valid_json_responses']:
                    logger.info(f"         - {response['symbol']}: {len(response['structured_fields'])} structured fields")
            
            # Calculate test success based on new structured approach requirements
            success_criteria = [
                parsing_results['successful_analyses'] >= 1,  # At least 1 successful analysis
                parsing_results['json_parsing_success'] >= 1,  # At least 1 JSON parsing success
                parsing_results['fallback_analyses'] == 0,  # No fallback analyses
                parsing_results['datetime_errors'] == 0,  # No datetime errors
                json_success_rate >= 0.8  # 80% JSON success rate
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold for structured approach
                self.log_test_result("JSON Parsing Success Verification", True, 
                                   f"Structured approach successful: {success_count}/{len(success_criteria)} criteria met. JSON success rate: {json_success_rate:.2f}, No fallbacks: {parsing_results['fallback_analyses'] == 0}, No datetime errors: {parsing_results['datetime_errors'] == 0}")
            else:
                self.log_test_result("JSON Parsing Success Verification", False, 
                                   f"Structured approach issues: {success_count}/{len(success_criteria)} criteria met. JSON parsing may still be failing or system has stability issues")
                
        except Exception as e:
            self.log_test_result("JSON Parsing Success Verification", False, f"Exception: {str(e)}")
    
    async def test_2_technical_indicators_integration_check(self):
        """Test 2: Technical Indicators Integration Check - Verify new structured format promotes technical indicators usage"""
        logger.info("\n🔍 TEST 2: Technical Indicators Integration Check")
        
        try:
            integration_results = {
                'structured_fields_analysis': {},
                'technical_indicators_mentioned': {},
                'justification_succincte_analysis': {},
                'confluence_analyse_found': False,
                'specific_indicator_values': {},
                'patterns_vs_indicators_ratio': 0.0,
                'structured_format_compliance': False,
                'analyses_with_indicators': 0,
                'total_valid_analyses': 0
            }
            
            logger.info("   🚀 Testing technical indicators integration in structured JSON format...")
            logger.info("   📊 Expected: IA1 mentions RSI, MACD, MFI, VWAP values in structured fields like 'justification_succincte' and 'confluence_analyse'")
            
            # Step 1: Analyze valid JSON responses from previous test
            logger.info("   📊 Analyzing valid JSON responses for technical indicators integration...")
            
            valid_analyses = []
            if hasattr(self, 'ia1_analyses') and self.ia1_analyses:
                for analysis in self.ia1_analyses:
                    if analysis.get('is_valid_json', False):
                        valid_analyses.append(analysis)
                        integration_results['total_valid_analyses'] += 1
                
                logger.info(f"      ✅ Found {len(valid_analyses)} valid JSON analyses to examine")
            else:
                logger.warning(f"      ⚠️ No valid JSON analyses available from previous test")
                return
            
            # Step 2: Analyze structured JSON fields for technical indicators
            logger.info("   📋 Analyzing structured JSON fields for technical indicators mentions...")
            
            for analysis in valid_analyses:
                symbol = analysis['symbol']
                analysis_data = analysis['analysis_data']
                ia1_response = analysis_data.get('ia1_analysis', {})
                
                logger.info(f"      📊 Analyzing {symbol} structured response...")
                
                # Check justification_succincte field
                justification = ia1_response.get('justification_succincte', '')
                if justification:
                    logger.info(f"         ✅ justification_succincte field found: {len(justification)} chars")
                    
                    # Look for technical indicators with values
                    indicators_found = {}
                    for indicator in self.core_technical_indicators:
                        # Look for patterns like "RSI 32.5" or "MACD -0.015"
                        patterns = [
                            rf'{indicator}\s*[:\s]+[\d\.\-]+',  # RSI: 32.5 or RSI 32.5
                            rf'{indicator}\s*[\d\.\-]+',        # RSI32.5
                            rf'{indicator}.*?[\d\.\-]+%?'       # RSI shows 32.5%
                        ]
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, justification, re.IGNORECASE)
                            if matches:
                                indicators_found[indicator] = len(matches)
                                if indicator not in integration_results['technical_indicators_mentioned']:
                                    integration_results['technical_indicators_mentioned'][indicator] = 0
                                integration_results['technical_indicators_mentioned'][indicator] += len(matches)
                                logger.info(f"            ✅ {indicator} found with values: {matches}")
                                break
                    
                    integration_results['justification_succincte_analysis'][symbol] = {
                        'length': len(justification),
                        'indicators_found': indicators_found,
                        'sample': justification[:200] + "..." if len(justification) > 200 else justification
                    }
                
                # Check analyse_detaillee → confluence_analyse
                analyse_detaillee = ia1_response.get('analyse_detaillee', {})
                if isinstance(analyse_detaillee, dict):
                    confluence_analyse = analyse_detaillee.get('confluence_analyse', '')
                    if confluence_analyse:
                        logger.info(f"         ✅ confluence_analyse field found: {len(confluence_analyse)} chars")
                        integration_results['confluence_analyse_found'] = True
                        
                        # Look for technical indicators discussion
                        confluence_indicators = 0
                        for indicator in self.core_technical_indicators:
                            if indicator.lower() in confluence_analyse.lower():
                                confluence_indicators += 1
                        
                        logger.info(f"            📊 Technical indicators in confluence: {confluence_indicators}/{len(self.core_technical_indicators)}")
                
                # Check if this analysis has technical indicators
                if any(indicator in integration_results['technical_indicators_mentioned'] for indicator in self.core_technical_indicators):
                    integration_results['analyses_with_indicators'] += 1
            
            # Final analysis
            logger.info(f"\n   📊 TECHNICAL INDICATORS INTEGRATION CHECK RESULTS:")
            logger.info(f"      Total valid analyses: {integration_results['total_valid_analyses']}")
            logger.info(f"      Analyses with technical indicators: {integration_results['analyses_with_indicators']}")
            logger.info(f"      Confluence analyse found: {integration_results['confluence_analyse_found']}")
            
            if integration_results['technical_indicators_mentioned']:
                logger.info(f"      📊 Technical indicators mentioned:")
                for indicator, count in integration_results['technical_indicators_mentioned'].items():
                    logger.info(f"         - {indicator}: {count} mentions")
            else:
                logger.warning(f"      ❌ NO technical indicators found in structured fields")
            
            # Calculate success metrics
            indicators_usage_rate = integration_results['analyses_with_indicators'] / max(integration_results['total_valid_analyses'], 1)
            indicators_coverage = len(integration_results['technical_indicators_mentioned']) / len(self.core_technical_indicators)
            
            logger.info(f"      📊 Integration metrics:")
            logger.info(f"         Technical indicators usage rate: {indicators_usage_rate:.2f}")
            logger.info(f"         Technical indicators coverage: {indicators_coverage:.2f} ({len(integration_results['technical_indicators_mentioned'])}/{len(self.core_technical_indicators)})")
            
            # Calculate test success
            success_criteria = [
                integration_results['total_valid_analyses'] > 0,  # Have valid analyses
                integration_results['analyses_with_indicators'] > 0,  # At least 1 analysis with indicators
                len(integration_results['technical_indicators_mentioned']) >= 2,  # At least 2 indicators mentioned
                indicators_usage_rate >= 0.5  # 50% of analyses mention indicators
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.75:  # 75% success threshold
                self.log_test_result("Technical Indicators Integration Check", True, 
                                   f"Integration successful: {success_count}/{len(success_criteria)} criteria met. Usage rate: {indicators_usage_rate:.2f}, Coverage: {indicators_coverage:.2f}")
            else:
                self.log_test_result("Technical Indicators Integration Check", False, 
                                   f"Integration issues: {success_count}/{len(success_criteria)} criteria met. Technical indicators may not be properly integrated in structured format")
                
        except Exception as e:
            self.log_test_result("Technical Indicators Integration Check", False, f"Exception: {str(e)}")
    
    async def test_3_structured_response_validation(self):
        """Test 3: Structured Response Validation - Verify new JSON format is followed"""
        logger.info("\n🔍 TEST 3: Structured Response Validation")
        
        try:
            validation_results = {
                'total_analyses': 0,
                'structured_fields_compliance': {},
                'required_fields_present': {},
                'field_quality_analysis': {},
                'complete_structured_responses': 0,
                'partial_structured_responses': 0,
                'non_structured_responses': 0
            }
            
            logger.info("   🚀 Validating structured JSON format compliance...")
            logger.info("   📊 Expected: Responses contain decision_trading, strategie, analyse_detaillee, indicateurs_techniques_utilises, etc.")
            
            # Analyze all IA1 analyses
            if hasattr(self, 'ia1_analyses') and self.ia1_analyses:
                validation_results['total_analyses'] = len(self.ia1_analyses)
                
                for analysis in self.ia1_analyses:
                    symbol = analysis['symbol']
                    analysis_data = analysis['analysis_data']
                    ia1_response = analysis_data.get('ia1_analysis', {})
                    
                    logger.info(f"      📊 Validating {symbol} structured format...")
                    
                    # Check for structured fields
                    fields_present = []
                    fields_missing = []
                    
                    for field in self.structured_json_fields:
                        if field in ia1_response and ia1_response[field] not in [None, '', {}]:
                            fields_present.append(field)
                            if field not in validation_results['structured_fields_compliance']:
                                validation_results['structured_fields_compliance'][field] = 0
                            validation_results['structured_fields_compliance'][field] += 1
                        else:
                            fields_missing.append(field)
                    
                    logger.info(f"         ✅ Fields present: {fields_present}")
                    if fields_missing:
                        logger.info(f"         ❌ Fields missing: {fields_missing}")
                    
                    # Categorize response completeness
                    completeness_ratio = len(fields_present) / len(self.structured_json_fields)
                    if completeness_ratio >= 0.8:  # 80% of fields present
                        validation_results['complete_structured_responses'] += 1
                        logger.info(f"         ✅ Complete structured response ({completeness_ratio:.2f})")
                    elif completeness_ratio >= 0.4:  # 40% of fields present
                        validation_results['partial_structured_responses'] += 1
                        logger.info(f"         ⚠️ Partial structured response ({completeness_ratio:.2f})")
                    else:
                        validation_results['non_structured_responses'] += 1
                        logger.info(f"         ❌ Non-structured response ({completeness_ratio:.2f})")
                    
                    # Quality analysis of specific fields
                    field_quality = {}
                    
                    # Check decision_trading
                    decision = ia1_response.get('decision_trading', '')
                    if decision:
                        field_quality['decision_trading'] = {
                            'present': True,
                            'valid': decision.upper() in ['LONG', 'SHORT', 'HOLD'],
                            'value': decision
                        }
                    
                    # Check ratio_risque_rendement
                    rr_ratio = ia1_response.get('ratio_risque_rendement', 0)
                    if rr_ratio:
                        field_quality['ratio_risque_rendement'] = {
                            'present': True,
                            'valid': isinstance(rr_ratio, (int, float)) and rr_ratio > 0,
                            'value': rr_ratio
                        }
                    
                    # Check indicateurs_techniques_utilises
                    indicators_used = ia1_response.get('indicateurs_techniques_utilises', [])
                    if indicators_used:
                        field_quality['indicateurs_techniques_utilises'] = {
                            'present': True,
                            'valid': isinstance(indicators_used, list) and len(indicators_used) > 0,
                            'count': len(indicators_used) if isinstance(indicators_used, list) else 0
                        }
                    
                    validation_results['field_quality_analysis'][symbol] = field_quality
                
                # Final analysis
                logger.info(f"\n   📊 STRUCTURED RESPONSE VALIDATION RESULTS:")
                logger.info(f"      Total analyses: {validation_results['total_analyses']}")
                logger.info(f"      Complete structured responses: {validation_results['complete_structured_responses']}")
                logger.info(f"      Partial structured responses: {validation_results['partial_structured_responses']}")
                logger.info(f"      Non-structured responses: {validation_results['non_structured_responses']}")
                
                logger.info(f"      📊 Field compliance:")
                for field, count in validation_results['structured_fields_compliance'].items():
                    compliance_rate = count / validation_results['total_analyses']
                    logger.info(f"         - {field}: {count}/{validation_results['total_analyses']} ({compliance_rate:.2f})")
                
                # Calculate success metrics
                complete_response_rate = validation_results['complete_structured_responses'] / max(validation_results['total_analyses'], 1)
                structured_response_rate = (validation_results['complete_structured_responses'] + validation_results['partial_structured_responses']) / max(validation_results['total_analyses'], 1)
                
                logger.info(f"      📊 Validation metrics:")
                logger.info(f"         Complete structured response rate: {complete_response_rate:.2f}")
                logger.info(f"         Overall structured response rate: {structured_response_rate:.2f}")
                
                # Calculate test success
                success_criteria = [
                    validation_results['total_analyses'] > 0,  # Have analyses to validate
                    validation_results['complete_structured_responses'] > 0,  # At least 1 complete response
                    len(validation_results['structured_fields_compliance']) >= 4,  # At least 4 fields used
                    complete_response_rate >= 0.5  # 50% complete responses
                ]
                success_count = sum(success_criteria)
                success_rate = success_count / len(success_criteria)
                
                if success_rate >= 0.75:  # 75% success threshold
                    self.log_test_result("Structured Response Validation", True, 
                                       f"Validation successful: {success_count}/{len(success_criteria)} criteria met. Complete rate: {complete_response_rate:.2f}, Fields used: {len(validation_results['structured_fields_compliance'])}")
                else:
                    self.log_test_result("Structured Response Validation", False, 
                                       f"Validation issues: {success_count}/{len(success_criteria)} criteria met. Structured format may not be properly followed")
            
            else:
                logger.warning(f"      ❌ No IA1 analyses available for validation")
                self.log_test_result("Structured Response Validation", False, "No analyses available for validation")
                
        except Exception as e:
            self.log_test_result("Structured Response Validation", False, f"Exception: {str(e)}")
    
    async def test_4_system_stability_check(self):
        """Test 4: System Stability Check - Ensure new approach doesn't break IA1 system"""
        logger.info("\n🔍 TEST 4: System Stability Check")
        
        try:
            stability_results = {
                'backend_logs_captured': False,
                'error_count': 0,
                'datetime_errors': 0,
                'json_parsing_errors': 0,
                'system_exceptions': 0,
                'database_storage_working': False,
                'api_endpoints_working': False,
                'performance_metrics': {},
                'stability_score': 0.0
            }
            
            logger.info("   🚀 Checking system stability with new structured approach...")
            logger.info("   📊 Expected: No datetime timezone errors, stable performance, database storage working")
            
            # Step 1: Analyze backend logs for stability issues
            logger.info("   📋 Analyzing backend logs for stability issues...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    stability_results['backend_logs_captured'] = True
                    logger.info(f"      ✅ Captured {len(backend_logs)} backend log lines")
                    
                    # Count different types of errors
                    error_patterns = {
                        'datetime_errors': ['datetime', 'timezone', 'offset-naive', 'offset-aware', 'can\'t subtract'],
                        'json_parsing_errors': ['json parsing failed', 'json decode error', 'invalid json', 'malformed json'],
                        'system_exceptions': ['exception', 'traceback', 'error:', 'failed:', 'critical']
                    }
                    
                    for error_type, patterns in error_patterns.items():
                        count = 0
                        for log_line in backend_logs:
                            for pattern in patterns:
                                if pattern.lower() in log_line.lower():
                                    count += 1
                                    break
                        stability_results[error_type] = count
                        logger.info(f"         {error_type}: {count} occurrences")
                    
                    stability_results['error_count'] = sum(stability_results[key] for key in error_patterns.keys())
                    
                else:
                    logger.warning(f"      ⚠️ Could not capture backend logs")
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Backend log analysis error: {e}")
            
            # Step 2: Test API endpoints functionality
            logger.info("   📞 Testing API endpoints functionality...")
            
            try:
                # Test /api/analyses endpoint
                response = requests.get(f"{self.api_url}/analyses", timeout=30)
                if response.status_code == 200:
                    analyses = response.json()
                    stability_results['api_endpoints_working'] = True
                    logger.info(f"      ✅ /api/analyses working: {len(analyses)} analyses retrieved")
                    
                    # Check if recent analyses are stored
                    if analyses:
                        stability_results['database_storage_working'] = True
                        logger.info(f"      ✅ Database storage working: Recent analyses found")
                    else:
                        logger.warning(f"      ⚠️ No analyses in database")
                else:
                    logger.warning(f"      ⚠️ /api/analyses failed: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"      ⚠️ API endpoint test error: {e}")
            
            # Step 3: Calculate performance metrics
            logger.info("   📊 Calculating performance metrics...")
            
            if hasattr(self, 'ia1_analyses') and self.ia1_analyses:
                response_times = []
                for analysis in self.ia1_analyses:
                    if 'response_time' in analysis:
                        response_times.append(analysis['response_time'])
                
                if response_times:
                    stability_results['performance_metrics'] = {
                        'avg_response_time': sum(response_times) / len(response_times),
                        'max_response_time': max(response_times),
                        'min_response_time': min(response_times),
                        'total_analyses': len(response_times)
                    }
                    
                    logger.info(f"      📊 Performance metrics:")
                    logger.info(f"         Average response time: {stability_results['performance_metrics']['avg_response_time']:.2f}s")
                    logger.info(f"         Max response time: {stability_results['performance_metrics']['max_response_time']:.2f}s")
                    logger.info(f"         Min response time: {stability_results['performance_metrics']['min_response_time']:.2f}s")
            
            # Step 4: Calculate stability score
            stability_factors = [
                stability_results['backend_logs_captured'],  # Logs accessible
                stability_results['datetime_errors'] == 0,  # No datetime errors
                stability_results['json_parsing_errors'] <= 1,  # Minimal JSON errors
                stability_results['system_exceptions'] <= 5,  # Minimal system exceptions
                stability_results['api_endpoints_working'],  # API working
                stability_results['database_storage_working']  # Database working
            ]
            
            stability_results['stability_score'] = sum(stability_factors) / len(stability_factors)
            
            # Final analysis
            logger.info(f"\n   📊 SYSTEM STABILITY CHECK RESULTS:")
            logger.info(f"      Backend logs captured: {stability_results['backend_logs_captured']}")
            logger.info(f"      Total errors: {stability_results['error_count']}")
            logger.info(f"      Datetime errors: {stability_results['datetime_errors']}")
            logger.info(f"      JSON parsing errors: {stability_results['json_parsing_errors']}")
            logger.info(f"      System exceptions: {stability_results['system_exceptions']}")
            logger.info(f"      API endpoints working: {stability_results['api_endpoints_working']}")
            logger.info(f"      Database storage working: {stability_results['database_storage_working']}")
            logger.info(f"      Stability score: {stability_results['stability_score']:.2f}")
            
            # Calculate test success
            success_criteria = [
                stability_results['datetime_errors'] == 0,  # No datetime errors
                stability_results['json_parsing_errors'] <= 1,  # Minimal JSON errors
                stability_results['api_endpoints_working'],  # API working
                stability_results['database_storage_working'],  # Database working
                stability_results['stability_score'] >= 0.7  # Good stability score
            ]
            success_count = sum(success_criteria)
            success_rate = success_count / len(success_criteria)
            
            if success_rate >= 0.8:  # 80% success threshold
                self.log_test_result("System Stability Check", True, 
                                   f"System stable: {success_count}/{len(success_criteria)} criteria met. Stability score: {stability_results['stability_score']:.2f}, Datetime errors: {stability_results['datetime_errors']}")
            else:
                self.log_test_result("System Stability Check", False, 
                                   f"System stability issues: {success_count}/{len(success_criteria)} criteria met. May have datetime errors or system instability")
                
        except Exception as e:
            self.log_test_result("System Stability Check", False, f"Exception: {str(e)}")
    
    async def test_5_comparison_with_previous_approach(self):
        """Test 5: Comparison with Previous Approach - Assess improvement over complex mandatory format"""
        logger.info("\n🔍 TEST 5: Comparison with Previous Approach")
        
        try:
            comparison_results = {
                'current_json_success_rate': 0.0,
                'current_technical_indicators_usage': 0.0,
                'current_system_stability': 0.0,
                'improvement_metrics': {},
                'overall_improvement_score': 0.0,
                'key_improvements': [],
                'remaining_issues': []
            }
            
            logger.info("   🚀 Comparing new structured approach with previous complex mandatory format...")
            logger.info("   📊 Expected: JSON parsing >80%, technical indicators >50%, improved system stability")
            
            # Calculate current performance metrics from previous tests
            if hasattr(self, 'test_results') and self.test_results:
                # Extract metrics from test results
                json_parsing_test = next((test for test in self.test_results if 'JSON Parsing' in test['test']), None)
                indicators_test = next((test for test in self.test_results if 'Technical Indicators' in test['test']), None)
                stability_test = next((test for test in self.test_results if 'System Stability' in test['test']), None)
                
                # Calculate JSON success rate
                if hasattr(self, 'ia1_analyses') and self.ia1_analyses:
                    valid_json_count = sum(1 for analysis in self.ia1_analyses if analysis.get('is_valid_json', False))
                    comparison_results['current_json_success_rate'] = valid_json_count / len(self.ia1_analyses)
                    
                    # Calculate technical indicators usage
                    indicators_count = 0
                    for analysis in self.ia1_analyses:
                        analysis_data = analysis.get('analysis_data', {})
                        ia1_response = analysis_data.get('ia1_analysis', {})
                        justification = ia1_response.get('justification_succincte', '')
                        
                        if any(indicator.lower() in justification.lower() for indicator in self.core_technical_indicators):
                            indicators_count += 1
                    
                    comparison_results['current_technical_indicators_usage'] = indicators_count / len(self.ia1_analyses)
                
                # System stability from previous test
                comparison_results['current_system_stability'] = 1.0 if stability_test and stability_test['success'] else 0.5
                
                logger.info(f"      📊 Current performance metrics:")
                logger.info(f"         JSON parsing success rate: {comparison_results['current_json_success_rate']:.2f}")
                logger.info(f"         Technical indicators usage: {comparison_results['current_technical_indicators_usage']:.2f}")
                logger.info(f"         System stability: {comparison_results['current_system_stability']:.2f}")
                
                # Compare with previous approach (from test_result.md context)
                previous_metrics = {
                    'json_success_rate': 0.0,  # Previous: 0% due to JSON parsing failures
                    'technical_indicators_usage': 0.0,  # Previous: 0% technical indicator mentions
                    'system_stability': 0.3  # Previous: Had datetime errors and fallback issues
                }
                
                # Calculate improvements
                comparison_results['improvement_metrics'] = {
                    'json_parsing_improvement': comparison_results['current_json_success_rate'] - previous_metrics['json_success_rate'],
                    'technical_indicators_improvement': comparison_results['current_technical_indicators_usage'] - previous_metrics['technical_indicators_usage'],
                    'system_stability_improvement': comparison_results['current_system_stability'] - previous_metrics['system_stability']
                }
                
                logger.info(f"      📊 Improvement metrics:")
                for metric, improvement in comparison_results['improvement_metrics'].items():
                    logger.info(f"         {metric}: {improvement:+.2f}")
                
                # Identify key improvements
                if comparison_results['current_json_success_rate'] >= 0.8:
                    comparison_results['key_improvements'].append("JSON parsing success rate >80%")
                
                if comparison_results['current_technical_indicators_usage'] >= 0.5:
                    comparison_results['key_improvements'].append("Technical indicators usage >50%")
                
                if comparison_results['current_system_stability'] >= 0.8:
                    comparison_results['key_improvements'].append("System stability improved")
                
                # Identify remaining issues
                if comparison_results['current_json_success_rate'] < 0.8:
                    comparison_results['remaining_issues'].append("JSON parsing still below 80%")
                
                if comparison_results['current_technical_indicators_usage'] < 0.5:
                    comparison_results['remaining_issues'].append("Technical indicators usage below 50%")
                
                # Calculate overall improvement score
                improvement_factors = [
                    comparison_results['current_json_success_rate'] >= 0.8,  # Target: >80%
                    comparison_results['current_technical_indicators_usage'] >= 0.5,  # Target: >50%
                    comparison_results['current_system_stability'] >= 0.8,  # Target: stable
                    len(comparison_results['key_improvements']) >= 2,  # At least 2 key improvements
                    len(comparison_results['remaining_issues']) <= 1  # Minimal remaining issues
                ]
                
                comparison_results['overall_improvement_score'] = sum(improvement_factors) / len(improvement_factors)
                
                # Final analysis
                logger.info(f"\n   📊 COMPARISON WITH PREVIOUS APPROACH RESULTS:")
                logger.info(f"      Overall improvement score: {comparison_results['overall_improvement_score']:.2f}")
                
                if comparison_results['key_improvements']:
                    logger.info(f"      ✅ Key improvements:")
                    for improvement in comparison_results['key_improvements']:
                        logger.info(f"         - {improvement}")
                
                if comparison_results['remaining_issues']:
                    logger.info(f"      ❌ Remaining issues:")
                    for issue in comparison_results['remaining_issues']:
                        logger.info(f"         - {issue}")
                
                # Calculate test success
                success_criteria = [
                    comparison_results['current_json_success_rate'] >= 0.8,  # JSON success >80%
                    comparison_results['current_technical_indicators_usage'] >= 0.5,  # Indicators >50%
                    len(comparison_results['key_improvements']) >= 2,  # At least 2 improvements
                    comparison_results['overall_improvement_score'] >= 0.6  # Good overall improvement
                ]
                success_count = sum(success_criteria)
                success_rate = success_count / len(success_criteria)
                
                if success_rate >= 0.75:  # 75% success threshold
                    self.log_test_result("Comparison with Previous Approach", True, 
                                       f"Significant improvement: {success_count}/{len(success_criteria)} criteria met. JSON: {comparison_results['current_json_success_rate']:.2f}, Indicators: {comparison_results['current_technical_indicators_usage']:.2f}")
                else:
                    self.log_test_result("Comparison with Previous Approach", False, 
                                       f"Limited improvement: {success_count}/{len(success_criteria)} criteria met. May not have achieved target improvements over previous approach")
            
            else:
                logger.warning(f"      ❌ No test results available for comparison")
                self.log_test_result("Comparison with Previous Approach", False, "No test results available for comparison")
                
        except Exception as e:
            self.log_test_result("Comparison with Previous Approach", False, f"Exception: {str(e)}")
    
    async def _capture_backend_logs(self):
        """Capture recent backend logs"""
        try:
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            all_logs = []
            for log_file in log_files:
                try:
                    result = subprocess.run(['tail', '-n', '100', log_file], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        all_logs.extend(result.stdout.strip().split('\n'))
                except Exception as e:
                    logger.debug(f"Could not read {log_file}: {e}")
            
            return [log for log in all_logs if log.strip()]
            
        except Exception as e:
            logger.warning(f"Failed to capture backend logs: {e}")
            return []
    
    def generate_summary(self):
        """Generate comprehensive test summary"""
        logger.info("\n" + "="*80)
        logger.info("NEW IA1 STRUCTURED PROMPT APPROACH - COMPREHENSIVE TEST SUMMARY")
        logger.info("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test['success'])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"📊 OVERALL RESULTS: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        # Detailed results
        for test in self.test_results:
            status = "✅ PASS" if test['success'] else "❌ FAIL"
            logger.info(f"{status}: {test['test']}")
            if test['details']:
                logger.info(f"   {test['details']}")
        
        # Final assessment
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            logger.info("\n🎉 NEW IA1 STRUCTURED PROMPT APPROACH: SUCCESSFUL")
            logger.info("   The new structured approach has significantly improved IA1 system:")
            logger.info("   - JSON parsing issues resolved")
            logger.info("   - Technical indicators integration improved")
            logger.info("   - System stability enhanced")
        else:
            logger.info("\n⚠️ NEW IA1 STRUCTURED PROMPT APPROACH: NEEDS IMPROVEMENT")
            logger.info("   Some issues remain with the new structured approach:")
            logger.info("   - May still have JSON parsing failures")
            logger.info("   - Technical indicators integration may be incomplete")
            logger.info("   - System stability issues may persist")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'test_results': self.test_results
        }

async def main():
    """Main test execution function"""
    logger.info("🚀 Starting NEW IA1 Structured Prompt Approach Verification Test Suite")
    
    test_suite = NewIA1StructuredPromptTestSuite()
    
    try:
        # Execute all tests in sequence
        await test_suite.test_1_json_parsing_success_verification()
        await test_suite.test_2_technical_indicators_integration_check()
        await test_suite.test_3_structured_response_validation()
        await test_suite.test_4_system_stability_check()
        await test_suite.test_5_comparison_with_previous_approach()
        
        # Generate final summary
        summary = test_suite.generate_summary()
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    asyncio.run(main())
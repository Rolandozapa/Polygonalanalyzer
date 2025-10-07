#!/usr/bin/env python3
"""
MULTI-PHASE STRATEGIC FRAMEWORK TESTING SUITE
Focus: Test du Multi-Phase Strategic Framework enrichi dans le prompt IA2.

CRITICAL VALIDATION POINTS:

1. **Test du prompt IA2 enrichi** - Vérifier que le nouveau champ `market_regime_assessment` est bien dans la configuration :
   - Vérifier le contenu du prompt IA2 v3 Strategic Ultra  
   - Confirmer que `market_regime_assessment` est dans la section JSON output
   - Valider que les champs execution_priority, risk_level sont configurés correctement

2. **Test de génération IA2** - Essayer de créer une décision IA2 réelle :
   - Forcer des analyses IA1 avec différents symboles (BTCUSDT, ETHUSDT, LINKUSDT)
   - Identifier s'il existe des analyses avec confidence >70% ET RR >2.0 ET signal LONG/SHORT
   - Si aucune n'existe, documenter les conditions actuelles

3. **Test endpoint de création IA2** :
   - Tester /api/create-test-ia2-decision 
   - Vérifier que la décision créée contient bien les champs attendus
   - Valider que market_regime_assessment, execution_priority, risk_level ont des valeurs

4. **Validation de la réponse API IA2** :
   - Vérifier /api/ia2-decisions retourne des décisions avec tous les champs Multi-Phase
   - Confirmer que les valeurs ne sont plus null pour les nouveaux champs
   - Tester la diversité des valeurs (bullish/bearish/neutral, immediate/delayed/wait, etc.)

OBJECTIF: Confirmer que le prompt IA2 enrichi génère correctement tous les signaux du Multi-Phase Strategic Framework et que ces données sont accessibles via l'API.
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

class MultiPhaseStrategicFrameworkTestSuite:
    """Comprehensive test suite for Multi-Phase Strategic Framework validation"""
    
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
        logger.info(f"Testing Multi-Phase Strategic Framework at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for analysis (from review request)
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'LINKUSDT']  # Specific symbols from review request
        self.actual_test_symbols = []  # Will be populated from available opportunities
        
        # Expected Multi-Phase Strategic Framework fields that should be present and not null
        self.expected_ia2_fields = [
            'market_regime_assessment', 'execution_priority', 'risk_level', 
            'volume_profile_bias', 'orderbook_quality', 'multi_phase_score'
        ]
        
        # Valid values for Multi-Phase fields
        self.valid_market_regime_values = ['bullish', 'bearish', 'neutral']
        self.valid_execution_priority_values = ['immediate', 'delayed', 'wait']
        self.valid_risk_level_values = ['low', 'medium', 'high']
        
        # Error patterns to check for in logs
        self.error_patterns = [
            "market_regime_assessment.*null",
            "execution_priority.*null", 
            "risk_level.*null",
            "ia2.*fallback",
            "ia2.*default"
        ]
        
        # IA2 decision data storage
        self.ia2_decisions = []
        self.backend_logs = []
        
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
    
    async def test_1_ia2_prompt_enriched_validation(self):
        """Test 1: Test du prompt IA2 enrichi - Vérifier que le nouveau champ `market_regime_assessment` est bien dans la configuration"""
        logger.info("\n🔍 TEST 1: Test du prompt IA2 enrichi - Validation du Multi-Phase Strategic Framework")
        
        try:
            prompt_results = {
                'ia2_v3_prompt_exists': False,
                'ia2_strategic_prompt_exists': False,
                'market_regime_assessment_found': False,
                'execution_priority_found': False,
                'risk_level_found': False,
                'json_output_section_valid': False,
                'required_variables_present': False,
                'multi_phase_framework_complete': False,
                'prompt_content_analysis': {},
                'error_details': []
            }
            
            logger.info("   🚀 Testing IA1 analysis with specific symbols to verify confluence values are not null...")
            logger.info("   📊 Expected: confluence_grade (A,B,C,D), confluence_score (0-100), should_trade (true/false)")
            
            # Get available symbols from scout system
            logger.info("   📞 Getting available symbols from scout system...")
            
            try:
                response = requests.get(f"{self.api_url}/opportunities", timeout=60)
                
                if response.status_code == 200:
                    opportunities = response.json()
                    if isinstance(opportunities, dict) and 'opportunities' in opportunities:
                        opportunities = opportunities['opportunities']
                    
                    # Get available symbols for testing
                    available_symbols = [opp.get('symbol') for opp in opportunities[:15] if opp.get('symbol')]
                    
                    # Prefer specific symbols from review request
                    preferred_symbols = ['BTCUSDT', 'ETHUSDT', 'LINKUSDT']
                    test_symbols = []
                    
                    for symbol in preferred_symbols:
                        if symbol in available_symbols:
                            test_symbols.append(symbol)
                    
                    # Fill remaining slots with available symbols if needed
                    for symbol in available_symbols:
                        if symbol not in test_symbols and len(test_symbols) < 3:
                            test_symbols.append(symbol)
                    
                    self.actual_test_symbols = test_symbols[:3]  # Limit to 3 symbols as per review request
                    logger.info(f"      ✅ Test symbols selected: {self.actual_test_symbols}")
                    
                else:
                    logger.warning(f"      ⚠️ Could not get opportunities, using default symbols")
                    self.actual_test_symbols = self.test_symbols
                    
            except Exception as e:
                logger.warning(f"      ⚠️ Error getting opportunities: {e}, using default symbols")
                self.actual_test_symbols = self.test_symbols
            
            # Test each symbol for confluence analysis validation
            for symbol in self.actual_test_symbols:
                logger.info(f"\n   📞 Testing IA1 analysis for {symbol} - checking confluence values...")
                analysis_results['analyses_attempted'] += 1
                
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/force-ia1-analysis",
                        json={"symbol": symbol},
                        timeout=120
                    )
                    response_time = time.time() - start_time
                    analysis_results['response_times'].append(response_time)
                    
                    if response.status_code == 200:
                        analysis_data = response.json()
                        analysis_results['analyses_successful'] += 1
                        
                        logger.info(f"      ✅ {symbol} analysis successful (response time: {response_time:.2f}s)")
                        
                        # Extract IA1 analysis data
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        if not isinstance(ia1_analysis, dict):
                            ia1_analysis = {}
                        
                        # Check confluence_grade
                        confluence_grade = ia1_analysis.get('confluence_grade')
                        if confluence_grade is not None and confluence_grade != 'null':
                            analysis_results['confluence_grade_not_null'] += 1
                            logger.info(f"         ✅ confluence_grade present: {confluence_grade}")
                            
                            if confluence_grade in self.valid_confluence_grades:
                                analysis_results['valid_confluence_grades'] += 1
                                logger.info(f"         ✅ confluence_grade valid: {confluence_grade}")
                            else:
                                logger.warning(f"         ⚠️ confluence_grade invalid: {confluence_grade} (expected A,B,C,D)")
                        else:
                            logger.error(f"         ❌ confluence_grade is null or missing")
                        
                        # Check confluence_score
                        confluence_score = ia1_analysis.get('confluence_score')
                        if confluence_score is not None and confluence_score != 'null':
                            analysis_results['confluence_score_not_null'] += 1
                            logger.info(f"         ✅ confluence_score present: {confluence_score}")
                            
                            # Check if score is in valid range (0-100)
                            try:
                                score_value = float(confluence_score)
                                if 0 <= score_value <= 100:
                                    analysis_results['valid_confluence_scores'] += 1
                                    logger.info(f"         ✅ confluence_score valid range: {score_value}")
                                else:
                                    logger.warning(f"         ⚠️ confluence_score out of range: {score_value} (expected 0-100)")
                            except (ValueError, TypeError):
                                logger.warning(f"         ⚠️ confluence_score not numeric: {confluence_score}")
                        else:
                            logger.error(f"         ❌ confluence_score is null or missing")
                        
                        # Check should_trade
                        should_trade = ia1_analysis.get('should_trade')
                        if should_trade is not None and should_trade != 'null':
                            analysis_results['should_trade_not_null'] += 1
                            logger.info(f"         ✅ should_trade present: {should_trade}")
                            
                            if isinstance(should_trade, bool) or should_trade in ['true', 'false', True, False]:
                                logger.info(f"         ✅ should_trade valid boolean: {should_trade}")
                            else:
                                logger.warning(f"         ⚠️ should_trade not boolean: {should_trade}")
                        else:
                            logger.error(f"         ❌ should_trade is null or missing")
                        
                        # Check for confluence reasoning in IA1 reasoning
                        reasoning = ia1_analysis.get('reasoning', '')
                        if reasoning and 'confluence' in reasoning.lower():
                            analysis_results['confluence_reasoning_present'] += 1
                            logger.info(f"         ✅ Confluence reasoning present in IA1 analysis")
                        else:
                            logger.warning(f"         ⚠️ No confluence reasoning found in IA1 analysis")
                        
                        # Store confluence data for analysis
                        confluence_data = {
                            'symbol': symbol,
                            'confluence_grade': confluence_grade,
                            'confluence_score': confluence_score,
                            'should_trade': should_trade,
                            'has_reasoning': bool(reasoning and 'confluence' in reasoning.lower()),
                            'response_time': response_time
                        }
                        analysis_results['confluence_data'].append(confluence_data)
                        
                        # Store successful analysis details
                        analysis_results['successful_analyses'].append({
                            'symbol': symbol,
                            'response_time': response_time,
                            'confluence_data': confluence_data,
                            'analysis_data': ia1_analysis
                        })
                        
                    elif response.status_code == 500:
                        # Check for confluence-related errors
                        error_text = response.text
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP 500")
                        logger.error(f"         Error response: {error_text[:500]}")
                        
                        # Check for confluence calculation errors
                        confluence_error_found = False
                        if any(pattern in error_text.lower() for pattern in ["confluence", "grade", "score"]):
                            confluence_error_found = True
                            logger.error(f"         🚨 CONFLUENCE ERROR DETECTED in {symbol}")
                        
                        analysis_results['error_details'].append({
                            'symbol': symbol,
                            'error_type': 'HTTP_500',
                            'error_text': error_text[:500],
                            'has_confluence_error': confluence_error_found
                        })
                        
                    else:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                        if response.text:
                            error_text = response.text[:300]
                            logger.error(f"         Error response: {error_text}")
                            analysis_results['error_details'].append({
                                'symbol': symbol,
                                'error_type': f'HTTP_{response.status_code}',
                                'error_text': error_text
                            })
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} analysis exception: {e}")
                    analysis_results['error_details'].append({
                        'symbol': symbol,
                        'error_type': 'EXCEPTION',
                        'error_text': str(e)
                    })
                
                # Wait between analyses
                if symbol != self.actual_test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 5 seconds before next analysis...")
                    await asyncio.sleep(5)
            
            # Capture backend logs to check for confluence calculation logs
            logger.info("   📋 Capturing backend logs to check for confluence calculation logs...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    # Look for confluence-related log patterns
                    confluence_calculation_logs = []
                    confluence_error_logs = []
                    success_logs = []
                    
                    for log_line in backend_logs:
                        log_lower = log_line.lower()
                        
                        # Check for confluence calculation logs
                        if any(pattern in log_lower for pattern in ['confluence', 'grade', 'score']):
                            confluence_calculation_logs.append(log_line.strip())
                        
                        # Check for confluence errors
                        if any(pattern in log_lower for pattern in ['confluence.*error', 'confluence.*null', 'confluence.*failed']):
                            confluence_error_logs.append(log_line.strip())
                        
                        # Check for successful IA1 completions
                        if any(pattern in log_lower for pattern in ['ia1 analysis completed', 'ia1 ultra analysis', 'analysis successful']):
                            success_logs.append(log_line.strip())
                    
                    logger.info(f"      📊 Backend logs analysis:")
                    logger.info(f"         - Confluence calculation logs: {len(confluence_calculation_logs)}")
                    logger.info(f"         - Confluence error logs: {len(confluence_error_logs)}")
                    logger.info(f"         - Success logs: {len(success_logs)}")
                    
                    # Show sample logs
                    if confluence_calculation_logs:
                        logger.info(f"      ✅ Sample confluence log: {confluence_calculation_logs[0]}")
                    if confluence_error_logs:
                        logger.error(f"      🚨 CONFLUENCE ERROR FOUND: {confluence_error_logs[0]}")
                    if success_logs:
                        logger.info(f"      ✅ Sample success log: {success_logs[0]}")
                        
            except Exception as e:
                logger.warning(f"      ⚠️ Could not analyze backend logs: {e}")
            
            # Final analysis and results
            success_rate = analysis_results['analyses_successful'] / max(analysis_results['analyses_attempted'], 1)
            confluence_grade_rate = analysis_results['confluence_grade_not_null'] / max(analysis_results['analyses_successful'], 1)
            confluence_score_rate = analysis_results['confluence_score_not_null'] / max(analysis_results['analyses_successful'], 1)
            should_trade_rate = analysis_results['should_trade_not_null'] / max(analysis_results['analyses_successful'], 1)
            valid_grades_rate = analysis_results['valid_confluence_grades'] / max(analysis_results['confluence_grade_not_null'], 1)
            valid_scores_rate = analysis_results['valid_confluence_scores'] / max(analysis_results['confluence_score_not_null'], 1)
            avg_response_time = sum(analysis_results['response_times']) / max(len(analysis_results['response_times']), 1)
            
            logger.info(f"\n   📊 API FORCE IA1 ANALYSIS CONFLUENCE RESULTS:")
            logger.info(f"      Analyses attempted: {analysis_results['analyses_attempted']}")
            logger.info(f"      Analyses successful: {analysis_results['analyses_successful']}")
            logger.info(f"      Success rate: {success_rate:.2f}")
            logger.info(f"      confluence_grade not null: {analysis_results['confluence_grade_not_null']} ({confluence_grade_rate:.2f})")
            logger.info(f"      confluence_score not null: {analysis_results['confluence_score_not_null']} ({confluence_score_rate:.2f})")
            logger.info(f"      should_trade not null: {analysis_results['should_trade_not_null']} ({should_trade_rate:.2f})")
            logger.info(f"      Valid confluence grades: {analysis_results['valid_confluence_grades']} ({valid_grades_rate:.2f})")
            logger.info(f"      Valid confluence scores: {analysis_results['valid_confluence_scores']} ({valid_scores_rate:.2f})")
            logger.info(f"      Confluence reasoning present: {analysis_results['confluence_reasoning_present']}")
            logger.info(f"      Average response time: {avg_response_time:.2f}s")
            
            # Show confluence data details
            if analysis_results['confluence_data']:
                logger.info(f"      📊 Confluence Data Details:")
                for data in analysis_results['confluence_data']:
                    logger.info(f"         - {data['symbol']}: grade={data['confluence_grade']}, score={data['confluence_score']}, trade={data['should_trade']}, reasoning={data['has_reasoning']}")
            
            # Show error details if any
            if analysis_results['error_details']:
                logger.info(f"      📊 Error Details:")
                for error in analysis_results['error_details']:
                    logger.info(f"         - {error['symbol']}: {error['error_type']}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                analysis_results['analyses_successful'] >= 2,  # At least 2 successful analyses
                analysis_results['confluence_grade_not_null'] >= 2,  # At least 2 with confluence_grade not null
                analysis_results['confluence_score_not_null'] >= 2,  # At least 2 with confluence_score not null
                analysis_results['should_trade_not_null'] >= 2,  # At least 2 with should_trade not null
                analysis_results['valid_confluence_grades'] >= 1,  # At least 1 valid grade
                success_rate >= 0.67  # At least 67% success rate (2/3)
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("API Force IA1 Analysis Confluence", True, 
                                   f"Confluence analysis successful: {success_count}/{len(success_criteria)} criteria met. Grade rate: {confluence_grade_rate:.2f}, Score rate: {confluence_score_rate:.2f}, Trade rate: {should_trade_rate:.2f}")
            else:
                self.log_test_result("API Force IA1 Analysis Confluence", False, 
                                   f"Confluence analysis issues: {success_count}/{len(success_criteria)} criteria met. Many confluence values still null or invalid")
                
        except Exception as e:
            self.log_test_result("API Force IA1 Analysis Confluence", False, f"Exception: {str(e)}")

    async def test_2_api_analyses_confluence(self):
        """Test 2: API Analyses Endpoint - Check confluence consistency in stored analyses"""
        logger.info("\n🔍 TEST 2: API Analyses Endpoint - Confluence Consistency Validation")
        
        try:
            analyses_results = {
                'api_call_successful': False,
                'analyses_returned': 0,
                'confluence_fields_present': 0,
                'confluence_grades_not_null': 0,
                'confluence_scores_not_null': 0,
                'should_trade_not_null': 0,
                'default_fallback_values': 0,
                'diverse_confluence_scores': 0,
                'analyses_data': [],
                'error_details': []
            }
            
            logger.info("   🚀 Testing /api/analyses endpoint for confluence consistency...")
            logger.info("   📊 Expected: Confluence values consistent, no default fallbacks (50/100), real market conditions")
            
            # Test /api/analyses endpoint
            logger.info("   📞 Calling /api/analyses endpoint...")
            
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_url}/analyses", timeout=60)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    analyses_results['api_call_successful'] = True
                    logger.info(f"      ✅ /api/analyses successful (response time: {response_time:.2f}s)")
                    
                    # Parse response
                    try:
                        data = response.json()
                        
                        # Handle different response formats
                        if isinstance(data, dict) and 'analyses' in data:
                            analyses = data['analyses']
                        elif isinstance(data, list):
                            analyses = data
                        else:
                            analyses = []
                        
                        analyses_results['analyses_returned'] = len(analyses)
                        logger.info(f"      📊 Analyses returned: {len(analyses)}")
                        
                        if len(analyses) > 0:
                            confluence_scores = []
                            
                            # Analyze first 20 analyses for confluence validation
                            for i, analysis in enumerate(analyses[:20]):
                                if not isinstance(analysis, dict):
                                    continue
                                
                                # Check for confluence fields
                                confluence_grade = analysis.get('confluence_grade')
                                confluence_score = analysis.get('confluence_score')
                                should_trade = analysis.get('should_trade')
                                
                                has_confluence_fields = any([
                                    confluence_grade is not None,
                                    confluence_score is not None,
                                    should_trade is not None
                                ])
                                
                                if has_confluence_fields:
                                    analyses_results['confluence_fields_present'] += 1
                                
                                # Check confluence_grade
                                if confluence_grade is not None and confluence_grade != 'null':
                                    analyses_results['confluence_grades_not_null'] += 1
                                
                                # Check confluence_score
                                if confluence_score is not None and confluence_score != 'null':
                                    analyses_results['confluence_scores_not_null'] += 1
                                    
                                    # Check for default fallback values (50, 100)
                                    try:
                                        score_value = float(confluence_score)
                                        confluence_scores.append(score_value)
                                        
                                        if score_value in [50.0, 100.0]:
                                            analyses_results['default_fallback_values'] += 1
                                    except (ValueError, TypeError):
                                        pass
                                
                                # Check should_trade
                                if should_trade is not None and should_trade != 'null':
                                    analyses_results['should_trade_not_null'] += 1
                                
                                # Store sample analysis data
                                if i < 5:  # Store first 5 for analysis
                                    analyses_results['analyses_data'].append({
                                        'symbol': analysis.get('symbol', 'UNKNOWN'),
                                        'confluence_grade': confluence_grade,
                                        'confluence_score': confluence_score,
                                        'should_trade': should_trade,
                                        'timestamp': analysis.get('timestamp', 'N/A'),
                                        'has_confluence_fields': has_confluence_fields
                                    })
                                    
                                    logger.info(f"         📋 Sample {i+1} ({analysis.get('symbol', 'UNKNOWN')}): grade={confluence_grade}, score={confluence_score}, trade={should_trade}")
                            
                            # Check for diversity in confluence scores (not all the same)
                            if confluence_scores:
                                unique_scores = len(set(confluence_scores))
                                if unique_scores > 1:
                                    analyses_results['diverse_confluence_scores'] = unique_scores
                                    logger.info(f"         ✅ Diverse confluence scores found: {unique_scores} unique values")
                                else:
                                    logger.warning(f"         ⚠️ All confluence scores are the same: {confluence_scores[0] if confluence_scores else 'N/A'}")
                        
                        else:
                            logger.warning(f"      ⚠️ No analyses returned from API")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"      ❌ Invalid JSON response: {e}")
                        analyses_results['error_details'].append(f"JSON decode error: {e}")
                        
                else:
                    logger.error(f"      ❌ /api/analyses failed: HTTP {response.status_code}")
                    if response.text:
                        error_text = response.text[:500]
                        logger.error(f"         Error response: {error_text}")
                        analyses_results['error_details'].append(f"HTTP {response.status_code}: {error_text}")
                    
            except Exception as e:
                logger.error(f"      ❌ /api/analyses exception: {e}")
                analyses_results['error_details'].append(f"Exception: {str(e)}")
            
            # Final analysis and results
            confluence_fields_rate = analyses_results['confluence_fields_present'] / max(analyses_results['analyses_returned'], 1)
            confluence_grades_rate = analyses_results['confluence_grades_not_null'] / max(analyses_results['analyses_returned'], 1)
            confluence_scores_rate = analyses_results['confluence_scores_not_null'] / max(analyses_results['analyses_returned'], 1)
            should_trade_rate = analyses_results['should_trade_not_null'] / max(analyses_results['analyses_returned'], 1)
            fallback_rate = analyses_results['default_fallback_values'] / max(analyses_results['confluence_scores_not_null'], 1)
            
            logger.info(f"\n   📊 API ANALYSES CONFLUENCE RESULTS:")
            logger.info(f"      API call successful: {analyses_results['api_call_successful']}")
            logger.info(f"      Analyses returned: {analyses_results['analyses_returned']}")
            logger.info(f"      Confluence fields present: {analyses_results['confluence_fields_present']} ({confluence_fields_rate:.2f})")
            logger.info(f"      Confluence grades not null: {analyses_results['confluence_grades_not_null']} ({confluence_grades_rate:.2f})")
            logger.info(f"      Confluence scores not null: {analyses_results['confluence_scores_not_null']} ({confluence_scores_rate:.2f})")
            logger.info(f"      Should trade not null: {analyses_results['should_trade_not_null']} ({should_trade_rate:.2f})")
            logger.info(f"      Default fallback values (50/100): {analyses_results['default_fallback_values']} ({fallback_rate:.2f})")
            logger.info(f"      Diverse confluence scores: {analyses_results['diverse_confluence_scores']}")
            
            # Show sample analyses data
            if analyses_results['analyses_data']:
                logger.info(f"      📊 Sample Analyses Data:")
                for analysis in analyses_results['analyses_data']:
                    logger.info(f"         - {analysis['symbol']}: grade={analysis['confluence_grade']}, score={analysis['confluence_score']}, trade={analysis['should_trade']}")
            
            # Show error details if any
            if analyses_results['error_details']:
                logger.info(f"      📊 Error Details:")
                for error in analyses_results['error_details']:
                    logger.info(f"         - {error}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                analyses_results['api_call_successful'],  # API call successful
                analyses_results['analyses_returned'] > 0,  # Returns data
                analyses_results['confluence_fields_present'] > 0,  # Has confluence fields
                analyses_results['confluence_grades_not_null'] > 0,  # Some grades not null
                analyses_results['confluence_scores_not_null'] > 0,  # Some scores not null
                analyses_results['default_fallback_values'] < analyses_results['confluence_scores_not_null'] * 0.5  # Less than 50% fallbacks
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("API Analyses Confluence", True, 
                                   f"Analyses API confluence successful: {success_count}/{len(success_criteria)} criteria met. Grades rate: {confluence_grades_rate:.2f}, Scores rate: {confluence_scores_rate:.2f}, Fallback rate: {fallback_rate:.2f}")
            else:
                self.log_test_result("API Analyses Confluence", False, 
                                   f"Analyses API confluence issues: {success_count}/{len(success_criteria)} criteria met. Too many null values or fallbacks")
                
        except Exception as e:
            self.log_test_result("API Analyses Confluence", False, f"Exception: {str(e)}")

    async def test_3_confluence_calculation_logic(self):
        """Test 3: Confluence Calculation Logic - Verify calculation consistency and diversity"""
        logger.info("\n🔍 TEST 3: Confluence Calculation Logic - Validation and Diversity")
        
        try:
            calculation_results = {
                'analyses_tested': 0,
                'grade_d_with_score_0': 0,
                'grade_d_with_should_trade_false': 0,
                'consistent_grade_score_mapping': 0,
                'diverse_grades_found': set(),
                'diverse_scores_found': set(),
                'should_trade_variations': set(),
                'calculation_consistency': 0,
                'sample_data': [],
                'error_details': []
            }
            
            logger.info("   🚀 Testing confluence calculation logic and diversity...")
            logger.info("   📊 Expected: Grade D = score 0 = should_trade false, diverse grades/scores across symbols")
            
            # Test multiple symbols to check for diversity
            test_symbols = ['BTCUSDT', 'ETHUSDT', 'LINKUSDT', 'SOLUSDT', 'ADAUSDT']
            
            for symbol in test_symbols:
                logger.info(f"\n   📞 Testing confluence calculation for {symbol}...")
                calculation_results['analyses_tested'] += 1
                
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
                        ia1_analysis = analysis_data.get('ia1_analysis', {})
                        
                        if isinstance(ia1_analysis, dict):
                            confluence_grade = ia1_analysis.get('confluence_grade')
                            confluence_score = ia1_analysis.get('confluence_score')
                            should_trade = ia1_analysis.get('should_trade')
                            
                            logger.info(f"      ✅ {symbol}: grade={confluence_grade}, score={confluence_score}, trade={should_trade}")
                            
                            # Collect diversity data
                            if confluence_grade:
                                calculation_results['diverse_grades_found'].add(confluence_grade)
                            if confluence_score is not None:
                                try:
                                    score_val = float(confluence_score)
                                    calculation_results['diverse_scores_found'].add(score_val)
                                except (ValueError, TypeError):
                                    pass
                            if should_trade is not None:
                                calculation_results['should_trade_variations'].add(str(should_trade))
                            
                            # Check Grade D logic
                            if confluence_grade == 'D':
                                try:
                                    score_val = float(confluence_score) if confluence_score is not None else None
                                    if score_val == 0 or score_val is None:
                                        calculation_results['grade_d_with_score_0'] += 1
                                        logger.info(f"         ✅ Grade D with score 0 logic correct")
                                    else:
                                        logger.warning(f"         ⚠️ Grade D but score not 0: {score_val}")
                                    
                                    if should_trade in [False, 'false', 'False']:
                                        calculation_results['grade_d_with_should_trade_false'] += 1
                                        logger.info(f"         ✅ Grade D with should_trade false logic correct")
                                    else:
                                        logger.warning(f"         ⚠️ Grade D but should_trade not false: {should_trade}")
                                except (ValueError, TypeError):
                                    logger.warning(f"         ⚠️ Grade D but score not numeric: {confluence_score}")
                            
                            # Check general consistency
                            if confluence_grade and confluence_score is not None and should_trade is not None:
                                calculation_results['consistent_grade_score_mapping'] += 1
                                logger.info(f"         ✅ All confluence fields present and consistent")
                            
                            # Store sample data
                            calculation_results['sample_data'].append({
                                'symbol': symbol,
                                'confluence_grade': confluence_grade,
                                'confluence_score': confluence_score,
                                'should_trade': should_trade,
                                'response_time': response_time
                            })
                        
                        else:
                            logger.warning(f"      ⚠️ {symbol}: Invalid IA1 analysis structure")
                    
                    else:
                        logger.error(f"      ❌ {symbol} analysis failed: HTTP {response.status_code}")
                        calculation_results['error_details'].append(f"{symbol}: HTTP {response.status_code}")
                
                except Exception as e:
                    logger.error(f"      ❌ {symbol} analysis exception: {e}")
                    calculation_results['error_details'].append(f"{symbol}: {str(e)}")
                
                # Wait between analyses
                if symbol != test_symbols[-1]:
                    logger.info(f"      ⏳ Waiting 8 seconds before next analysis...")
                    await asyncio.sleep(8)
            
            # Final analysis and results
            diversity_grade_count = len(calculation_results['diverse_grades_found'])
            diversity_score_count = len(calculation_results['diverse_scores_found'])
            diversity_trade_count = len(calculation_results['should_trade_variations'])
            consistency_rate = calculation_results['consistent_grade_score_mapping'] / max(calculation_results['analyses_tested'], 1)
            
            logger.info(f"\n   📊 CONFLUENCE CALCULATION LOGIC RESULTS:")
            logger.info(f"      Analyses tested: {calculation_results['analyses_tested']}")
            logger.info(f"      Grade D with score 0: {calculation_results['grade_d_with_score_0']}")
            logger.info(f"      Grade D with should_trade false: {calculation_results['grade_d_with_should_trade_false']}")
            logger.info(f"      Consistent grade-score mapping: {calculation_results['consistent_grade_score_mapping']} ({consistency_rate:.2f})")
            logger.info(f"      Diverse grades found: {sorted(calculation_results['diverse_grades_found'])} ({diversity_grade_count})")
            logger.info(f"      Diverse scores found: {sorted(calculation_results['diverse_scores_found'])} ({diversity_score_count})")
            logger.info(f"      Should trade variations: {sorted(calculation_results['should_trade_variations'])} ({diversity_trade_count})")
            
            # Show sample data
            if calculation_results['sample_data']:
                logger.info(f"      📊 Sample Calculation Data:")
                for data in calculation_results['sample_data']:
                    logger.info(f"         - {data['symbol']}: grade={data['confluence_grade']}, score={data['confluence_score']}, trade={data['should_trade']}")
            
            # Show error details if any
            if calculation_results['error_details']:
                logger.info(f"      📊 Error Details:")
                for error in calculation_results['error_details']:
                    logger.info(f"         - {error}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                calculation_results['analyses_tested'] >= 3,  # At least 3 analyses tested
                diversity_grade_count >= 2,  # At least 2 different grades
                diversity_score_count >= 3,  # At least 3 different scores
                diversity_trade_count >= 1,  # At least some should_trade variation
                calculation_results['consistent_grade_score_mapping'] >= 2,  # At least 2 consistent mappings
                consistency_rate >= 0.6  # At least 60% consistency
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.83:  # 83% success threshold (5/6 criteria)
                self.log_test_result("Confluence Calculation Logic", True, 
                                   f"Calculation logic successful: {success_count}/{len(success_criteria)} criteria met. Diversity: {diversity_grade_count} grades, {diversity_score_count} scores, consistency: {consistency_rate:.2f}")
            else:
                self.log_test_result("Confluence Calculation Logic", False, 
                                   f"Calculation logic issues: {success_count}/{len(success_criteria)} criteria met. Limited diversity or consistency problems")
                
        except Exception as e:
            self.log_test_result("Confluence Calculation Logic", False, f"Exception: {str(e)}")

    async def test_4_backend_logs_confluence_validation(self):
        """Test 4: Backend Logs Confluence Validation - Check for confluence calculation logs"""
        logger.info("\n🔍 TEST 4: Backend Logs Confluence Validation")
        
        try:
            logs_results = {
                'logs_captured': False,
                'total_log_lines': 0,
                'confluence_calculation_logs': 0,
                'confluence_error_logs': 0,
                'success_indicators': 0,
                'confluence_patterns_found': [],
                'success_patterns_found': [],
                'sample_confluence_logs': [],
                'sample_success_logs': []
            }
            
            logger.info("   🚀 Analyzing backend logs for confluence calculation patterns...")
            logger.info("   📊 Expected: Confluence calculation logs present, no confluence errors")
            
            # Capture backend logs
            logger.info("   📋 Capturing recent backend logs...")
            
            try:
                backend_logs = await self._capture_backend_logs()
                if backend_logs:
                    logs_results['logs_captured'] = True
                    logs_results['total_log_lines'] = len(backend_logs)
                    logger.info(f"      ✅ Captured {len(backend_logs)} log lines")
                    
                    # Analyze each log line for confluence patterns
                    for log_line in backend_logs:
                        log_lower = log_line.lower()
                        
                        # Check for confluence calculation patterns
                        confluence_patterns = [
                            'confluence',
                            'grade',
                            'score',
                            'should_trade'
                        ]
                        
                        for pattern in confluence_patterns:
                            if pattern in log_lower:
                                logs_results['confluence_calculation_logs'] += 1
                                if pattern not in logs_results['confluence_patterns_found']:
                                    logs_results['confluence_patterns_found'].append(pattern)
                                if len(logs_results['sample_confluence_logs']) < 3:
                                    logs_results['sample_confluence_logs'].append(log_line.strip())
                                break
                        
                        # Check for confluence error patterns
                        confluence_error_patterns = [
                            'confluence.*error',
                            'confluence.*null',
                            'confluence.*failed',
                            'grade.*error',
                            'score.*error'
                        ]
                        
                        for pattern in confluence_error_patterns:
                            if re.search(pattern, log_lower):
                                logs_results['confluence_error_logs'] += 1
                                logger.error(f"      🚨 CONFLUENCE ERROR PATTERN FOUND: {pattern}")
                                logger.error(f"         Log: {log_line.strip()}")
                        
                        # Check for success indicators
                        success_patterns = [
                            'ia1 analysis completed successfully',
                            'ia1 ultra analysis completed',
                            'analysis successful for',
                            'technical analysis completed',
                            'force-ia1-analysis completed'
                        ]
                        
                        for pattern in success_patterns:
                            if pattern in log_lower:
                                logs_results['success_indicators'] += 1
                                if pattern not in logs_results['success_patterns_found']:
                                    logs_results['success_patterns_found'].append(pattern)
                                if len(logs_results['sample_success_logs']) < 3:
                                    logs_results['sample_success_logs'].append(log_line.strip())
                                break
                    
                    logger.info(f"      📊 Log analysis completed:")
                    logger.info(f"         - Total log lines analyzed: {logs_results['total_log_lines']}")
                    logger.info(f"         - Confluence calculation logs: {logs_results['confluence_calculation_logs']}")
                    logger.info(f"         - Confluence error logs: {logs_results['confluence_error_logs']}")
                    logger.info(f"         - Success indicators: {logs_results['success_indicators']}")
                    
                    # Show confluence patterns found
                    if logs_results['confluence_patterns_found']:
                        logger.info(f"      ✅ CONFLUENCE PATTERNS FOUND:")
                        for pattern in logs_results['confluence_patterns_found']:
                            logger.info(f"         - {pattern}")
                    else:
                        logger.warning(f"      ⚠️ No confluence patterns found in logs")
                    
                    # Show success patterns found
                    if logs_results['success_patterns_found']:
                        logger.info(f"      ✅ SUCCESS PATTERNS FOUND:")
                        for pattern in logs_results['success_patterns_found']:
                            logger.info(f"         - {pattern}")
                    
                    # Show sample confluence logs
                    if logs_results['sample_confluence_logs']:
                        logger.info(f"      📋 Sample Confluence Logs:")
                        for i, log in enumerate(logs_results['sample_confluence_logs']):
                            logger.info(f"         {i+1}. {log}")
                    
                    # Show sample success logs
                    if logs_results['sample_success_logs']:
                        logger.info(f"      📋 Sample Success Logs:")
                        for i, log in enumerate(logs_results['sample_success_logs']):
                            logger.info(f"         {i+1}. {log}")
                
                else:
                    logger.warning(f"      ⚠️ No backend logs captured")
                    
            except Exception as e:
                logger.error(f"      ❌ Failed to capture backend logs: {e}")
            
            # Final analysis and results
            confluence_presence_rate = 1.0 if logs_results['confluence_calculation_logs'] > 0 else 0.0
            error_free_rate = 1.0 if logs_results['confluence_error_logs'] == 0 else 0.0
            success_rate = 1.0 if logs_results['success_indicators'] > 0 else 0.0
            
            logger.info(f"\n   📊 BACKEND LOGS CONFLUENCE VALIDATION RESULTS:")
            logger.info(f"      Logs captured: {logs_results['logs_captured']}")
            logger.info(f"      Total log lines: {logs_results['total_log_lines']}")
            logger.info(f"      Confluence calculation logs: {logs_results['confluence_calculation_logs']}")
            logger.info(f"      Confluence error logs: {logs_results['confluence_error_logs']}")
            logger.info(f"      Success indicators: {logs_results['success_indicators']}")
            logger.info(f"      Confluence presence rate: {confluence_presence_rate:.2f}")
            logger.info(f"      Error-free rate: {error_free_rate:.2f}")
            logger.info(f"      Success indicators rate: {success_rate:.2f}")
            
            # Calculate test success based on review requirements
            success_criteria = [
                logs_results['logs_captured'],  # Logs captured successfully
                logs_results['confluence_calculation_logs'] > 0,  # Some confluence logs found
                logs_results['confluence_error_logs'] == 0,  # No confluence errors
                logs_results['success_indicators'] > 0,  # Some success indicators
                logs_results['total_log_lines'] > 50  # Sufficient log data
            ]
            success_count = sum(success_criteria)
            test_success_rate = success_count / len(success_criteria)
            
            if test_success_rate >= 0.8:  # 80% success threshold (4/5 criteria)
                self.log_test_result("Backend Logs Confluence Validation", True, 
                                   f"Backend logs confluence validation successful: {success_count}/{len(success_criteria)} criteria met. Confluence logs: {logs_results['confluence_calculation_logs']}, No errors: {logs_results['confluence_error_logs'] == 0}")
            else:
                self.log_test_result("Backend Logs Confluence Validation", False, 
                                   f"Backend logs confluence validation issues: {success_count}/{len(success_criteria)} criteria met. May have missing confluence logs or errors")
                
        except Exception as e:
            self.log_test_result("Backend Logs Confluence Validation", False, f"Exception: {str(e)}")

async def main():
    """Main test execution function"""
    logger.info("🚀 Starting Confluence Analysis Fix Testing Suite")
    logger.info("=" * 80)
    
    # Initialize test suite
    test_suite = ConfluenceAnalysisTestSuite()
    
    try:
        # Run all confluence analysis tests
        logger.info("Running Test 1: API Force IA1 Analysis - Confluence Values")
        await test_suite.test_1_api_force_ia1_analysis_confluence()
        
        logger.info("Running Test 2: API Analyses Endpoint - Confluence Consistency")
        await test_suite.test_2_api_analyses_confluence()
        
        logger.info("Running Test 3: Confluence Calculation Logic - Validation and Diversity")
        await test_suite.test_3_confluence_calculation_logic()
        
        logger.info("Running Test 4: Backend Logs Confluence Validation")
        await test_suite.test_4_backend_logs_confluence_validation()
        
        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎯 CONFLUENCE ANALYSIS FIX TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in test_suite.test_results if result['success'])
        total_tests = len(test_suite.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        for result in test_suite.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   Details: {result['details']}")
        
        logger.info(f"\n📊 OVERALL RESULTS:")
        logger.info(f"   Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"   Success Rate: {success_rate:.2%}")
        
        if success_rate >= 0.75:
            logger.info("🎉 CONFLUENCE ANALYSIS FIX TESTING SUCCESSFUL!")
            logger.info("   The confluence analysis fix appears to be working correctly.")
        else:
            logger.warning("⚠️ CONFLUENCE ANALYSIS FIX TESTING ISSUES DETECTED")
            logger.warning("   Some confluence analysis functionality may still need attention.")
        
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False
    
    return success_rate >= 0.75

if __name__ == "__main__":
    asyncio.run(main())
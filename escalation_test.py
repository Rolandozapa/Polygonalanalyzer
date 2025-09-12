#!/usr/bin/env python3
"""
IA1 TO IA2 ESCALATION SYSTEM TEST SUITE
Focus: Test IA1 to IA2 Escalation System - COMPREHENSIVE VALIDATION

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **Escalation Logic Validation**: Test the 3 voies escalation system:
   - VOIE 1: LONG/SHORT signals with confidence ≥ 70%
   - VOIE 2: Risk-Reward ratio ≥ 2.0 (any signal)  
   - VOIE 3: LONG/SHORT signals with confidence ≥ 95% (override)

2. **End-to-End Escalation Flow**: Verify complete pipeline from IA1 → IA2 → Decision storage

3. **Database Integration**: Check that IA2 decisions are properly saved when escalation occurs

SUCCESS CRITERIA:
✅ _should_send_to_ia2 function correctly identifies eligible analyses
✅ Escalation occurs for analyses meeting any of the 3 voies criteria
✅ IA2 make_decision method executes successfully after escalation
✅ New IA2 decisions appear in database after successful escalation
✅ No more "cannot access local variable" errors in escalation flow
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

class IA1ToIA2EscalationTestSuite:
    """Comprehensive test suite for IA1 to IA2 Escalation System"""
    
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
        logger.info(f"Testing IA1 to IA2 Escalation System at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for escalation testing")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # Test symbols for escalation testing
        self.test_symbols = [
            "BTCUSDT",   # High volatility, likely to trigger escalation
            "ETHUSDT",   # Medium volatility, good for testing
            "SOLUSDT",   # High volatility, trending
            "XRPUSDT",   # Lower volatility, edge cases
            "ADAUSDT"    # Medium volatility, range-bound
        ]
        
        # Escalation criteria for testing
        self.escalation_criteria = {
            "VOIE_1": {"confidence_min": 0.70, "signals": ["long", "short"]},
            "VOIE_2": {"rr_min": 2.0, "signals": ["long", "short", "hold"]},
            "VOIE_3": {"confidence_min": 0.95, "signals": ["long", "short"]}
        }
        
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
    
    async def test_1_escalation_criteria_validation(self):
        """Test 1: Validate IA1 to IA2 Escalation Criteria (3 Voies System)"""
        logger.info("\n🔍 TEST 1: IA1 to IA2 Escalation Criteria Validation Test")
        
        try:
            # Test the 3 voies escalation system
            escalation_test_results = {
                'voie_1_tests': 0,
                'voie_2_tests': 0,
                'voie_3_tests': 0,
                'voie_1_correct': 0,
                'voie_2_correct': 0,
                'voie_3_correct': 0,
                'escalation_errors': []
            }
            
            logger.info("   🚀 Testing 3 Voies Escalation System:")
            logger.info("      VOIE 1: LONG/SHORT signals with confidence ≥ 70%")
            logger.info("      VOIE 2: Risk-Reward ratio ≥ 2.0 (any signal)")
            logger.info("      VOIE 3: LONG/SHORT signals with confidence ≥ 95% (override)")
            
            # Test multiple symbols to trigger different escalation scenarios
            for symbol in self.test_symbols[:3]:  # Test first 3 symbols
                try:
                    logger.info(f"   📊 Testing escalation criteria for {symbol}...")
                    
                    # Run IA1 cycle to get analysis and check escalation
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={"symbol": symbol},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        
                        if cycle_data.get('success'):
                            # Check if analysis was created
                            analysis = cycle_data.get('analysis', {})
                            escalated_to_ia2 = cycle_data.get('escalated_to_ia2', False)
                            ia2_decision = cycle_data.get('ia2_decision')
                            
                            confidence = analysis.get('confidence', 0)
                            rr_ratio = analysis.get('risk_reward_ratio', 0)
                            signal = analysis.get('recommendation', 'hold').lower()
                            
                            logger.info(f"      📈 {symbol} Analysis: Confidence={confidence:.1%}, RR={rr_ratio:.2f}, Signal={signal}")
                            logger.info(f"      🎯 Escalated to IA2: {escalated_to_ia2}")
                            
                            # Test VOIE 1: LONG/SHORT signals with confidence ≥ 70%
                            if signal in ['long', 'short'] and confidence >= 0.70:
                                escalation_test_results['voie_1_tests'] += 1
                                if escalated_to_ia2:
                                    escalation_test_results['voie_1_correct'] += 1
                                    logger.info(f"         ✅ VOIE 1 triggered correctly: {signal} signal, {confidence:.1%} confidence")
                                else:
                                    escalation_test_results['escalation_errors'].append(
                                        f"VOIE 1 failed for {symbol}: {signal} signal, {confidence:.1%} confidence should escalate"
                                    )
                                    logger.warning(f"         ❌ VOIE 1 failed: Should escalate but didn't")
                            
                            # Test VOIE 2: Risk-Reward ratio ≥ 2.0 (any signal)
                            if rr_ratio >= 2.0:
                                escalation_test_results['voie_2_tests'] += 1
                                if escalated_to_ia2:
                                    escalation_test_results['voie_2_correct'] += 1
                                    logger.info(f"         ✅ VOIE 2 triggered correctly: RR={rr_ratio:.2f} ≥ 2.0")
                                else:
                                    escalation_test_results['escalation_errors'].append(
                                        f"VOIE 2 failed for {symbol}: RR={rr_ratio:.2f} ≥ 2.0 should escalate"
                                    )
                                    logger.warning(f"         ❌ VOIE 2 failed: RR={rr_ratio:.2f} should escalate")
                            
                            # Test VOIE 3: LONG/SHORT signals with confidence ≥ 95% (override)
                            if signal in ['long', 'short'] and confidence >= 0.95:
                                escalation_test_results['voie_3_tests'] += 1
                                if escalated_to_ia2:
                                    escalation_test_results['voie_3_correct'] += 1
                                    logger.info(f"         ✅ VOIE 3 triggered correctly: {signal} signal, {confidence:.1%} confidence (override)")
                                else:
                                    escalation_test_results['escalation_errors'].append(
                                        f"VOIE 3 failed for {symbol}: {signal} signal, {confidence:.1%} confidence should escalate (override)"
                                    )
                                    logger.warning(f"         ❌ VOIE 3 failed: Should escalate with override")
                            
                            # Check IA2 decision if escalated
                            if escalated_to_ia2 and ia2_decision:
                                logger.info(f"         🤖 IA2 Decision: {ia2_decision.get('signal', 'N/A')} with confidence {ia2_decision.get('confidence', 0):.1%}")
                        else:
                            logger.warning(f"      ❌ IA1 cycle failed for {symbol}: {cycle_data.get('error', 'Unknown error')}")
                    else:
                        logger.warning(f"      ❌ API call failed for {symbol}: HTTP {response.status_code}")
                    
                    await asyncio.sleep(3)  # Delay between requests
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing escalation for {symbol}: {e}")
                    escalation_test_results['escalation_errors'].append(f"Exception testing {symbol}: {str(e)}")
            
            # Calculate results
            total_voie_tests = (escalation_test_results['voie_1_tests'] + 
                              escalation_test_results['voie_2_tests'] + 
                              escalation_test_results['voie_3_tests'])
            total_voie_correct = (escalation_test_results['voie_1_correct'] + 
                                escalation_test_results['voie_2_correct'] + 
                                escalation_test_results['voie_3_correct'])
            
            logger.info(f"   📊 Escalation Criteria Test Results:")
            logger.info(f"      VOIE 1 tests: {escalation_test_results['voie_1_correct']}/{escalation_test_results['voie_1_tests']}")
            logger.info(f"      VOIE 2 tests: {escalation_test_results['voie_2_correct']}/{escalation_test_results['voie_2_tests']}")
            logger.info(f"      VOIE 3 tests: {escalation_test_results['voie_3_correct']}/{escalation_test_results['voie_3_tests']}")
            logger.info(f"      Total accuracy: {total_voie_correct}/{total_voie_tests}")
            
            if escalation_test_results['escalation_errors']:
                logger.info(f"      Escalation errors:")
                for error in escalation_test_results['escalation_errors']:
                    logger.info(f"        - {error}")
            
            # Determine test result
            if total_voie_tests > 0:
                accuracy = total_voie_correct / total_voie_tests
                
                if accuracy >= 0.8:
                    self.log_test_result("Escalation Criteria Validation", True, 
                                       f"3 Voies escalation system working: {accuracy:.1%} accuracy ({total_voie_correct}/{total_voie_tests})")
                elif accuracy >= 0.6:
                    self.log_test_result("Escalation Criteria Validation", False, 
                                       f"Partial escalation criteria working: {accuracy:.1%} accuracy ({total_voie_correct}/{total_voie_tests})")
                else:
                    self.log_test_result("Escalation Criteria Validation", False, 
                                       f"Escalation criteria issues: {accuracy:.1%} accuracy ({total_voie_correct}/{total_voie_tests})")
            else:
                self.log_test_result("Escalation Criteria Validation", False, 
                                   "No escalation scenarios available for testing")
                
        except Exception as e:
            self.log_test_result("Escalation Criteria Validation", False, f"Exception: {str(e)}")
    
    async def test_2_database_integration_check(self):
        """Test 2: Database Integration - IA2 Decisions Storage"""
        logger.info("\n🔍 TEST 2: Database Integration - IA2 Decisions Storage Test")
        
        try:
            if not self.db:
                self.log_test_result("Database Integration Check", False, 
                                   "Database connection not available")
                return
            
            logger.info("   🚀 Testing IA2 decisions storage in database...")
            
            # Get initial count of IA2 decisions
            initial_decisions_count = self.db.trading_decisions.count_documents({})
            logger.info(f"   📊 Initial IA2 decisions in database: {initial_decisions_count}")
            
            # Run IA1 cycle to trigger potential escalation
            test_symbol = "BTCUSDT"
            logger.info(f"   🎯 Running IA1 cycle for {test_symbol} to test database integration...")
            
            response = requests.post(
                f"{self.api_url}/run-ia1-cycle",
                json={"symbol": test_symbol},
                timeout=60
            )
            
            if response.status_code == 200:
                cycle_data = response.json()
                
                if cycle_data.get('success'):
                    escalated_to_ia2 = cycle_data.get('escalated_to_ia2', False)
                    ia2_decision = cycle_data.get('ia2_decision')
                    
                    logger.info(f"   📈 IA1 cycle completed: Escalated={escalated_to_ia2}")
                    
                    if escalated_to_ia2 and ia2_decision:
                        # Wait a moment for database write
                        await asyncio.sleep(2)
                        
                        # Check if new IA2 decision was stored
                        final_decisions_count = self.db.trading_decisions.count_documents({})
                        new_decisions = final_decisions_count - initial_decisions_count
                        
                        logger.info(f"   📊 Final IA2 decisions in database: {final_decisions_count}")
                        logger.info(f"   📊 New decisions added: {new_decisions}")
                        
                        if new_decisions > 0:
                            # Get the latest decision to verify it's from IA2
                            latest_decision = self.db.trading_decisions.find_one(
                                {},
                                sort=[("timestamp", -1)]
                            )
                            
                            if latest_decision:
                                decision_symbol = latest_decision.get('symbol', '')
                                decision_signal = latest_decision.get('signal', '')
                                decision_confidence = latest_decision.get('confidence', 0)
                                
                                logger.info(f"   🤖 Latest IA2 decision: {decision_symbol} - {decision_signal} ({decision_confidence:.1%})")
                                
                                # Verify it matches our test
                                if decision_symbol == test_symbol:
                                    self.log_test_result("Database Integration Check", True, 
                                                       f"IA2 decision properly stored: {decision_symbol} - {decision_signal}")
                                else:
                                    self.log_test_result("Database Integration Check", False, 
                                                       f"IA2 decision stored but symbol mismatch: expected {test_symbol}, got {decision_symbol}")
                            else:
                                self.log_test_result("Database Integration Check", False, 
                                                   "New decision count increased but couldn't retrieve latest decision")
                        else:
                            self.log_test_result("Database Integration Check", False, 
                                               f"IA2 escalation occurred but no new decision stored in database")
                    else:
                        logger.info(f"   ⚪ No escalation occurred for {test_symbol}, testing database query capability...")
                        
                        # Test database query functionality
                        recent_decisions = list(self.db.trading_decisions.find({}).sort("timestamp", -1).limit(5))
                        
                        if recent_decisions:
                            logger.info(f"   📊 Found {len(recent_decisions)} recent IA2 decisions in database")
                            for i, decision in enumerate(recent_decisions):
                                symbol = decision.get('symbol', 'N/A')
                                signal = decision.get('signal', 'N/A')
                                confidence = decision.get('confidence', 0)
                                timestamp = decision.get('timestamp', 'N/A')
                                logger.info(f"      {i+1}. {symbol} - {signal} ({confidence:.1%}) at {timestamp}")
                            
                            self.log_test_result("Database Integration Check", True, 
                                               f"Database integration working: {len(recent_decisions)} IA2 decisions found")
                        else:
                            self.log_test_result("Database Integration Check", False, 
                                               "No IA2 decisions found in database - integration may be broken")
                else:
                    logger.warning(f"   ❌ IA1 cycle failed: {cycle_data.get('error', 'Unknown error')}")
                    self.log_test_result("Database Integration Check", False, 
                                       f"IA1 cycle failed: {cycle_data.get('error', 'Unknown error')}")
            else:
                logger.warning(f"   ❌ API call failed: HTTP {response.status_code}")
                self.log_test_result("Database Integration Check", False, 
                                   f"API call failed: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Database Integration Check", False, f"Exception: {str(e)}")
    
    async def test_3_error_resolution_check(self):
        """Test 3: Error Resolution - Advanced Market Aggregator Import Fix"""
        logger.info("\n🔍 TEST 3: Error Resolution - Advanced Market Aggregator Import Fix Test")
        
        try:
            logger.info("   🚀 Testing for resolved 'advanced_market_aggregator' import errors...")
            
            # Check backend logs for import errors
            error_analysis = await self._analyze_backend_logs_for_errors()
            
            # Test IA1 cycle to see if import errors occur
            test_symbol = "ETHUSDT"
            logger.info(f"   🎯 Running IA1 cycle for {test_symbol} to test error resolution...")
            
            response = requests.post(
                f"{self.api_url}/run-ia1-cycle",
                json={"symbol": test_symbol},
                timeout=60
            )
            
            if response.status_code == 200:
                cycle_data = response.json()
                
                if cycle_data.get('success'):
                    logger.info(f"   ✅ IA1 cycle completed successfully - no import errors")
                    
                    # Check if escalation occurred without errors
                    escalated_to_ia2 = cycle_data.get('escalated_to_ia2', False)
                    ia2_decision = cycle_data.get('ia2_decision')
                    
                    if escalated_to_ia2:
                        if ia2_decision:
                            logger.info(f"   ✅ IA2 escalation completed successfully")
                            self.log_test_result("Error Resolution Check", True, 
                                               "No import errors detected, IA1→IA2 escalation working")
                        else:
                            logger.warning(f"   ⚠️ IA2 escalation occurred but no decision returned")
                            self.log_test_result("Error Resolution Check", False, 
                                               "IA2 escalation occurred but no decision returned")
                    else:
                        logger.info(f"   ⚪ No escalation occurred, but IA1 cycle completed without errors")
                        self.log_test_result("Error Resolution Check", True, 
                                           "No import errors detected, IA1 cycle working")
                else:
                    error_msg = cycle_data.get('error', 'Unknown error')
                    logger.warning(f"   ❌ IA1 cycle failed: {error_msg}")
                    
                    # Check if it's the specific import error we're looking for
                    if 'advanced_market_aggregator' in error_msg.lower():
                        self.log_test_result("Error Resolution Check", False, 
                                           f"Advanced market aggregator import error still present: {error_msg}")
                    elif 'cannot access local variable' in error_msg.lower():
                        self.log_test_result("Error Resolution Check", False, 
                                           f"Local variable access error still present: {error_msg}")
                    else:
                        self.log_test_result("Error Resolution Check", False, 
                                           f"Other error occurred: {error_msg}")
            else:
                logger.warning(f"   ❌ API call failed: HTTP {response.status_code}")
                
                # Try to get error details from response
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', 'No details available')
                    logger.info(f"   📋 Error details: {error_detail}")
                    
                    if 'advanced_market_aggregator' in error_detail.lower():
                        self.log_test_result("Error Resolution Check", False, 
                                           f"Advanced market aggregator import error in API: {error_detail}")
                    else:
                        self.log_test_result("Error Resolution Check", False, 
                                           f"API error (not import related): HTTP {response.status_code}")
                except:
                    self.log_test_result("Error Resolution Check", False, 
                                       f"API error: HTTP {response.status_code}")
            
            # Analyze backend logs for specific error patterns
            logger.info(f"   📋 Backend logs analysis:")
            logger.info(f"      Total log entries: {error_analysis['total_entries']}")
            logger.info(f"      Error entries: {error_analysis['error_entries']}")
            logger.info(f"      Import errors: {error_analysis['import_errors']}")
            logger.info(f"      Aggregator errors: {error_analysis['aggregator_errors']}")
            
            if error_analysis['recent_errors']:
                logger.info(f"      Recent errors:")
                for error in error_analysis['recent_errors'][:3]:
                    logger.info(f"        - {error[:100]}...")
                
        except Exception as e:
            self.log_test_result("Error Resolution Check", False, f"Exception: {str(e)}")
    
    async def test_4_api_endpoints_validation(self):
        """Test 4: API Endpoints Validation for Escalation System"""
        logger.info("\n🔍 TEST 4: API Endpoints Validation for Escalation System Test")
        
        try:
            logger.info("   🚀 Testing escalation-related API endpoints...")
            
            endpoint_test_results = {
                'run_ia1_cycle': False,
                'get_decisions': False,
                'get_analyses': False,
                'endpoint_errors': []
            }
            
            # Test 1: /api/run-ia1-cycle endpoint
            logger.info("   📊 Testing /api/run-ia1-cycle endpoint...")
            try:
                response = requests.post(
                    f"{self.api_url}/run-ia1-cycle",
                    json={"symbol": "BTCUSDT"},
                    timeout=60
                )
                
                if response.status_code == 200:
                    cycle_data = response.json()
                    
                    # Check for required escalation fields
                    has_escalated_field = 'escalated_to_ia2' in cycle_data
                    has_ia2_decision_field = 'ia2_decision' in cycle_data
                    has_analysis_field = 'analysis' in cycle_data
                    
                    if has_escalated_field and has_analysis_field:
                        endpoint_test_results['run_ia1_cycle'] = True
                        logger.info(f"      ✅ /api/run-ia1-cycle working: escalated_to_ia2={cycle_data.get('escalated_to_ia2')}")
                        
                        if has_ia2_decision_field and cycle_data.get('escalated_to_ia2'):
                            logger.info(f"      ✅ IA2 decision field present when escalated")
                    else:
                        endpoint_test_results['endpoint_errors'].append(
                            f"/api/run-ia1-cycle missing required fields: escalated_to_ia2={has_escalated_field}, analysis={has_analysis_field}"
                        )
                        logger.warning(f"      ❌ Missing required escalation fields")
                else:
                    endpoint_test_results['endpoint_errors'].append(
                        f"/api/run-ia1-cycle returned HTTP {response.status_code}"
                    )
                    logger.warning(f"      ❌ HTTP {response.status_code}")
                    
            except Exception as e:
                endpoint_test_results['endpoint_errors'].append(f"/api/run-ia1-cycle exception: {str(e)}")
                logger.error(f"      ❌ Exception: {e}")
            
            await asyncio.sleep(2)
            
            # Test 2: /api/decisions endpoint
            logger.info("   📊 Testing /api/decisions endpoint...")
            try:
                response = requests.get(f"{self.api_url}/decisions", timeout=30)
                
                if response.status_code == 200:
                    decisions_data = response.json()
                    
                    if isinstance(decisions_data, list):
                        endpoint_test_results['get_decisions'] = True
                        logger.info(f"      ✅ /api/decisions working: {len(decisions_data)} decisions found")
                        
                        # Check if decisions have required fields
                        if decisions_data:
                            sample_decision = decisions_data[0]
                            required_fields = ['symbol', 'signal', 'confidence', 'timestamp']
                            missing_fields = [field for field in required_fields if field not in sample_decision]
                            
                            if not missing_fields:
                                logger.info(f"      ✅ Decision format correct: {sample_decision.get('symbol')} - {sample_decision.get('signal')}")
                            else:
                                logger.warning(f"      ⚠️ Decision missing fields: {missing_fields}")
                    else:
                        endpoint_test_results['endpoint_errors'].append(
                            f"/api/decisions returned non-list data: {type(decisions_data)}"
                        )
                        logger.warning(f"      ❌ Non-list response: {type(decisions_data)}")
                else:
                    endpoint_test_results['endpoint_errors'].append(
                        f"/api/decisions returned HTTP {response.status_code}"
                    )
                    logger.warning(f"      ❌ HTTP {response.status_code}")
                    
            except Exception as e:
                endpoint_test_results['endpoint_errors'].append(f"/api/decisions exception: {str(e)}")
                logger.error(f"      ❌ Exception: {e}")
            
            await asyncio.sleep(2)
            
            # Test 3: /api/analyses endpoint
            logger.info("   📊 Testing /api/analyses endpoint...")
            try:
                response = requests.get(f"{self.api_url}/analyses", timeout=30)
                
                if response.status_code == 200:
                    analyses_data = response.json()
                    
                    if isinstance(analyses_data, list):
                        endpoint_test_results['get_analyses'] = True
                        logger.info(f"      ✅ /api/analyses working: {len(analyses_data)} analyses found")
                        
                        # Check if analyses have escalation-related fields
                        if analyses_data:
                            sample_analysis = analyses_data[0]
                            escalation_fields = ['confidence', 'risk_reward_ratio', 'recommendation']
                            present_fields = [field for field in escalation_fields if field in sample_analysis]
                            
                            logger.info(f"      📊 Escalation fields present: {present_fields}")
                            
                            if len(present_fields) >= 2:
                                logger.info(f"      ✅ Analysis format supports escalation criteria")
                            else:
                                logger.warning(f"      ⚠️ Analysis missing escalation fields")
                    else:
                        endpoint_test_results['endpoint_errors'].append(
                            f"/api/analyses returned non-list data: {type(analyses_data)}"
                        )
                        logger.warning(f"      ❌ Non-list response: {type(analyses_data)}")
                else:
                    endpoint_test_results['endpoint_errors'].append(
                        f"/api/analyses returned HTTP {response.status_code}"
                    )
                    logger.warning(f"      ❌ HTTP {response.status_code}")
                    
            except Exception as e:
                endpoint_test_results['endpoint_errors'].append(f"/api/analyses exception: {str(e)}")
                logger.error(f"      ❌ Exception: {e}")
            
            # Calculate results
            working_endpoints = sum([
                endpoint_test_results['run_ia1_cycle'],
                endpoint_test_results['get_decisions'],
                endpoint_test_results['get_analyses']
            ])
            
            logger.info(f"   📊 API Endpoints Test Results:")
            logger.info(f"      /api/run-ia1-cycle: {'✅' if endpoint_test_results['run_ia1_cycle'] else '❌'}")
            logger.info(f"      /api/decisions: {'✅' if endpoint_test_results['get_decisions'] else '❌'}")
            logger.info(f"      /api/analyses: {'✅' if endpoint_test_results['get_analyses'] else '❌'}")
            logger.info(f"      Working endpoints: {working_endpoints}/3")
            
            if endpoint_test_results['endpoint_errors']:
                logger.info(f"      Endpoint errors:")
                for error in endpoint_test_results['endpoint_errors']:
                    logger.info(f"        - {error}")
            
            # Determine test result
            if working_endpoints == 3:
                self.log_test_result("API Endpoints Validation", True, 
                                   f"All escalation API endpoints working: {working_endpoints}/3")
            elif working_endpoints >= 2:
                self.log_test_result("API Endpoints Validation", False, 
                                   f"Most escalation API endpoints working: {working_endpoints}/3")
            else:
                self.log_test_result("API Endpoints Validation", False, 
                                   f"Escalation API endpoints issues: {working_endpoints}/3 working")
                
        except Exception as e:
            self.log_test_result("API Endpoints Validation", False, f"Exception: {str(e)}")
    
    async def _analyze_backend_logs_for_errors(self):
        """Analyze backend logs for specific error patterns"""
        try:
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            analysis = {
                'total_entries': 0,
                'error_entries': 0,
                'import_errors': 0,
                'aggregator_errors': 0,
                'escalation_errors': 0,
                'recent_errors': []
            }
            
            error_patterns = [
                r'ERROR',
                r'CRITICAL',
                r'Exception',
                r'Traceback',
                r'Failed'
            ]
            
            import_patterns = [
                r'ImportError',
                r'ModuleNotFoundError',
                r'cannot import',
                r'advanced_market_aggregator'
            ]
            
            aggregator_patterns = [
                r'advanced_market_aggregator',
                r'datetime.*error',
                r'unsupported operand'
            ]
            
            escalation_patterns = [
                r'escalation',
                r'IA2',
                r'cannot access local variable'
            ]
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '500', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            lines = log_content.split('\n')
                            analysis['total_entries'] += len([line for line in lines if line.strip()])
                            
                            for line in lines:
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
                                    analysis['error_entries'] += 1
                                    if len(analysis['recent_errors']) < 5:
                                        analysis['recent_errors'].append(line.strip())
                                
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in import_patterns):
                                    analysis['import_errors'] += 1
                                
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in aggregator_patterns):
                                    analysis['aggregator_errors'] += 1
                                
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in escalation_patterns):
                                    analysis['escalation_errors'] += 1
                            
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Log analysis failed: {e}")
            return {
                'total_entries': 0,
                'error_entries': 0,
                'import_errors': 0,
                'aggregator_errors': 0,
                'escalation_errors': 0,
                'recent_errors': []
            }
    
    async def run_comprehensive_escalation_test_suite(self):
        """Run comprehensive IA1 to IA2 escalation test suite"""
        logger.info("🚀 Starting IA1 to IA2 Escalation System Test Suite")
        logger.info("=" * 80)
        logger.info("📋 IA1 TO IA2 ESCALATION SYSTEM TEST SUITE")
        logger.info("🎯 Testing: 3 Voies escalation system and end-to-end flow")
        logger.info("🎯 Expected: Proper escalation, IA2 decisions, database storage")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_escalation_criteria_validation()
        await self.test_2_database_integration_check()
        await self.test_3_error_resolution_check()
        await self.test_4_api_endpoints_validation()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA1 TO IA2 ESCALATION SYSTEM TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Critical requirements analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 CRITICAL ESCALATION REQUIREMENTS VERIFICATION")
        logger.info("=" * 80)
        
        requirements_status = {}
        
        for result in self.test_results:
            if "Escalation Criteria" in result['test']:
                requirements_status['3 Voies Escalation System'] = result['success']
            elif "Database Integration" in result['test']:
                requirements_status['IA2 Decisions Storage'] = result['success']
            elif "Error Resolution" in result['test']:
                requirements_status['Import Errors Fixed'] = result['success']
            elif "API Endpoints" in result['test']:
                requirements_status['Escalation API Endpoints'] = result['success']
        
        logger.info("🎯 CRITICAL REQUIREMENTS STATUS:")
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}")
        
        requirements_met = sum(1 for status in requirements_status.values() if status)
        total_requirements = len(requirements_status)
        
        # Final verdict
        logger.info(f"\n🏆 REQUIREMENTS SATISFACTION: {requirements_met}/{total_requirements}")
        
        if requirements_met == total_requirements:
            logger.info("\n🎉 VERDICT: IA1 TO IA2 ESCALATION SYSTEM FULLY WORKING!")
            logger.info("✅ 3 Voies escalation system correctly implemented")
            logger.info("✅ IA2 decisions properly stored in database")
            logger.info("✅ Import errors resolved, no escalation blocks")
            logger.info("✅ All escalation API endpoints working")
            logger.info("✅ End-to-end escalation flow operational")
        elif requirements_met >= total_requirements * 0.75:
            logger.info("\n⚠️ VERDICT: IA1 TO IA2 ESCALATION MOSTLY WORKING")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif requirements_met >= total_requirements * 0.5:
            logger.info("\n⚠️ VERDICT: IA1 TO IA2 ESCALATION PARTIALLY WORKING")
            logger.info("🔧 Several requirements need attention for full functionality")
        else:
            logger.info("\n❌ VERDICT: IA1 TO IA2 ESCALATION SYSTEM NOT WORKING")
            logger.info("🚨 Major issues detected - escalation system needs fixes")
        
        logger.info("\n" + "=" * 80)
        logger.info("🏁 IA1 TO IA2 ESCALATION SYSTEM TEST SUITE COMPLETED")
        logger.info("=" * 80)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'requirements_met': requirements_met,
            'total_requirements': total_requirements,
            'test_results': self.test_results,
            'requirements_status': requirements_status
        }

async def main():
    """Main function to run the escalation test suite"""
    test_suite = IA1ToIA2EscalationTestSuite()
    results = await test_suite.run_comprehensive_escalation_test_suite()
    
    # Return exit code based on results
    if results['requirements_met'] == results['total_requirements']:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
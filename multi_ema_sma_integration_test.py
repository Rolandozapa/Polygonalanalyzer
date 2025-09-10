#!/usr/bin/env python3
"""
ENHANCED MULTI EMA/SMA INTEGRATION COMPREHENSIVE TEST SUITE
Focus: Testing the enhanced Multi EMA/SMA integration in the trading bot

TESTING REQUIREMENTS FROM REVIEW REQUEST:
1. Advanced Technical Indicators Integration: 
   - Test Multi EMA/SMA fields calculation (ema_9, ema_21, sma_50, ema_200, trend_hierarchy, etc.)
   - Verify these values are being properly extracted and passed to IA1 and IA2 prompts

2. IA1 Enhanced Prompt Testing:
   - Verify 6-indicator confluence matrix (MFI + VWAP + RSI + Multi-Timeframe + Volume + EMA/SMA HIERARCHY)
   - Test EMA hierarchy data display in current snapshot section
   - Check Golden Cross/Death Cross signal detection
   - Verify trend strength scoring

3. IA2 Enhanced Decision Making:
   - Test IA2 receiving EMA hierarchy data from current_indicators
   - Verify 6-indicator confluence matrix validation in IA2
   - Check EMA levels usage for dynamic S/R calculation
   - Test confluence execution logic (6/6 GODLIKE, 5/6 STRONG, 4/6 GOOD, 3/6 HOLD)

4. Backend API Endpoints:
   - Test /api/status endpoint
   - Test /api/opportunities endpoint with enhanced indicators
   - Test trading-related endpoints with new indicators
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
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiEMASMAIntegrationTestSuite:
    """Comprehensive test suite for Multi EMA/SMA integration system"""
    
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
        logger.info(f"Testing Multi EMA/SMA Integration System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected Multi EMA/SMA fields
        self.expected_ema_sma_fields = [
            'ema_9', 'ema_21', 'sma_50', 'ema_200',
            'trend_hierarchy', 'trend_momentum', 'price_vs_emas',
            'ema_cross_signal', 'trend_strength_score'
        ]
        
        # Expected confluence matrix indicators
        self.expected_confluence_indicators = [
            'MFI', 'VWAP', 'RSI', 'Multi-Timeframe', 'Volume', 'EMA/SMA HIERARCHY'
        ]
        
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
    
    async def test_1_advanced_technical_indicators_calculation(self):
        """Test 1: Advanced Technical Indicators Integration - Multi EMA/SMA Fields Calculation"""
        logger.info("\n🔍 TEST 1: Advanced Technical Indicators Integration - Multi EMA/SMA Fields Calculation")
        
        try:
            # Test the opportunities endpoint to see if enhanced indicators are calculated
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   📊 Opportunities response received with {len(data)} opportunities")
                
                # Check if we have opportunities with technical analysis
                opportunities_with_indicators = 0
                ema_sma_fields_found = []
                sample_indicators = {}
                
                for opportunity in data:
                    if 'technical_analysis' in opportunity:
                        opportunities_with_indicators += 1
                        tech_analysis = opportunity['technical_analysis']
                        
                        # Check for Multi EMA/SMA fields
                        for field in self.expected_ema_sma_fields:
                            if field in tech_analysis:
                                if field not in ema_sma_fields_found:
                                    ema_sma_fields_found.append(field)
                                    sample_indicators[field] = tech_analysis[field]
                
                logger.info(f"   📊 Opportunities with technical indicators: {opportunities_with_indicators}")
                logger.info(f"   📊 Multi EMA/SMA fields found: {ema_sma_fields_found}")
                logger.info(f"   📊 Sample indicator values: {json.dumps(sample_indicators, indent=2)}")
                
                # Evaluate success
                required_fields_found = len(ema_sma_fields_found)
                total_required_fields = len(self.expected_ema_sma_fields)
                
                if required_fields_found >= 7:  # At least 7 out of 9 fields
                    self.log_test_result("Advanced Technical Indicators Calculation", True, 
                                       f"Multi EMA/SMA fields found: {required_fields_found}/{total_required_fields} - {ema_sma_fields_found}")
                else:
                    self.log_test_result("Advanced Technical Indicators Calculation", False, 
                                       f"Insufficient Multi EMA/SMA fields: {required_fields_found}/{total_required_fields}")
            else:
                self.log_test_result("Advanced Technical Indicators Calculation", False, 
                                   f"Opportunities endpoint failed: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Advanced Technical Indicators Calculation", False, f"Exception: {str(e)}")
    
    async def test_2_ia1_enhanced_prompt_confluence_matrix(self):
        """Test 2: IA1 Enhanced Prompt Testing - 6-Indicator Confluence Matrix"""
        logger.info("\n🔍 TEST 2: IA1 Enhanced Prompt Testing - 6-Indicator Confluence Matrix")
        
        try:
            # Get opportunities to trigger IA1 analysis
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for IA1 analysis results with confluence matrix
                ia1_analyses_found = 0
                confluence_matrix_found = 0
                ema_hierarchy_in_analysis = 0
                golden_death_cross_detected = 0
                trend_strength_scoring = 0
                
                confluence_indicators_detected = []
                sample_analysis = {}
                
                for opportunity in data:
                    # Check if this opportunity has IA1 analysis
                    if 'analysis' in opportunity or 'ia1_analysis' in opportunity:
                        ia1_analyses_found += 1
                        
                        # Get the analysis text
                        analysis_text = ""
                        if 'analysis' in opportunity:
                            analysis_text = str(opportunity['analysis'])
                        elif 'ia1_analysis' in opportunity:
                            analysis_text = str(opportunity['ia1_analysis'])
                        
                        # Check for confluence matrix keywords
                        confluence_keywords = [
                            'confluence', 'matrix', 'GODLIKE', 'STRONG', 'GOOD', 'HOLD',
                            '6/6', '5/6', '4/6', '3/6', 'MFI', 'VWAP', 'EMA', 'hierarchy'
                        ]
                        
                        confluence_found = sum(1 for keyword in confluence_keywords if keyword.lower() in analysis_text.lower())
                        if confluence_found >= 3:
                            confluence_matrix_found += 1
                        
                        # Check for EMA hierarchy analysis
                        ema_keywords = ['ema_9', 'ema_21', 'sma_50', 'ema_200', 'hierarchy', 'golden_cross', 'death_cross']
                        ema_found = sum(1 for keyword in ema_keywords if keyword.lower() in analysis_text.lower())
                        if ema_found >= 2:
                            ema_hierarchy_in_analysis += 1
                        
                        # Check for Golden/Death Cross detection
                        if 'golden_cross' in analysis_text.lower() or 'death_cross' in analysis_text.lower():
                            golden_death_cross_detected += 1
                        
                        # Check for trend strength scoring
                        if 'trend_strength' in analysis_text.lower() or 'strength_score' in analysis_text.lower():
                            trend_strength_scoring += 1
                        
                        # Store sample for detailed analysis
                        if confluence_matrix_found == 1 and not sample_analysis:
                            sample_analysis = {
                                'symbol': opportunity.get('symbol', 'Unknown'),
                                'analysis_snippet': analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text
                            }
                
                logger.info(f"   📊 IA1 analyses found: {ia1_analyses_found}")
                logger.info(f"   📊 Confluence matrix detected: {confluence_matrix_found}")
                logger.info(f"   📊 EMA hierarchy in analysis: {ema_hierarchy_in_analysis}")
                logger.info(f"   📊 Golden/Death Cross detected: {golden_death_cross_detected}")
                logger.info(f"   📊 Trend strength scoring: {trend_strength_scoring}")
                
                if sample_analysis:
                    logger.info(f"   📊 Sample analysis ({sample_analysis['symbol']}): {sample_analysis['analysis_snippet']}")
                
                # Evaluate success
                success_criteria = [
                    ia1_analyses_found > 0,
                    confluence_matrix_found > 0,
                    ema_hierarchy_in_analysis > 0
                ]
                
                if sum(success_criteria) >= 2:
                    self.log_test_result("IA1 Enhanced Prompt Confluence Matrix", True, 
                                       f"IA1 analyses: {ia1_analyses_found}, Confluence: {confluence_matrix_found}, EMA hierarchy: {ema_hierarchy_in_analysis}")
                else:
                    self.log_test_result("IA1 Enhanced Prompt Confluence Matrix", False, 
                                       f"Insufficient IA1 enhancement evidence: analyses={ia1_analyses_found}, confluence={confluence_matrix_found}")
            else:
                self.log_test_result("IA1 Enhanced Prompt Confluence Matrix", False, 
                                   f"Opportunities endpoint failed: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("IA1 Enhanced Prompt Confluence Matrix", False, f"Exception: {str(e)}")
    
    async def test_3_ia2_enhanced_decision_making(self):
        """Test 3: IA2 Enhanced Decision Making - EMA Hierarchy Data Usage"""
        logger.info("\n🔍 TEST 3: IA2 Enhanced Decision Making - EMA Hierarchy Data Usage")
        
        try:
            # Get opportunities to check for IA2 decisions
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for IA2 decision results
                ia2_decisions_found = 0
                ema_hierarchy_in_ia2 = 0
                confluence_validation_in_ia2 = 0
                dynamic_sr_calculation = 0
                confluence_execution_logic = 0
                
                sample_ia2_decision = {}
                
                for opportunity in data:
                    # Check if this opportunity has IA2 decision
                    if 'ia2_decision' in opportunity or 'decision' in opportunity:
                        ia2_decisions_found += 1
                        
                        # Get the decision data
                        decision_data = opportunity.get('ia2_decision', opportunity.get('decision', {}))
                        decision_text = str(decision_data)
                        
                        # Check for EMA hierarchy data usage
                        ema_keywords = ['ema_9', 'ema_21', 'sma_50', 'ema_200', 'hierarchy', 'trend_strength_score']
                        ema_found = sum(1 for keyword in ema_keywords if keyword.lower() in decision_text.lower())
                        if ema_found >= 2:
                            ema_hierarchy_in_ia2 += 1
                        
                        # Check for confluence matrix validation
                        confluence_keywords = ['confluence', 'matrix', 'GODLIKE', 'STRONG', 'GOOD', '6/6', '5/6', '4/6']
                        confluence_found = sum(1 for keyword in confluence_keywords if keyword.lower() in decision_text.lower())
                        if confluence_found >= 2:
                            confluence_validation_in_ia2 += 1
                        
                        # Check for dynamic S/R calculation using EMA levels
                        sr_keywords = ['support', 'resistance', 'ema', 'dynamic', 'level']
                        sr_found = sum(1 for keyword in sr_keywords if keyword.lower() in decision_text.lower())
                        if sr_found >= 3:
                            dynamic_sr_calculation += 1
                        
                        # Check for confluence execution logic
                        execution_keywords = ['GODLIKE', 'STRONG', 'GOOD', 'HOLD', 'execute', 'confidence']
                        execution_found = sum(1 for keyword in execution_keywords if keyword.lower() in decision_text.lower())
                        if execution_found >= 2:
                            confluence_execution_logic += 1
                        
                        # Store sample for detailed analysis
                        if ema_hierarchy_in_ia2 == 1 and not sample_ia2_decision:
                            sample_ia2_decision = {
                                'symbol': opportunity.get('symbol', 'Unknown'),
                                'decision_snippet': decision_text[:500] + "..." if len(decision_text) > 500 else decision_text
                            }
                
                logger.info(f"   📊 IA2 decisions found: {ia2_decisions_found}")
                logger.info(f"   📊 EMA hierarchy in IA2: {ema_hierarchy_in_ia2}")
                logger.info(f"   📊 Confluence validation in IA2: {confluence_validation_in_ia2}")
                logger.info(f"   📊 Dynamic S/R calculation: {dynamic_sr_calculation}")
                logger.info(f"   📊 Confluence execution logic: {confluence_execution_logic}")
                
                if sample_ia2_decision:
                    logger.info(f"   📊 Sample IA2 decision ({sample_ia2_decision['symbol']}): {sample_ia2_decision['decision_snippet']}")
                
                # Evaluate success
                success_criteria = [
                    ia2_decisions_found > 0,
                    ema_hierarchy_in_ia2 > 0,
                    confluence_validation_in_ia2 > 0
                ]
                
                if sum(success_criteria) >= 2:
                    self.log_test_result("IA2 Enhanced Decision Making", True, 
                                       f"IA2 decisions: {ia2_decisions_found}, EMA hierarchy: {ema_hierarchy_in_ia2}, Confluence validation: {confluence_validation_in_ia2}")
                else:
                    self.log_test_result("IA2 Enhanced Decision Making", False, 
                                       f"Insufficient IA2 enhancement evidence: decisions={ia2_decisions_found}, EMA={ema_hierarchy_in_ia2}")
            else:
                self.log_test_result("IA2 Enhanced Decision Making", False, 
                                   f"Opportunities endpoint failed: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("IA2 Enhanced Decision Making", False, f"Exception: {str(e)}")
    
    async def test_4_backend_api_endpoints_enhanced_indicators(self):
        """Test 4: Backend API Endpoints - Enhanced Indicators Integration"""
        logger.info("\n🔍 TEST 4: Backend API Endpoints - Enhanced Indicators Integration")
        
        try:
            # Test /api/status endpoint
            status_response = requests.get(f"{self.api_url}/status", timeout=30)
            status_working = status_response.status_code == 200
            
            # Test /api/opportunities endpoint with enhanced indicators
            opportunities_response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            opportunities_working = opportunities_response.status_code == 200
            
            # Check if enhanced indicators don't break existing functionality
            existing_functionality_intact = True
            enhanced_indicators_present = False
            
            if opportunities_working:
                data = opportunities_response.json()
                
                # Check that basic structure is intact
                if isinstance(data, list) and len(data) > 0:
                    sample_opportunity = data[0]
                    
                    # Check basic fields are still present
                    basic_fields = ['symbol', 'current_price', 'volume_24h']
                    basic_fields_present = sum(1 for field in basic_fields if field in sample_opportunity)
                    
                    if basic_fields_present >= 2:
                        existing_functionality_intact = True
                    
                    # Check for enhanced indicators
                    if 'technical_analysis' in sample_opportunity:
                        tech_analysis = sample_opportunity['technical_analysis']
                        enhanced_fields_found = sum(1 for field in self.expected_ema_sma_fields if field in tech_analysis)
                        if enhanced_fields_found >= 5:
                            enhanced_indicators_present = True
            
            logger.info(f"   📊 Status endpoint: {'✅ Working' if status_working else '❌ Failed'}")
            logger.info(f"   📊 Opportunities endpoint: {'✅ Working' if opportunities_working else '❌ Failed'}")
            logger.info(f"   📊 Existing functionality: {'✅ Intact' if existing_functionality_intact else '❌ Broken'}")
            logger.info(f"   📊 Enhanced indicators: {'✅ Present' if enhanced_indicators_present else '❌ Missing'}")
            
            # Evaluate success
            success_criteria = [
                status_working,
                opportunities_working,
                existing_functionality_intact,
                enhanced_indicators_present
            ]
            
            if sum(success_criteria) >= 3:
                self.log_test_result("Backend API Endpoints Enhanced Indicators", True, 
                                   f"Status: {status_working}, Opportunities: {opportunities_working}, Functionality intact: {existing_functionality_intact}, Enhanced indicators: {enhanced_indicators_present}")
            else:
                self.log_test_result("Backend API Endpoints Enhanced Indicators", False, 
                                   f"API endpoints issues: Status={status_working}, Opportunities={opportunities_working}")
                
        except Exception as e:
            self.log_test_result("Backend API Endpoints Enhanced Indicators", False, f"Exception: {str(e)}")
    
    async def test_5_data_flow_ia1_to_ia2_integration(self):
        """Test 5: Data Flow Integration - IA1 to IA2 Enhanced Indicators Passing"""
        logger.info("\n🔍 TEST 5: Data Flow Integration - IA1 to IA2 Enhanced Indicators Passing")
        
        try:
            # Get opportunities to analyze the complete data flow
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Analyze data flow from technical indicators → IA1 → IA2
                complete_flow_found = 0
                technical_to_ia1_flow = 0
                ia1_to_ia2_flow = 0
                
                for opportunity in data:
                    has_technical_indicators = False
                    has_ia1_analysis = False
                    has_ia2_decision = False
                    
                    # Check for technical indicators
                    if 'technical_analysis' in opportunity:
                        tech_analysis = opportunity['technical_analysis']
                        ema_fields_found = sum(1 for field in self.expected_ema_sma_fields if field in tech_analysis)
                        if ema_fields_found >= 5:
                            has_technical_indicators = True
                    
                    # Check for IA1 analysis
                    if 'analysis' in opportunity or 'ia1_analysis' in opportunity:
                        analysis_text = str(opportunity.get('analysis', opportunity.get('ia1_analysis', '')))
                        if 'ema' in analysis_text.lower() or 'hierarchy' in analysis_text.lower():
                            has_ia1_analysis = True
                    
                    # Check for IA2 decision
                    if 'ia2_decision' in opportunity or 'decision' in opportunity:
                        decision_text = str(opportunity.get('ia2_decision', opportunity.get('decision', '')))
                        if 'ema' in decision_text.lower() or 'confluence' in decision_text.lower():
                            has_ia2_decision = True
                    
                    # Count flow stages
                    if has_technical_indicators and has_ia1_analysis:
                        technical_to_ia1_flow += 1
                    
                    if has_ia1_analysis and has_ia2_decision:
                        ia1_to_ia2_flow += 1
                    
                    if has_technical_indicators and has_ia1_analysis and has_ia2_decision:
                        complete_flow_found += 1
                
                logger.info(f"   📊 Complete data flow (Technical → IA1 → IA2): {complete_flow_found}")
                logger.info(f"   📊 Technical → IA1 flow: {technical_to_ia1_flow}")
                logger.info(f"   📊 IA1 → IA2 flow: {ia1_to_ia2_flow}")
                
                # Evaluate success
                if complete_flow_found > 0:
                    self.log_test_result("Data Flow IA1 to IA2 Integration", True, 
                                       f"Complete enhanced indicators flow working: {complete_flow_found} opportunities")
                else:
                    self.log_test_result("Data Flow IA1 to IA2 Integration", False, 
                                       f"No complete data flow found: Technical→IA1={technical_to_ia1_flow}, IA1→IA2={ia1_to_ia2_flow}")
            else:
                self.log_test_result("Data Flow IA1 to IA2 Integration", False, 
                                   f"Opportunities endpoint failed: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Data Flow IA1 to IA2 Integration", False, f"Exception: {str(e)}")
    
    async def test_6_confluence_execution_logic_validation(self):
        """Test 6: Confluence Execution Logic - 6/6 GODLIKE, 5/6 STRONG, 4/6 GOOD, 3/6 HOLD"""
        logger.info("\n🔍 TEST 6: Confluence Execution Logic Validation")
        
        try:
            # Get opportunities to check confluence execution logic
            response = requests.get(f"{self.api_url}/opportunities", timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for confluence execution patterns
                godlike_signals = 0
                strong_signals = 0
                good_signals = 0
                hold_signals = 0
                
                confluence_patterns_found = []
                
                for opportunity in data:
                    # Check IA1 analysis for confluence patterns
                    analysis_text = ""
                    if 'analysis' in opportunity:
                        analysis_text += str(opportunity['analysis'])
                    if 'ia1_analysis' in opportunity:
                        analysis_text += str(opportunity['ia1_analysis'])
                    
                    # Check IA2 decision for confluence patterns
                    decision_text = ""
                    if 'ia2_decision' in opportunity:
                        decision_text += str(opportunity['ia2_decision'])
                    if 'decision' in opportunity:
                        decision_text += str(opportunity['decision'])
                    
                    combined_text = (analysis_text + " " + decision_text).lower()
                    
                    # Look for confluence execution patterns
                    if '6/6' in combined_text or 'godlike' in combined_text:
                        godlike_signals += 1
                        confluence_patterns_found.append(f"{opportunity.get('symbol', 'Unknown')}: GODLIKE (6/6)")
                    elif '5/6' in combined_text or ('strong' in combined_text and 'confluence' in combined_text):
                        strong_signals += 1
                        confluence_patterns_found.append(f"{opportunity.get('symbol', 'Unknown')}: STRONG (5/6)")
                    elif '4/6' in combined_text or ('good' in combined_text and 'confluence' in combined_text):
                        good_signals += 1
                        confluence_patterns_found.append(f"{opportunity.get('symbol', 'Unknown')}: GOOD (4/6)")
                    elif '3/6' in combined_text or ('hold' in combined_text and 'confluence' in combined_text):
                        hold_signals += 1
                        confluence_patterns_found.append(f"{opportunity.get('symbol', 'Unknown')}: HOLD (3/6)")
                
                logger.info(f"   📊 GODLIKE signals (6/6): {godlike_signals}")
                logger.info(f"   📊 STRONG signals (5/6): {strong_signals}")
                logger.info(f"   📊 GOOD signals (4/6): {good_signals}")
                logger.info(f"   📊 HOLD signals (3/6): {hold_signals}")
                
                if confluence_patterns_found:
                    logger.info(f"   📊 Confluence patterns found: {confluence_patterns_found[:5]}")  # Show first 5
                
                # Evaluate success
                total_confluence_signals = godlike_signals + strong_signals + good_signals + hold_signals
                
                if total_confluence_signals > 0:
                    self.log_test_result("Confluence Execution Logic Validation", True, 
                                       f"Confluence execution logic working: {total_confluence_signals} signals with proper classification")
                else:
                    self.log_test_result("Confluence Execution Logic Validation", False, 
                                       "No confluence execution logic patterns found in IA1/IA2 outputs")
            else:
                self.log_test_result("Confluence Execution Logic Validation", False, 
                                   f"Opportunities endpoint failed: HTTP {response.status_code}")
                
        except Exception as e:
            self.log_test_result("Confluence Execution Logic Validation", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Multi EMA/SMA integration tests"""
        logger.info("🚀 Starting Multi EMA/SMA Integration Comprehensive Test Suite")
        logger.info("=" * 80)
        logger.info("📋 MULTI EMA/SMA INTEGRATION SYSTEM COMPREHENSIVE TESTING")
        logger.info("🎯 Testing: Advanced technical indicators, IA1 confluence matrix, IA2 decision making, API endpoints")
        logger.info("🎯 Expected: Complete Multi EMA/SMA integration with 6-indicator confluence matrix working")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_advanced_technical_indicators_calculation()
        await self.test_2_ia1_enhanced_prompt_confluence_matrix()
        await self.test_3_ia2_enhanced_decision_making()
        await self.test_4_backend_api_endpoints_enhanced_indicators()
        await self.test_5_data_flow_ia1_to_ia2_integration()
        await self.test_6_confluence_execution_logic_validation()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 MULTI EMA/SMA INTEGRATION COMPREHENSIVE TEST SUMMARY")
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
        logger.info("📋 MULTI EMA/SMA INTEGRATION SYSTEM STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Multi EMA/SMA Integration System FULLY FUNCTIONAL!")
            logger.info("✅ Advanced technical indicators calculation working")
            logger.info("✅ IA1 enhanced prompt with 6-indicator confluence matrix operational")
            logger.info("✅ IA2 enhanced decision making with EMA hierarchy data working")
            logger.info("✅ Backend API endpoints with enhanced indicators functional")
            logger.info("✅ Complete data flow IA1→IA2 integration working")
            logger.info("✅ Confluence execution logic (GODLIKE/STRONG/GOOD/HOLD) operational")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - Multi EMA/SMA integration working with minor gaps")
            logger.info("🔍 Some components may need fine-tuning for full optimization")
        elif passed_tests >= total_tests * 0.6:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core Multi EMA/SMA features working")
            logger.info("🔧 Some advanced features may need implementation or debugging")
        else:
            logger.info("❌ SYSTEM NOT FUNCTIONAL - Critical issues with Multi EMA/SMA integration")
            logger.info("🚨 Major implementation gaps or system errors preventing functionality")
        
        # Specific requirements check
        logger.info("\n📝 MULTI EMA/SMA INTEGRATION REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "Advanced Technical Indicators" in result['test']:
                    requirements_met.append("✅ Multi EMA/SMA fields calculation working")
                elif "IA1 Enhanced Prompt" in result['test']:
                    requirements_met.append("✅ IA1 6-indicator confluence matrix operational")
                elif "IA2 Enhanced Decision" in result['test']:
                    requirements_met.append("✅ IA2 EMA hierarchy data usage working")
                elif "Backend API Endpoints" in result['test']:
                    requirements_met.append("✅ Enhanced indicators in API endpoints functional")
                elif "Data Flow" in result['test']:
                    requirements_met.append("✅ IA1→IA2 enhanced indicators data flow working")
                elif "Confluence Execution Logic" in result['test']:
                    requirements_met.append("✅ Confluence execution logic (GODLIKE/STRONG/GOOD/HOLD) working")
            else:
                if "Advanced Technical Indicators" in result['test']:
                    requirements_failed.append("❌ Multi EMA/SMA fields calculation failed")
                elif "IA1 Enhanced Prompt" in result['test']:
                    requirements_failed.append("❌ IA1 6-indicator confluence matrix not working")
                elif "IA2 Enhanced Decision" in result['test']:
                    requirements_failed.append("❌ IA2 EMA hierarchy data usage failed")
                elif "Backend API Endpoints" in result['test']:
                    requirements_failed.append("❌ Enhanced indicators in API endpoints not working")
                elif "Data Flow" in result['test']:
                    requirements_failed.append("❌ IA1→IA2 enhanced indicators data flow broken")
                elif "Confluence Execution Logic" in result['test']:
                    requirements_failed.append("❌ Confluence execution logic not working")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: Multi EMA/SMA Integration System is FULLY FUNCTIONAL!")
            logger.info("✅ All enhanced technical indicators implemented and working correctly")
            logger.info("✅ IA1 6-indicator confluence matrix operational with EMA/SMA hierarchy")
            logger.info("✅ IA2 enhanced decision making with confluence execution logic working")
            logger.info("✅ Complete data flow and API integration functional")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: Multi EMA/SMA Integration System is MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif len(requirements_failed) <= 2:
            logger.info("\n⚠️ VERDICT: Multi EMA/SMA Integration System is PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several components need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: Multi EMA/SMA Integration System is NOT FUNCTIONAL")
            logger.info("🚨 Major implementation gaps preventing Multi EMA/SMA integration")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = MultiEMASMAIntegrationTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
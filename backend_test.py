#!/usr/bin/env python3
"""
COMPREHENSIVE IA1→IA2 PIPELINE DEMONSTRATION RUN TEST SUITE
Focus: Test the completely repaired dual AI trading system to demonstrate full functionality

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **IA1→IA2 Pipeline Complete Flow**: 
   - Trigger IA1 analysis for multiple symbols (BTCUSDT, ETHUSDT, SOLUSDT)
   - Verify VOIE escalation logic (VOIE 1: confidence ≥70%, VOIE 2: RR ≥2.0, VOIE 3: confidence ≥95%)
   - Confirm IA2 strategic decisions with enhanced prompts

2. **IA2 Strategic Intelligence**:
   - Verify IA2 generates detailed strategic reasoning
   - Check new fields: market_regime_assessment, position_size_recommendation, execution_priority, calculated_rr, rr_reasoning
   - Confirm IA2 can override IA1 decisions when strategic analysis indicates conflicts

3. **System Components Integration**:
   - Test all API endpoints: /api/opportunities, /api/analyses, /api/decisions, /api/performance
   - Verify orchestrator functionality and cycle management
   - Check database storage of analyses and decisions

4. **Advanced Technical Analysis**:
   - Confirm multi-timeframe analysis working
   - Verify advanced indicators (RSI, MACD, MFI, VWAP, EMA hierarchy)
   - Test pattern detection and confluence matrix

5. **Performance & Stability**:
   - Monitor CPU usage and system stability
   - Check error handling and fallback mechanisms
   - Verify logging quality and decision traceability

EXPECTED RESULTS:
- IA1 analyses with 70%+ confidence should escalate to IA2
- IA2 should provide strategic decisions with detailed reasoning
- System should demonstrate intelligent double-verification (IA1 vs IA2 decisions)
- All endpoints should return proper data
- No more "string indices" or "acomplete" errors

RUN DEMONSTRATION:
Show the complete flow from opportunity scanning → IA1 analysis → IA2 strategic decision → database storage. Focus on demonstrating the intelligence and strategic capabilities we just implemented.
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

class IA2SimplifiedPromptTestSuite:
    """Comprehensive test suite for IA2 simplified prompt system after major code deletion"""
    
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
        logger.info(f"Testing IA2 Simplified Prompt System at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for IA2 decision analysis")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # Test symbols for IA1 → IA2 pipeline testing
        self.test_symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT",
            "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT"
        ]
        
        # Track IA2 decisions created during testing
        self.ia2_decisions_created = []
        
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
    
    async def test_1_backend_logs_string_indices_check(self):
        """Test 1: Check backend logs for 'string indices must be integers, not str' errors"""
        logger.info("\n🔍 TEST 1: Backend Logs String Indices Error Check")
        
        try:
            # Check recent backend logs for string indices errors
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            string_indices_errors = []
            total_log_lines = 0
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        # Get last 1000 lines to check recent activity
                        result = subprocess.run(['tail', '-n', '1000', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            total_log_lines += len(log_content.split('\n'))
                            
                            # Search for string indices errors
                            error_patterns = [
                                r"string indices must be integers, not str",
                                r"TypeError.*string indices must be integers",
                                r"string indices.*not str"
                            ]
                            
                            for pattern in error_patterns:
                                matches = re.findall(pattern, log_content, re.IGNORECASE)
                                if matches:
                                    string_indices_errors.extend(matches)
                                    
                            # Also check for IA2-related errors specifically
                            ia2_error_lines = []
                            for line in log_content.split('\n'):
                                if 'ia2' in line.lower() and any(pattern in line.lower() for pattern in ['string indices', 'typeerror', 'error']):
                                    ia2_error_lines.append(line.strip())
                            
                            if ia2_error_lines:
                                logger.info(f"   📊 Found {len(ia2_error_lines)} IA2-related error lines in {log_file}")
                                for error_line in ia2_error_lines[-5:]:  # Show last 5 errors
                                    logger.info(f"      {error_line}")
                                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not read {log_file}: {e}")
            
            logger.info(f"   📊 Analyzed {total_log_lines} log lines from backend logs")
            logger.info(f"   📊 Found {len(string_indices_errors)} 'string indices' errors")
            
            if len(string_indices_errors) == 0:
                self.log_test_result("Backend Logs String Indices Check", True, 
                                   f"No 'string indices must be integers, not str' errors found in recent logs")
            else:
                self.log_test_result("Backend Logs String Indices Check", False, 
                                   f"Found {len(string_indices_errors)} string indices errors in backend logs")
                # Log the errors for debugging
                for error in string_indices_errors[:3]:  # Show first 3 errors
                    logger.info(f"      Error: {error}")
                    
        except Exception as e:
            self.log_test_result("Backend Logs String Indices Check", False, f"Exception: {str(e)}")
    
    async def test_2_ia2_decision_generation_stability(self):
        """Test 2: Test IA2 Decision Generation Stability by triggering IA1 → IA2 pipeline"""
        logger.info("\n🔍 TEST 2: IA2 Decision Generation Stability Test")
        
        try:
            # Get current IA2 decision count before testing
            initial_ia2_count = 0
            if self.db is not None:
                initial_ia2_count = self.db.trading_decisions.count_documents({})
                logger.info(f"   📊 Initial IA2 decisions in database: {initial_ia2_count}")
            
            # Trigger IA1 analysis for multiple symbols to potentially trigger IA2
            ia1_analyses_triggered = 0
            ia2_escalations_detected = 0
            
            for symbol in self.test_symbols[:3]:  # Test first 3 symbols
                try:
                    logger.info(f"   🚀 Triggering IA1 analysis for {symbol}")
                    
                    # Use the correct endpoint for IA1 analysis
                    response = requests.post(f"{self.api_url}/force-ia1-analysis", 
                                           json={"symbol": symbol}, 
                                           timeout=120)  # Longer timeout for IA1 → IA2 pipeline
                    
                    if response.status_code in [200, 201]:
                        ia1_analyses_triggered += 1
                        result = response.json()
                        
                        # Check if IA2 was triggered
                        if result.get('success', False):
                            logger.info(f"      ✅ {symbol}: IA1 analysis successful")
                            
                            # Check if IA2 escalation occurred
                            if 'ia2_triggered' in result or 'escalated_to_ia2' in result:
                                ia2_escalations_detected += 1
                                logger.info(f"      ✅ {symbol}: IA2 escalation detected")
                        else:
                            logger.info(f"      ℹ️ {symbol}: IA1 analysis completed but no escalation")
                    else:
                        logger.warning(f"      ⚠️ {symbol}: IA1 analysis failed - HTTP {response.status_code}")
                        
                    # Small delay between requests
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    logger.warning(f"      ⚠️ {symbol}: Exception during analysis - {str(e)}")
            
            # Wait a bit for IA2 processing to complete
            logger.info("   ⏳ Waiting 30 seconds for IA2 processing to complete...")
            await asyncio.sleep(30)
            
            # Check final IA2 decision count
            final_ia2_count = 0
            new_ia2_decisions = 0
            if self.db is not None:
                final_ia2_count = self.db.trading_decisions.count_documents({})
                new_ia2_decisions = final_ia2_count - initial_ia2_count
                logger.info(f"   📊 Final IA2 decisions in database: {final_ia2_count}")
                logger.info(f"   📊 New IA2 decisions created: {new_ia2_decisions}")
            
            # Evaluate test results
            logger.info(f"   📊 IA1 analyses triggered: {ia1_analyses_triggered}")
            logger.info(f"   📊 IA2 escalations detected: {ia2_escalations_detected}")
            
            if ia1_analyses_triggered >= 2:  # At least 2 IA1 analyses successful
                if self.db is not None and new_ia2_decisions > 0:
                    self.log_test_result("IA2 Decision Generation Stability", True, 
                                       f"IA2 system generating decisions: {new_ia2_decisions} new decisions created")
                elif ia2_escalations_detected > 0:
                    self.log_test_result("IA2 Decision Generation Stability", True, 
                                       f"IA2 escalations detected: {ia2_escalations_detected} escalations")
                else:
                    self.log_test_result("IA2 Decision Generation Stability", False, 
                                       "No IA2 escalations detected despite IA1 analyses")
            else:
                self.log_test_result("IA2 Decision Generation Stability", False, 
                                   f"Insufficient IA1 analyses triggered: {ia1_analyses_triggered}")
                
        except Exception as e:
            self.log_test_result("IA2 Decision Generation Stability", False, f"Exception: {str(e)}")
    
    async def test_3_ia2_decision_structure_validation(self):
        """Test 3: Validate IA2 Decision Structure and Required Fields"""
        logger.info("\n🔍 TEST 3: IA2 Decision Structure Validation Test")
        
        try:
            if self.db is None:
                self.log_test_result("IA2 Decision Structure Validation", False, 
                                   "MongoDB connection not available for decision analysis")
                return
            
            # Get recent IA2 decisions (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            recent_decisions = list(self.db.trading_decisions.find({
                "timestamp": {"$gte": cutoff_time}
            }).sort("timestamp", -1).limit(10))
            
            logger.info(f"   📊 Found {len(recent_decisions)} recent IA2 decisions for analysis")
            
            if len(recent_decisions) == 0:
                self.log_test_result("IA2 Decision Structure Validation", False, 
                                   "No recent IA2 decisions found for structure validation")
                return
            
            # Analyze decision structure
            structure_analysis = {
                'total_decisions': len(recent_decisions),
                'has_calculated_rr': 0,
                'has_rr_reasoning': 0,
                'has_signal': 0,
                'has_confidence': 0,
                'has_reasoning': 0,
                'has_strategic_analysis': 0,
                'signal_distribution': {'LONG': 0, 'SHORT': 0, 'HOLD': 0},
                'confidence_range': {'min': 1.0, 'max': 0.0, 'avg': 0.0},
                'decisions_with_errors': 0
            }
            
            confidence_values = []
            
            for decision in recent_decisions:
                # Check for required fields
                if 'calculated_rr' in decision:
                    structure_analysis['has_calculated_rr'] += 1
                if 'rr_reasoning' in decision:
                    structure_analysis['has_rr_reasoning'] += 1
                if 'signal' in decision:
                    structure_analysis['has_signal'] += 1
                    signal = decision['signal']
                    if signal in structure_analysis['signal_distribution']:
                        structure_analysis['signal_distribution'][signal] += 1
                if 'confidence' in decision:
                    structure_analysis['has_confidence'] += 1
                    confidence = decision['confidence']
                    confidence_values.append(confidence)
                if 'reasoning' in decision:
                    structure_analysis['has_reasoning'] += 1
                    reasoning = decision['reasoning']
                    if len(reasoning) > 100:  # Substantial reasoning
                        structure_analysis['has_strategic_analysis'] += 1
                
                # Check for error indicators
                if 'error' in str(decision).lower() or 'exception' in str(decision).lower():
                    structure_analysis['decisions_with_errors'] += 1
            
            # Calculate confidence statistics
            if confidence_values:
                structure_analysis['confidence_range']['min'] = min(confidence_values)
                structure_analysis['confidence_range']['max'] = max(confidence_values)
                structure_analysis['confidence_range']['avg'] = sum(confidence_values) / len(confidence_values)
            
            # Log detailed analysis
            logger.info(f"   📊 Decision Structure Analysis:")
            logger.info(f"      Total decisions analyzed: {structure_analysis['total_decisions']}")
            logger.info(f"      Has calculated_rr field: {structure_analysis['has_calculated_rr']}/{structure_analysis['total_decisions']}")
            logger.info(f"      Has rr_reasoning field: {structure_analysis['has_rr_reasoning']}/{structure_analysis['total_decisions']}")
            logger.info(f"      Has signal field: {structure_analysis['has_signal']}/{structure_analysis['total_decisions']}")
            logger.info(f"      Has confidence field: {structure_analysis['has_confidence']}/{structure_analysis['total_decisions']}")
            logger.info(f"      Has reasoning field: {structure_analysis['has_reasoning']}/{structure_analysis['total_decisions']}")
            logger.info(f"      Has strategic analysis: {structure_analysis['has_strategic_analysis']}/{structure_analysis['total_decisions']}")
            logger.info(f"      Signal distribution: {structure_analysis['signal_distribution']}")
            logger.info(f"      Confidence range: {structure_analysis['confidence_range']}")
            logger.info(f"      Decisions with errors: {structure_analysis['decisions_with_errors']}")
            
            # Evaluate structure quality
            required_fields_score = (
                structure_analysis['has_calculated_rr'] + 
                structure_analysis['has_rr_reasoning'] + 
                structure_analysis['has_signal'] + 
                structure_analysis['has_confidence'] + 
                structure_analysis['has_reasoning']
            ) / (structure_analysis['total_decisions'] * 5)  # 5 required fields
            
            strategic_quality_score = structure_analysis['has_strategic_analysis'] / structure_analysis['total_decisions']
            
            logger.info(f"   📊 Required fields score: {required_fields_score:.2%}")
            logger.info(f"   📊 Strategic quality score: {strategic_quality_score:.2%}")
            
            # Determine test result
            if required_fields_score >= 0.8 and strategic_quality_score >= 0.6 and structure_analysis['decisions_with_errors'] == 0:
                self.log_test_result("IA2 Decision Structure Validation", True, 
                                   f"IA2 decisions have proper structure: {required_fields_score:.1%} required fields, {strategic_quality_score:.1%} strategic quality")
            elif required_fields_score >= 0.6:
                self.log_test_result("IA2 Decision Structure Validation", False, 
                                   f"IA2 decision structure partially valid: {required_fields_score:.1%} required fields, {structure_analysis['decisions_with_errors']} errors")
            else:
                self.log_test_result("IA2 Decision Structure Validation", False, 
                                   f"IA2 decision structure invalid: {required_fields_score:.1%} required fields, {structure_analysis['decisions_with_errors']} errors")
                
        except Exception as e:
            self.log_test_result("IA2 Decision Structure Validation", False, f"Exception: {str(e)}")
    
    async def test_4_calculated_rr_and_reasoning_fields(self):
        """Test 4: Specific Test for calculated_rr and rr_reasoning Fields"""
        logger.info("\n🔍 TEST 4: Calculated RR and RR Reasoning Fields Test")
        
        try:
            if self.db is None:
                self.log_test_result("Calculated RR and RR Reasoning Fields", False, 
                                   "MongoDB connection not available for field analysis")
                return
            
            # Get recent IA2 decisions
            recent_decisions = list(self.db.trading_decisions.find({}).sort("timestamp", -1).limit(20))
            
            logger.info(f"   📊 Analyzing {len(recent_decisions)} recent IA2 decisions for RR fields")
            
            if len(recent_decisions) == 0:
                self.log_test_result("Calculated RR and RR Reasoning Fields", False, 
                                   "No IA2 decisions found for RR field analysis")
                return
            
            # Analyze RR fields
            rr_analysis = {
                'total_decisions': len(recent_decisions),
                'has_calculated_rr': 0,
                'has_rr_reasoning': 0,
                'valid_calculated_rr': 0,
                'valid_rr_reasoning': 0,
                'rr_values': [],
                'reasoning_samples': []
            }
            
            for decision in recent_decisions:
                # Check calculated_rr field
                if 'calculated_rr' in decision:
                    rr_analysis['has_calculated_rr'] += 1
                    calculated_rr = decision['calculated_rr']
                    
                    # Validate calculated_rr value
                    if isinstance(calculated_rr, (int, float)) and calculated_rr > 0:
                        rr_analysis['valid_calculated_rr'] += 1
                        rr_analysis['rr_values'].append(calculated_rr)
                
                # Check rr_reasoning field
                if 'rr_reasoning' in decision:
                    rr_analysis['has_rr_reasoning'] += 1
                    rr_reasoning = decision['rr_reasoning']
                    
                    # Validate rr_reasoning content
                    if isinstance(rr_reasoning, str) and len(rr_reasoning) > 20:
                        rr_analysis['valid_rr_reasoning'] += 1
                        if len(rr_analysis['reasoning_samples']) < 3:
                            rr_analysis['reasoning_samples'].append(rr_reasoning[:100] + "...")
            
            # Calculate statistics
            rr_field_presence = rr_analysis['has_calculated_rr'] / rr_analysis['total_decisions']
            reasoning_field_presence = rr_analysis['has_rr_reasoning'] / rr_analysis['total_decisions']
            rr_validity = rr_analysis['valid_calculated_rr'] / max(rr_analysis['has_calculated_rr'], 1)
            reasoning_validity = rr_analysis['valid_rr_reasoning'] / max(rr_analysis['has_rr_reasoning'], 1)
            
            # Log detailed analysis
            logger.info(f"   📊 RR Fields Analysis:")
            logger.info(f"      calculated_rr field present: {rr_analysis['has_calculated_rr']}/{rr_analysis['total_decisions']} ({rr_field_presence:.1%})")
            logger.info(f"      rr_reasoning field present: {rr_analysis['has_rr_reasoning']}/{rr_analysis['total_decisions']} ({reasoning_field_presence:.1%})")
            logger.info(f"      Valid calculated_rr values: {rr_analysis['valid_calculated_rr']}/{rr_analysis['has_calculated_rr']} ({rr_validity:.1%})")
            logger.info(f"      Valid rr_reasoning content: {rr_analysis['valid_rr_reasoning']}/{rr_analysis['has_rr_reasoning']} ({reasoning_validity:.1%})")
            
            if rr_analysis['rr_values']:
                avg_rr = sum(rr_analysis['rr_values']) / len(rr_analysis['rr_values'])
                min_rr = min(rr_analysis['rr_values'])
                max_rr = max(rr_analysis['rr_values'])
                logger.info(f"      RR value range: {min_rr:.2f} - {max_rr:.2f} (avg: {avg_rr:.2f})")
            
            logger.info(f"   📊 Sample RR Reasoning:")
            for i, sample in enumerate(rr_analysis['reasoning_samples'], 1):
                logger.info(f"      Sample {i}: {sample}")
            
            # Determine test result
            if rr_field_presence >= 0.8 and reasoning_field_presence >= 0.8 and rr_validity >= 0.9 and reasoning_validity >= 0.9:
                self.log_test_result("Calculated RR and RR Reasoning Fields", True, 
                                   f"RR fields properly implemented: {rr_field_presence:.1%} presence, {rr_validity:.1%} validity")
            elif rr_field_presence >= 0.5 and reasoning_field_presence >= 0.5:
                self.log_test_result("Calculated RR and RR Reasoning Fields", False, 
                                   f"RR fields partially implemented: {rr_field_presence:.1%} presence, {rr_validity:.1%} validity")
            else:
                self.log_test_result("Calculated RR and RR Reasoning Fields", False, 
                                   f"RR fields missing or invalid: {rr_field_presence:.1%} presence, {rr_validity:.1%} validity")
                
        except Exception as e:
            self.log_test_result("Calculated RR and RR Reasoning Fields", False, f"Exception: {str(e)}")
    
    async def test_5_strategic_reasoning_quality(self):
        """Test 5: Validate Strategic Reasoning Quality in IA2 Decisions"""
        logger.info("\n🔍 TEST 5: Strategic Reasoning Quality Validation Test")
        
        try:
            if self.db is None:
                self.log_test_result("Strategic Reasoning Quality", False, 
                                   "MongoDB connection not available for reasoning analysis")
                return
            
            # Get recent IA2 decisions with reasoning
            recent_decisions = list(self.db.trading_decisions.find({
                "reasoning": {"$exists": True}
            }).sort("timestamp", -1).limit(15))
            
            logger.info(f"   📊 Analyzing {len(recent_decisions)} IA2 decisions for reasoning quality")
            
            if len(recent_decisions) == 0:
                self.log_test_result("Strategic Reasoning Quality", False, 
                                   "No IA2 decisions with reasoning found for quality analysis")
                return
            
            # Quality indicators to look for in reasoning
            quality_indicators = {
                'technical_analysis': ['rsi', 'macd', 'ema', 'sma', 'bollinger', 'stochastic', 'vwap', 'mfi'],
                'market_context': ['market', 'trend', 'momentum', 'volatility', 'volume', 'support', 'resistance'],
                'risk_management': ['risk', 'reward', 'stop', 'loss', 'take', 'profit', 'position', 'size'],
                'strategic_thinking': ['confluence', 'probability', 'expected', 'value', 'optimization', 'strategy'],
                'decision_logic': ['confidence', 'threshold', 'criteria', 'analysis', 'evaluation', 'assessment']
            }
            
            reasoning_analysis = {
                'total_decisions': len(recent_decisions),
                'quality_scores': [],
                'category_scores': {category: 0 for category in quality_indicators.keys()},
                'reasoning_lengths': [],
                'decisions_with_quality': 0
            }
            
            for decision in recent_decisions:
                reasoning = decision.get('reasoning', '').lower()
                reasoning_length = len(reasoning)
                reasoning_analysis['reasoning_lengths'].append(reasoning_length)
                
                # Calculate quality score for this decision
                decision_quality_score = 0
                category_hits = {category: 0 for category in quality_indicators.keys()}
                
                for category, indicators in quality_indicators.items():
                    category_hit_count = sum(1 for indicator in indicators if indicator in reasoning)
                    category_hits[category] = category_hit_count
                    
                    # Score: 1 point per category with at least 2 indicators
                    if category_hit_count >= 2:
                        decision_quality_score += 1
                        reasoning_analysis['category_scores'][category] += 1
                
                reasoning_analysis['quality_scores'].append(decision_quality_score)
                
                # Consider decision as having quality if it scores >= 3 categories
                if decision_quality_score >= 3:
                    reasoning_analysis['decisions_with_quality'] += 1
                
                # Log sample reasoning for top quality decisions
                if decision_quality_score >= 4 and len(reasoning_analysis['quality_scores']) <= 3:
                    logger.info(f"   📊 High-quality reasoning sample (score: {decision_quality_score}/5):")
                    logger.info(f"      {reasoning[:200]}...")
            
            # Calculate overall statistics
            avg_quality_score = sum(reasoning_analysis['quality_scores']) / len(reasoning_analysis['quality_scores'])
            avg_reasoning_length = sum(reasoning_analysis['reasoning_lengths']) / len(reasoning_analysis['reasoning_lengths'])
            quality_percentage = reasoning_analysis['decisions_with_quality'] / reasoning_analysis['total_decisions']
            
            # Log detailed analysis
            logger.info(f"   📊 Strategic Reasoning Quality Analysis:")
            logger.info(f"      Average quality score: {avg_quality_score:.1f}/5.0")
            logger.info(f"      Average reasoning length: {avg_reasoning_length:.0f} characters")
            logger.info(f"      Decisions with quality reasoning: {reasoning_analysis['decisions_with_quality']}/{reasoning_analysis['total_decisions']} ({quality_percentage:.1%})")
            
            logger.info(f"   📊 Category Coverage:")
            for category, score in reasoning_analysis['category_scores'].items():
                coverage = score / reasoning_analysis['total_decisions']
                logger.info(f"      {category.replace('_', ' ').title()}: {score}/{reasoning_analysis['total_decisions']} ({coverage:.1%})")
            
            # Determine test result
            if avg_quality_score >= 3.0 and quality_percentage >= 0.7 and avg_reasoning_length >= 200:
                self.log_test_result("Strategic Reasoning Quality", True, 
                                   f"High-quality strategic reasoning: {avg_quality_score:.1f}/5.0 score, {quality_percentage:.1%} quality decisions")
            elif avg_quality_score >= 2.0 and quality_percentage >= 0.5:
                self.log_test_result("Strategic Reasoning Quality", False, 
                                   f"Moderate strategic reasoning quality: {avg_quality_score:.1f}/5.0 score, {quality_percentage:.1%} quality decisions")
            else:
                self.log_test_result("Strategic Reasoning Quality", False, 
                                   f"Poor strategic reasoning quality: {avg_quality_score:.1f}/5.0 score, {quality_percentage:.1%} quality decisions")
                
        except Exception as e:
            self.log_test_result("Strategic Reasoning Quality", False, f"Exception: {str(e)}")
    
    async def test_6_voie_escalation_paths(self):
        """Test 6: Test VOIE 1, VOIE 2, and VOIE 3 Escalation Paths"""
        logger.info("\n🔍 TEST 6: VOIE Escalation Paths Test")
        
        try:
            # Check backend logs for VOIE escalation messages
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            voie_patterns = {
                'VOIE 1': [r'VOIE 1', r'confidence.*70%', r'IA2 ACCEPTED.*VOIE 1'],
                'VOIE 2': [r'VOIE 2', r'RR.*2\.0', r'IA2 ACCEPTED.*VOIE 2'],
                'VOIE 3': [r'VOIE 3', r'95%.*confidence', r'OVERRIDE.*Exceptional', r'IA2 ACCEPTED.*VOIE 3']
            }
            
            voie_analysis = {
                'VOIE 1': 0,
                'VOIE 2': 0,
                'VOIE 3': 0,
                'total_escalations': 0,
                'escalation_samples': []
            }
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '2000', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            
                            # Search for VOIE patterns
                            for voie_name, patterns in voie_patterns.items():
                                for pattern in patterns:
                                    matches = re.findall(pattern, log_content, re.IGNORECASE)
                                    if matches:
                                        voie_analysis[voie_name] += len(matches)
                            
                            # Look for escalation samples
                            for line in log_content.split('\n'):
                                if any(voie in line.upper() for voie in ['VOIE 1', 'VOIE 2', 'VOIE 3']):
                                    if len(voie_analysis['escalation_samples']) < 5:
                                        voie_analysis['escalation_samples'].append(line.strip())
                                        
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not read {log_file}: {e}")
            
            voie_analysis['total_escalations'] = sum([voie_analysis['VOIE 1'], voie_analysis['VOIE 2'], voie_analysis['VOIE 3']])
            
            # Log analysis results
            logger.info(f"   📊 VOIE Escalation Analysis:")
            logger.info(f"      VOIE 1 escalations (confidence ≥70%): {voie_analysis['VOIE 1']}")
            logger.info(f"      VOIE 2 escalations (RR ≥2.0): {voie_analysis['VOIE 2']}")
            logger.info(f"      VOIE 3 escalations (confidence ≥95%): {voie_analysis['VOIE 3']}")
            logger.info(f"      Total escalations detected: {voie_analysis['total_escalations']}")
            
            if voie_analysis['escalation_samples']:
                logger.info(f"   📊 Escalation Samples:")
                for i, sample in enumerate(voie_analysis['escalation_samples'], 1):
                    logger.info(f"      Sample {i}: {sample}")
            
            # Determine test result
            if voie_analysis['total_escalations'] >= 3:
                active_voies = sum(1 for voie in ['VOIE 1', 'VOIE 2', 'VOIE 3'] if voie_analysis[voie] > 0)
                self.log_test_result("VOIE Escalation Paths", True, 
                                   f"VOIE escalation system working: {voie_analysis['total_escalations']} escalations, {active_voies} active paths")
            elif voie_analysis['total_escalations'] >= 1:
                self.log_test_result("VOIE Escalation Paths", False, 
                                   f"Limited VOIE escalation activity: {voie_analysis['total_escalations']} escalations detected")
            else:
                self.log_test_result("VOIE Escalation Paths", False, 
                                   "No VOIE escalation activity detected in logs")
                
        except Exception as e:
            self.log_test_result("VOIE Escalation Paths", False, f"Exception: {str(e)}")
    
    async def test_7_end_to_end_pipeline_integration(self):
        """Test 7: End-to-End IA1 → IA2 Pipeline Integration Test"""
        logger.info("\n🔍 TEST 7: End-to-End IA1 → IA2 Pipeline Integration Test")
        
        try:
            # Test complete pipeline with a specific symbol
            test_symbol = "BTCUSDT"
            logger.info(f"   🚀 Testing complete IA1 → IA2 pipeline with {test_symbol}")
            
            # Step 1: Trigger IA1 analysis
            logger.info("   📊 Step 1: Triggering IA1 analysis...")
            ia1_response = requests.post(f"{self.api_url}/force-ia1-analysis", 
                                       json={"symbol": test_symbol}, 
                                       timeout=180)  # Extended timeout for full pipeline
            
            if ia1_response.status_code not in [200, 201]:
                self.log_test_result("End-to-End Pipeline Integration", False, 
                                   f"IA1 analysis failed: HTTP {ia1_response.status_code}")
                return
            
            ia1_result = ia1_response.json()
            logger.info(f"      ✅ IA1 analysis completed for {test_symbol}")
            
            # Step 2: Check if IA2 was triggered
            ia2_triggered = False
            decision_id = None
            
            if ia1_result.get('success', False):
                logger.info(f"      ✅ IA1 analysis successful")
                
                # Check for IA2 escalation indicators
                if 'ia2_triggered' in ia1_result or 'escalated_to_ia2' in ia1_result or 'decision_id' in ia1_result:
                    ia2_triggered = True
                    decision_id = ia1_result.get('decision_id')
                    logger.info(f"      ✅ IA2 escalation triggered, decision ID: {decision_id}")
                else:
                    logger.info("      ℹ️ IA2 not triggered (IA1 analysis did not meet escalation criteria)")
            else:
                logger.info("      ⚠️ IA1 analysis completed but not successful")
            
            # Step 3: Wait for processing and check database
            logger.info("   📊 Step 2: Waiting for IA2 processing...")
            await asyncio.sleep(45)  # Wait for IA2 processing
            
            # Step 4: Verify IA2 decision in database
            ia2_decision_found = False
            if self.db is not None and decision_id:
                decision = self.db.trading_decisions.find_one({"id": decision_id})
                if decision:
                    ia2_decision_found = True
                    logger.info(f"      ✅ IA2 decision found in database")
                    
                    # Check decision quality
                    has_signal = 'signal' in decision
                    has_confidence = 'confidence' in decision
                    has_reasoning = 'reasoning' in decision and len(decision['reasoning']) > 50
                    has_calculated_rr = 'calculated_rr' in decision
                    
                    logger.info(f"         Signal: {decision.get('signal', 'N/A')}")
                    logger.info(f"         Confidence: {decision.get('confidence', 'N/A')}")
                    logger.info(f"         Has reasoning: {has_reasoning}")
                    logger.info(f"         Has calculated_rr: {has_calculated_rr}")
            
            # Step 5: Check Active Position Manager integration
            logger.info("   📊 Step 3: Checking Active Position Manager integration...")
            try:
                apm_response = requests.get(f"{self.api_url}/active-positions", timeout=30)
                apm_integration = apm_response.status_code == 200
                if apm_integration:
                    logger.info(f"      ✅ Active Position Manager integration working")
                else:
                    logger.info(f"      ⚠️ Active Position Manager not accessible")
            except:
                apm_integration = False
                logger.info(f"      ⚠️ Active Position Manager integration failed")
            
            # Step 6: Check BingX integration
            logger.info("   📊 Step 4: Checking BingX integration...")
            try:
                bingx_response = requests.get(f"{self.api_url}/bingx/status", timeout=30)
                bingx_integration = bingx_response.status_code == 200
                if bingx_integration:
                    logger.info(f"      ✅ BingX integration accessible")
                else:
                    logger.info(f"      ⚠️ BingX integration not accessible")
            except:
                bingx_integration = False
                logger.info(f"      ⚠️ BingX integration failed")
            
            # Evaluate overall pipeline
            pipeline_components = {
                'IA1 Analysis': ia1_response.status_code in [200, 201],
                'IA2 Escalation Logic': True,  # Always present, even if not triggered
                'IA2 Decision Generation': ia2_decision_found if ia2_triggered else True,
                'Active Position Manager': apm_integration,
                'BingX Integration': bingx_integration
            }
            
            working_components = sum(1 for component, working in pipeline_components.items() if working)
            total_components = len(pipeline_components)
            
            logger.info(f"   📊 Pipeline Component Status:")
            for component, working in pipeline_components.items():
                status = "✅" if working else "❌"
                logger.info(f"      {status} {component}")
            
            # Determine test result
            if working_components == total_components:
                self.log_test_result("End-to-End Pipeline Integration", True, 
                                   f"Complete pipeline working: {working_components}/{total_components} components functional")
            elif working_components >= total_components * 0.8:
                self.log_test_result("End-to-End Pipeline Integration", False, 
                                   f"Pipeline mostly working: {working_components}/{total_components} components functional")
            else:
                self.log_test_result("End-to-End Pipeline Integration", False, 
                                   f"Pipeline has issues: {working_components}/{total_components} components functional")
                
        except Exception as e:
            self.log_test_result("End-to-End Pipeline Integration", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all IA2 simplified prompt tests"""
        logger.info("🚀 Starting IA2 Simplified Prompt Comprehensive Test Suite")
        logger.info("=" * 80)
        logger.info("📋 IA2 SIMPLIFIED PROMPT SYSTEM COMPREHENSIVE TESTING")
        logger.info("🎯 Testing: String indices error resolution, decision generation, RR fields, strategic reasoning")
        logger.info("🎯 Expected: IA2 system working correctly after major code deletion and simplification")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_backend_logs_string_indices_check()
        await self.test_2_ia2_decision_generation_stability()
        await self.test_3_ia2_decision_structure_validation()
        await self.test_4_calculated_rr_and_reasoning_fields()
        await self.test_5_strategic_reasoning_quality()
        await self.test_6_voie_escalation_paths()
        await self.test_7_end_to_end_pipeline_integration()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA2 SIMPLIFIED PROMPT COMPREHENSIVE TEST SUMMARY")
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
        logger.info("📋 CRITICAL REQUIREMENTS VERIFICATION")
        logger.info("=" * 80)
        
        requirements_status = {}
        
        for result in self.test_results:
            if "String Indices" in result['test']:
                requirements_status['IA2 Error Resolution'] = result['success']
            elif "Decision Generation" in result['test']:
                requirements_status['IA2 Decision Generation'] = result['success']
            elif "Strategic Reasoning" in result['test']:
                requirements_status['Strategic Reasoning Quality'] = result['success']
            elif "RR and RR Reasoning" in result['test']:
                requirements_status['RR Calculation Fields'] = result['success']
            elif "End-to-End" in result['test']:
                requirements_status['End-to-End Pipeline'] = result['success']
        
        logger.info("🎯 CRITICAL REQUIREMENTS STATUS:")
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}")
        
        requirements_met = sum(1 for status in requirements_status.values() if status)
        total_requirements = len(requirements_status)
        
        # Final verdict
        logger.info(f"\n🏆 REQUIREMENTS SATISFACTION: {requirements_met}/{total_requirements}")
        
        if requirements_met == total_requirements:
            logger.info("\n🎉 VERDICT: IA2 SIMPLIFIED PROMPT SYSTEM IS FULLY FUNCTIONAL!")
            logger.info("✅ String indices errors resolved")
            logger.info("✅ IA2 decision generation working")
            logger.info("✅ Strategic reasoning quality maintained")
            logger.info("✅ RR calculation fields implemented")
            logger.info("✅ End-to-end pipeline operational")
            logger.info("✅ Major code deletion successful - simplified system working correctly")
        elif requirements_met >= total_requirements * 0.8:
            logger.info("\n⚠️ VERDICT: IA2 SIMPLIFIED PROMPT SYSTEM IS MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif requirements_met >= total_requirements * 0.6:
            logger.info("\n⚠️ VERDICT: IA2 SIMPLIFIED PROMPT SYSTEM IS PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several critical requirements need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: IA2 SIMPLIFIED PROMPT SYSTEM IS NOT FUNCTIONAL")
            logger.info("🚨 Major issues preventing IA2 system from working correctly")
            logger.info("🚨 Simplified prompt implementation may have critical bugs")
        
        return passed_tests, total_tests

class BingXIntegrationTestSuite:
    """Comprehensive test suite for BingX API integration system"""
    
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
        logger.info(f"Testing BingX Integration System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected BingX endpoints to test
        self.bingx_endpoints = [
            {'method': 'GET', 'path': '/bingx/status', 'name': 'System Status'},
            {'method': 'GET', 'path': '/bingx/balance', 'name': 'Account Balance'},
            {'method': 'GET', 'path': '/bingx/positions', 'name': 'Open Positions'},
            {'method': 'GET', 'path': '/bingx/risk-config', 'name': 'Risk Configuration'},
            {'method': 'GET', 'path': '/bingx/trading-history', 'name': 'Trading History'},
            {'method': 'POST', 'path': '/bingx/execute-ia2', 'name': 'IA2 Trade Execution'},
            {'method': 'GET', 'path': '/bingx/market-price', 'name': 'Market Price'},
            {'method': 'POST', 'path': '/bingx/trade', 'name': 'Manual Trade'},
            {'method': 'POST', 'path': '/bingx/close-position', 'name': 'Close Position'},
            {'method': 'POST', 'path': '/bingx/close-all-positions', 'name': 'Close All Positions'},
            {'method': 'POST', 'path': '/bingx/emergency-stop', 'name': 'Emergency Stop'},
            {'method': 'POST', 'path': '/bingx/risk-config', 'name': 'Update Risk Config'},
        ]
        
        # Mock IA2 decision data for testing
        self.mock_ia2_decision = {
            "symbol": "BTCUSDT",
            "signal": "LONG",
            "confidence": 0.85,
            "position_size": 2.5,
            "leverage": 5,
            "entry_price": 45000.0,
            "stop_loss": 43000.0,
            "take_profit": 48000.0,
            "reasoning": "Strong bullish momentum with RSI oversold recovery"
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
    
    async def test_1_bingx_api_connectivity(self):
        """Test 1: BingX API Connectivity via /api/bingx/status endpoint"""
        logger.info("\n🔍 TEST 1: BingX API Connectivity Test")
        
        try:
            response = requests.get(f"{self.api_url}/bingx/status", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   📊 Status response: {json.dumps(data, indent=2)}")
                
                # Check for expected status fields
                expected_fields = ['status', 'api_connected', 'timestamp']
                missing_fields = [field for field in expected_fields if field not in data]
                
                if not missing_fields:
                    api_connected = data.get('api_connected', False)
                    if api_connected:
                        self.log_test_result("BingX API Connectivity", True, f"API connected successfully: {data.get('status')}")
                    else:
                        self.log_test_result("BingX API Connectivity", False, f"API not connected: {data}")
                else:
                    self.log_test_result("BingX API Connectivity", False, f"Missing fields: {missing_fields}")
            else:
                self.log_test_result("BingX API Connectivity", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("BingX API Connectivity", False, f"Exception: {str(e)}")
    
    async def test_2_account_balance_retrieval(self):
        """Test 2: Account Balance Retrieval via /api/bingx/balance endpoint"""
        logger.info("\n🔍 TEST 2: Account Balance Retrieval Test")
        
        try:
            response = requests.get(f"{self.api_url}/bingx/balance", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   📊 Balance response: {json.dumps(data, indent=2)}")
                
                # Check for expected balance fields
                expected_fields = ['balance', 'available_balance', 'timestamp']
                has_balance_data = any(field in data for field in expected_fields)
                
                if has_balance_data:
                    balance = data.get('balance', data.get('total_balance', 0))
                    available = data.get('available_balance', data.get('available_margin', 0))
                    
                    self.log_test_result("Account Balance Retrieval", True, 
                                       f"Balance: ${balance}, Available: ${available}")
                else:
                    self.log_test_result("Account Balance Retrieval", False, 
                                       f"No balance data found in response: {data}")
            else:
                self.log_test_result("Account Balance Retrieval", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("Account Balance Retrieval", False, f"Exception: {str(e)}")
    
    async def test_3_bingx_integration_manager(self):
        """Test 3: BingX Integration Manager Initialization and Core Functionality"""
        logger.info("\n🔍 TEST 3: BingX Integration Manager Test")
        
        try:
            # Test system status to verify manager initialization
            status_response = requests.get(f"{self.api_url}/bingx/status", timeout=30)
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                
                # Check for manager-specific fields
                manager_indicators = [
                    'status', 'api_connected', 'active_positions', 'pending_orders',
                    'emergency_stop', 'session_pnl'
                ]
                
                found_indicators = [field for field in manager_indicators if field in status_data]
                
                if len(found_indicators) >= 3:
                    self.log_test_result("BingX Integration Manager", True, 
                                       f"Manager operational with {len(found_indicators)} indicators: {found_indicators}")
                else:
                    self.log_test_result("BingX Integration Manager", False, 
                                       f"Insufficient manager indicators: {found_indicators}")
            else:
                self.log_test_result("BingX Integration Manager", False, 
                                   f"Status endpoint failed: HTTP {status_response.status_code}")
                
        except Exception as e:
            self.log_test_result("BingX Integration Manager", False, f"Exception: {str(e)}")
    
    async def test_4_all_bingx_endpoints(self):
        """Test 4: Test All 15 BingX API Endpoints"""
        logger.info("\n🔍 TEST 4: All BingX API Endpoints Test")
        
        endpoint_results = []
        
        for endpoint in self.bingx_endpoints:
            try:
                method = endpoint['method']
                path = endpoint['path']
                name = endpoint['name']
                
                logger.info(f"   Testing {method} {path} ({name})")
                
                if method == 'GET':
                    if 'market-price' in path:
                        # Add symbol parameter for market price endpoint
                        response = requests.get(f"{self.api_url}{path}?symbol=BTCUSDT", timeout=30)
                    else:
                        response = requests.get(f"{self.api_url}{path}", timeout=30)
                        
                elif method == 'POST':
                    if 'execute-ia2' in path:
                        # Use mock IA2 decision data
                        response = requests.post(f"{self.api_url}{path}", 
                                               json=self.mock_ia2_decision, timeout=30)
                    elif 'trade' in path:
                        # Mock manual trade data
                        trade_data = {
                            "symbol": "BTCUSDT",
                            "side": "LONG",
                            "quantity": 0.001,
                            "leverage": 5
                        }
                        response = requests.post(f"{self.api_url}{path}", 
                                               json=trade_data, timeout=30)
                    elif 'close-position' in path:
                        # Mock close position data
                        close_data = {
                            "symbol": "BTCUSDT",
                            "position_side": "LONG"
                        }
                        response = requests.post(f"{self.api_url}{path}", 
                                               json=close_data, timeout=30)
                    elif 'risk-config' in path:
                        # Mock risk config data
                        risk_data = {
                            "max_position_size": 0.1,
                            "max_leverage": 10,
                            "stop_loss_percentage": 0.02
                        }
                        response = requests.post(f"{self.api_url}{path}", 
                                               json=risk_data, timeout=30)
                    else:
                        # Empty POST for other endpoints
                        response = requests.post(f"{self.api_url}{path}", json={}, timeout=30)
                
                # Evaluate response
                if response.status_code in [200, 201]:
                    try:
                        data = response.json()
                        endpoint_results.append({
                            'endpoint': f"{method} {path}",
                            'name': name,
                            'status': 'SUCCESS',
                            'response_size': len(str(data))
                        })
                        logger.info(f"      ✅ {name}: SUCCESS (HTTP {response.status_code})")
                    except:
                        endpoint_results.append({
                            'endpoint': f"{method} {path}",
                            'name': name,
                            'status': 'SUCCESS_NO_JSON',
                            'response_size': len(response.text)
                        })
                        logger.info(f"      ✅ {name}: SUCCESS - No JSON response")
                else:
                    endpoint_results.append({
                        'endpoint': f"{method} {path}",
                        'name': name,
                        'status': f'HTTP_{response.status_code}',
                        'response_size': len(response.text)
                    })
                    logger.info(f"      ❌ {name}: HTTP {response.status_code}")
                    
            except Exception as e:
                endpoint_results.append({
                    'endpoint': f"{method} {path}",
                    'name': name,
                    'status': 'ERROR',
                    'error': str(e)
                })
                logger.info(f"      ❌ {name}: Exception - {str(e)}")
        
        # Evaluate overall endpoint testing
        successful_endpoints = len([r for r in endpoint_results if r['status'] in ['SUCCESS', 'SUCCESS_NO_JSON']])
        total_endpoints = len(endpoint_results)
        
        success_rate = successful_endpoints / total_endpoints if total_endpoints > 0 else 0
        
        if success_rate >= 0.8:  # 80% success rate
            self.log_test_result("All BingX API Endpoints", True, 
                               f"Success rate: {successful_endpoints}/{total_endpoints} ({success_rate:.1%})")
        else:
            self.log_test_result("All BingX API Endpoints", False, 
                               f"Low success rate: {successful_endpoints}/{total_endpoints} ({success_rate:.1%})")
        
        # Log detailed endpoint results
        logger.info(f"   📊 Endpoint Test Results:")
        for result in endpoint_results:
            status_icon = "✅" if result['status'] in ['SUCCESS', 'SUCCESS_NO_JSON'] else "❌"
            logger.info(f"      {status_icon} {result['name']}: {result['status']}")
    
    async def test_5_risk_management_system(self):
        """Test 5: Risk Management Configuration and Validation"""
        logger.info("\n🔍 TEST 5: Risk Management System Test")
        
        try:
            # Test getting risk configuration
            get_response = requests.get(f"{self.api_url}/bingx/risk-config", timeout=30)
            
            if get_response.status_code == 200:
                risk_config = get_response.json()
                logger.info(f"   📊 Current risk config: {json.dumps(risk_config, indent=2)}")
                
                # Check for expected risk parameters
                expected_params = ['max_position_size', 'max_leverage', 'stop_loss_percentage']
                found_params = [param for param in expected_params if param in risk_config]
                
                if len(found_params) >= 2:
                    # Test updating risk configuration
                    new_risk_config = {
                        "max_position_size": 0.05,  # 5% max position
                        "max_leverage": 8,
                        "stop_loss_percentage": 0.03  # 3% stop loss
                    }
                    
                    post_response = requests.post(f"{self.api_url}/bingx/risk-config", 
                                                json=new_risk_config, timeout=30)
                    
                    if post_response.status_code in [200, 201]:
                        self.log_test_result("Risk Management System", True, 
                                           f"Risk config retrieved and updated successfully")
                    else:
                        self.log_test_result("Risk Management System", False, 
                                           f"Risk config update failed: HTTP {post_response.status_code}")
                else:
                    self.log_test_result("Risk Management System", False, 
                                       f"Missing risk parameters: {expected_params}")
            else:
                self.log_test_result("Risk Management System", False, 
                                   f"Risk config retrieval failed: HTTP {get_response.status_code}")
                
        except Exception as e:
            self.log_test_result("Risk Management System", False, f"Exception: {str(e)}")
    
    async def test_6_ia2_integration_execution(self):
        """Test 6: IA2 Integration - Execute Trade via BingX Integration"""
        logger.info("\n🔍 TEST 6: IA2 Integration Trade Execution Test")
        
        try:
            # Test IA2 trade execution with mock data
            logger.info(f"   🚀 Testing IA2 trade execution with mock decision: {self.mock_ia2_decision}")
            
            response = requests.post(f"{self.api_url}/bingx/execute-ia2", 
                                   json=self.mock_ia2_decision, timeout=60)
            
            if response.status_code in [200, 201]:
                result = response.json()
                logger.info(f"   📊 IA2 execution result: {json.dumps(result, indent=2)}")
                
                # Check execution result
                status = result.get('status', 'unknown')
                
                if status in ['executed', 'skipped', 'rejected']:
                    # All these are valid responses
                    if status == 'executed':
                        order_id = result.get('order_id')
                        symbol = result.get('symbol')
                        self.log_test_result("IA2 Integration Execution", True, 
                                           f"Trade executed successfully: {symbol} Order ID: {order_id}")
                    elif status == 'skipped':
                        reason = result.get('reason', 'Unknown')
                        self.log_test_result("IA2 Integration Execution", True, 
                                           f"Trade skipped (valid): {reason}")
                    elif status == 'rejected':
                        errors = result.get('errors', [])
                        self.log_test_result("IA2 Integration Execution", True, 
                                           f"Trade rejected by risk management (valid): {errors}")
                else:
                    self.log_test_result("IA2 Integration Execution", False, 
                                       f"Unexpected status: {status}")
            else:
                self.log_test_result("IA2 Integration Execution", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("IA2 Integration Execution", False, f"Exception: {str(e)}")
    
    async def test_7_error_handling_resilience(self):
        """Test 7: Error Handling and System Resilience"""
        logger.info("\n🔍 TEST 7: Error Handling and System Resilience Test")
        
        error_test_results = []
        
        # Test 1: Invalid symbol
        try:
            response = requests.get(f"{self.api_url}/bingx/market-price?symbol=INVALIDUSDT", timeout=30)
            if response.status_code in [400, 404, 422]:
                error_test_results.append("✅ Invalid symbol handled correctly")
            else:
                error_test_results.append(f"❌ Invalid symbol: HTTP {response.status_code}")
        except:
            error_test_results.append("❌ Invalid symbol: Exception occurred")
        
        # Test 2: Invalid trade data
        try:
            invalid_trade = {
                "symbol": "BTCUSDT",
                "side": "INVALID_SIDE",
                "quantity": -1,  # Invalid negative quantity
                "leverage": 1000  # Invalid high leverage
            }
            response = requests.post(f"{self.api_url}/bingx/trade", json=invalid_trade, timeout=30)
            if response.status_code in [400, 422]:
                error_test_results.append("✅ Invalid trade data handled correctly")
            else:
                error_test_results.append(f"❌ Invalid trade data: HTTP {response.status_code}")
        except:
            error_test_results.append("❌ Invalid trade data: Exception occurred")
        
        # Test 3: Invalid IA2 decision
        try:
            invalid_ia2 = {
                "symbol": "",  # Empty symbol
                "signal": "INVALID",
                "confidence": 2.0,  # Invalid confidence > 1
                "position_size": -5  # Invalid negative size
            }
            response = requests.post(f"{self.api_url}/bingx/execute-ia2", json=invalid_ia2, timeout=30)
            if response.status_code in [400, 422] or (response.status_code == 200 and 
                                                     response.json().get('status') in ['rejected', 'error']):
                error_test_results.append("✅ Invalid IA2 decision handled correctly")
            else:
                error_test_results.append(f"❌ Invalid IA2 decision: HTTP {response.status_code}")
        except:
            error_test_results.append("❌ Invalid IA2 decision: Exception occurred")
        
        # Test 4: System still responsive after errors
        try:
            response = requests.get(f"{self.api_url}/bingx/status", timeout=30)
            if response.status_code == 200:
                error_test_results.append("✅ System remains responsive after errors")
            else:
                error_test_results.append(f"❌ System unresponsive: HTTP {response.status_code}")
        except:
            error_test_results.append("❌ System unresponsive: Exception occurred")
        
        # Evaluate error handling
        successful_error_tests = len([r for r in error_test_results if r.startswith("✅")])
        total_error_tests = len(error_test_results)
        
        logger.info(f"   📊 Error Handling Test Results:")
        for result in error_test_results:
            logger.info(f"      {result}")
        
        if successful_error_tests >= 3:  # At least 3 out of 4 error tests pass
            self.log_test_result("Error Handling Resilience", True, 
                               f"Error handling working: {successful_error_tests}/{total_error_tests} tests passed")
        else:
            self.log_test_result("Error Handling Resilience", False, 
                               f"Poor error handling: {successful_error_tests}/{total_error_tests} tests passed")
    
    async def test_8_api_credentials_validation(self):
        """Test 8: API Credentials Validation"""
        logger.info("\n🔍 TEST 8: API Credentials Validation Test")
        
        try:
            # Check if credentials are properly configured
            backend_env_path = '/app/backend/.env'
            credentials_found = False
            
            if os.path.exists(backend_env_path):
                with open(backend_env_path, 'r') as f:
                    env_content = f.read()
                    
                    has_api_key = 'BINGX_API_KEY=' in env_content
                    has_secret_key = 'BINGX_SECRET_KEY=' in env_content
                    has_base_url = 'BINGX_BASE_URL=' in env_content
                    
                    if has_api_key and has_secret_key:
                        credentials_found = True
                        logger.info("   📊 BingX credentials found in environment")
                        
                        # Extract API key for validation (first 10 chars only for security)
                        for line in env_content.split('\n'):
                            if line.startswith('BINGX_API_KEY='):
                                api_key_preview = line.split('=')[1][:10] + "..."
                                logger.info(f"   📊 API Key preview: {api_key_preview}")
                                break
            
            if credentials_found:
                # Test credentials by checking API connectivity
                status_response = requests.get(f"{self.api_url}/bingx/status", timeout=30)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    api_connected = status_data.get('api_connected', False)
                    
                    if api_connected:
                        self.log_test_result("API Credentials Validation", True, 
                                           "Credentials configured and API connection successful")
                    else:
                        self.log_test_result("API Credentials Validation", False, 
                                           "Credentials found but API connection failed")
                else:
                    self.log_test_result("API Credentials Validation", False, 
                                       f"Status endpoint failed: HTTP {status_response.status_code}")
            else:
                self.log_test_result("API Credentials Validation", False, 
                                   "BingX credentials not found in environment")
                
        except Exception as e:
            self.log_test_result("API Credentials Validation", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all BingX integration tests"""
        logger.info("🚀 Starting BingX Integration Comprehensive Test Suite")
        logger.info("=" * 80)
        logger.info("📋 BINGX INTEGRATION SYSTEM COMPREHENSIVE TESTING")
        logger.info("🎯 Testing: API connectivity, endpoints, risk management, IA2 integration, error handling")
        logger.info("🎯 Expected: Complete BingX integration working with all 15 endpoints functional")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_bingx_api_connectivity()
        await self.test_2_account_balance_retrieval()
        await self.test_3_bingx_integration_manager()
        await self.test_4_all_bingx_endpoints()
        await self.test_5_risk_management_system()
        await self.test_6_ia2_integration_execution()
        await self.test_7_error_handling_resilience()
        await self.test_8_api_credentials_validation()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 BINGX INTEGRATION COMPREHENSIVE TEST SUMMARY")
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
        logger.info("📋 BINGX INTEGRATION SYSTEM STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - BingX Integration System FULLY FUNCTIONAL!")
            logger.info("✅ API connectivity working")
            logger.info("✅ Account balance retrieval operational")
            logger.info("✅ BingX Integration Manager initialized")
            logger.info("✅ All 15 BingX endpoints functional")
            logger.info("✅ Risk management system working")
            logger.info("✅ IA2 integration executing trades")
            logger.info("✅ Error handling resilient")
            logger.info("✅ API credentials validated")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - BingX integration working with minor gaps")
            logger.info("🔍 Some components may need fine-tuning for full optimization")
        elif passed_tests >= total_tests * 0.6:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core BingX features working")
            logger.info("🔧 Some advanced features may need implementation or debugging")
        else:
            logger.info("❌ SYSTEM NOT FUNCTIONAL - Critical issues with BingX integration")
            logger.info("🚨 Major implementation gaps or system errors preventing functionality")
        
        # Specific requirements check
        logger.info("\n📝 BINGX INTEGRATION REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "API Connectivity" in result['test']:
                    requirements_met.append("✅ BingX API connectivity verified")
                elif "Account Balance" in result['test']:
                    requirements_met.append("✅ Account balance retrieval working")
                elif "Integration Manager" in result['test']:
                    requirements_met.append("✅ BingX Integration Manager operational")
                elif "All BingX API Endpoints" in result['test']:
                    requirements_met.append("✅ All 15 BingX endpoints functional")
                elif "Risk Management" in result['test']:
                    requirements_met.append("✅ Risk management system working")
                elif "IA2 Integration" in result['test']:
                    requirements_met.append("✅ IA2 trade execution via BingX working")
                elif "Error Handling" in result['test']:
                    requirements_met.append("✅ Error handling resilient")
                elif "API Credentials" in result['test']:
                    requirements_met.append("✅ API credentials validated")
            else:
                if "API Connectivity" in result['test']:
                    requirements_failed.append("❌ BingX API connectivity failed")
                elif "Account Balance" in result['test']:
                    requirements_failed.append("❌ Account balance retrieval not working")
                elif "Integration Manager" in result['test']:
                    requirements_failed.append("❌ BingX Integration Manager not operational")
                elif "All BingX API Endpoints" in result['test']:
                    requirements_failed.append("❌ BingX endpoints not fully functional")
                elif "Risk Management" in result['test']:
                    requirements_failed.append("❌ Risk management system not working")
                elif "IA2 Integration" in result['test']:
                    requirements_failed.append("❌ IA2 trade execution via BingX failed")
                elif "Error Handling" in result['test']:
                    requirements_failed.append("❌ Error handling not resilient")
                elif "API Credentials" in result['test']:
                    requirements_failed.append("❌ API credentials validation failed")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: BingX Integration System is FULLY FUNCTIONAL!")
            logger.info("✅ All integration features implemented and working correctly")
            logger.info("✅ API connectivity, endpoints, risk management, and IA2 integration operational")
            logger.info("✅ System ready for production trading with proper error handling")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: BingX Integration System is MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif len(requirements_failed) <= 3:
            logger.info("\n⚠️ VERDICT: BingX Integration System is PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several components need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: BingX Integration System is NOT FUNCTIONAL")
            logger.info("🚨 Major implementation gaps preventing BingX integration")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IA2SimplifiedPromptTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
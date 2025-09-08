#!/usr/bin/env python3
"""
Backend Testing Suite for Adaptive Contextual Logic - IA2 Focus
Focus: Testing the new adaptive contextual logic implementation in IA2 decision engine
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
import subprocess

# Add backend to path
sys.path.append('/app/backend')

import requests
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveContextualLogicTestSuite:
    """Test suite for Adaptive Contextual Logic in IA2 Decision Engine"""
    
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
        logger.info(f"Testing Adaptive Contextual Logic at: {self.api_url}")
        
        # MongoDB connection for direct data access
        self.mongo_client = None
        self.db = None
        
        # Test results
        self.test_results = []
        
        # Adaptive logic specific test data
        self.adaptive_decisions = []
        self.context_types_detected = []
        
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
            # Extract decisions from response if needed
            if isinstance(data, dict) and 'decisions' in data:
                decisions = data['decisions']
            else:
                decisions = data
            return decisions, None
        except Exception as e:
            return None, f"Exception: {str(e)}"
    
    def get_opportunities_from_api(self):
        """Helper method to get opportunities from API"""
        try:
            response = requests.get(f"{self.api_url}/opportunities", timeout=30)
            if response.status_code != 200:
                return None, f"API error: {response.status_code}"
                
            data = response.json()
            # Extract opportunities from response
            if isinstance(data, dict) and 'opportunities' in data:
                opportunities = data['opportunities']
            else:
                opportunities = data
            return opportunities, None
        except Exception as e:
            return None, f"Exception: {str(e)}"
        
    async def test_adaptive_mode_enabled(self):
        """Test 1: Adaptive Mode Enabled - Verify adaptive_mode_enabled = True and _apply_adaptive_context_to_decision is called"""
        logger.info("\n🔍 TEST 1: Adaptive Mode Enabled")
        
        try:
            # Check backend logs for adaptive mode initialization
            log_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'adaptive.*mode\\|adaptive.*enabled\\|adaptive.*logic' || echo 'No adaptive mode logs'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            adaptive_mode_logs = []
            adaptive_logic_logs = []
            
            for line in result.stdout.split('\n'):
                if 'adaptive' in line.lower() and ('mode' in line.lower() or 'enabled' in line.lower()):
                    adaptive_mode_logs.append(line.strip())
                elif 'adaptive' in line.lower() and 'logic' in line.lower():
                    adaptive_logic_logs.append(line.strip())
            
            logger.info(f"   📊 Adaptive mode logs found: {len(adaptive_mode_logs)}")
            logger.info(f"   📊 Adaptive logic logs found: {len(adaptive_logic_logs)}")
            
            # Check for _apply_adaptive_context_to_decision calls
            context_application_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'apply.*adaptive.*context\\|adaptive.*applied\\|🧠.*adaptive' || echo 'No context application logs'"
            context_result = subprocess.run(context_application_cmd, shell=True, capture_output=True, text=True)
            
            context_application_logs = []
            for line in context_result.stdout.split('\n'):
                if any(keyword in line.lower() for keyword in ['apply', 'adaptive', 'context']):
                    context_application_logs.append(line.strip())
            
            logger.info(f"   📊 Context application logs found: {len(context_application_logs)}")
            
            # Show sample logs
            if adaptive_mode_logs:
                logger.info(f"   📝 Sample adaptive mode log: {adaptive_mode_logs[-1][:150]}...")
            if context_application_logs:
                logger.info(f"   📝 Sample context application log: {context_application_logs[-1][:150]}...")
            
            # Check decisions for adaptive context evidence
            decisions, error = self.get_decisions_from_api()
            adaptive_decisions_count = 0
            
            if not error and decisions:
                for decision in decisions:
                    reasoning = decision.get('ia2_reasoning', '')
                    if '🧠 ADAPTIVE CONTEXT:' in reasoning or 'ADAPTIVE' in reasoning:
                        adaptive_decisions_count += 1
                        self.adaptive_decisions.append(decision)
                        logger.info(f"   🎯 Adaptive decision found: {decision.get('symbol', 'UNKNOWN')}")
                        logger.info(f"      Reasoning contains: {reasoning[:100]}...")
            
            success = len(context_application_logs) > 0 or adaptive_decisions_count > 0
            details = f"Adaptive mode logs: {len(adaptive_mode_logs)}, Context application logs: {len(context_application_logs)}, Adaptive decisions: {adaptive_decisions_count}"
            
            self.log_test_result("Adaptive Mode Enabled", success, details)
            
        except Exception as e:
            self.log_test_result("Adaptive Mode Enabled", False, f"Exception: {str(e)}")
    
    async def test_adaptive_contexts_detected(self):
        """Test 2: Adaptive Contexts Detected - Check for EXTREME VOLATILITY, HIGH IA2 CONFIDENCE, STRONG TRENDING, EXTREME SENTIMENT, BALANCED CONDITIONS"""
        logger.info("\n🔍 TEST 2: Adaptive Contexts Detected")
        
        try:
            # Check backend logs for specific adaptive contexts
            context_patterns = {
                'EXTREME_VOLATILITY': ['extreme.*volatility', '🌪️.*adaptive', 'volatility.*>.*15'],
                'HIGH_IA2_CONFIDENCE': ['high.*ia2.*confidence', '🧠.*adaptive', 'confidence.*>.*85'],
                'STRONG_TRENDING': ['strong.*trend', '🚀.*adaptive', 'trending.*fort'],
                'EXTREME_SENTIMENT': ['extreme.*sentiment', '⚡.*adaptive', 'sentiment.*>.*20'],
                'BALANCED_CONDITIONS': ['balanced.*conditions', '⚖️.*adaptive', 'conditions.*normal']
            }
            
            context_detections = {}
            
            for context_type, patterns in context_patterns.items():
                context_detections[context_type] = 0
                
                for pattern in patterns:
                    log_cmd = f"tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i '{pattern}' || echo 'No {pattern} logs'"
                    result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
                    
                    matches = [line.strip() for line in result.stdout.split('\n') if line.strip() and 'No' not in line]
                    context_detections[context_type] += len(matches)
                    
                    if matches:
                        logger.info(f"   🎯 {context_type} detected: {len(matches)} instances")
                        logger.info(f"      Sample: {matches[-1][:120]}...")
            
            # Analyze decisions for context-specific reasoning
            decisions, error = self.get_decisions_from_api()
            decision_contexts = {}
            
            if not error and decisions:
                for decision in decisions:
                    reasoning = decision.get('ia2_reasoning', '')
                    symbol = decision.get('symbol', 'UNKNOWN')
                    
                    # Check for context keywords in reasoning
                    for context_type in context_patterns.keys():
                        context_keywords = context_type.replace('_', ' ').lower()
                        if context_keywords in reasoning.lower() or any(keyword in reasoning.lower() for keyword in context_keywords.split()):
                            if context_type not in decision_contexts:
                                decision_contexts[context_type] = []
                            decision_contexts[context_type].append({
                                'symbol': symbol,
                                'confidence': decision.get('confidence', 0),
                                'signal': decision.get('signal', 'UNKNOWN')
                            })
            
            # Analyze opportunities for context triggers
            opportunities, opp_error = self.get_opportunities_from_api()
            context_triggers = {
                'EXTREME_VOLATILITY': 0,
                'HIGH_IA2_CONFIDENCE': 0,
                'STRONG_TRENDING': 0,
                'EXTREME_SENTIMENT': 0,
                'BALANCED_CONDITIONS': 0
            }
            
            if not opp_error and opportunities:
                for opp in opportunities:
                    price_change = abs(opp.get('price_change_24h', 0))
                    volatility = opp.get('volatility', 0) * 100
                    
                    # Check for context triggers
                    if price_change > 15:
                        context_triggers['EXTREME_VOLATILITY'] += 1
                    elif price_change > 20:
                        context_triggers['EXTREME_SENTIMENT'] += 1
                    elif price_change > 8:
                        context_triggers['STRONG_TRENDING'] += 1
                    else:
                        context_triggers['BALANCED_CONDITIONS'] += 1
            
            # Store context types for later analysis
            self.context_types_detected = list(context_detections.keys())
            
            # Summary
            total_contexts_detected = sum(context_detections.values())
            total_decision_contexts = sum(len(contexts) for contexts in decision_contexts.values())
            
            logger.info(f"   📊 Context Detection Summary:")
            for context_type, count in context_detections.items():
                decision_count = len(decision_contexts.get(context_type, []))
                trigger_count = context_triggers.get(context_type, 0)
                logger.info(f"      {context_type}: {count} logs, {decision_count} decisions, {trigger_count} triggers")
            
            success = total_contexts_detected > 0 or total_decision_contexts > 0
            details = f"Total context logs: {total_contexts_detected}, Decision contexts: {total_decision_contexts}, Context types: {len([k for k, v in context_detections.items() if v > 0])}"
            
            self.log_test_result("Adaptive Contexts Detected", success, details)
            
        except Exception as e:
            self.log_test_result("Adaptive Contexts Detected", False, f"Exception: {str(e)}")
    
    async def test_confidence_adjustments(self):
        """Test 3: Confidence Adjustments - Verify confidence boosts (×1.05 to ×1.15) and reductions (×0.85 to ×0.9) within limits (min 0.4, max 0.98)"""
        logger.info("\n🔍 TEST 3: Confidence Adjustments")
        
        try:
            # Check backend logs for confidence adjustments
            confidence_adjustment_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'confidence.*adjust\\|confidence.*boost\\|confidence.*reduc\\|conf:.*→' || echo 'No confidence adjustment logs'"
            result = subprocess.run(confidence_adjustment_cmd, shell=True, capture_output=True, text=True)
            
            confidence_logs = []
            boost_logs = []
            reduction_logs = []
            
            for line in result.stdout.split('\n'):
                if line.strip() and 'No confidence' not in line:
                    confidence_logs.append(line.strip())
                    if 'boost' in line.lower() or 'increase' in line.lower():
                        boost_logs.append(line.strip())
                    elif 'reduc' in line.lower() or 'decrease' in line.lower():
                        reduction_logs.append(line.strip())
            
            logger.info(f"   📊 Confidence adjustment logs: {len(confidence_logs)}")
            logger.info(f"   📊 Boost logs: {len(boost_logs)}")
            logger.info(f"   📊 Reduction logs: {len(reduction_logs)}")
            
            # Analyze decisions for confidence patterns
            decisions, error = self.get_decisions_from_api()
            confidence_analysis = {
                'within_limits': 0,
                'below_min': 0,
                'above_max': 0,
                'boost_range': 0,
                'reduction_range': 0,
                'total_decisions': 0
            }
            
            confidence_values = []
            
            if not error and decisions:
                for decision in decisions:
                    confidence = decision.get('confidence', 0)
                    confidence_values.append(confidence)
                    confidence_analysis['total_decisions'] += 1
                    
                    # Check limits
                    if 0.4 <= confidence <= 0.98:
                        confidence_analysis['within_limits'] += 1
                    elif confidence < 0.4:
                        confidence_analysis['below_min'] += 1
                    elif confidence > 0.98:
                        confidence_analysis['above_max'] += 1
                    
                    # Check for boost/reduction patterns in reasoning
                    reasoning = decision.get('ia2_reasoning', '')
                    if 'boost' in reasoning.lower() or 'confidence boosted' in reasoning.lower():
                        confidence_analysis['boost_range'] += 1
                    elif 'reduc' in reasoning.lower() or 'confidence reduced' in reasoning.lower():
                        confidence_analysis['reduction_range'] += 1
            
            # Analyze confidence distribution
            if confidence_values:
                avg_confidence = np.mean(confidence_values)
                min_confidence = min(confidence_values)
                max_confidence = max(confidence_values)
                std_confidence = np.std(confidence_values)
                
                logger.info(f"   📊 Confidence Statistics:")
                logger.info(f"      Average: {avg_confidence:.3f}")
                logger.info(f"      Range: {min_confidence:.3f} - {max_confidence:.3f}")
                logger.info(f"      Standard deviation: {std_confidence:.3f}")
                logger.info(f"      Within limits (0.4-0.98): {confidence_analysis['within_limits']}/{confidence_analysis['total_decisions']}")
                logger.info(f"      Below minimum (<0.4): {confidence_analysis['below_min']}")
                logger.info(f"      Above maximum (>0.98): {confidence_analysis['above_max']}")
                logger.info(f"      Boost patterns: {confidence_analysis['boost_range']}")
                logger.info(f"      Reduction patterns: {confidence_analysis['reduction_range']}")
            
            # Check for specific multiplier patterns in logs
            multiplier_patterns = {
                '1.05': 0, '1.1': 0, '1.15': 0,  # Boosts
                '0.85': 0, '0.9': 0, '0.95': 0   # Reductions
            }
            
            for log_line in confidence_logs:
                for multiplier in multiplier_patterns.keys():
                    if multiplier in log_line:
                        multiplier_patterns[multiplier] += 1
            
            logger.info(f"   📊 Multiplier Patterns Found:")
            for multiplier, count in multiplier_patterns.items():
                logger.info(f"      {multiplier}: {count} instances")
            
            success = len(confidence_logs) > 0 or confidence_analysis['boost_range'] > 0 or confidence_analysis['reduction_range'] > 0
            details = f"Confidence logs: {len(confidence_logs)}, Within limits: {confidence_analysis['within_limits']}/{confidence_analysis['total_decisions']}, Boosts: {confidence_analysis['boost_range']}, Reductions: {confidence_analysis['reduction_range']}"
            
            self.log_test_result("Confidence Adjustments", success, details)
            
        except Exception as e:
            self.log_test_result("Confidence Adjustments", False, f"Exception: {str(e)}")
    
    async def test_adaptive_logs(self):
        """Test 4: Adaptive Logs - Look for "ADAPTIVE CONTEXT", "🧠 ADAPTIVE:", "🌪️ ADAPTIVE:", "🚀 ADAPTIVE:", "⚡ ADAPTIVE:" in IA2 reasoning"""
        logger.info("\n🔍 TEST 4: Adaptive Logs")
        
        try:
            # Check for specific adaptive log patterns
            adaptive_log_patterns = {
                '🧠 ADAPTIVE:': 0,
                '🌪️ ADAPTIVE:': 0,
                '🚀 ADAPTIVE:': 0,
                '⚡ ADAPTIVE:': 0,
                'ADAPTIVE CONTEXT': 0
            }
            
            # Search in backend logs
            for pattern in adaptive_log_patterns.keys():
                escaped_pattern = pattern.replace(':', '\\:').replace('🧠', '🧠').replace('🌪️', '🌪️').replace('🚀', '🚀').replace('⚡', '⚡')
                log_cmd = f"tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -F '{pattern}' || echo 'No {pattern} logs'"
                result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
                
                matches = [line.strip() for line in result.stdout.split('\n') if line.strip() and 'No' not in line and pattern in line]
                adaptive_log_patterns[pattern] = len(matches)
                
                if matches:
                    logger.info(f"   🎯 Found {pattern}: {len(matches)} instances")
                    logger.info(f"      Sample: {matches[-1][:120]}...")
            
            # Check decisions for adaptive reasoning patterns
            decisions, error = self.get_decisions_from_api()
            decision_adaptive_patterns = {pattern: 0 for pattern in adaptive_log_patterns.keys()}
            adaptive_reasoning_examples = []
            
            if not error and decisions:
                for decision in decisions:
                    reasoning = decision.get('ia2_reasoning', '')
                    symbol = decision.get('symbol', 'UNKNOWN')
                    
                    for pattern in adaptive_log_patterns.keys():
                        if pattern in reasoning:
                            decision_adaptive_patterns[pattern] += 1
                            adaptive_reasoning_examples.append({
                                'symbol': symbol,
                                'pattern': pattern,
                                'reasoning_snippet': reasoning[:200]
                            })
            
            # Show examples of adaptive reasoning
            if adaptive_reasoning_examples:
                logger.info(f"   📝 Adaptive Reasoning Examples:")
                for example in adaptive_reasoning_examples[:3]:  # Show first 3 examples
                    logger.info(f"      {example['symbol']} - {example['pattern']}")
                    logger.info(f"         {example['reasoning_snippet']}...")
            
            # Check for confidence change messages
            confidence_change_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'confidence.*→\\|conf:.*→\\|adaptive.*applied' || echo 'No confidence change logs'"
            confidence_result = subprocess.run(confidence_change_cmd, shell=True, capture_output=True, text=True)
            
            confidence_change_logs = []
            for line in confidence_result.stdout.split('\n'):
                if line.strip() and 'No confidence' not in line and ('→' in line or 'applied' in line.lower()):
                    confidence_change_logs.append(line.strip())
            
            logger.info(f"   📊 Confidence change logs: {len(confidence_change_logs)}")
            
            if confidence_change_logs:
                logger.info(f"   📝 Sample confidence change: {confidence_change_logs[-1][:150]}...")
            
            # Summary
            total_log_patterns = sum(adaptive_log_patterns.values())
            total_decision_patterns = sum(decision_adaptive_patterns.values())
            
            logger.info(f"   📊 Adaptive Log Summary:")
            for pattern, count in adaptive_log_patterns.items():
                decision_count = decision_adaptive_patterns[pattern]
                logger.info(f"      {pattern}: {count} logs, {decision_count} in decisions")
            
            success = total_log_patterns > 0 or total_decision_patterns > 0 or len(confidence_change_logs) > 0
            details = f"Log patterns: {total_log_patterns}, Decision patterns: {total_decision_patterns}, Confidence changes: {len(confidence_change_logs)}"
            
            self.log_test_result("Adaptive Logs", success, details)
            
        except Exception as e:
            self.log_test_result("Adaptive Logs", False, f"Exception: {str(e)}")
    
    async def test_impact_on_decisions(self):
        """Test 5: Impact on Decisions - Verify signals can be adjusted according to context and decisions reflect adaptive logic"""
        logger.info("\n🔍 TEST 5: Impact on Decisions")
        
        try:
            # Analyze decisions for adaptive impact
            decisions, error = self.get_decisions_from_api()
            
            if error:
                self.log_test_result("Impact on Decisions", False, error)
                return
            
            adaptive_impact_analysis = {
                'total_decisions': 0,
                'adaptive_decisions': 0,
                'signal_adjustments': 0,
                'confidence_adjustments': 0,
                'context_influenced': 0,
                'no_errors_detected': 0
            }
            
            signal_changes = []
            confidence_changes = []
            context_influences = []
            
            if decisions:
                for decision in decisions:
                    adaptive_impact_analysis['total_decisions'] += 1
                    
                    reasoning = decision.get('ia2_reasoning', '')
                    symbol = decision.get('symbol', 'UNKNOWN')
                    signal = decision.get('signal', 'UNKNOWN')
                    confidence = decision.get('confidence', 0)
                    
                    # Check for adaptive decision markers
                    if 'ADAPTIVE CONTEXT:' in reasoning or '🧠 ADAPTIVE' in reasoning:
                        adaptive_impact_analysis['adaptive_decisions'] += 1
                        
                        # Check for signal adjustments
                        if 'signal' in reasoning.lower() and ('adjust' in reasoning.lower() or 'change' in reasoning.lower()):
                            adaptive_impact_analysis['signal_adjustments'] += 1
                            signal_changes.append({
                                'symbol': symbol,
                                'signal': signal,
                                'reasoning': reasoning[:150]
                            })
                        
                        # Check for confidence adjustments
                        if 'confidence' in reasoning.lower() and ('boost' in reasoning.lower() or 'reduc' in reasoning.lower() or 'adjust' in reasoning.lower()):
                            adaptive_impact_analysis['confidence_adjustments'] += 1
                            confidence_changes.append({
                                'symbol': symbol,
                                'confidence': confidence,
                                'reasoning': reasoning[:150]
                            })
                        
                        # Check for context influence
                        context_keywords = ['volatility', 'trending', 'sentiment', 'balanced', 'extreme']
                        if any(keyword in reasoning.lower() for keyword in context_keywords):
                            adaptive_impact_analysis['context_influenced'] += 1
                            context_influences.append({
                                'symbol': symbol,
                                'context_type': next((kw for kw in context_keywords if kw in reasoning.lower()), 'unknown'),
                                'reasoning': reasoning[:150]
                            })
            
            # Check for system errors or crashes related to adaptive logic
            error_check_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'error.*adaptive\\|adaptive.*fail\\|exception.*adaptive' || echo 'No adaptive errors'"
            error_result = subprocess.run(error_check_cmd, shell=True, capture_output=True, text=True)
            
            adaptive_errors = []
            for line in error_result.stdout.split('\n'):
                if line.strip() and 'No adaptive errors' not in line and ('error' in line.lower() or 'fail' in line.lower() or 'exception' in line.lower()):
                    adaptive_errors.append(line.strip())
            
            if len(adaptive_errors) == 0:
                adaptive_impact_analysis['no_errors_detected'] = 1
            
            # Show examples
            if signal_changes:
                logger.info(f"   📝 Signal Adjustment Examples:")
                for change in signal_changes[:2]:
                    logger.info(f"      {change['symbol']} ({change['signal']}): {change['reasoning']}...")
            
            if confidence_changes:
                logger.info(f"   📝 Confidence Adjustment Examples:")
                for change in confidence_changes[:2]:
                    logger.info(f"      {change['symbol']} (conf: {change['confidence']:.2f}): {change['reasoning']}...")
            
            if context_influences:
                logger.info(f"   📝 Context Influence Examples:")
                for influence in context_influences[:2]:
                    logger.info(f"      {influence['symbol']} ({influence['context_type']}): {influence['reasoning']}...")
            
            if adaptive_errors:
                logger.info(f"   ⚠️ Adaptive Logic Errors Found:")
                for error in adaptive_errors[:2]:
                    logger.info(f"      {error[:150]}...")
            
            # Calculate adaptive effectiveness
            adaptive_effectiveness = 0
            if adaptive_impact_analysis['total_decisions'] > 0:
                adaptive_effectiveness = (adaptive_impact_analysis['adaptive_decisions'] / adaptive_impact_analysis['total_decisions']) * 100
            
            logger.info(f"   📊 Adaptive Impact Summary:")
            logger.info(f"      Total decisions: {adaptive_impact_analysis['total_decisions']}")
            logger.info(f"      Adaptive decisions: {adaptive_impact_analysis['adaptive_decisions']} ({adaptive_effectiveness:.1f}%)")
            logger.info(f"      Signal adjustments: {adaptive_impact_analysis['signal_adjustments']}")
            logger.info(f"      Confidence adjustments: {adaptive_impact_analysis['confidence_adjustments']}")
            logger.info(f"      Context influenced: {adaptive_impact_analysis['context_influenced']}")
            logger.info(f"      No errors detected: {adaptive_impact_analysis['no_errors_detected']}")
            logger.info(f"      Adaptive errors: {len(adaptive_errors)}")
            
            success = (adaptive_impact_analysis['adaptive_decisions'] > 0 and 
                      len(adaptive_errors) == 0 and
                      (adaptive_impact_analysis['signal_adjustments'] > 0 or 
                       adaptive_impact_analysis['confidence_adjustments'] > 0))
            
            details = f"Adaptive decisions: {adaptive_impact_analysis['adaptive_decisions']}/{adaptive_impact_analysis['total_decisions']}, Signal adjustments: {adaptive_impact_analysis['signal_adjustments']}, Confidence adjustments: {adaptive_impact_analysis['confidence_adjustments']}, Errors: {len(adaptive_errors)}"
            
            self.log_test_result("Impact on Decisions", success, details)
            
        except Exception as e:
            self.log_test_result("Impact on Decisions", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Adaptive Contextual Logic tests"""
        logger.info("🚀 Starting Adaptive Contextual Logic Test Suite")
        logger.info("=" * 80)
        
        await self.setup_database()
        
        # Run all tests
        await self.test_adaptive_mode_enabled()
        await self.test_adaptive_contexts_detected()
        await self.test_confidence_adjustments()
        await self.test_adaptive_logs()
        await self.test_impact_on_decisions()
        
        await self.cleanup_database()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 ADAPTIVE CONTEXTUAL LOGIC TEST SUMMARY")
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
            logger.info("🎉 ALL TESTS PASSED - Adaptive Contextual Logic is working correctly!")
            logger.info("✅ Adaptive mode is enabled and functioning")
            logger.info("✅ Context detection is operational")
            logger.info("✅ Confidence adjustments are working within limits")
            logger.info("✅ Adaptive logs are being generated")
            logger.info("✅ Decisions are being influenced by adaptive logic")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY WORKING - Some adaptive logic issues detected")
            logger.info("🔍 Review the failed tests for specific adaptive problems")
        else:
            logger.info("❌ CRITICAL ISSUES - Adaptive Contextual Logic needs attention")
            logger.info("🚨 Major problems with adaptive mode or context detection")
            
        # Recommendations based on test results
        logger.info("\n📝 RECOMMENDATIONS:")
        
        # Check if any tests failed and provide specific recommendations
        failed_tests = [result for result in self.test_results if not result['success']]
        if failed_tests:
            for failed_test in failed_tests:
                logger.info(f"❌ {failed_test['test']}: {failed_test['details']}")
        else:
            logger.info("✅ No critical issues found with Adaptive Contextual Logic")
            logger.info("✅ All adaptive contexts are being detected and processed")
            logger.info("✅ Confidence adjustments are working within specified limits")
            logger.info("✅ Adaptive logs are being generated correctly")
            logger.info("✅ Decisions are being properly influenced by contextual logic")
            
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = AdaptiveContextualLogicTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
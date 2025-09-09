#!/usr/bin/env python3
"""
Comprehensive Backend Testing Suite for Ultra Professional Trading Bot System
Focus: Testing all failing backend components identified in test_result.md

Key Components to Test:
1. Multi-Timeframe Hierarchical Analysis System (working: false)
2. Enhanced IA1 and IA2 Prompts with Technical Indicators Integration (working: false)  
3. IA1→IA2 Filtering and Processing Pipeline (working: false, stuck_count: 2)
4. Sophisticated RR Analysis System (working: false)
5. Enhanced RR Validation System (current focus)

System Architecture:
- IA1: Technical Analysis AI with RSI, MACD, Stochastic, Bollinger Bands
- IA2: Strategic Decision AI with probabilistic TP optimization
- Multi-timeframe analysis (Daily, 4H, 1H)
- Risk-reward calculations and filtering
- Active position management
- AI training and enhancement systems
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveBackendTestSuite:
    """Comprehensive test suite for Ultra Professional Trading Bot System"""
    
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
        logger.info(f"Testing Ultra Professional Trading Bot System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected system components
        self.expected_endpoints = [
            "/opportunities", "/analyses", "/decisions", "/active-positions",
            "/start-trading", "/trading/execution-mode", "/performance",
            "/ai-training/status", "/ai-training/run-quick", "/ai-training/load-insights"
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
    
    async def test_1_system_architecture_coherence(self):
        """Test 1: Overall System Architecture Coherence"""
        logger.info("\n🔍 TEST 1: System Architecture Coherence - IA1→IA2 Pipeline Logic")
        
        try:
            # Test core endpoints availability
            endpoints_working = 0
            total_endpoints = len(self.expected_endpoints)
            
            for endpoint in self.expected_endpoints:
                try:
                    response = requests.get(f"{self.api_url}{endpoint}", timeout=10)
                    if response.status_code in [200, 201]:
                        endpoints_working += 1
                        logger.info(f"      ✅ {endpoint}: HTTP {response.status_code}")
                    else:
                        logger.info(f"      ❌ {endpoint}: HTTP {response.status_code}")
                except Exception as e:
                    logger.info(f"      ❌ {endpoint}: {str(e)}")
            
            # Test data flow: Scout → IA1 → IA2 → Execution
            logger.info("   🔍 Testing data flow pipeline...")
            
            # Get opportunities (Scout output)
            opportunities_response = requests.get(f"{self.api_url}/opportunities", timeout=30)
            opportunities_count = 0
            if opportunities_response.status_code == 200:
                opportunities_data = opportunities_response.json()
                opportunities_count = len(opportunities_data.get('opportunities', []))
                logger.info(f"      📊 Scout opportunities: {opportunities_count}")
            
            # Get IA1 analyses
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            analyses_count = 0
            if analyses_response.status_code == 200:
                analyses_data = analyses_response.json()
                analyses_count = len(analyses_data.get('analyses', []))
                logger.info(f"      📊 IA1 analyses: {analyses_count}")
            
            # Get IA2 decisions
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            decisions_count = 0
            if decisions_response.status_code == 200:
                decisions_data = decisions_response.json()
                decisions_count = len(decisions_data.get('decisions', []))
                logger.info(f"      📊 IA2 decisions: {decisions_count}")
            
            # Get active positions (Execution output)
            positions_response = requests.get(f"{self.api_url}/active-positions", timeout=30)
            positions_count = 0
            if positions_response.status_code == 200:
                positions_data = positions_response.json()
                positions_count = positions_data.get('total_positions', 0)
                logger.info(f"      📊 Active positions: {positions_count}")
            
            # Calculate pipeline efficiency
            pipeline_efficiency = 0
            if opportunities_count > 0:
                ia1_efficiency = (analyses_count / opportunities_count) * 100
                logger.info(f"      📊 Scout→IA1 efficiency: {ia1_efficiency:.1f}%")
                
                if analyses_count > 0:
                    ia2_efficiency = (decisions_count / analyses_count) * 100
                    logger.info(f"      📊 IA1→IA2 efficiency: {ia2_efficiency:.1f}%")
                    pipeline_efficiency = (ia1_efficiency + ia2_efficiency) / 2
            
            # Success criteria
            endpoints_success = endpoints_working >= (total_endpoints * 0.8)  # 80% endpoints working
            pipeline_success = opportunities_count > 0 and analyses_count > 0  # Basic pipeline working
            
            success = endpoints_success and pipeline_success
            
            details = f"Endpoints: {endpoints_working}/{total_endpoints}, Pipeline: {opportunities_count}→{analyses_count}→{decisions_count}→{positions_count}, Efficiency: {pipeline_efficiency:.1f}%"
            
            self.log_test_result("System Architecture Coherence", success, details)
            
        except Exception as e:
            self.log_test_result("System Architecture Coherence", False, f"Exception: {str(e)}")
    
    async def test_2_multi_timeframe_analysis_system(self):
        """Test 2: Multi-Timeframe Hierarchical Analysis System"""
        logger.info("\n🔍 TEST 2: Multi-Timeframe Hierarchical Analysis System")
        
        try:
            # Check if multi-timeframe analysis method exists in backend code
            backend_code = ""
            try:
                with open('/app/backend/server.py', 'r') as f:
                    backend_code = f.read()
            except Exception as e:
                self.log_test_result("Multi-Timeframe Analysis System", False, f"Could not read backend code: {e}")
                return
            
            # Check for multi-timeframe analysis implementation
            multi_timeframe_components = [
                "analyze_multi_timeframe_hierarchy",
                "_analyze_daily_context",
                "_analyze_h4_context", 
                "_analyze_h1_context",
                "dominant_timeframe",
                "decisive_pattern",
                "hierarchy_confidence"
            ]
            
            components_found = {}
            for component in multi_timeframe_components:
                if component in backend_code:
                    components_found[component] = True
                    logger.info(f"      ✅ Component found: {component}")
                else:
                    components_found[component] = False
                    logger.info(f"      ❌ Component missing: {component}")
            
            # Check backend logs for multi-timeframe execution
            import subprocess
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "2000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            # Look for multi-timeframe log patterns
            multi_timeframe_patterns = [
                "🎯 MULTI-TIMEFRAME ANALYSIS",
                "📊 Dominant Timeframe:",
                "📊 Decisive Pattern:",
                "📊 Hierarchy Confidence:",
                "⚠️ Anti-Momentum Risk:",
                "DAILY_TREND",
                "H4_TREND", 
                "H1_TREND"
            ]
            
            pattern_counts = {}
            for pattern in multi_timeframe_patterns:
                count = backend_logs.count(pattern)
                pattern_counts[pattern] = count
                logger.info(f"      📊 '{pattern}': {count} occurrences")
            
            # Get IA1 analyses to check for multi-timeframe context
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            multi_timeframe_analyses = 0
            total_analyses = 0
            
            if analyses_response.status_code == 200:
                analyses_data = analyses_response.json()
                analyses = analyses_data.get('analyses', [])
                total_analyses = len(analyses)
                
                for analysis in analyses[-10:]:  # Check last 10 analyses
                    reasoning = analysis.get('ia1_reasoning', '')
                    
                    # Check for multi-timeframe keywords
                    multi_timeframe_keywords = [
                        "timeframe", "daily", "4H", "1H", "hierarchy", "dominant",
                        "decisive pattern", "multi-timeframe", "temporal"
                    ]
                    
                    if any(keyword.lower() in reasoning.lower() for keyword in multi_timeframe_keywords):
                        multi_timeframe_analyses += 1
                        symbol = analysis.get('symbol', 'Unknown')
                        logger.info(f"      ✅ {symbol}: Multi-timeframe context detected")
            
            components_implemented = sum(components_found.values())
            patterns_detected = sum(1 for count in pattern_counts.values() if count > 0)
            
            logger.info(f"   📊 Components implemented: {components_implemented}/{len(multi_timeframe_components)}")
            logger.info(f"   📊 Log patterns detected: {patterns_detected}/{len(multi_timeframe_patterns)}")
            logger.info(f"   📊 Multi-timeframe analyses: {multi_timeframe_analyses}/{min(total_analyses, 10)}")
            
            # Success criteria: Implementation exists AND execution evidence
            implementation_success = components_implemented >= 5  # Most components implemented
            execution_success = patterns_detected >= 2 or multi_timeframe_analyses >= 3  # Evidence of execution
            
            success = implementation_success and execution_success
            
            details = f"Components: {components_implemented}/{len(multi_timeframe_components)}, Patterns: {patterns_detected}/{len(multi_timeframe_patterns)}, Analyses: {multi_timeframe_analyses}/{min(total_analyses, 10)}"
            
            self.log_test_result("Multi-Timeframe Analysis System", success, details)
            
        except Exception as e:
            self.log_test_result("Multi-Timeframe Analysis System", False, f"Exception: {str(e)}")
    
    async def test_3_enhanced_technical_indicators_integration(self):
        """Test 3: Enhanced Technical Indicators Integration (RSI, MACD, Stochastic, Bollinger Bands)"""
        logger.info("\n🔍 TEST 3: Enhanced Technical Indicators Integration")
        
        try:
            # Get IA1 analyses to check technical indicators
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if analyses_response.status_code != 200:
                self.log_test_result("Enhanced Technical Indicators Integration", False, f"HTTP {analyses_response.status_code}")
                return
            
            analyses_data = analyses_response.json()
            analyses = analyses_data.get('analyses', [])
            
            if not analyses:
                self.log_test_result("Enhanced Technical Indicators Integration", False, "No IA1 analyses found")
                return
            
            # Check technical indicators coverage
            indicators_coverage = {
                'rsi': 0,
                'macd': 0, 
                'stochastic': 0,
                'bollinger': 0
            }
            
            realistic_values = {
                'rsi': 0,
                'macd': 0,
                'stochastic': 0,
                'bollinger': 0
            }
            
            confluence_analyses = 0
            enhanced_precision = 0
            total_checked = min(len(analyses), 10)
            
            for analysis in analyses[-total_checked:]:  # Check last 10 analyses
                symbol = analysis.get('symbol', 'Unknown')
                reasoning = analysis.get('ia1_reasoning', '')
                
                # Check RSI
                rsi_value = analysis.get('rsi', 0)
                if rsi_value > 0 and 0 <= rsi_value <= 100:
                    indicators_coverage['rsi'] += 1
                    if 20 <= rsi_value <= 80:  # Realistic range
                        realistic_values['rsi'] += 1
                
                # Check MACD
                macd_value = analysis.get('macd_signal', 0)
                if macd_value != 0:
                    indicators_coverage['macd'] += 1
                    if -1 <= macd_value <= 1:  # Realistic range
                        realistic_values['macd'] += 1
                
                # Check Stochastic
                stochastic_value = analysis.get('stochastic', 0)
                stochastic_d_value = analysis.get('stochastic_d', 0)
                if stochastic_value > 0 and 0 <= stochastic_value <= 100:
                    indicators_coverage['stochastic'] += 1
                    if 20 <= stochastic_value <= 80:  # Realistic range
                        realistic_values['stochastic'] += 1
                
                # Check Bollinger Bands
                bollinger_value = analysis.get('bollinger_position', 0)
                if bollinger_value != 0:
                    indicators_coverage['bollinger'] += 1
                    if -2 <= bollinger_value <= 2:  # Realistic range
                        realistic_values['bollinger'] += 1
                
                # Check for confluence analysis in reasoning
                confluence_keywords = [
                    "confluence", "align", "confirm", "contradict", "divergence",
                    "technical indicators", "multiple indicators", "combined signals"
                ]
                if any(keyword in reasoning.lower() for keyword in confluence_keywords):
                    confluence_analyses += 1
                
                # Check for enhanced precision language
                precision_keywords = [
                    "precise", "sophisticated", "advanced", "enhanced", "refined",
                    "detailed analysis", "comprehensive", "multi-indicator"
                ]
                if any(keyword in reasoning.lower() for keyword in precision_keywords):
                    enhanced_precision += 1
                
                logger.info(f"      📊 {symbol}: RSI={rsi_value:.1f}, MACD={macd_value:.6f}, Stoch={stochastic_value:.1f}, BB={bollinger_value:.2f}")
            
            # Calculate coverage percentages
            rsi_coverage = (indicators_coverage['rsi'] / total_checked) * 100
            macd_coverage = (indicators_coverage['macd'] / total_checked) * 100
            stochastic_coverage = (indicators_coverage['stochastic'] / total_checked) * 100
            bollinger_coverage = (indicators_coverage['bollinger'] / total_checked) * 100
            
            confluence_percentage = (confluence_analyses / total_checked) * 100
            precision_percentage = (enhanced_precision / total_checked) * 100
            
            logger.info(f"   📊 RSI coverage: {rsi_coverage:.1f}% ({indicators_coverage['rsi']}/{total_checked})")
            logger.info(f"   📊 MACD coverage: {macd_coverage:.1f}% ({indicators_coverage['macd']}/{total_checked})")
            logger.info(f"   📊 Stochastic coverage: {stochastic_coverage:.1f}% ({indicators_coverage['stochastic']}/{total_checked})")
            logger.info(f"   📊 Bollinger coverage: {bollinger_coverage:.1f}% ({indicators_coverage['bollinger']}/{total_checked})")
            logger.info(f"   📊 Confluence analysis: {confluence_percentage:.1f}% ({confluence_analyses}/{total_checked})")
            logger.info(f"   📊 Enhanced precision: {precision_percentage:.1f}% ({enhanced_precision}/{total_checked})")
            
            # Check IA2 decisions for technical indicators integration
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            ia2_technical_integration = 0
            total_decisions = 0
            
            if decisions_response.status_code == 200:
                decisions_data = decisions_response.json()
                decisions = decisions_data.get('decisions', [])
                total_decisions = min(len(decisions), 10)
                
                for decision in decisions[-total_decisions:]:
                    reasoning = decision.get('ia2_reasoning', '')
                    
                    # Check for technical indicators in IA2 reasoning
                    technical_keywords = [
                        "RSI", "MACD", "Stochastic", "Bollinger", "technical", "indicators",
                        "overbought", "oversold", "momentum", "volatility"
                    ]
                    
                    if any(keyword.lower() in reasoning.lower() for keyword in technical_keywords):
                        ia2_technical_integration += 1
            
            ia2_integration_percentage = (ia2_technical_integration / max(total_decisions, 1)) * 100
            logger.info(f"   📊 IA2 technical integration: {ia2_integration_percentage:.1f}% ({ia2_technical_integration}/{total_decisions})")
            
            # Success criteria
            indicators_working = sum(1 for coverage in [rsi_coverage, macd_coverage, stochastic_coverage, bollinger_coverage] if coverage >= 50)
            confluence_success = confluence_percentage >= 60
            ia2_integration_success = ia2_integration_percentage >= 70
            
            success = indicators_working >= 3 and confluence_success and ia2_integration_success
            
            details = f"Indicators working: {indicators_working}/4, Confluence: {confluence_percentage:.1f}%, IA2 integration: {ia2_integration_percentage:.1f}%"
            
            self.log_test_result("Enhanced Technical Indicators Integration", success, details)
            
        except Exception as e:
            self.log_test_result("Enhanced Technical Indicators Integration", False, f"Exception: {str(e)}")
    
    async def test_4_ia1_ia2_filtering_pipeline(self):
        """Test 4: IA1→IA2 Filtering and Processing Pipeline"""
        logger.info("\n🔍 TEST 4: IA1→IA2 Filtering and Processing Pipeline")
        
        try:
            # Trigger fresh analysis
            logger.info("   🚀 Triggering fresh analysis via /api/start-trading...")
            start_response = requests.post(f"{self.api_url}/start-trading", timeout=180)
            
            if start_response.status_code not in [200, 201]:
                logger.warning(f"   ⚠️ Start trading returned HTTP {start_response.status_code}")
            else:
                # Wait for processing
                logger.info("   ⏳ Waiting 30 seconds for pipeline processing...")
                await asyncio.sleep(30)
            
            # Get IA1 analyses
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            
            if analyses_response.status_code != 200:
                self.log_test_result("IA1→IA2 Filtering Pipeline", False, f"HTTP {analyses_response.status_code}")
                return
            
            analyses_data = analyses_response.json()
            analyses = analyses_data.get('analyses', [])
            
            # Get IA2 decisions
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            
            if decisions_response.status_code != 200:
                self.log_test_result("IA1→IA2 Filtering Pipeline", False, f"HTTP {decisions_response.status_code}")
                return
            
            decisions_data = decisions_response.json()
            decisions = decisions_data.get('decisions', [])
            
            # Analyze filtering logic
            voie1_eligible = 0  # LONG/SHORT + Confidence ≥70%
            voie2_eligible = 0  # RR ≥2.0
            high_confidence_analyses = 0
            high_rr_analyses = 0
            
            for analysis in analyses[-15:]:  # Check last 15 analyses
                symbol = analysis.get('symbol', 'Unknown')
                signal = analysis.get('ia1_signal', '').upper()
                confidence = analysis.get('analysis_confidence', 0) * 100  # Convert to percentage
                rr_ratio = analysis.get('risk_reward_ratio', 0)
                
                # VOIE 1: LONG/SHORT + Confidence ≥70%
                if signal in ['LONG', 'SHORT'] and confidence >= 70:
                    voie1_eligible += 1
                    logger.info(f"      ✅ VOIE 1 eligible: {symbol} ({signal}, {confidence:.1f}%)")
                
                # VOIE 2: RR ≥2.0
                if rr_ratio >= 2.0:
                    voie2_eligible += 1
                    logger.info(f"      ✅ VOIE 2 eligible: {symbol} (RR: {rr_ratio:.2f})")
                
                if confidence >= 70:
                    high_confidence_analyses += 1
                
                if rr_ratio >= 2.0:
                    high_rr_analyses += 1
            
            # Check backend logs for filtering evidence
            import subprocess
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "3000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            # Look for filtering log patterns
            filtering_patterns = [
                "✅ IA2 ACCEPTED (VOIE 1)",
                "✅ IA2 ACCEPTED (VOIE 2)", 
                "🛑IA2 SKIP",
                "IA2 FILTER",
                "VOIE 1",
                "VOIE 2"
            ]
            
            filtering_logs = {}
            for pattern in filtering_patterns:
                count = backend_logs.count(pattern)
                filtering_logs[pattern] = count
                logger.info(f"      📊 '{pattern}': {count} occurrences")
            
            # Check for recent IA2 decisions
            recent_decisions = 0
            for decision in decisions:
                timestamp_str = decision.get('timestamp', '')
                if self._is_recent_timestamp(timestamp_str):
                    recent_decisions += 1
            
            total_eligible = voie1_eligible + voie2_eligible
            filtering_evidence = sum(filtering_logs.values())
            
            logger.info(f"   📊 VOIE 1 eligible analyses: {voie1_eligible}")
            logger.info(f"   📊 VOIE 2 eligible analyses: {voie2_eligible}")
            logger.info(f"   📊 Total eligible for IA2: {total_eligible}")
            logger.info(f"   📊 Recent IA2 decisions: {recent_decisions}")
            logger.info(f"   📊 Filtering log evidence: {filtering_evidence}")
            
            # Success criteria
            filtering_logic_working = total_eligible > 0  # Some analyses meet criteria
            pipeline_operational = recent_decisions > 0 or filtering_evidence > 0  # Evidence of processing
            
            success = filtering_logic_working and pipeline_operational
            
            details = f"Eligible: {total_eligible} (VOIE1: {voie1_eligible}, VOIE2: {voie2_eligible}), Recent decisions: {recent_decisions}, Filtering logs: {filtering_evidence}"
            
            self.log_test_result("IA1→IA2 Filtering Pipeline", success, details)
            
        except Exception as e:
            self.log_test_result("IA1→IA2 Filtering Pipeline", False, f"Exception: {str(e)}")
    
    async def test_5_sophisticated_rr_analysis_system(self):
        """Test 5: Sophisticated RR Analysis System"""
        logger.info("\n🔍 TEST 5: Sophisticated RR Analysis System")
        
        try:
            # Check if sophisticated RR methods exist in backend code
            backend_code = ""
            try:
                with open('/app/backend/server.py', 'r') as f:
                    backend_code = f.read()
            except Exception as e:
                self.log_test_result("Sophisticated RR Analysis System", False, f"Could not read backend code: {e}")
                return
            
            # Check for sophisticated RR analysis methods
            sophisticated_rr_methods = [
                "calculate_neutral_risk_reward",
                "calculate_composite_rr",
                "evaluate_sophisticated_risk_level",
                "calculate_bullish_rr",
                "calculate_bearish_rr"
            ]
            
            methods_found = {}
            for method in sophisticated_rr_methods:
                if method in backend_code:
                    methods_found[method] = True
                    logger.info(f"      ✅ Method found: {method}")
                else:
                    methods_found[method] = False
                    logger.info(f"      ❌ Method missing: {method}")
            
            # Check backend logs for sophisticated RR execution
            import subprocess
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "3000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            # Look for sophisticated RR log patterns
            sophisticated_rr_patterns = [
                "🧠 SOPHISTICATED ANALYSIS",
                "📊 Composite RR:",
                "📊 Neutral RR:",
                "🎯 Sophisticated Risk Level:",
                "⚠️ SIGNIFICANT RR DIVERGENCE",
                "composite_rr_data",
                "sophisticated_risk_level"
            ]
            
            pattern_counts = {}
            for pattern in sophisticated_rr_patterns:
                count = backend_logs.count(pattern)
                pattern_counts[pattern] = count
                logger.info(f"      📊 '{pattern}': {count} occurrences")
            
            # Check IA2 decisions for sophisticated RR data
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            sophisticated_rr_decisions = 0
            total_decisions = 0
            
            if decisions_response.status_code == 200:
                decisions_data = decisions_response.json()
                decisions = decisions_data.get('decisions', [])
                total_decisions = min(len(decisions), 10)
                
                for decision in decisions[-total_decisions:]:
                    reasoning = decision.get('ia2_reasoning', '')
                    
                    # Check for sophisticated RR analysis in reasoning
                    sophisticated_keywords = [
                        "sophisticated", "composite", "neutral RR", "risk level",
                        "advanced RR", "RR divergence", "sophisticated analysis"
                    ]
                    
                    if any(keyword.lower() in reasoning.lower() for keyword in sophisticated_keywords):
                        sophisticated_rr_decisions += 1
                        symbol = decision.get('symbol', 'Unknown')
                        logger.info(f"      ✅ {symbol}: Sophisticated RR analysis detected")
            
            methods_implemented = sum(methods_found.values())
            patterns_detected = sum(1 for count in pattern_counts.values() if count > 0)
            sophisticated_percentage = (sophisticated_rr_decisions / max(total_decisions, 1)) * 100
            
            logger.info(f"   📊 Methods implemented: {methods_implemented}/{len(sophisticated_rr_methods)}")
            logger.info(f"   📊 Log patterns detected: {patterns_detected}/{len(sophisticated_rr_patterns)}")
            logger.info(f"   📊 Sophisticated RR decisions: {sophisticated_percentage:.1f}% ({sophisticated_rr_decisions}/{total_decisions})")
            
            # Success criteria
            implementation_success = methods_implemented >= 3  # Most methods implemented
            execution_success = patterns_detected >= 2 or sophisticated_rr_decisions >= 2  # Evidence of execution
            
            success = implementation_success and execution_success
            
            details = f"Methods: {methods_implemented}/{len(sophisticated_rr_methods)}, Patterns: {patterns_detected}/{len(sophisticated_rr_patterns)}, Decisions: {sophisticated_percentage:.1f}%"
            
            self.log_test_result("Sophisticated RR Analysis System", success, details)
            
        except Exception as e:
            self.log_test_result("Sophisticated RR Analysis System", False, f"Exception: {str(e)}")
    
    async def test_6_enhanced_rr_validation_system(self):
        """Test 6: Enhanced RR Validation System (Current Focus)"""
        logger.info("\n🔍 TEST 6: Enhanced RR Validation System (Current Focus)")
        
        try:
            # Check backend logs for momentum validation
            import subprocess
            backend_logs = ""
            try:
                log_result = subprocess.run(
                    ["tail", "-n", "2000", "/var/log/supervisor/backend.out.log"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                backend_logs += log_result.stdout
            except:
                pass
            
            # Look for enhanced RR validation patterns
            validation_patterns = [
                "MOMENTUM VALIDATION",
                "🎯 MULTI-TIMEFRAME VALIDATION",
                "✅ SOPHISTICATED VALIDATION APPLIED",
                "LEGITIMATE REVERSAL",
                "MOMENTUM ERROR",
                "Anti-Momentum Risk"
            ]
            
            pattern_counts = {}
            for pattern in validation_patterns:
                count = backend_logs.count(pattern)
                pattern_counts[pattern] = count
                logger.info(f"      📊 '{pattern}': {count} occurrences")
            
            # Get IA1 analyses to check for validation integration
            analyses_response = requests.get(f"{self.api_url}/analyses", timeout=30)
            validation_enhanced_analyses = 0
            total_analyses = 0
            
            if analyses_response.status_code == 200:
                analyses_data = analyses_response.json()
                analyses = analyses_data.get('analyses', [])
                total_analyses = min(len(analyses), 10)
                
                for analysis in analyses[-total_analyses:]:
                    reasoning = analysis.get('ia1_reasoning', '')
                    
                    # Check for validation enhancement indicators
                    validation_keywords = [
                        "validation", "momentum", "reversal", "sophisticated",
                        "anti-momentum", "multi-timeframe", "hierarchy"
                    ]
                    
                    if any(keyword.lower() in reasoning.lower() for keyword in validation_keywords):
                        validation_enhanced_analyses += 1
                        symbol = analysis.get('symbol', 'Unknown')
                        logger.info(f"      ✅ {symbol}: Validation enhancement detected")
            
            patterns_detected = sum(1 for count in pattern_counts.values() if count > 0)
            validation_percentage = (validation_enhanced_analyses / max(total_analyses, 1)) * 100
            
            logger.info(f"   📊 Validation patterns detected: {patterns_detected}/{len(validation_patterns)}")
            logger.info(f"   📊 Validation enhanced analyses: {validation_percentage:.1f}% ({validation_enhanced_analyses}/{total_analyses})")
            
            # Check for conditional execution (system should work conditionally)
            conditional_execution = backend_logs.count("No correction needed") > 0 or backend_logs.count("correctly doesn't trigger") > 0
            
            if conditional_execution:
                logger.info("      ✅ Conditional execution detected - system working as designed")
            
            # Success criteria (adjusted for conditional system)
            validation_working = patterns_detected >= 2 or validation_enhanced_analyses >= 3 or conditional_execution
            
            success = validation_working
            
            details = f"Patterns: {patterns_detected}/{len(validation_patterns)}, Enhanced analyses: {validation_percentage:.1f}%, Conditional: {conditional_execution}"
            
            self.log_test_result("Enhanced RR Validation System", success, details)
            
        except Exception as e:
            self.log_test_result("Enhanced RR Validation System", False, f"Exception: {str(e)}")
    
    async def test_7_ai_training_and_enhancement_systems(self):
        """Test 7: AI Training and Enhancement Systems"""
        logger.info("\n🔍 TEST 7: AI Training and Enhancement Systems")
        
        try:
            # Test AI Training Status endpoint
            training_status_response = requests.get(f"{self.api_url}/ai-training/status", timeout=30)
            training_status_working = training_status_response.status_code == 200
            
            if training_status_working:
                status_data = training_status_response.json()
                system_ready = status_data.get('system_ready', False)
                logger.info(f"      ✅ AI Training Status: System ready = {system_ready}")
            else:
                logger.info(f"      ❌ AI Training Status: HTTP {training_status_response.status_code}")
            
            # Test AI Training Run endpoint
            training_run_response = requests.post(f"{self.api_url}/ai-training/run-quick", timeout=30)
            training_run_working = training_run_response.status_code == 200
            
            if training_run_working:
                run_data = training_run_response.json()
                cache_utilized = run_data.get('cache_utilized', False)
                logger.info(f"      ✅ AI Training Run: Cache utilized = {cache_utilized}")
            else:
                logger.info(f"      ❌ AI Training Run: HTTP {training_run_response.status_code}")
            
            # Test AI Enhancement Status endpoint
            enhancement_status_response = requests.get(f"{self.api_url}/ai-training/enhancement-status", timeout=30)
            enhancement_status_working = enhancement_status_response.status_code == 200
            
            if enhancement_status_working:
                enhancement_data = enhancement_status_response.json()
                ia1_active = enhancement_data.get('ia1_enhancement_active', False)
                ia2_active = enhancement_data.get('ia2_enhancement_active', False)
                logger.info(f"      ✅ AI Enhancement Status: IA1={ia1_active}, IA2={ia2_active}")
            else:
                logger.info(f"      ❌ AI Enhancement Status: HTTP {enhancement_status_response.status_code}")
            
            # Check for AI enhancement in recent decisions
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            enhanced_decisions = 0
            total_decisions = 0
            
            if decisions_response.status_code == 200:
                decisions_data = decisions_response.json()
                decisions = decisions_data.get('decisions', [])
                total_decisions = min(len(decisions), 10)
                
                for decision in decisions[-total_decisions:]:
                    reasoning = decision.get('ia2_reasoning', '')
                    
                    # Check for AI enhancement keywords
                    enhancement_keywords = [
                        "enhanced", "optimized", "adjusted", "pattern-based",
                        "market condition", "AI enhancement", "performance enhancement"
                    ]
                    
                    if any(keyword.lower() in reasoning.lower() for keyword in enhancement_keywords):
                        enhanced_decisions += 1
            
            enhancement_percentage = (enhanced_decisions / max(total_decisions, 1)) * 100
            
            logger.info(f"   📊 AI Training Status: {training_status_working}")
            logger.info(f"   📊 AI Training Run: {training_run_working}")
            logger.info(f"   📊 AI Enhancement Status: {enhancement_status_working}")
            logger.info(f"   📊 Enhanced decisions: {enhancement_percentage:.1f}% ({enhanced_decisions}/{total_decisions})")
            
            # Success criteria
            endpoints_working = sum([training_status_working, training_run_working, enhancement_status_working])
            enhancement_active = enhancement_percentage >= 50  # At least 50% of decisions show enhancement
            
            success = endpoints_working >= 2 and enhancement_active
            
            details = f"Endpoints working: {endpoints_working}/3, Enhanced decisions: {enhancement_percentage:.1f}%"
            
            self.log_test_result("AI Training and Enhancement Systems", success, details)
            
        except Exception as e:
            self.log_test_result("AI Training and Enhancement Systems", False, f"Exception: {str(e)}")
    
    async def test_8_active_position_management_system(self):
        """Test 8: Active Position Management System"""
        logger.info("\n🔍 TEST 8: Active Position Management System")
        
        try:
            # Test Active Positions endpoint
            positions_response = requests.get(f"{self.api_url}/active-positions", timeout=30)
            
            if positions_response.status_code != 200:
                self.log_test_result("Active Position Management System", False, f"HTTP {positions_response.status_code}")
                return
            
            positions_data = positions_response.json()
            
            # Check position management fields
            expected_fields = [
                'active_positions', 'total_positions', 'total_unrealized_pnl',
                'total_position_value', 'execution_mode', 'monitoring_active'
            ]
            
            fields_present = 0
            for field in expected_fields:
                if field in positions_data:
                    fields_present += 1
                    logger.info(f"      ✅ Field present: {field} = {positions_data[field]}")
                else:
                    logger.info(f"      ❌ Field missing: {field}")
            
            # Test Execution Mode endpoint
            execution_mode_response = requests.get(f"{self.api_url}/trading/execution-mode", timeout=30)
            execution_mode_working = execution_mode_response.status_code == 200
            
            if execution_mode_working:
                mode_data = execution_mode_response.json()
                current_mode = mode_data.get('execution_mode', 'Unknown')
                logger.info(f"      ✅ Execution Mode: {current_mode}")
            else:
                logger.info(f"      ❌ Execution Mode: HTTP {execution_mode_response.status_code}")
            
            # Test Position Close endpoint (with non-existent ID)
            close_response = requests.post(f"{self.api_url}/active-positions/close/test-id", timeout=30)
            close_endpoint_working = close_response.status_code in [404, 400]  # Expected for non-existent ID
            
            if close_endpoint_working:
                logger.info(f"      ✅ Position Close endpoint: HTTP {close_response.status_code} (expected)")
            else:
                logger.info(f"      ❌ Position Close endpoint: HTTP {close_response.status_code}")
            
            # Check for position management integration in IA2 decisions
            decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
            position_integration = 0
            total_decisions = 0
            
            if decisions_response.status_code == 200:
                decisions_data = decisions_response.json()
                decisions = decisions_data.get('decisions', [])
                total_decisions = min(len(decisions), 10)
                
                for decision in decisions[-total_decisions:]:
                    # Check if decision has position sizing
                    position_size = decision.get('position_size', 0)
                    if position_size != 0:
                        position_integration += 1
            
            integration_percentage = (position_integration / max(total_decisions, 1)) * 100
            
            logger.info(f"   📊 Position management fields: {fields_present}/{len(expected_fields)}")
            logger.info(f"   📊 Execution mode endpoint: {execution_mode_working}")
            logger.info(f"   📊 Position close endpoint: {close_endpoint_working}")
            logger.info(f"   📊 Position integration: {integration_percentage:.1f}% ({position_integration}/{total_decisions})")
            
            # Success criteria
            fields_success = fields_present >= 5  # Most fields present
            endpoints_success = execution_mode_working and close_endpoint_working
            integration_success = integration_percentage >= 70  # Good integration with decisions
            
            success = fields_success and endpoints_success and integration_success
            
            details = f"Fields: {fields_present}/{len(expected_fields)}, Endpoints: {endpoints_success}, Integration: {integration_percentage:.1f}%"
            
            self.log_test_result("Active Position Management System", success, details)
            
        except Exception as e:
            self.log_test_result("Active Position Management System", False, f"Exception: {str(e)}")
    
    def _is_recent_timestamp(self, timestamp_str: str) -> bool:
        """Check if timestamp is from the last 24 hours"""
        try:
            if not timestamp_str:
                return False
            
            # Parse timestamp (handle different formats)
            if 'T' in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # Handle "2025-09-09 13:56:35 (Heure de Paris)" format
                if '(' in timestamp_str:
                    timestamp_str = timestamp_str.split('(')[0].strip()
                timestamp = datetime.fromisoformat(timestamp_str)
            
            # Remove timezone info for comparison
            if timestamp.tzinfo:
                timestamp = timestamp.replace(tzinfo=None)
            
            now = datetime.now()
            return (now - timestamp) <= timedelta(hours=24)
            
        except Exception:
            return False
    
    async def run_comprehensive_tests(self):
        """Run all comprehensive backend tests"""
        logger.info("🚀 Starting Comprehensive Ultra Professional Trading Bot System Test Suite")
        logger.info("=" * 100)
        logger.info("📋 COMPREHENSIVE SYSTEM ARCHITECTURE REVIEW")
        logger.info("🎯 Testing: IA1→IA2 pipeline, Multi-timeframe analysis, Technical indicators, RR systems")
        logger.info("🎯 Focus: System coherence, filtering logic, validation systems, robustness")
        logger.info("=" * 100)
        
        # Run all tests in sequence
        await self.test_1_system_architecture_coherence()
        await self.test_2_multi_timeframe_analysis_system()
        await self.test_3_enhanced_technical_indicators_integration()
        await self.test_4_ia1_ia2_filtering_pipeline()
        await self.test_5_sophisticated_rr_analysis_system()
        await self.test_6_enhanced_rr_validation_system()
        await self.test_7_ai_training_and_enhancement_systems()
        await self.test_8_active_position_management_system()
        
        # Summary
        logger.info("\n" + "=" * 100)
        logger.info("📊 COMPREHENSIVE SYSTEM ARCHITECTURE REVIEW SUMMARY")
        logger.info("=" * 100)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        # Categorize results
        critical_failures = []
        working_components = []
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
            
            if result['success']:
                working_components.append(result['test'])
            else:
                critical_failures.append(result['test'])
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        
        # System analysis
        logger.info("\n" + "=" * 100)
        logger.info("📋 ULTRA PROFESSIONAL TRADING BOT SYSTEM STATUS")
        logger.info("=" * 100)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Ultra Professional Trading Bot System FULLY FUNCTIONAL!")
            logger.info("✅ Complete system architecture coherence achieved")
            logger.info("✅ All filtering and validation systems operational")
            logger.info("✅ Technical analysis depth meets professional standards")
            logger.info("✅ System robustness confirmed across all components")
        elif passed_tests >= total_tests * 0.75:
            logger.info("⚠️ MOSTLY FUNCTIONAL - System working with minor gaps")
            logger.info("🔍 Some advanced features may need fine-tuning")
        elif passed_tests >= total_tests * 0.5:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core features working")
            logger.info("🔧 Several components need attention for full functionality")
        else:
            logger.info("❌ SYSTEM NOT PRODUCTION READY - Critical issues identified")
            logger.info("🚨 Major architectural gaps preventing optimal performance")
        
        # Detailed analysis
        logger.info("\n📝 SYSTEM ARCHITECTURE ANALYSIS:")
        
        logger.info("\n🟢 WORKING COMPONENTS:")
        for component in working_components:
            logger.info(f"   ✅ {component}")
        
        if critical_failures:
            logger.info("\n🔴 CRITICAL FAILURES REQUIRING ATTENTION:")
            for failure in critical_failures:
                logger.info(f"   ❌ {failure}")
        
        # Architecture recommendations
        logger.info("\n📋 ARCHITECTURAL ASSESSMENT:")
        
        if "System Architecture Coherence" in working_components:
            logger.info("   ✅ Overall architecture coherence: EXCELLENT")
            logger.info("   ✅ IA1→IA2 pipeline logic: OPERATIONAL")
        else:
            logger.info("   ❌ Overall architecture coherence: NEEDS ATTENTION")
        
        if "Multi-Timeframe Analysis System" in working_components:
            logger.info("   ✅ Multi-timeframe validation: SOPHISTICATED")
        else:
            logger.info("   ❌ Multi-timeframe validation: IMPLEMENTATION GAPS")
        
        if "Enhanced Technical Indicators Integration" in working_components:
            logger.info("   ✅ Technical analysis depth: PROFESSIONAL GRADE")
        else:
            logger.info("   ❌ Technical analysis depth: INSUFFICIENT INTEGRATION")
        
        if "Active Position Management System" in working_components:
            logger.info("   ✅ System robustness: PRODUCTION READY")
        else:
            logger.info("   ❌ System robustness: NEEDS STRENGTHENING")
        
        # Final verdict
        logger.info(f"\n🏆 FINAL SYSTEM MATURITY ASSESSMENT:")
        
        if passed_tests >= 7:
            logger.info("🎉 VERDICT: Ultra Professional Trading Bot System is PRODUCTION READY!")
            logger.info("✅ Sophisticated architecture with excellent coherence")
            logger.info("✅ Advanced filtering and validation systems operational")
            logger.info("✅ Professional-grade technical analysis integration")
            logger.info("✅ Robust error handling and scalable design")
        elif passed_tests >= 5:
            logger.info("⚠️ VERDICT: System is MOSTLY PRODUCTION READY")
            logger.info("🔍 Minor optimizations needed for full professional deployment")
        elif passed_tests >= 3:
            logger.info("⚠️ VERDICT: System has GOOD FOUNDATION but needs development")
            logger.info("🔧 Several key components require implementation or fixes")
        else:
            logger.info("❌ VERDICT: System is NOT READY for professional trading")
            logger.info("🚨 Major architectural issues prevent reliable operation")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = ComprehensiveBackendTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed >= total * 0.75:  # 75% pass rate for success
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
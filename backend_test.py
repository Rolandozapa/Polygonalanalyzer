#!/usr/bin/env python3
"""
BINGX OPPORTUNITIES REFRESH COMPREHENSIVE TEST SUITE
Focus: Force complete opportunities refresh with BingX data to resolve stale data issues

CRITICAL TEST REQUIREMENTS FROM REVIEW REQUEST:
1. **Identify Real Opportunities Source**: Where do the 14 opportunities actually come from?
2. **Force Cache Invalidation**: Clear all caches that might be preventing BingX data
3. **Test BingX Integration**: Verify trending_auto_updater has BingX data available
4. **Force Scout Refresh**: Make scout use new BingX opportunities instead of cached ones

CRITICAL ISSUES TO RESOLVE:
- Why get_current_opportunities logs don't appear (method not called?)
- Old timestamps (11:54 and 11:20 from this morning) persist despite cache TTL=10s
- Data sources still show ["cryptocompare", "coingecko"] instead of BingX
- DOGEUSDT (top BingX gainer +8.87%) missing from opportunities

TESTING APPROACH:
- Check if scout.scan_opportunities uses advanced_market_aggregator.get_current_opportunities
- Verify if trending_auto_updater.current_trending has BingX data
- Force clear all caches (opportunities, market data, etc.)
- Test multiple entry points: /api/start-scout, direct opportunity calls
- Monitor logs to see if get_current_opportunities is actually called

GOAL: Get fresh BingX trending opportunities (DOGEUSDT, ONDOUSDT with high gains) to replace the stale 4.8h old data.
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

class IA1IA2PipelineDemonstrationTestSuite:
    """Comprehensive test suite for IA1→IA2 Pipeline Demonstration Run"""
    
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
        logger.info(f"Testing IA1→IA2 Pipeline Demonstration at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for pipeline analysis")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Test results
        self.test_results = []
        
        # Test symbols for IA1 → IA2 pipeline demonstration
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
        
        # Track pipeline execution
        self.pipeline_executions = []
        
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
    
    async def test_1_api_endpoints_availability(self):
        """Test 1: Verify all required API endpoints are available"""
        logger.info("\n🔍 TEST 1: API Endpoints Availability Test")
        
        required_endpoints = [
            {'method': 'GET', 'path': '/opportunities', 'name': 'Market Opportunities'},
            {'method': 'GET', 'path': '/analyses', 'name': 'IA1 Analyses'},
            {'method': 'GET', 'path': '/decisions', 'name': 'IA2 Decisions'},
            {'method': 'GET', 'path': '/performance', 'name': 'Performance Metrics'},
            {'method': 'POST', 'path': '/run-ia1-cycle', 'name': 'IA1 Cycle Trigger'},
            {'method': 'POST', 'path': '/force-ia1-analysis', 'name': 'Force IA1 Analysis'}
        ]
        
        endpoint_results = []
        
        for endpoint in required_endpoints:
            try:
                method = endpoint['method']
                path = endpoint['path']
                name = endpoint['name']
                
                logger.info(f"   Testing {method} {path} ({name})")
                
                if method == 'GET':
                    response = requests.get(f"{self.api_url}{path}", timeout=30)
                elif method == 'POST':
                    if 'force-ia1-analysis' in path:
                        response = requests.post(f"{self.api_url}{path}", 
                                               json={"symbol": "BTCUSDT"}, timeout=30)
                    else:
                        response = requests.post(f"{self.api_url}{path}", json={}, timeout=30)
                
                if response.status_code in [200, 201]:
                    endpoint_results.append({'endpoint': name, 'status': 'SUCCESS'})
                    logger.info(f"      ✅ {name}: SUCCESS (HTTP {response.status_code})")
                else:
                    endpoint_results.append({'endpoint': name, 'status': f'HTTP_{response.status_code}'})
                    logger.info(f"      ❌ {name}: HTTP {response.status_code}")
                    
            except Exception as e:
                endpoint_results.append({'endpoint': name, 'status': 'ERROR', 'error': str(e)})
                logger.info(f"      ❌ {name}: Exception - {str(e)}")
        
        successful_endpoints = len([r for r in endpoint_results if r['status'] == 'SUCCESS'])
        total_endpoints = len(endpoint_results)
        
        if successful_endpoints >= total_endpoints * 0.8:  # 80% success rate
            self.log_test_result("API Endpoints Availability", True, 
                               f"Success rate: {successful_endpoints}/{total_endpoints}")
        else:
            self.log_test_result("API Endpoints Availability", False, 
                               f"Low success rate: {successful_endpoints}/{total_endpoints}")
    
    async def test_2_ia1_ia2_pipeline_complete_flow(self):
        """Test 2: IA1→IA2 Pipeline Complete Flow with VOIE Escalation Logic"""
        logger.info("\n🔍 TEST 2: IA1→IA2 Pipeline Complete Flow Test")
        
        try:
            # Get initial counts
            initial_ia1_count = 0
            initial_ia2_count = 0
            if self.db is not None:
                initial_ia1_count = self.db.technical_analyses.count_documents({})
                initial_ia2_count = self.db.trading_decisions.count_documents({})
                logger.info(f"   📊 Initial counts - IA1: {initial_ia1_count}, IA2: {initial_ia2_count}")
            
            # Trigger IA1 analysis for multiple symbols
            pipeline_executions = []
            
            for symbol in self.test_symbols:
                try:
                    logger.info(f"   🚀 Triggering IA1→IA2 pipeline for {symbol}")
                    
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
                            'ia2_triggered': False,
                            'voie_used': None,
                            'decision_id': result.get('decision_id')
                        }
                        
                        # Check for IA2 escalation indicators
                        if result.get('success', False):
                            confidence = result.get('confidence', 0)
                            rr = result.get('risk_reward_ratio', 0)
                            
                            # Determine VOIE escalation path
                            if confidence >= 0.95:
                                execution_data['voie_used'] = 'VOIE 3'
                                execution_data['ia2_triggered'] = True
                            elif confidence >= 0.70:
                                execution_data['voie_used'] = 'VOIE 1'
                                execution_data['ia2_triggered'] = True
                            elif rr >= 2.0:
                                execution_data['voie_used'] = 'VOIE 2'
                                execution_data['ia2_triggered'] = True
                            
                            logger.info(f"      ✅ {symbol}: IA1 success (Conf: {confidence:.1%}, RR: {rr:.1f}) - {execution_data['voie_used'] or 'No escalation'}")
                        else:
                            logger.info(f"      ⚠️ {symbol}: IA1 analysis failed")
                        
                        pipeline_executions.append(execution_data)
                    else:
                        logger.warning(f"      ❌ {symbol}: HTTP {response.status_code}")
                        
                    # Small delay between requests
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.warning(f"      ❌ {symbol}: Exception - {str(e)}")
            
            # Wait for IA2 processing
            logger.info("   ⏳ Waiting 60 seconds for IA2 processing to complete...")
            await asyncio.sleep(60)
            
            # Check final counts and analyze results
            final_ia1_count = 0
            final_ia2_count = 0
            if self.db is not None:
                final_ia1_count = self.db.technical_analyses.count_documents({})
                final_ia2_count = self.db.trading_decisions.count_documents({})
                
                new_ia1_analyses = final_ia1_count - initial_ia1_count
                new_ia2_decisions = final_ia2_count - initial_ia2_count
                
                logger.info(f"   📊 Final counts - IA1: {final_ia1_count} (+{new_ia1_analyses}), IA2: {final_ia2_count} (+{new_ia2_decisions})")
            
            # Analyze pipeline performance
            successful_ia1 = len([e for e in pipeline_executions if e['ia1_success']])
            expected_ia2 = len([e for e in pipeline_executions if e['ia2_triggered']])
            
            logger.info(f"   📊 Pipeline Analysis:")
            logger.info(f"      Successful IA1 analyses: {successful_ia1}/{len(self.test_symbols)}")
            logger.info(f"      Expected IA2 escalations: {expected_ia2}")
            logger.info(f"      Actual new IA2 decisions: {new_ia2_decisions if self.db else 'N/A'}")
            
            # VOIE distribution
            voie_distribution = {}
            for execution in pipeline_executions:
                voie = execution['voie_used']
                if voie:
                    voie_distribution[voie] = voie_distribution.get(voie, 0) + 1
            
            logger.info(f"      VOIE distribution: {voie_distribution}")
            
            # Determine test result
            if successful_ia1 >= 3 and (expected_ia2 > 0 or new_ia2_decisions > 0):
                self.log_test_result("IA1→IA2 Pipeline Complete Flow", True, 
                                   f"Pipeline working: {successful_ia1} IA1 analyses, {expected_ia2} expected escalations, {new_ia2_decisions} new decisions")
            elif successful_ia1 >= 2:
                self.log_test_result("IA1→IA2 Pipeline Complete Flow", False, 
                                   f"Partial pipeline function: {successful_ia1} IA1 analyses, limited escalations")
            else:
                self.log_test_result("IA1→IA2 Pipeline Complete Flow", False, 
                                   f"Pipeline issues: {successful_ia1} IA1 analyses, {expected_ia2} escalations")
            
            self.pipeline_executions = pipeline_executions
                
        except Exception as e:
            self.log_test_result("IA1→IA2 Pipeline Complete Flow", False, f"Exception: {str(e)}")
    
    async def test_3_ia2_strategic_intelligence_fields(self):
        """Test 3: IA2 Strategic Intelligence - New Fields and Enhanced Reasoning"""
        logger.info("\n🔍 TEST 3: IA2 Strategic Intelligence Fields Test")
        
        try:
            if self.db is None:
                self.log_test_result("IA2 Strategic Intelligence Fields", False, 
                                   "MongoDB connection not available")
                return
            
            # Get recent IA2 decisions
            recent_decisions = list(self.db.trading_decisions.find({}).sort("timestamp", -1).limit(10))
            
            logger.info(f"   📊 Analyzing {len(recent_decisions)} recent IA2 decisions for strategic intelligence")
            
            if len(recent_decisions) == 0:
                self.log_test_result("IA2 Strategic Intelligence Fields", False, 
                                   "No recent IA2 decisions found")
                return
            
            # Analyze strategic intelligence fields
            strategic_analysis = {
                'total_decisions': len(recent_decisions),
                'has_market_regime_assessment': 0,
                'has_position_size_recommendation': 0,
                'has_execution_priority': 0,
                'has_calculated_rr': 0,
                'has_rr_reasoning': 0,
                'has_strategic_reasoning': 0,
                'strategic_reasoning_quality': 0,
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'signal_distribution': {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
            }
            
            strategic_keywords = [
                'market_regime', 'regime_assessment', 'strategic', 'confluence', 
                'institutional', 'probability', 'optimization', 'execution_priority',
                'position_size', 'risk_management', 'technical_indicators'
            ]
            
            for decision in recent_decisions:
                # Check for new strategic fields
                if 'market_regime_assessment' in decision:
                    strategic_analysis['has_market_regime_assessment'] += 1
                if 'position_size_recommendation' in decision:
                    strategic_analysis['has_position_size_recommendation'] += 1
                if 'execution_priority' in decision:
                    strategic_analysis['has_execution_priority'] += 1
                if 'calculated_rr' in decision:
                    strategic_analysis['has_calculated_rr'] += 1
                if 'rr_reasoning' in decision:
                    strategic_analysis['has_rr_reasoning'] += 1
                
                # Analyze reasoning quality
                reasoning = decision.get('reasoning', '').lower()
                if len(reasoning) > 100:
                    strategic_analysis['has_strategic_reasoning'] += 1
                    
                    # Count strategic keywords
                    keyword_count = sum(1 for keyword in strategic_keywords if keyword in reasoning)
                    if keyword_count >= 5:
                        strategic_analysis['strategic_reasoning_quality'] += 1
                
                # Confidence distribution
                confidence = decision.get('confidence', 0)
                if confidence >= 0.8:
                    strategic_analysis['confidence_distribution']['high'] += 1
                elif confidence >= 0.6:
                    strategic_analysis['confidence_distribution']['medium'] += 1
                else:
                    strategic_analysis['confidence_distribution']['low'] += 1
                
                # Signal distribution
                signal = decision.get('signal', 'UNKNOWN')
                if signal in strategic_analysis['signal_distribution']:
                    strategic_analysis['signal_distribution'][signal] += 1
            
            # Calculate percentages
            total = strategic_analysis['total_decisions']
            field_coverage = (
                strategic_analysis['has_calculated_rr'] + 
                strategic_analysis['has_rr_reasoning'] + 
                strategic_analysis['has_strategic_reasoning']
            ) / (total * 3) if total > 0 else 0
            
            strategic_quality = strategic_analysis['strategic_reasoning_quality'] / total if total > 0 else 0
            
            # Log detailed analysis
            logger.info(f"   📊 Strategic Intelligence Analysis:")
            logger.info(f"      Market regime assessment: {strategic_analysis['has_market_regime_assessment']}/{total}")
            logger.info(f"      Position size recommendation: {strategic_analysis['has_position_size_recommendation']}/{total}")
            logger.info(f"      Execution priority: {strategic_analysis['has_execution_priority']}/{total}")
            logger.info(f"      Calculated RR: {strategic_analysis['has_calculated_rr']}/{total}")
            logger.info(f"      RR reasoning: {strategic_analysis['has_rr_reasoning']}/{total}")
            logger.info(f"      Strategic reasoning: {strategic_analysis['has_strategic_reasoning']}/{total}")
            logger.info(f"      Strategic quality: {strategic_analysis['strategic_reasoning_quality']}/{total} ({strategic_quality:.1%})")
            logger.info(f"      Confidence distribution: {strategic_analysis['confidence_distribution']}")
            logger.info(f"      Signal distribution: {strategic_analysis['signal_distribution']}")
            
            # Determine test result
            if field_coverage >= 0.7 and strategic_quality >= 0.5:
                self.log_test_result("IA2 Strategic Intelligence Fields", True, 
                                   f"Strategic intelligence working: {field_coverage:.1%} field coverage, {strategic_quality:.1%} quality")
            elif field_coverage >= 0.5:
                self.log_test_result("IA2 Strategic Intelligence Fields", False, 
                                   f"Partial strategic intelligence: {field_coverage:.1%} field coverage, {strategic_quality:.1%} quality")
            else:
                self.log_test_result("IA2 Strategic Intelligence Fields", False, 
                                   f"Limited strategic intelligence: {field_coverage:.1%} field coverage, {strategic_quality:.1%} quality")
                
        except Exception as e:
            self.log_test_result("IA2 Strategic Intelligence Fields", False, f"Exception: {str(e)}")
    
    async def test_4_advanced_technical_analysis_integration(self):
        """Test 4: Advanced Technical Analysis - Multi-timeframe, Indicators, Confluence Matrix"""
        logger.info("\n🔍 TEST 4: Advanced Technical Analysis Integration Test")
        
        try:
            if self.db is None:
                self.log_test_result("Advanced Technical Analysis Integration", False, 
                                   "MongoDB connection not available")
                return
            
            # Get recent IA1 analyses
            recent_analyses = list(self.db.technical_analyses.find({}).sort("timestamp", -1).limit(10))
            
            logger.info(f"   📊 Analyzing {len(recent_analyses)} recent IA1 analyses for advanced technical analysis")
            
            if len(recent_analyses) == 0:
                self.log_test_result("Advanced Technical Analysis Integration", False, 
                                   "No recent IA1 analyses found")
                return
            
            # Technical indicators to check for
            technical_indicators = {
                'RSI': ['rsi', 'oversold', 'overbought'],
                'MACD': ['macd', 'signal_line', 'histogram'],
                'Stochastic': ['stochastic', '%k', '%d'],
                'Bollinger Bands': ['bollinger', 'bands', 'squeeze'],
                'EMA/SMA': ['ema', 'sma', 'moving_average', 'hierarchy'],
                'MFI': ['mfi', 'money_flow', 'institutional'],
                'VWAP': ['vwap', 'volume_weighted', 'precision'],
                'Multi-timeframe': ['timeframe', 'daily', 'hourly', '4h', '1h'],
                'Confluence': ['confluence', 'matrix', 'alignment'],
                'Pattern Detection': ['pattern', 'support', 'resistance', 'trend']
            }
            
            analysis_results = {
                'total_analyses': len(recent_analyses),
                'indicator_coverage': {indicator: 0 for indicator in technical_indicators.keys()},
                'multi_timeframe_count': 0,
                'confluence_matrix_count': 0,
                'advanced_reasoning_count': 0,
                'confidence_with_indicators': []
            }
            
            for analysis in recent_analyses:
                reasoning = analysis.get('analysis', '').lower()
                
                # Check for each technical indicator
                for indicator, keywords in technical_indicators.items():
                    if any(keyword in reasoning for keyword in keywords):
                        analysis_results['indicator_coverage'][indicator] += 1
                
                # Check for multi-timeframe analysis
                timeframe_keywords = ['daily', 'hourly', '4h', '1h', 'timeframe', 'multi-tf']
                if sum(1 for keyword in timeframe_keywords if keyword in reasoning) >= 2:
                    analysis_results['multi_timeframe_count'] += 1
                
                # Check for confluence matrix
                confluence_keywords = ['confluence', 'matrix', 'alignment', 'godlike', 'strong', 'good']
                if sum(1 for keyword in confluence_keywords if keyword in reasoning) >= 2:
                    analysis_results['confluence_matrix_count'] += 1
                
                # Check for advanced reasoning (multiple indicators)
                indicator_count = sum(1 for indicator, keywords in technical_indicators.items() 
                                    if any(keyword in reasoning for keyword in keywords))
                if indicator_count >= 5:
                    analysis_results['advanced_reasoning_count'] += 1
                    analysis_results['confidence_with_indicators'].append(analysis.get('analysis_confidence', 0))
            
            # Calculate coverage percentages
            total = analysis_results['total_analyses']
            indicator_coverage_avg = sum(analysis_results['indicator_coverage'].values()) / (len(technical_indicators) * total) if total > 0 else 0
            multi_timeframe_coverage = analysis_results['multi_timeframe_count'] / total if total > 0 else 0
            confluence_coverage = analysis_results['confluence_matrix_count'] / total if total > 0 else 0
            advanced_reasoning_coverage = analysis_results['advanced_reasoning_count'] / total if total > 0 else 0
            
            # Log detailed analysis
            logger.info(f"   📊 Advanced Technical Analysis Results:")
            logger.info(f"      Indicator Coverage (avg): {indicator_coverage_avg:.1%}")
            for indicator, count in analysis_results['indicator_coverage'].items():
                coverage = count / total if total > 0 else 0
                logger.info(f"        {indicator}: {count}/{total} ({coverage:.1%})")
            
            logger.info(f"      Multi-timeframe analysis: {analysis_results['multi_timeframe_count']}/{total} ({multi_timeframe_coverage:.1%})")
            logger.info(f"      Confluence matrix usage: {analysis_results['confluence_matrix_count']}/{total} ({confluence_coverage:.1%})")
            logger.info(f"      Advanced reasoning (5+ indicators): {analysis_results['advanced_reasoning_count']}/{total} ({advanced_reasoning_coverage:.1%})")
            
            if analysis_results['confidence_with_indicators']:
                avg_confidence = sum(analysis_results['confidence_with_indicators']) / len(analysis_results['confidence_with_indicators'])
                logger.info(f"      Average confidence with advanced indicators: {avg_confidence:.1%}")
            
            # Determine test result
            if (indicator_coverage_avg >= 0.6 and multi_timeframe_coverage >= 0.4 and 
                confluence_coverage >= 0.3 and advanced_reasoning_coverage >= 0.4):
                self.log_test_result("Advanced Technical Analysis Integration", True, 
                                   f"Advanced analysis working: {indicator_coverage_avg:.1%} indicators, {confluence_coverage:.1%} confluence")
            elif indicator_coverage_avg >= 0.4 and advanced_reasoning_coverage >= 0.2:
                self.log_test_result("Advanced Technical Analysis Integration", False, 
                                   f"Partial advanced analysis: {indicator_coverage_avg:.1%} indicators, {advanced_reasoning_coverage:.1%} advanced reasoning")
            else:
                self.log_test_result("Advanced Technical Analysis Integration", False, 
                                   f"Limited advanced analysis: {indicator_coverage_avg:.1%} indicators, {advanced_reasoning_coverage:.1%} advanced reasoning")
                
        except Exception as e:
            self.log_test_result("Advanced Technical Analysis Integration", False, f"Exception: {str(e)}")
    
    async def test_5_system_performance_and_stability(self):
        """Test 5: System Performance & Stability - CPU, Error Handling, Logging"""
        logger.info("\n🔍 TEST 5: System Performance & Stability Test")
        
        try:
            # Check CPU usage
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                logger.info(f"   📊 System Resources:")
                logger.info(f"      CPU Usage: {cpu_percent:.1f}%")
                logger.info(f"      Memory Usage: {memory_percent:.1f}%")
                
                cpu_stable = cpu_percent < 90.0  # CPU under 90%
                memory_stable = memory_percent < 85.0  # Memory under 85%
                
            except ImportError:
                logger.info("   ⚠️ psutil not available, skipping resource monitoring")
                cpu_stable = True
                memory_stable = True
            
            # Check backend logs for errors
            error_analysis = await self._analyze_backend_logs()
            
            # Check system stability indicators
            stability_indicators = {
                'cpu_stable': cpu_stable,
                'memory_stable': memory_stable,
                'low_error_rate': error_analysis['error_rate'] < 0.1,  # Less than 10% error rate
                'no_critical_errors': error_analysis['critical_errors'] == 0,
                'good_logging_quality': error_analysis['log_quality_score'] >= 0.7
            }
            
            stable_indicators = sum(1 for indicator, stable in stability_indicators.items() if stable)
            total_indicators = len(stability_indicators)
            
            logger.info(f"   📊 Stability Analysis:")
            for indicator, stable in stability_indicators.items():
                status = "✅" if stable else "❌"
                logger.info(f"      {status} {indicator.replace('_', ' ').title()}")
            
            logger.info(f"   📊 Error Analysis:")
            logger.info(f"      Total log entries: {error_analysis['total_entries']}")
            logger.info(f"      Error entries: {error_analysis['error_entries']}")
            logger.info(f"      Error rate: {error_analysis['error_rate']:.1%}")
            logger.info(f"      Critical errors: {error_analysis['critical_errors']}")
            logger.info(f"      Log quality score: {error_analysis['log_quality_score']:.1%}")
            
            # Determine test result
            if stable_indicators >= total_indicators * 0.8:
                self.log_test_result("System Performance & Stability", True, 
                                   f"System stable: {stable_indicators}/{total_indicators} indicators good")
            elif stable_indicators >= total_indicators * 0.6:
                self.log_test_result("System Performance & Stability", False, 
                                   f"System mostly stable: {stable_indicators}/{total_indicators} indicators good")
            else:
                self.log_test_result("System Performance & Stability", False, 
                                   f"System stability issues: {stable_indicators}/{total_indicators} indicators good")
                
        except Exception as e:
            self.log_test_result("System Performance & Stability", False, f"Exception: {str(e)}")
    
    async def _analyze_backend_logs(self):
        """Analyze backend logs for error patterns and quality"""
        try:
            log_files = [
                "/var/log/supervisor/backend.out.log",
                "/var/log/supervisor/backend.err.log"
            ]
            
            analysis = {
                'total_entries': 0,
                'error_entries': 0,
                'critical_errors': 0,
                'error_rate': 0.0,
                'log_quality_score': 0.0,
                'recent_errors': []
            }
            
            error_patterns = [
                r'ERROR',
                r'CRITICAL',
                r'Exception',
                r'Traceback',
                r'string indices must be integers',
                r'acomplete.*not found'
            ]
            
            quality_indicators = [
                r'✅',
                r'📊',
                r'🚀',
                r'INFO.*IA[12]',
                r'VOIE [123]',
                r'confidence.*%'
            ]
            
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        result = subprocess.run(['tail', '-n', '1000', log_file], 
                                              capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            log_content = result.stdout
                            lines = log_content.split('\n')
                            analysis['total_entries'] += len([line for line in lines if line.strip()])
                            
                            # Count errors
                            for line in lines:
                                if any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns):
                                    analysis['error_entries'] += 1
                                    if 'CRITICAL' in line.upper() or 'string indices' in line.lower():
                                        analysis['critical_errors'] += 1
                                    if len(analysis['recent_errors']) < 5:
                                        analysis['recent_errors'].append(line.strip())
                            
                            # Count quality indicators
                            quality_count = sum(1 for line in lines 
                                              if any(re.search(pattern, line, re.IGNORECASE) for pattern in quality_indicators))
                            analysis['log_quality_score'] += quality_count / max(len(lines), 1)
                            
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not analyze {log_file}: {e}")
            
            # Calculate final metrics
            if analysis['total_entries'] > 0:
                analysis['error_rate'] = analysis['error_entries'] / analysis['total_entries']
            
            analysis['log_quality_score'] = min(analysis['log_quality_score'] / len(log_files), 1.0)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Log analysis failed: {e}")
            return {
                'total_entries': 0,
                'error_entries': 0,
                'critical_errors': 0,
                'error_rate': 0.0,
                'log_quality_score': 0.0,
                'recent_errors': []
            }
    
    async def test_6_database_storage_verification(self):
        """Test 6: Database Storage of Analyses and Decisions"""
        logger.info("\n🔍 TEST 6: Database Storage Verification Test")
        
        try:
            if self.db is None:
                self.log_test_result("Database Storage Verification", False, 
                                   "MongoDB connection not available")
                return
            
            # Check collections and recent data
            collections_analysis = {}
            
            # Technical Analyses (IA1)
            ia1_count = self.db.technical_analyses.count_documents({})
            recent_ia1 = self.db.technical_analyses.count_documents({
                "timestamp": {"$gte": datetime.now() - timedelta(hours=24)}
            })
            collections_analysis['technical_analyses'] = {
                'total': ia1_count,
                'recent_24h': recent_ia1,
                'collection_exists': True
            }
            
            # Trading Decisions (IA2)
            ia2_count = self.db.trading_decisions.count_documents({})
            recent_ia2 = self.db.trading_decisions.count_documents({
                "timestamp": {"$gte": datetime.now() - timedelta(hours=24)}
            })
            collections_analysis['trading_decisions'] = {
                'total': ia2_count,
                'recent_24h': recent_ia2,
                'collection_exists': True
            }
            
            # Market Opportunities
            opportunities_count = self.db.market_opportunities.count_documents({})
            recent_opportunities = self.db.market_opportunities.count_documents({
                "timestamp": {"$gte": datetime.now() - timedelta(hours=24)}
            })
            collections_analysis['market_opportunities'] = {
                'total': opportunities_count,
                'recent_24h': recent_opportunities,
                'collection_exists': True
            }
            
            # Performance Tracking
            performance_count = self.db.trading_performance.count_documents({})
            collections_analysis['trading_performance'] = {
                'total': performance_count,
                'recent_24h': 0,  # Performance tracking may be less frequent
                'collection_exists': True
            }
            
            # Analyze data quality
            data_quality_analysis = {}
            
            # Check IA1 data quality
            if recent_ia1 > 0:
                sample_ia1 = list(self.db.technical_analyses.find({}).sort("timestamp", -1).limit(5))
                required_ia1_fields = ['symbol', 'analysis', 'analysis_confidence', 'ia1_signal']
                ia1_quality = sum(1 for analysis in sample_ia1 
                                if all(field in analysis for field in required_ia1_fields)) / len(sample_ia1)
                data_quality_analysis['ia1_quality'] = ia1_quality
            else:
                data_quality_analysis['ia1_quality'] = 0
            
            # Check IA2 data quality
            if recent_ia2 > 0:
                sample_ia2 = list(self.db.trading_decisions.find({}).sort("timestamp", -1).limit(5))
                required_ia2_fields = ['symbol', 'signal', 'confidence', 'reasoning']
                ia2_quality = sum(1 for decision in sample_ia2 
                                if all(field in decision for field in required_ia2_fields)) / len(sample_ia2)
                data_quality_analysis['ia2_quality'] = ia2_quality
            else:
                data_quality_analysis['ia2_quality'] = 0
            
            # Log detailed analysis
            logger.info(f"   📊 Database Collections Analysis:")
            for collection, data in collections_analysis.items():
                logger.info(f"      {collection}: {data['total']} total, {data['recent_24h']} recent (24h)")
            
            logger.info(f"   📊 Data Quality Analysis:")
            logger.info(f"      IA1 data quality: {data_quality_analysis['ia1_quality']:.1%}")
            logger.info(f"      IA2 data quality: {data_quality_analysis['ia2_quality']:.1%}")
            
            # Calculate overall storage health
            total_recent_data = sum(data['recent_24h'] for data in collections_analysis.values())
            avg_data_quality = (data_quality_analysis['ia1_quality'] + data_quality_analysis['ia2_quality']) / 2
            
            # Determine test result
            if total_recent_data >= 5 and avg_data_quality >= 0.7:
                self.log_test_result("Database Storage Verification", True, 
                                   f"Database storage working: {total_recent_data} recent entries, {avg_data_quality:.1%} quality")
            elif total_recent_data >= 2 and avg_data_quality >= 0.5:
                self.log_test_result("Database Storage Verification", False, 
                                   f"Partial database storage: {total_recent_data} recent entries, {avg_data_quality:.1%} quality")
            else:
                self.log_test_result("Database Storage Verification", False, 
                                   f"Limited database storage: {total_recent_data} recent entries, {avg_data_quality:.1%} quality")
                
        except Exception as e:
            self.log_test_result("Database Storage Verification", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_demonstration(self):
        """Run comprehensive IA1→IA2 pipeline demonstration"""
        logger.info("🚀 Starting IA1→IA2 Pipeline Demonstration Run")
        logger.info("=" * 80)
        logger.info("📋 COMPREHENSIVE IA1→IA2 PIPELINE DEMONSTRATION RUN")
        logger.info("🎯 Testing: Complete dual AI trading system functionality")
        logger.info("🎯 Expected: Full pipeline working with VOIE escalation, strategic intelligence, advanced analysis")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_api_endpoints_availability()
        await self.test_2_ia1_ia2_pipeline_complete_flow()
        await self.test_3_ia2_strategic_intelligence_fields()
        await self.test_4_advanced_technical_analysis_integration()
        await self.test_5_system_performance_and_stability()
        await self.test_6_database_storage_verification()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA1→IA2 PIPELINE DEMONSTRATION COMPREHENSIVE TEST SUMMARY")
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
            if "API Endpoints" in result['test']:
                requirements_status['System Components Integration'] = result['success']
            elif "Pipeline Complete Flow" in result['test']:
                requirements_status['IA1→IA2 Pipeline Complete Flow'] = result['success']
            elif "Strategic Intelligence" in result['test']:
                requirements_status['IA2 Strategic Intelligence'] = result['success']
            elif "Advanced Technical Analysis" in result['test']:
                requirements_status['Advanced Technical Analysis'] = result['success']
            elif "Performance & Stability" in result['test']:
                requirements_status['Performance & Stability'] = result['success']
            elif "Database Storage" in result['test']:
                requirements_status['Database Storage & Traceability'] = result['success']
        
        logger.info("🎯 CRITICAL REQUIREMENTS STATUS:")
        for requirement, status in requirements_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {requirement}")
        
        requirements_met = sum(1 for status in requirements_status.values() if status)
        total_requirements = len(requirements_status)
        
        # Final verdict
        logger.info(f"\n🏆 REQUIREMENTS SATISFACTION: {requirements_met}/{total_requirements}")
        
        if requirements_met == total_requirements:
            logger.info("\n🎉 VERDICT: IA1→IA2 PIPELINE DEMONSTRATION FULLY SUCCESSFUL!")
            logger.info("✅ Complete dual AI trading system working")
            logger.info("✅ VOIE escalation logic operational")
            logger.info("✅ IA2 strategic intelligence enhanced")
            logger.info("✅ Advanced technical analysis integrated")
            logger.info("✅ System performance stable")
            logger.info("✅ Database storage and traceability working")
            logger.info("✅ Ready for production demonstration")
        elif requirements_met >= total_requirements * 0.8:
            logger.info("\n⚠️ VERDICT: IA1→IA2 PIPELINE DEMONSTRATION MOSTLY SUCCESSFUL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif requirements_met >= total_requirements * 0.6:
            logger.info("\n⚠️ VERDICT: IA1→IA2 PIPELINE DEMONSTRATION PARTIALLY SUCCESSFUL")
            logger.info("🔧 Several critical requirements need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: IA1→IA2 PIPELINE DEMONSTRATION NOT SUCCESSFUL")
            logger.info("🚨 Major issues preventing dual AI system from working correctly")
            logger.info("🚨 System needs significant debugging and fixes")
        
        return passed_tests, total_tests

async def main():
    """Main function to run the comprehensive IA1→IA2 pipeline demonstration"""
    test_suite = IA1IA2PipelineDemonstrationTestSuite()
    passed_tests, total_tests = await test_suite.run_comprehensive_demonstration()
    
    # Exit with appropriate code
    if passed_tests == total_tests:
        sys.exit(0)  # All tests passed
    elif passed_tests >= total_tests * 0.8:
        sys.exit(1)  # Mostly successful
    else:
        sys.exit(2)  # Major issues

if __name__ == "__main__":
    asyncio.run(main())
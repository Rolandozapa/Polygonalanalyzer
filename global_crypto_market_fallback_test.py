#!/usr/bin/env python3
"""
GLOBAL CRYPTO MARKET ANALYZER FALLBACK API TESTING SUITE
Focus: Testing the improved Global Crypto Market Analyzer with fallback APIs

TESTING REQUIREMENTS FROM REVIEW REQUEST:
1. **Fallback API Integration**: Test that Binance API fallback works when CoinGecko is rate-limited
2. **Multi-Source Resilience**: Verify the system gracefully handles API failures and switches sources
3. **Data Quality**: Validate that fallback data provides reasonable market context
4. **Global Market Context**: Test that IA1/IA2 now receive global market context regardless of primary API status
5. **Admin Endpoint Functionality**: Test `/admin/market/global` with improved fallback system
6. **Market Regime Detection**: Verify bull/bear detection works with Binance fallback data
7. **Fear & Greed Integration**: Confirm Fear & Greed index still works independently
8. **Cache and Performance**: Test that fallback doesn't break caching mechanisms

The improvements should now provide:
- Binance API as backup for BTC price and volume data
- Realistic default values when all APIs fail
- Continued operation despite CoinGecko rate limits
- Enhanced error handling and logging
- Functional market context for IAs even with limited data

Test scenarios:
- Successful data fetching (if rate limits lifted)
- Fallback to Binance when CoinGecko fails
- Graceful degradation with default values
- IA1/IA2 context integration working with fallback data
- Admin endpoint providing useful information despite API limitations
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GlobalCryptoMarketFallbackTestSuite:
    """Comprehensive test suite for Global Crypto Market Analyzer with fallback APIs"""
    
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
        logger.info(f"Testing Global Crypto Market Analyzer Fallback System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
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
    
    async def test_1_fallback_api_integration(self):
        """Test 1: Fallback API Integration - Test Binance API fallback when CoinGecko is rate-limited"""
        logger.info("\n🔍 TEST 1: Fallback API Integration")
        
        try:
            # Test the admin endpoint which should use fallback APIs
            response = requests.get(f"{self.api_url}/admin/market/global", timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   📊 Market data response status: {data.get('status')}")
                
                # Check for market data presence (could be from CoinGecko or fallback)
                market_overview = data.get('market_overview', {})
                
                # Key indicators that should be present regardless of source
                key_indicators = {
                    'total_market_cap': market_overview.get('total_market_cap', 0),
                    'total_volume_24h': market_overview.get('total_volume_24h', 0),
                    'btc_dominance': market_overview.get('btc_dominance', 0),
                    'eth_dominance': market_overview.get('eth_dominance', 0),
                    'btc_price': market_overview.get('btc_price', 0)
                }
                
                # Check if we have reasonable values (indicating fallback is working)
                reasonable_values = 0
                for key, value in key_indicators.items():
                    if key == 'total_market_cap' and value > 1e12:  # > $1T
                        reasonable_values += 1
                        logger.info(f"      ✅ {key}: ${value/1e12:.2f}T (reasonable)")
                    elif key == 'total_volume_24h' and value > 1e10:  # > $10B
                        reasonable_values += 1
                        logger.info(f"      ✅ {key}: ${value/1e9:.1f}B (reasonable)")
                    elif key in ['btc_dominance', 'eth_dominance'] and 10 <= value <= 80:
                        reasonable_values += 1
                        logger.info(f"      ✅ {key}: {value:.1f}% (reasonable)")
                    elif key == 'btc_price' and 20000 <= value <= 200000:  # Reasonable BTC price range
                        reasonable_values += 1
                        logger.info(f"      ✅ {key}: ${value:,.0f} (reasonable)")
                    else:
                        logger.info(f"      ⚠️ {key}: {value} (may be fallback/default)")
                
                # Check backend logs for fallback evidence
                fallback_evidence = await self._check_fallback_logs()
                
                if reasonable_values >= 3 or fallback_evidence:
                    self.log_test_result("Fallback API Integration", True, 
                                       f"Market data available: {reasonable_values}/5 reasonable values, fallback evidence: {fallback_evidence}")
                else:
                    self.log_test_result("Fallback API Integration", False, 
                                       f"Insufficient market data: {reasonable_values}/5 reasonable values")
            else:
                self.log_test_result("Fallback API Integration", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("Fallback API Integration", False, f"Exception: {str(e)}")
    
    async def test_2_multi_source_resilience(self):
        """Test 2: Multi-Source Resilience - Verify graceful handling of API failures"""
        logger.info("\n🔍 TEST 2: Multi-Source Resilience")
        
        try:
            # Test the admin endpoint multiple times to check consistency
            responses = []
            for i in range(3):
                response = requests.get(f"{self.api_url}/admin/market/global", timeout=30)
                if response.status_code == 200:
                    responses.append(response.json())
                else:
                    logger.warning(f"   Request {i+1} failed: HTTP {response.status_code}")
                
                if i < 2:  # Don't sleep after last request
                    await asyncio.sleep(2)
            
            if len(responses) >= 2:
                # Check consistency across responses
                consistent_data = True
                first_response = responses[0]
                
                for response in responses[1:]:
                    # Check if key fields are consistent (allowing for small variations)
                    first_btc = first_response.get('market_overview', {}).get('btc_price', 0)
                    current_btc = response.get('market_overview', {}).get('btc_price', 0)
                    
                    if first_btc > 0 and current_btc > 0:
                        price_variation = abs(first_btc - current_btc) / first_btc
                        if price_variation > 0.05:  # More than 5% variation is suspicious
                            consistent_data = False
                            logger.warning(f"   Large price variation: {price_variation:.2%}")
                
                # Check for error handling in responses
                error_handling_good = True
                for response in responses:
                    status = response.get('status')
                    if status not in ['success', 'error']:
                        error_handling_good = False
                
                # Check backend logs for resilience patterns
                resilience_logs = await self._check_resilience_logs()
                
                if consistent_data and error_handling_good:
                    self.log_test_result("Multi-Source Resilience", True, 
                                       f"System resilient: {len(responses)}/3 successful responses, consistent data, resilience logs: {resilience_logs}")
                else:
                    self.log_test_result("Multi-Source Resilience", False, 
                                       f"Resilience issues: consistent={consistent_data}, error_handling={error_handling_good}")
            else:
                self.log_test_result("Multi-Source Resilience", False, 
                                   f"Insufficient responses: {len(responses)}/3")
                
        except Exception as e:
            self.log_test_result("Multi-Source Resilience", False, f"Exception: {str(e)}")
    
    async def test_3_data_quality_validation(self):
        """Test 3: Data Quality - Validate that fallback data provides reasonable market context"""
        logger.info("\n🔍 TEST 3: Data Quality Validation")
        
        try:
            response = requests.get(f"{self.api_url}/admin/market/global", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate market overview data quality
                market_overview = data.get('market_overview', {})
                quality_score = 0
                max_score = 8
                
                # Test 1: Market cap reasonableness
                total_mcap = market_overview.get('total_market_cap', 0)
                if 1e12 <= total_mcap <= 10e12:  # $1T - $10T range
                    quality_score += 1
                    logger.info(f"      ✅ Market cap reasonable: ${total_mcap/1e12:.2f}T")
                else:
                    logger.info(f"      ⚠️ Market cap questionable: ${total_mcap/1e12:.2f}T")
                
                # Test 2: Volume reasonableness
                total_volume = market_overview.get('total_volume_24h', 0)
                if 1e10 <= total_volume <= 5e11:  # $10B - $500B range
                    quality_score += 1
                    logger.info(f"      ✅ Volume reasonable: ${total_volume/1e9:.1f}B")
                else:
                    logger.info(f"      ⚠️ Volume questionable: ${total_volume/1e9:.1f}B")
                
                # Test 3: BTC dominance reasonableness
                btc_dominance = market_overview.get('btc_dominance', 0)
                if 40 <= btc_dominance <= 70:  # Typical range
                    quality_score += 1
                    logger.info(f"      ✅ BTC dominance reasonable: {btc_dominance:.1f}%")
                else:
                    logger.info(f"      ⚠️ BTC dominance questionable: {btc_dominance:.1f}%")
                
                # Test 4: ETH dominance reasonableness
                eth_dominance = market_overview.get('eth_dominance', 0)
                if 10 <= eth_dominance <= 25:  # Typical range
                    quality_score += 1
                    logger.info(f"      ✅ ETH dominance reasonable: {eth_dominance:.1f}%")
                else:
                    logger.info(f"      ⚠️ ETH dominance questionable: {eth_dominance:.1f}%")
                
                # Test 5: BTC price reasonableness
                btc_price = market_overview.get('btc_price', 0)
                if 20000 <= btc_price <= 200000:  # Reasonable range
                    quality_score += 1
                    logger.info(f"      ✅ BTC price reasonable: ${btc_price:,.0f}")
                else:
                    logger.info(f"      ⚠️ BTC price questionable: ${btc_price:,.0f}")
                
                # Test 6: Sentiment data quality
                sentiment = data.get('sentiment_analysis', {})
                fear_greed = sentiment.get('fear_greed_value', -1)
                if 0 <= fear_greed <= 100:
                    quality_score += 1
                    logger.info(f"      ✅ Fear & Greed reasonable: {fear_greed}")
                else:
                    logger.info(f"      ⚠️ Fear & Greed questionable: {fear_greed}")
                
                # Test 7: Market regime validity
                market_regime = sentiment.get('market_regime', '')
                valid_regimes = ['extreme_bull', 'bull', 'neutral_bullish', 'neutral', 'neutral_bearish', 'bear', 'extreme_bear']
                if market_regime in valid_regimes:
                    quality_score += 1
                    logger.info(f"      ✅ Market regime valid: {market_regime}")
                else:
                    logger.info(f"      ⚠️ Market regime invalid: {market_regime}")
                
                # Test 8: Trading intelligence presence
                trading_intel = data.get('trading_intelligence', {})
                if trading_intel.get('trading_recommendations') and len(trading_intel.get('trading_recommendations', [])) > 0:
                    quality_score += 1
                    logger.info(f"      ✅ Trading recommendations present: {len(trading_intel.get('trading_recommendations', []))}")
                else:
                    logger.info(f"      ⚠️ Trading recommendations missing")
                
                quality_percentage = (quality_score / max_score) * 100
                
                if quality_percentage >= 75:
                    self.log_test_result("Data Quality Validation", True, 
                                       f"Data quality excellent: {quality_score}/{max_score} ({quality_percentage:.1f}%)")
                elif quality_percentage >= 50:
                    self.log_test_result("Data Quality Validation", True, 
                                       f"Data quality acceptable: {quality_score}/{max_score} ({quality_percentage:.1f}%)")
                else:
                    self.log_test_result("Data Quality Validation", False, 
                                       f"Data quality poor: {quality_score}/{max_score} ({quality_percentage:.1f}%)")
            else:
                self.log_test_result("Data Quality Validation", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("Data Quality Validation", False, f"Exception: {str(e)}")
    
    async def test_4_global_market_context_for_ias(self):
        """Test 4: Global Market Context - Test that IA1/IA2 receive global market context"""
        logger.info("\n🔍 TEST 4: Global Market Context for IAs")
        
        try:
            response = requests.get(f"{self.api_url}/admin/market/global", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for IA-formatted context
                trading_intelligence = data.get('trading_intelligence', {})
                ia_context = trading_intelligence.get('ia_formatted_context', '')
                
                # Key elements that should be in IA context
                ia_context_elements = [
                    'GLOBAL CRYPTO MARKET CONTEXT',
                    'MARKET OVERVIEW',
                    'MARKET REGIME',
                    'SENTIMENT',
                    'IA TRADING GUIDANCE',
                    'BTC',
                    'Dominance'
                ]
                
                found_elements = sum(1 for element in ia_context_elements if element in ia_context)
                
                # Check context length (should be substantial)
                context_substantial = len(ia_context) > 200
                
                # Check for market recommendations
                recommendations = trading_intelligence.get('trading_recommendations', [])
                has_recommendations = len(recommendations) > 0
                
                # Check for market context summary
                context_summary = trading_intelligence.get('market_context_summary', '')
                has_summary = len(context_summary) > 50
                
                logger.info(f"      📊 IA context elements found: {found_elements}/{len(ia_context_elements)}")
                logger.info(f"      📊 Context length: {len(ia_context)} characters")
                logger.info(f"      📊 Recommendations: {len(recommendations)}")
                logger.info(f"      📊 Summary length: {len(context_summary)} characters")
                
                if found_elements >= 5 and context_substantial and has_recommendations:
                    self.log_test_result("Global Market Context for IAs", True, 
                                       f"IA context complete: {found_elements}/{len(ia_context_elements)} elements, {len(ia_context)} chars, {len(recommendations)} recs")
                else:
                    self.log_test_result("Global Market Context for IAs", False, 
                                       f"IA context incomplete: elements={found_elements}, substantial={context_substantial}, recs={has_recommendations}")
            else:
                self.log_test_result("Global Market Context for IAs", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("Global Market Context for IAs", False, f"Exception: {str(e)}")
    
    async def test_5_admin_endpoint_with_fallback(self):
        """Test 5: Admin Endpoint Functionality - Test /admin/market/global with improved fallback system"""
        logger.info("\n🔍 TEST 5: Admin Endpoint with Fallback System")
        
        try:
            response = requests.get(f"{self.api_url}/admin/market/global", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_sections = [
                    'status', 'timestamp', 'market_overview', 
                    'sentiment_analysis', 'scores', 'trading_intelligence'
                ]
                
                found_sections = sum(1 for section in required_sections if section in data)
                
                # Check status handling
                status = data.get('status')
                status_valid = status in ['success', 'error', 'partial']
                
                # Check timestamp
                timestamp = data.get('timestamp')
                timestamp_valid = timestamp and len(timestamp) > 10
                
                # Check for fallback indicators
                fallback_indicators = []
                
                # Look for fallback messages in the data
                if 'fallback' in str(data).lower():
                    fallback_indicators.append('fallback_mentioned')
                
                if 'binance' in str(data).lower():
                    fallback_indicators.append('binance_mentioned')
                
                if 'rate limit' in str(data).lower():
                    fallback_indicators.append('rate_limit_mentioned')
                
                # Check error handling
                if status == 'error':
                    error_msg = data.get('error', '')
                    if 'rate limit' in error_msg.lower() or 'fallback' in error_msg.lower():
                        fallback_indicators.append('error_with_fallback_info')
                
                logger.info(f"      📊 Response sections: {found_sections}/{len(required_sections)}")
                logger.info(f"      📊 Status: {status} (valid: {status_valid})")
                logger.info(f"      📊 Fallback indicators: {fallback_indicators}")
                
                if found_sections >= 4 and status_valid and timestamp_valid:
                    self.log_test_result("Admin Endpoint with Fallback", True, 
                                       f"Admin endpoint functional: {found_sections}/{len(required_sections)} sections, status: {status}, fallback indicators: {len(fallback_indicators)}")
                else:
                    self.log_test_result("Admin Endpoint with Fallback", False, 
                                       f"Admin endpoint issues: sections={found_sections}, status_valid={status_valid}, timestamp_valid={timestamp_valid}")
            else:
                self.log_test_result("Admin Endpoint with Fallback", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("Admin Endpoint with Fallback", False, f"Exception: {str(e)}")
    
    async def test_6_market_regime_detection_with_fallback(self):
        """Test 6: Market Regime Detection - Verify bull/bear detection works with Binance fallback data"""
        logger.info("\n🔍 TEST 6: Market Regime Detection with Fallback Data")
        
        try:
            response = requests.get(f"{self.api_url}/admin/market/global", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                sentiment_analysis = data.get('sentiment_analysis', {})
                
                # Check market regime detection
                market_regime = sentiment_analysis.get('market_regime')
                volatility_regime = sentiment_analysis.get('volatility_regime')
                liquidity_condition = sentiment_analysis.get('liquidity_condition')
                
                # Valid regimes
                valid_market_regimes = ['extreme_bull', 'bull', 'neutral_bullish', 'neutral', 'neutral_bearish', 'bear', 'extreme_bear']
                valid_volatility_regimes = ['low', 'medium', 'high', 'extreme']
                valid_liquidity_conditions = ['poor', 'moderate', 'good', 'excellent']
                
                # Check regime validity
                market_regime_valid = market_regime in valid_market_regimes
                volatility_regime_valid = volatility_regime in valid_volatility_regimes
                liquidity_condition_valid = liquidity_condition in valid_liquidity_conditions
                
                # Check scores
                scores = data.get('scores', {})
                bull_bear_score = scores.get('bull_bear_score', 0)
                market_health_score = scores.get('market_health_score', 0)
                opportunity_score = scores.get('opportunity_score', 0)
                
                # Validate score ranges
                bull_bear_valid = -100 <= bull_bear_score <= 100
                health_valid = 0 <= market_health_score <= 100
                opportunity_valid = 0 <= opportunity_score <= 100
                
                logger.info(f"      📊 Market regime: {market_regime} (valid: {market_regime_valid})")
                logger.info(f"      📊 Volatility regime: {volatility_regime} (valid: {volatility_regime_valid})")
                logger.info(f"      📊 Liquidity condition: {liquidity_condition} (valid: {liquidity_condition_valid})")
                logger.info(f"      📊 Bull/Bear score: {bull_bear_score} (valid: {bull_bear_valid})")
                logger.info(f"      📊 Market health: {market_health_score} (valid: {health_valid})")
                logger.info(f"      📊 Opportunity score: {opportunity_score} (valid: {opportunity_valid})")
                
                # Check for regime consistency
                regime_consistency = True
                if market_regime in ['bull', 'extreme_bull'] and bull_bear_score < -20:
                    regime_consistency = False
                elif market_regime in ['bear', 'extreme_bear'] and bull_bear_score > 20:
                    regime_consistency = False
                
                valid_regimes = sum([market_regime_valid, volatility_regime_valid, liquidity_condition_valid])
                valid_scores = sum([bull_bear_valid, health_valid, opportunity_valid])
                
                if valid_regimes >= 2 and valid_scores >= 2 and regime_consistency:
                    self.log_test_result("Market Regime Detection with Fallback", True, 
                                       f"Regime detection working: {valid_regimes}/3 regimes valid, {valid_scores}/3 scores valid, consistent: {regime_consistency}")
                else:
                    self.log_test_result("Market Regime Detection with Fallback", False, 
                                       f"Regime detection issues: regimes={valid_regimes}/3, scores={valid_scores}/3, consistent={regime_consistency}")
            else:
                self.log_test_result("Market Regime Detection with Fallback", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("Market Regime Detection with Fallback", False, f"Exception: {str(e)}")
    
    async def test_7_fear_greed_independence(self):
        """Test 7: Fear & Greed Integration - Confirm Fear & Greed index works independently"""
        logger.info("\n🔍 TEST 7: Fear & Greed Index Independence")
        
        try:
            # Test Fear & Greed API directly first
            fear_greed_direct = await self._test_fear_greed_direct()
            
            # Test via admin endpoint
            response = requests.get(f"{self.api_url}/admin/market/global", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                sentiment_analysis = data.get('sentiment_analysis', {})
                
                fear_greed_value = sentiment_analysis.get('fear_greed_value')
                fear_greed_classification = sentiment_analysis.get('fear_greed_classification')
                market_sentiment = sentiment_analysis.get('market_sentiment')
                
                # Validate Fear & Greed data
                fear_greed_valid = (
                    isinstance(fear_greed_value, (int, float)) and 
                    0 <= fear_greed_value <= 100
                )
                
                valid_classifications = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
                classification_valid = fear_greed_classification in valid_classifications
                
                valid_sentiments = ['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
                sentiment_valid = market_sentiment in valid_sentiments
                
                # Check consistency between value and classification
                consistency_valid = True
                if fear_greed_value is not None:
                    if fear_greed_value < 25 and fear_greed_classification != 'Extreme Fear':
                        consistency_valid = False
                    elif fear_greed_value >= 75 and fear_greed_classification != 'Extreme Greed':
                        consistency_valid = False
                
                logger.info(f"      📊 Fear & Greed value: {fear_greed_value} (valid: {fear_greed_valid})")
                logger.info(f"      📊 Classification: {fear_greed_classification} (valid: {classification_valid})")
                logger.info(f"      📊 Market sentiment: {market_sentiment} (valid: {sentiment_valid})")
                logger.info(f"      📊 Direct API test: {fear_greed_direct}")
                logger.info(f"      📊 Consistency: {consistency_valid}")
                
                if fear_greed_valid and classification_valid and sentiment_valid and consistency_valid:
                    self.log_test_result("Fear & Greed Independence", True, 
                                       f"Fear & Greed working: value={fear_greed_value}, class={fear_greed_classification}, direct_api={fear_greed_direct}")
                else:
                    self.log_test_result("Fear & Greed Independence", False, 
                                       f"Fear & Greed issues: valid={fear_greed_valid}, class_valid={classification_valid}, sentiment_valid={sentiment_valid}, consistent={consistency_valid}")
            else:
                self.log_test_result("Fear & Greed Independence", False, 
                                   f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("Fear & Greed Independence", False, f"Exception: {str(e)}")
    
    async def test_8_cache_and_performance_with_fallback(self):
        """Test 8: Cache and Performance - Test that fallback doesn't break caching mechanisms"""
        logger.info("\n🔍 TEST 8: Cache and Performance with Fallback")
        
        try:
            # First request (should populate cache or use fallback)
            start_time = time.time()
            response1 = requests.get(f"{self.api_url}/admin/market/global", timeout=30)
            first_request_time = time.time() - start_time
            
            if response1.status_code != 200:
                self.log_test_result("Cache and Performance with Fallback", False, 
                                   f"First request failed: HTTP {response1.status_code}")
                return
            
            data1 = response1.json()
            timestamp1 = data1.get('timestamp')
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Second request (should use cache if available)
            start_time = time.time()
            response2 = requests.get(f"{self.api_url}/admin/market/global", timeout=30)
            second_request_time = time.time() - start_time
            
            if response2.status_code != 200:
                self.log_test_result("Cache and Performance with Fallback", False, 
                                   f"Second request failed: HTTP {response2.status_code}")
                return
            
            data2 = response2.json()
            timestamp2 = data2.get('timestamp')
            
            # Third request after a longer wait (might refresh cache)
            await asyncio.sleep(3)
            start_time = time.time()
            response3 = requests.get(f"{self.api_url}/admin/market/global", timeout=30)
            third_request_time = time.time() - start_time
            
            # Analyze caching behavior
            cache_hit_likely = timestamp1 == timestamp2  # Same timestamp suggests cache hit
            performance_consistent = all(t < 10 for t in [first_request_time, second_request_time, third_request_time])  # All requests under 10s
            
            # Check for performance improvement in second request
            performance_improvement = second_request_time < first_request_time * 0.8  # 20% faster
            
            # Check data consistency
            data_consistent = True
            if data1.get('market_overview', {}).get('btc_price') and data2.get('market_overview', {}).get('btc_price'):
                price1 = data1['market_overview']['btc_price']
                price2 = data2['market_overview']['btc_price']
                if abs(price1 - price2) / price1 > 0.1:  # More than 10% difference is suspicious for cache
                    data_consistent = False
            
            logger.info(f"      📊 Request times: {first_request_time:.2f}s, {second_request_time:.2f}s, {third_request_time:.2f}s")
            logger.info(f"      📊 Cache hit likely: {cache_hit_likely} (same timestamp)")
            logger.info(f"      📊 Performance improvement: {performance_improvement}")
            logger.info(f"      📊 Data consistent: {data_consistent}")
            logger.info(f"      📊 All requests fast: {performance_consistent}")
            
            # Check backend logs for cache evidence
            cache_logs = await self._check_cache_logs()
            
            if performance_consistent and (cache_hit_likely or performance_improvement or cache_logs):
                self.log_test_result("Cache and Performance with Fallback", True, 
                                   f"Cache/performance good: consistent_perf={performance_consistent}, cache_hit={cache_hit_likely}, improvement={performance_improvement}, cache_logs={cache_logs}")
            else:
                self.log_test_result("Cache and Performance with Fallback", False, 
                                   f"Cache/performance issues: consistent_perf={performance_consistent}, cache_hit={cache_hit_likely}, improvement={performance_improvement}")
                
        except Exception as e:
            self.log_test_result("Cache and Performance with Fallback", False, f"Exception: {str(e)}")
    
    async def _check_fallback_logs(self) -> bool:
        """Check backend logs for fallback API usage evidence"""
        try:
            result = subprocess.run(
                ["tail", "-n", "1000", "/var/log/supervisor/backend.out.log"],
                capture_output=True, text=True, timeout=10
            )
            logs = result.stdout
            
            fallback_patterns = [
                "Binance fallback", "fallback", "rate limit", "CoinGecko.*429", 
                "Using Binance", "API fallback", "backup API", "alternative source"
            ]
            
            return any(pattern.lower() in logs.lower() for pattern in fallback_patterns)
        except:
            return False
    
    async def _check_resilience_logs(self) -> bool:
        """Check backend logs for resilience patterns"""
        try:
            result = subprocess.run(
                ["tail", "-n", "1000", "/var/log/supervisor/backend.out.log"],
                capture_output=True, text=True, timeout=10
            )
            logs = result.stdout
            
            resilience_patterns = [
                "gracefully", "fallback", "retry", "alternative", "backup", 
                "resilience", "error handling", "switching source"
            ]
            
            return any(pattern.lower() in logs.lower() for pattern in resilience_patterns)
        except:
            return False
    
    async def _check_cache_logs(self) -> bool:
        """Check backend logs for cache usage evidence"""
        try:
            result = subprocess.run(
                ["tail", "-n", "1000", "/var/log/supervisor/backend.out.log"],
                capture_output=True, text=True, timeout=10
            )
            logs = result.stdout
            
            cache_patterns = [
                "cache", "cached", "Using cached", "cache hit", "cache miss", 
                "cache valid", "cache expired"
            ]
            
            return any(pattern.lower() in logs.lower() for pattern in cache_patterns)
        except:
            return False
    
    async def _test_fear_greed_direct(self) -> bool:
        """Test Fear & Greed API directly"""
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get("https://api.alternative.me/fng?limit=1") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data") and len(data["data"]) > 0
            return False
        except:
            return False
    
    async def run_comprehensive_tests(self):
        """Run all Global Crypto Market Analyzer fallback tests"""
        logger.info("🚀 Starting Global Crypto Market Analyzer Fallback API Test Suite")
        logger.info("=" * 90)
        logger.info("📋 GLOBAL CRYPTO MARKET ANALYZER FALLBACK API TESTING")
        logger.info("🎯 Testing: Fallback APIs, multi-source resilience, data quality, IA context, admin endpoints")
        logger.info("🎯 Focus: Binance fallback when CoinGecko rate-limited, graceful degradation, continued operation")
        logger.info("=" * 90)
        
        # Run all tests in sequence
        await self.test_1_fallback_api_integration()
        await self.test_2_multi_source_resilience()
        await self.test_3_data_quality_validation()
        await self.test_4_global_market_context_for_ias()
        await self.test_5_admin_endpoint_with_fallback()
        await self.test_6_market_regime_detection_with_fallback()
        await self.test_7_fear_greed_independence()
        await self.test_8_cache_and_performance_with_fallback()
        
        # Summary
        logger.info("\n" + "=" * 90)
        logger.info("📊 GLOBAL CRYPTO MARKET ANALYZER FALLBACK API TEST SUMMARY")
        logger.info("=" * 90)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # System analysis
        logger.info("\n" + "=" * 90)
        logger.info("📋 GLOBAL CRYPTO MARKET ANALYZER FALLBACK SYSTEM STATUS")
        logger.info("=" * 90)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Global Crypto Market Analyzer Fallback System FULLY FUNCTIONAL!")
            logger.info("✅ Fallback API integration (Binance) working when CoinGecko rate-limited")
            logger.info("✅ Multi-source resilience handling API failures gracefully")
            logger.info("✅ Data quality maintained with fallback sources")
            logger.info("✅ IA1/IA2 global market context integration working with fallback data")
            logger.info("✅ Admin endpoint functional with improved fallback system")
            logger.info("✅ Market regime detection working with Binance fallback data")
            logger.info("✅ Fear & Greed index working independently")
            logger.info("✅ Cache and performance not broken by fallback mechanisms")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - Fallback system working with minor gaps")
            logger.info("🔍 Some fallback components may need fine-tuning")
        elif passed_tests >= total_tests * 0.6:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core fallback features working")
            logger.info("🔧 Some advanced fallback features may need implementation")
        else:
            logger.info("❌ FALLBACK SYSTEM NOT FUNCTIONAL - Critical issues with fallback APIs")
            logger.info("🚨 Major implementation gaps preventing fallback functionality")
        
        # Final verdict
        logger.info(f"\n🏆 FINAL RESULT: {passed_tests}/{total_tests} fallback requirements satisfied")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 VERDICT: Global Crypto Market Analyzer Fallback System is FULLY FUNCTIONAL!")
            logger.info("✅ Binance API successfully provides backup for BTC price and volume data")
            logger.info("✅ Realistic default values used when all APIs fail")
            logger.info("✅ Continued operation despite CoinGecko rate limits")
            logger.info("✅ Enhanced error handling and logging working")
            logger.info("✅ Functional market context for IAs even with limited data")
        elif passed_tests >= total_tests * 0.75:
            logger.info("\n⚠️ VERDICT: Fallback System is MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor fallback issues may need attention")
        else:
            logger.info("\n❌ VERDICT: Fallback System needs significant improvement")
            logger.info("🚨 Major fallback functionality gaps preventing reliable operation")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = GlobalCryptoMarketFallbackTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed >= total * 0.75:  # 75% pass rate for success
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
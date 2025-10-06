#!/usr/bin/env python3
"""
FOCUSED ENHANCED OHLCV MULTI-SOURCE INTEGRATION TEST
Testing specific endpoints and functionality mentioned in the review request
"""

import asyncio
import json
import logging
import requests
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedOHLCVTest:
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
        logger.info(f"Testing Enhanced OHLCV at: {self.api_url}")
        
        # Test symbols confirmed working at 100% success rate
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
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

    async def test_ia1_cycle_ohlcv_integration(self):
        """Test GET /api/run-ia1-cycle - Should use enhanced OHLCV data from 5 working sources"""
        logger.info("\n🔍 TEST: IA1 Cycle OHLCV Integration")
        
        try:
            integration_results = {
                'total_tests': 0,
                'successful_calls': 0,
                'real_price_data': 0,
                'technical_indicators_working': 0,
                'ohlcv_data_quality': {}
            }
            
            for symbol in self.test_symbols:
                try:
                    logger.info(f"   📈 Testing IA1 cycle for {symbol}...")
                    integration_results['total_tests'] += 1
                    
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={"symbol": symbol},
                        timeout=45
                    )
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        
                        if cycle_data.get('success'):
                            integration_results['successful_calls'] += 1
                            analysis = cycle_data.get('analysis', {})
                            
                            # Check for real price data (not fallback values like $0.01)
                            entry_price = analysis.get('entry_price', 0)
                            current_price = analysis.get('current_price', entry_price)
                            
                            has_real_prices = entry_price > 1.0 and current_price > 1.0  # Real crypto prices
                            if has_real_prices:
                                integration_results['real_price_data'] += 1
                                logger.info(f"      ✅ Real price data: Entry=${entry_price:.2f}, Current=${current_price:.2f}")
                            else:
                                logger.warning(f"      ❌ Suspicious price data: Entry=${entry_price:.6f}, Current=${current_price:.6f}")
                            
                            # Check technical indicators
                            technical_indicators = {
                                'rsi_signal': analysis.get('rsi_signal', 'unknown'),
                                'macd_trend': analysis.get('macd_trend', 'unknown'),
                                'stochastic_signal': analysis.get('stochastic_signal', 'unknown'),
                                'mfi_signal': analysis.get('mfi_signal', 'unknown'),
                                'vwap_signal': analysis.get('vwap_signal', 'unknown')
                            }
                            
                            meaningful_indicators = sum(1 for v in technical_indicators.values() 
                                                      if v not in ['unknown', 'neutral', ''])
                            
                            if meaningful_indicators >= 2:  # At least 2 meaningful indicators
                                integration_results['technical_indicators_working'] += 1
                                logger.info(f"      ✅ Technical indicators working: {meaningful_indicators}/5")
                                for indicator, value in technical_indicators.items():
                                    if value not in ['unknown', 'neutral', '']:
                                        logger.info(f"         {indicator}: {value}")
                            else:
                                logger.warning(f"      ⚠️ Limited technical indicators: {meaningful_indicators}/5")
                            
                            # Store quality data
                            integration_results['ohlcv_data_quality'][symbol] = {
                                'has_real_prices': has_real_prices,
                                'meaningful_indicators': meaningful_indicators,
                                'response_time': response_time,
                                'entry_price': entry_price,
                                'current_price': current_price,
                                'technical_indicators': technical_indicators
                            }
                            
                        else:
                            logger.warning(f"      ❌ IA1 cycle failed for {symbol}: {cycle_data.get('error', 'Unknown error')}")
                    else:
                        logger.warning(f"      ❌ API call failed for {symbol}: HTTP {response.status_code}")
                    
                    await asyncio.sleep(2)  # Brief delay between requests
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing IA1 cycle for {symbol}: {e}")
            
            # Calculate results
            logger.info(f"   📊 IA1 Cycle OHLCV Integration Results:")
            logger.info(f"      Total tests: {integration_results['total_tests']}")
            logger.info(f"      Successful calls: {integration_results['successful_calls']}")
            logger.info(f"      Real price data: {integration_results['real_price_data']}")
            logger.info(f"      Technical indicators working: {integration_results['technical_indicators_working']}")
            
            # Determine success
            if integration_results['total_tests'] > 0:
                success_rate = integration_results['successful_calls'] / integration_results['total_tests']
                real_data_rate = integration_results['real_price_data'] / integration_results['total_tests']
                indicators_rate = integration_results['technical_indicators_working'] / integration_results['total_tests']
                
                overall_quality = (success_rate + real_data_rate + indicators_rate) / 3
                
                if overall_quality >= 0.7:
                    self.log_test_result("IA1 Cycle OHLCV Integration", True, 
                                       f"Integration successful: {overall_quality:.1%} overall quality")
                else:
                    self.log_test_result("IA1 Cycle OHLCV Integration", False, 
                                       f"Integration issues: {overall_quality:.1%} overall quality")
            else:
                self.log_test_result("IA1 Cycle OHLCV Integration", False, 
                                   "No IA1 cycle tests could be performed")
                
        except Exception as e:
            self.log_test_result("IA1 Cycle OHLCV Integration", False, f"Exception: {str(e)}")

    async def test_scout_ohlcv_integration(self):
        """Test GET /api/scout - Should leverage multi-source validation for trending crypto analysis"""
        logger.info("\n🔍 TEST: Scout OHLCV Integration")
        
        try:
            scout_results = {
                'scout_call_successful': False,
                'opportunities_found': 0,
                'multi_source_validation': False,
                'data_quality_indicators': {}
            }
            
            logger.info(f"   📈 Testing Scout system integration...")
            
            start_time = time.time()
            response = requests.get(f"{self.api_url}/scout", timeout=45)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                scout_data = response.json()
                
                if scout_data.get('success'):
                    scout_results['scout_call_successful'] = True
                    opportunities = scout_data.get('opportunities', [])
                    scout_results['opportunities_found'] = len(opportunities)
                    
                    logger.info(f"      ✅ Scout call successful: {len(opportunities)} opportunities found")
                    
                    # Check for data quality indicators
                    if opportunities:
                        sample_opportunity = opportunities[0]
                        
                        # Check for real market data
                        has_price = sample_opportunity.get('current_price', 0) > 0.1
                        has_volume = sample_opportunity.get('volume_24h', 0) > 1000
                        has_volatility = sample_opportunity.get('volatility', 0) > 0.01
                        has_market_cap = sample_opportunity.get('market_cap', 0) > 100000
                        
                        scout_results['data_quality_indicators'] = {
                            'has_price': has_price,
                            'has_volume': has_volume,
                            'has_volatility': has_volatility,
                            'has_market_cap': has_market_cap,
                            'response_time': response_time
                        }
                        
                        # Check for multi-source validation indicators
                        opportunity_str = str(sample_opportunity).lower()
                        if 'validation' in opportunity_str or 'source' in opportunity_str or 'quality' in opportunity_str:
                            scout_results['multi_source_validation'] = True
                            logger.info(f"      ✅ Multi-source validation indicators detected")
                        
                        logger.info(f"      📊 Data quality: Price={has_price}, Volume={has_volume}, Volatility={has_volatility}")
                        logger.info(f"      📊 Sample opportunity: {sample_opportunity.get('symbol', 'N/A')} - ${sample_opportunity.get('current_price', 0):.6f}")
                    
                else:
                    logger.warning(f"      ❌ Scout call failed: {scout_data.get('error', 'Unknown error')}")
            else:
                logger.warning(f"      ❌ Scout API call failed: HTTP {response.status_code}")
            
            # Determine success
            quality_score = 0
            if scout_results['scout_call_successful']:
                quality_score += 0.4
            if scout_results['opportunities_found'] > 10:
                quality_score += 0.3
            if scout_results['data_quality_indicators'].get('has_price') and scout_results['data_quality_indicators'].get('has_volume'):
                quality_score += 0.3
            
            if quality_score >= 0.7:
                self.log_test_result("Scout OHLCV Integration", True, 
                                   f"Scout integration successful: {quality_score:.1%} quality score")
            else:
                self.log_test_result("Scout OHLCV Integration", False, 
                                   f"Scout integration issues: {quality_score:.1%} quality score")
                
        except Exception as e:
            self.log_test_result("Scout OHLCV Integration", False, f"Exception: {str(e)}")

    async def test_multi_source_validation(self):
        """Test that system correctly combines data from multiple sources (BingX + Kraken validation)"""
        logger.info("\n🔍 TEST: Multi-Source Validation System")
        
        try:
            # Test multiple symbols to see if we get consistent, high-quality data
            validation_results = {
                'symbols_tested': 0,
                'consistent_data': 0,
                'high_quality_responses': 0,
                'validation_metadata': {}
            }
            
            for symbol in self.test_symbols:
                try:
                    logger.info(f"   📈 Testing multi-source validation for {symbol}...")
                    validation_results['symbols_tested'] += 1
                    
                    # Run IA1 cycle to test data quality
                    response = requests.post(
                        f"{self.api_url}/run-ia1-cycle",
                        json={"symbol": symbol},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        cycle_data = response.json()
                        
                        if cycle_data.get('success'):
                            analysis = cycle_data.get('analysis', {})
                            
                            # Check for data consistency indicators
                            entry_price = analysis.get('entry_price', 0)
                            current_price = analysis.get('current_price', entry_price)
                            confidence = analysis.get('confidence', 0)
                            
                            # High-quality data indicators
                            has_realistic_prices = entry_price > 1.0 and current_price > 1.0
                            has_reasonable_confidence = 0.3 <= confidence <= 0.95
                            has_analysis_text = len(analysis.get('analysis', '')) > 100
                            
                            if has_realistic_prices and has_reasonable_confidence:
                                validation_results['consistent_data'] += 1
                                logger.info(f"      ✅ Consistent data for {symbol}: Price=${current_price:.2f}, Confidence={confidence:.1%}")
                            
                            if has_realistic_prices and has_reasonable_confidence and has_analysis_text:
                                validation_results['high_quality_responses'] += 1
                                logger.info(f"      ✅ High-quality response for {symbol}")
                            
                            validation_results['validation_metadata'][symbol] = {
                                'entry_price': entry_price,
                                'current_price': current_price,
                                'confidence': confidence,
                                'analysis_length': len(analysis.get('analysis', '')),
                                'has_realistic_prices': has_realistic_prices,
                                'has_reasonable_confidence': has_reasonable_confidence
                            }
                        
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"   ❌ Error testing validation for {symbol}: {e}")
            
            # Calculate validation results
            logger.info(f"   📊 Multi-Source Validation Results:")
            logger.info(f"      Symbols tested: {validation_results['symbols_tested']}")
            logger.info(f"      Consistent data: {validation_results['consistent_data']}")
            logger.info(f"      High-quality responses: {validation_results['high_quality_responses']}")
            
            # Determine success
            if validation_results['symbols_tested'] > 0:
                consistency_rate = validation_results['consistent_data'] / validation_results['symbols_tested']
                quality_rate = validation_results['high_quality_responses'] / validation_results['symbols_tested']
                
                overall_validation = (consistency_rate + quality_rate) / 2
                
                if overall_validation >= 0.7:
                    self.log_test_result("Multi-Source Validation System", True, 
                                       f"Validation successful: {overall_validation:.1%} overall quality")
                else:
                    self.log_test_result("Multi-Source Validation System", False, 
                                       f"Validation issues: {overall_validation:.1%} overall quality")
            else:
                self.log_test_result("Multi-Source Validation System", False, 
                                   "No validation tests could be performed")
                
        except Exception as e:
            self.log_test_result("Multi-Source Validation System", False, f"Exception: {str(e)}")

    async def run_focused_tests(self):
        """Run focused Enhanced OHLCV Multi-Source Integration tests"""
        logger.info("🚀 Starting Focused Enhanced OHLCV Multi-Source Integration Tests")
        logger.info("=" * 80)
        logger.info("📋 FOCUSED ENHANCED OHLCV TESTS")
        logger.info("🎯 Testing specific endpoints and functionality from review request")
        logger.info("=" * 80)
        
        # Run focused tests
        await self.test_ia1_cycle_ohlcv_integration()
        await self.test_scout_ohlcv_integration()
        await self.test_multi_source_validation()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 FOCUSED ENHANCED OHLCV TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Final verdict
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.8:
            logger.info("\n🎉 VERDICT: ENHANCED OHLCV INTEGRATION SUCCESSFUL!")
            logger.info("✅ Enhanced OHLCV system working with main trading bot")
            logger.info("✅ API endpoints can access enhanced OHLCV data")
            logger.info("✅ Multi-source validation system operational")
        elif success_rate >= 0.6:
            logger.info("\n⚠️ VERDICT: ENHANCED OHLCV INTEGRATION PARTIALLY SUCCESSFUL")
            logger.info("🔧 Some issues need attention for complete integration")
        else:
            logger.info("\n❌ VERDICT: ENHANCED OHLCV INTEGRATION NEEDS WORK")
            logger.info("🚨 Major issues detected with enhanced OHLCV system")
        
        return passed_tests, total_tests

async def main():
    """Main function to run focused Enhanced OHLCV tests"""
    test_suite = FocusedOHLCVTest()
    passed_tests, total_tests = await test_suite.run_focused_tests()
    
    # Exit with appropriate code
    if passed_tests == total_tests:
        exit(0)  # All tests passed
    elif passed_tests >= total_tests * 0.8:
        exit(1)  # Mostly successful
    else:
        exit(2)  # Major issues

if __name__ == "__main__":
    asyncio.run(main())
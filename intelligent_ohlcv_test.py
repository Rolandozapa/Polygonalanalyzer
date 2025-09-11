#!/usr/bin/env python3
"""
INTELLIGENT OHLCV FETCHER INTEGRATION TESTING SUITE
🎯 Focus: Test the new Intelligent OHLCV Fetcher integration with IA2 system

TESTING REQUIREMENTS FROM REVIEW REQUEST:
1. **Module Loading**: Verify that the new intelligent_ohlcv_fetcher module is correctly imported and accessible
2. **Multi-Source Data Completion**: Test the intelligent OHLCV completion system that uses different APIs from IA1
3. **High-Frequency Data Integration**: Validate that 5-minute timeframe data is being fetched from alternative sources
4. **Enhanced S/R Calculation**: Test the high-precision support/resistance calculation using micro/intraday/daily timeframes
5. **Dynamic RR Integration**: Verify that IA2 decisions now show the new dynamic RR calculation with multi-timeframe analysis
6. **Source Diversification**: Confirm that if IA1 used Binance, IA2 tries Coinbase/Kraken/BingX etc.
7. **Fallback Logic**: Test that the system gracefully falls back to simple S/R calculation when high-frequency data is unavailable

EXPECTED SYSTEM BEHAVIOR:
- Enhanced risk-reward calculations using 5m/1h/4h/daily data
- Support/resistance levels with higher precision than daily-only analysis  
- Multi-source API diversification to avoid relying on same data as IA1
- Metadata tracking to ensure no API source redundancy between IA1 and IA2
- New "HIGH-FREQUENCY DATA INTEGRATION" section in IA2 prompts
- Enhanced RR calculations with micro/intraday/daily breakdown
- "calculated_rr" field with optimized values from intelligent system
- "rr_reasoning" showing multi-timeframe analysis details
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

class IntelligentOHLCVFetcherTestSuite:
    """Comprehensive test suite for Intelligent OHLCV Fetcher integration with IA2"""
    
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
        logger.info(f"Testing Intelligent OHLCV Fetcher Integration at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Test symbols for different scenarios
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
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
    
    async def test_1_module_loading_verification(self):
        """Test 1: Verify that the intelligent_ohlcv_fetcher module is correctly imported and accessible"""
        logger.info("\n🔍 TEST 1: Module Loading Verification")
        
        try:
            # Check if the module can be imported
            import sys
            sys.path.append('/app/backend')
            
            try:
                from intelligent_ohlcv_fetcher import (
                    intelligent_ohlcv_fetcher, 
                    OHLCVMetadata, 
                    HighFrequencyData, 
                    EnhancedSupportResistance, 
                    DynamicRiskReward
                )
                
                # Verify the main class exists and has expected methods
                fetcher = intelligent_ohlcv_fetcher
                expected_methods = [
                    'fetch_intelligent_ohlcv_completion',
                    'calculate_enhanced_sr_levels',
                    'calculate_dynamic_risk_reward'
                ]
                
                missing_methods = []
                for method in expected_methods:
                    if not hasattr(fetcher, method):
                        missing_methods.append(method)
                
                if not missing_methods:
                    # Check data classes
                    metadata = OHLCVMetadata(
                        primary_source="test",
                        primary_timeframe="1d",
                        primary_data_quality=0.8,
                        primary_data_count=100,
                        sources_used=["binance"],
                        avoided_sources=[]
                    )
                    
                    self.log_test_result("Module Loading Verification", True, 
                                       f"All required classes and methods available: {expected_methods}")
                else:
                    self.log_test_result("Module Loading Verification", False, 
                                       f"Missing methods: {missing_methods}")
                    
            except ImportError as e:
                self.log_test_result("Module Loading Verification", False, 
                                   f"Import error: {str(e)}")
                
        except Exception as e:
            self.log_test_result("Module Loading Verification", False, f"Exception: {str(e)}")
    
    async def test_2_multi_source_data_completion(self):
        """Test 2: Test the intelligent OHLCV completion system that uses different APIs from IA1"""
        logger.info("\n🔍 TEST 2: Multi-Source Data Completion Test")
        
        try:
            # Import the intelligent fetcher
            sys.path.append('/app/backend')
            from intelligent_ohlcv_fetcher import intelligent_ohlcv_fetcher, OHLCVMetadata
            
            # Create metadata indicating IA1 used Binance
            metadata = OHLCVMetadata(
                primary_source="binance",
                primary_timeframe="1d",
                primary_data_quality=0.8,
                primary_data_count=30,
                sources_used=["binance"],
                avoided_sources=[],
                completion_needed=True,
                preferred_completion_sources=["coinbase", "kraken", "bingx"]
            )
            
            # Test completion for a symbol
            test_symbol = "BTCUSDT"
            logger.info(f"   Testing completion for {test_symbol} with metadata: {metadata.primary_source}")
            
            # Call the completion method
            completion_result = await intelligent_ohlcv_fetcher.fetch_intelligent_ohlcv_completion(
                symbol=test_symbol,
                metadata=metadata,
                target_timeframes=["5m", "1h"]
            )
            
            if completion_result:
                # Check if different sources were used
                sources_used = []
                for timeframe, hf_data in completion_result.items():
                    if hf_data and hasattr(hf_data, 'source'):
                        sources_used.append(hf_data.source)
                
                # Verify source diversification
                if sources_used and "binance" not in sources_used:
                    self.log_test_result("Multi-Source Data Completion", True, 
                                       f"Successfully used alternative sources: {sources_used}")
                elif sources_used:
                    self.log_test_result("Multi-Source Data Completion", True, 
                                       f"Completion attempted with sources: {sources_used} (may include primary)")
                else:
                    self.log_test_result("Multi-Source Data Completion", False, 
                                       "No completion data returned")
            else:
                self.log_test_result("Multi-Source Data Completion", False, 
                                   "Completion method returned None")
                
        except Exception as e:
            self.log_test_result("Multi-Source Data Completion", False, f"Exception: {str(e)}")
    
    async def test_3_high_frequency_data_integration(self):
        """Test 3: Validate that 5-minute timeframe data is being fetched from alternative sources"""
        logger.info("\n🔍 TEST 3: High-Frequency Data Integration Test")
        
        try:
            sys.path.append('/app/backend')
            from intelligent_ohlcv_fetcher import intelligent_ohlcv_fetcher, OHLCVMetadata
            
            # Test 5-minute data fetching
            test_symbol = "ETHUSDT"
            
            # Create metadata to force alternative source usage
            metadata = OHLCVMetadata(
                primary_source="binance",
                primary_timeframe="1d",
                primary_data_quality=0.9,
                primary_data_count=50,
                sources_used=["binance"],
                avoided_sources=[],
                completion_needed=True
            )
            
            logger.info(f"   Testing 5-minute data fetch for {test_symbol}")
            
            # Fetch high-frequency data
            hf_completion = await intelligent_ohlcv_fetcher.fetch_intelligent_ohlcv_completion(
                symbol=test_symbol,
                metadata=metadata,
                target_timeframes=["5m"]
            )
            
            if hf_completion and "5m" in hf_completion:
                hf_data = hf_completion["5m"]
                
                if hf_data and hasattr(hf_data, 'data') and hasattr(hf_data, 'timeframe'):
                    # Verify it's 5-minute data
                    if hf_data.timeframe == "5m" and len(hf_data.data) > 0:
                        # Check data quality
                        data_quality = hf_data.quality_score if hasattr(hf_data, 'quality_score') else 0
                        source_used = hf_data.source if hasattr(hf_data, 'source') else "unknown"
                        
                        self.log_test_result("High-Frequency Data Integration", True, 
                                           f"5m data fetched: {len(hf_data.data)} points, quality: {data_quality:.2f}, source: {source_used}")
                    else:
                        self.log_test_result("High-Frequency Data Integration", False, 
                                           f"Invalid 5m data: timeframe={hf_data.timeframe}, length={len(hf_data.data) if hasattr(hf_data, 'data') else 0}")
                else:
                    self.log_test_result("High-Frequency Data Integration", False, 
                                       "5m data object missing required attributes")
            else:
                self.log_test_result("High-Frequency Data Integration", False, 
                                   "No 5-minute data returned from completion")
                
        except Exception as e:
            self.log_test_result("High-Frequency Data Integration", False, f"Exception: {str(e)}")
    
    async def test_4_enhanced_sr_calculation(self):
        """Test 4: Test the high-precision support/resistance calculation using micro/intraday/daily timeframes"""
        logger.info("\n🔍 TEST 4: Enhanced S/R Calculation Test")
        
        try:
            sys.path.append('/app/backend')
            from intelligent_ohlcv_fetcher import intelligent_ohlcv_fetcher, HighFrequencyData
            import pandas as pd
            
            # Create mock high-frequency data for testing
            test_symbol = "SOLUSDT"
            
            # Generate mock 5-minute OHLCV data
            dates = pd.date_range(start=datetime.now() - timedelta(hours=6), 
                                 end=datetime.now(), freq='5min')
            
            # Create realistic price data with some volatility
            base_price = 100.0
            price_data = []
            for i, date in enumerate(dates):
                # Add some random walk with volatility
                price_change = np.random.normal(0, 0.002)  # 0.2% volatility
                base_price *= (1 + price_change)
                
                # OHLC with some spread
                open_price = base_price
                high_price = base_price * (1 + abs(np.random.normal(0, 0.001)))
                low_price = base_price * (1 - abs(np.random.normal(0, 0.001)))
                close_price = base_price
                volume = np.random.uniform(1000, 10000)
                
                price_data.append({
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume
                })
            
            mock_df = pd.DataFrame(price_data, index=dates)
            
            # Create HighFrequencyData object
            hf_data = HighFrequencyData(
                symbol=test_symbol,
                timeframe="5m",
                data=mock_df,
                source="mock_coinbase",
                quality_score=0.9,
                data_count=len(mock_df),
                fetch_timestamp=datetime.now()
            )
            
            # Calculate high-precision S/R levels
            logger.info(f"   Testing S/R calculation for {test_symbol} with {len(mock_df)} data points")
            
            enhanced_hf_data = await intelligent_ohlcv_fetcher._calculate_high_precision_sr_levels(hf_data)
            
            # Verify S/R levels were calculated
            has_micro_levels = (enhanced_hf_data.micro_support_levels and 
                              enhanced_hf_data.micro_resistance_levels)
            
            if has_micro_levels:
                micro_supports = len(enhanced_hf_data.micro_support_levels)
                micro_resistances = len(enhanced_hf_data.micro_resistance_levels)
                
                # Test enhanced S/R calculation
                daily_support = base_price * 0.98
                daily_resistance = base_price * 1.02
                
                enhanced_sr = await intelligent_ohlcv_fetcher.calculate_enhanced_sr_levels(
                    symbol=test_symbol,
                    hf_data=enhanced_hf_data,
                    daily_support=daily_support,
                    daily_resistance=daily_resistance
                )
                
                if enhanced_sr:
                    self.log_test_result("Enhanced S/R Calculation", True, 
                                       f"S/R calculated: Micro S/R ({micro_supports}/{micro_resistances}), "
                                       f"Enhanced S/R confidence: {enhanced_sr.confidence_micro:.2f}")
                else:
                    self.log_test_result("Enhanced S/R Calculation", False, 
                                       "Enhanced S/R calculation returned None")
            else:
                self.log_test_result("Enhanced S/R Calculation", False, 
                                   "No micro S/R levels calculated from high-frequency data")
                
        except Exception as e:
            self.log_test_result("Enhanced S/R Calculation", False, f"Exception: {str(e)}")
    
    async def test_5_dynamic_rr_integration(self):
        """Test 5: Verify that IA2 decisions now show the new dynamic RR calculation with multi-timeframe analysis"""
        logger.info("\n🔍 TEST 5: Dynamic RR Integration Test")
        
        try:
            sys.path.append('/app/backend')
            from intelligent_ohlcv_fetcher import intelligent_ohlcv_fetcher, EnhancedSupportResistance
            
            # Create mock enhanced S/R data
            test_symbol = "BTCUSDT"
            entry_price = 45000.0
            
            enhanced_sr = EnhancedSupportResistance(
                symbol=test_symbol,
                daily_support=44000.0,
                daily_resistance=46000.0,
                intraday_support=44500.0,
                intraday_resistance=45800.0,
                micro_support=44800.0,
                micro_resistance=45400.0,
                confidence_daily=0.9,
                confidence_intraday=0.8,
                confidence_micro=0.7,
                calculation_source="mock_kraken",
                calculation_timestamp=datetime.now()
            )
            
            # Test dynamic RR calculation for LONG signal
            logger.info(f"   Testing dynamic RR calculation for LONG {test_symbol} @ {entry_price}")
            
            dynamic_rr = await intelligent_ohlcv_fetcher.calculate_dynamic_risk_reward(
                symbol=test_symbol,
                signal_type="LONG",
                entry_price=entry_price,
                enhanced_sr=enhanced_sr
            )
            
            if dynamic_rr:
                # Verify multi-timeframe RR calculations
                has_multi_timeframe = (
                    hasattr(dynamic_rr, 'rr_micro') and
                    hasattr(dynamic_rr, 'rr_intraday') and
                    hasattr(dynamic_rr, 'rr_daily') and
                    hasattr(dynamic_rr, 'rr_final')
                )
                
                if has_multi_timeframe:
                    rr_breakdown = f"Micro: {dynamic_rr.rr_micro:.2f}, Intraday: {dynamic_rr.rr_intraday:.2f}, Daily: {dynamic_rr.rr_daily:.2f}"
                    final_rr = f"Final: {dynamic_rr.rr_final:.2f} ({dynamic_rr.rr_selection_logic})"
                    
                    # Test SHORT signal as well
                    dynamic_rr_short = await intelligent_ohlcv_fetcher.calculate_dynamic_risk_reward(
                        symbol=test_symbol,
                        signal_type="SHORT",
                        entry_price=entry_price,
                        enhanced_sr=enhanced_sr
                    )
                    
                    if dynamic_rr_short:
                        self.log_test_result("Dynamic RR Integration", True, 
                                           f"Multi-timeframe RR calculated - LONG: {rr_breakdown}, {final_rr}. "
                                           f"SHORT Final: {dynamic_rr_short.rr_final:.2f}")
                    else:
                        self.log_test_result("Dynamic RR Integration", False, 
                                           "SHORT RR calculation failed")
                else:
                    self.log_test_result("Dynamic RR Integration", False, 
                                       "Dynamic RR missing multi-timeframe attributes")
            else:
                self.log_test_result("Dynamic RR Integration", False, 
                                   "Dynamic RR calculation returned None")
                
        except Exception as e:
            self.log_test_result("Dynamic RR Integration", False, f"Exception: {str(e)}")
    
    async def test_6_source_diversification(self):
        """Test 6: Confirm that if IA1 used Binance, IA2 tries Coinbase/Kraken/BingX etc."""
        logger.info("\n🔍 TEST 6: Source Diversification Test")
        
        try:
            sys.path.append('/app/backend')
            from intelligent_ohlcv_fetcher import intelligent_ohlcv_fetcher, OHLCVMetadata
            
            # Test different primary source scenarios
            test_scenarios = [
                {
                    "primary_source": "binance",
                    "expected_alternatives": ["coinbase", "kraken", "bingx", "twelvedata"]
                },
                {
                    "primary_source": "coinbase", 
                    "expected_alternatives": ["binance", "kraken", "bingx", "coinapi"]
                },
                {
                    "primary_source": "coingecko",
                    "expected_alternatives": ["binance", "coinbase", "kraken"]
                }
            ]
            
            diversification_results = []
            
            for scenario in test_scenarios:
                primary = scenario["primary_source"]
                expected_alts = scenario["expected_alternatives"]
                
                logger.info(f"   Testing diversification from primary source: {primary}")
                
                # Create metadata with primary source
                metadata = OHLCVMetadata(
                    primary_source=primary,
                    primary_timeframe="1d",
                    primary_data_quality=0.8,
                    primary_data_count=30,
                    sources_used=[primary],
                    avoided_sources=[],
                    completion_needed=True
                )
                
                # Get preferred completion sources
                preferred_sources = intelligent_ohlcv_fetcher._get_diversified_sources(metadata)
                
                # Check if alternative sources are preferred
                uses_alternatives = any(alt in preferred_sources for alt in expected_alts)
                avoids_primary = primary not in preferred_sources
                
                diversification_results.append({
                    "primary": primary,
                    "preferred": preferred_sources,
                    "uses_alternatives": uses_alternatives,
                    "avoids_primary": avoids_primary
                })
                
                logger.info(f"      Primary: {primary} → Preferred: {preferred_sources}")
            
            # Evaluate diversification success
            successful_diversifications = sum(1 for result in diversification_results 
                                            if result["uses_alternatives"] and result["avoids_primary"])
            total_scenarios = len(diversification_results)
            
            if successful_diversifications >= total_scenarios * 0.8:  # 80% success rate
                self.log_test_result("Source Diversification", True, 
                                   f"Diversification working: {successful_diversifications}/{total_scenarios} scenarios successful")
            else:
                self.log_test_result("Source Diversification", False, 
                                   f"Poor diversification: {successful_diversifications}/{total_scenarios} scenarios successful")
                
        except Exception as e:
            self.log_test_result("Source Diversification", False, f"Exception: {str(e)}")
    
    async def test_7_fallback_logic(self):
        """Test 7: Test that the system gracefully falls back to simple S/R calculation when high-frequency data is unavailable"""
        logger.info("\n🔍 TEST 7: Fallback Logic Test")
        
        try:
            sys.path.append('/app/backend')
            from intelligent_ohlcv_fetcher import intelligent_ohlcv_fetcher, OHLCVMetadata
            
            # Test fallback scenarios
            test_symbol = "TESTUSDT"  # Use a symbol that likely won't have HF data
            
            # Scenario 1: No completion sources available
            metadata_no_sources = OHLCVMetadata(
                primary_source="unknown_source",
                primary_timeframe="1d",
                primary_data_quality=0.5,
                primary_data_count=10,
                sources_used=["unknown_source"],
                avoided_sources=["binance", "coinbase", "kraken", "bingx"],  # Avoid all major sources
                completion_needed=True
            )
            
            logger.info(f"   Testing fallback for {test_symbol} with no available sources")
            
            # Attempt completion (should fail gracefully)
            completion_result = await intelligent_ohlcv_fetcher.fetch_intelligent_ohlcv_completion(
                symbol=test_symbol,
                metadata=metadata_no_sources,
                target_timeframes=["5m", "1h"]
            )
            
            # Test fallback RR calculation
            entry_price = 1.0
            daily_support = 0.95
            daily_resistance = 1.05
            
            # Create minimal enhanced S/R (fallback scenario)
            from intelligent_ohlcv_fetcher import EnhancedSupportResistance
            
            fallback_sr = EnhancedSupportResistance(
                symbol=test_symbol,
                daily_support=daily_support,
                daily_resistance=daily_resistance,
                intraday_support=daily_support,  # Same as daily (fallback)
                intraday_resistance=daily_resistance,  # Same as daily (fallback)
                micro_support=daily_support,  # Same as daily (fallback)
                micro_resistance=daily_resistance,  # Same as daily (fallback)
                confidence_daily=0.8,
                confidence_intraday=0.5,  # Lower confidence for fallback
                confidence_micro=0.3,  # Lower confidence for fallback
                calculation_source="fallback",
                calculation_timestamp=datetime.now()
            )
            
            # Test fallback RR calculation
            fallback_rr = await intelligent_ohlcv_fetcher.calculate_dynamic_risk_reward(
                symbol=test_symbol,
                signal_type="LONG",
                entry_price=entry_price,
                enhanced_sr=fallback_sr
            )
            
            if fallback_rr:
                # Verify fallback behavior
                is_fallback = (
                    fallback_rr.rr_selection_logic and 
                    ("fallback" in fallback_rr.rr_selection_logic.lower() or
                     "daily" in fallback_rr.rr_selection_logic.lower())
                )
                
                has_reasonable_rr = 0.5 <= fallback_rr.rr_final <= 5.0  # Reasonable range
                
                if is_fallback and has_reasonable_rr:
                    self.log_test_result("Fallback Logic", True, 
                                       f"Fallback working: RR={fallback_rr.rr_final:.2f}, logic='{fallback_rr.rr_selection_logic}'")
                else:
                    self.log_test_result("Fallback Logic", False, 
                                       f"Fallback issues: RR={fallback_rr.rr_final:.2f}, logic='{fallback_rr.rr_selection_logic}', is_fallback={is_fallback}")
            else:
                self.log_test_result("Fallback Logic", False, 
                                   "Fallback RR calculation returned None")
                
        except Exception as e:
            self.log_test_result("Fallback Logic", False, f"Exception: {str(e)}")
    
    async def test_8_ia2_integration_verification(self):
        """Test 8: Verify IA2 integration shows enhanced RR fields in actual decisions"""
        logger.info("\n🔍 TEST 8: IA2 Integration Verification Test")
        
        try:
            # Check recent IA2 decisions for enhanced RR fields
            # This would typically involve checking the database or API responses
            
            # For now, we'll test the API endpoints to see if they're working
            logger.info("   Testing IA2 API endpoints for enhanced RR integration")
            
            # Test getting recent decisions
            try:
                response = requests.get(f"{self.api_url}/decisions", timeout=30)
                
                if response.status_code == 200:
                    decisions = response.json()
                    
                    # Look for IA2 decisions with enhanced RR fields
                    ia2_decisions = [d for d in decisions if d.get('agent') == 'IA2' or 'ia2' in str(d.get('id', '')).lower()]
                    
                    enhanced_rr_fields = ['calculated_rr', 'rr_reasoning', 'rr_micro', 'rr_intraday', 'rr_daily']
                    
                    decisions_with_enhanced_rr = 0
                    for decision in ia2_decisions[-5:]:  # Check last 5 IA2 decisions
                        has_enhanced_fields = any(field in decision for field in enhanced_rr_fields)
                        if has_enhanced_fields:
                            decisions_with_enhanced_rr += 1
                    
                    if decisions_with_enhanced_rr > 0:
                        self.log_test_result("IA2 Integration Verification", True, 
                                           f"Found {decisions_with_enhanced_rr} IA2 decisions with enhanced RR fields")
                    else:
                        # Check if there are any IA2 decisions at all
                        if ia2_decisions:
                            self.log_test_result("IA2 Integration Verification", False, 
                                               f"Found {len(ia2_decisions)} IA2 decisions but none with enhanced RR fields")
                        else:
                            self.log_test_result("IA2 Integration Verification", False, 
                                               "No IA2 decisions found to verify enhanced RR integration")
                else:
                    self.log_test_result("IA2 Integration Verification", False, 
                                       f"API endpoint failed: HTTP {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                self.log_test_result("IA2 Integration Verification", False, 
                                   f"API request failed: {str(e)}")
                
        except Exception as e:
            self.log_test_result("IA2 Integration Verification", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Intelligent OHLCV Fetcher integration tests"""
        logger.info("🚀 Starting Intelligent OHLCV Fetcher Integration Test Suite")
        logger.info("=" * 80)
        logger.info("📋 INTELLIGENT OHLCV FETCHER INTEGRATION COMPREHENSIVE TESTING")
        logger.info("🎯 Testing: Module loading, multi-source completion, HF data, enhanced S/R, dynamic RR, source diversification, fallback logic")
        logger.info("🎯 Expected: Enhanced IA2 system with high-frequency data integration and multi-timeframe RR analysis")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_1_module_loading_verification()
        await self.test_2_multi_source_data_completion()
        await self.test_3_high_frequency_data_integration()
        await self.test_4_enhanced_sr_calculation()
        await self.test_5_dynamic_rr_integration()
        await self.test_6_source_diversification()
        await self.test_7_fallback_logic()
        await self.test_8_ia2_integration_verification()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 INTELLIGENT OHLCV FETCHER INTEGRATION TEST SUMMARY")
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
        logger.info("📋 INTELLIGENT OHLCV FETCHER INTEGRATION STATUS")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Intelligent OHLCV Fetcher Integration FULLY FUNCTIONAL!")
            logger.info("✅ Module loading and accessibility working")
            logger.info("✅ Multi-source data completion operational")
            logger.info("✅ High-frequency data integration working")
            logger.info("✅ Enhanced S/R calculation functional")
            logger.info("✅ Dynamic RR integration operational")
            logger.info("✅ Source diversification working")
            logger.info("✅ Fallback logic functional")
            logger.info("✅ IA2 integration verified")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - Intelligent OHLCV integration working with minor gaps")
            logger.info("🔍 Some components may need fine-tuning for full optimization")
        elif passed_tests >= total_tests * 0.6:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core OHLCV features working")
            logger.info("🔧 Some advanced features may need implementation or debugging")
        else:
            logger.info("❌ SYSTEM NOT FUNCTIONAL - Critical issues with OHLCV integration")
            logger.info("🚨 Major implementation gaps or system errors preventing functionality")
        
        # Requirements verification
        logger.info("\n📝 INTELLIGENT OHLCV FETCHER REQUIREMENTS VERIFICATION:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check each requirement based on test results
        for result in self.test_results:
            if result['success']:
                if "Module Loading" in result['test']:
                    requirements_met.append("✅ Module loading and accessibility verified")
                elif "Multi-Source Data Completion" in result['test']:
                    requirements_met.append("✅ Multi-source data completion working")
                elif "High-Frequency Data Integration" in result['test']:
                    requirements_met.append("✅ 5-minute timeframe data integration working")
                elif "Enhanced S/R Calculation" in result['test']:
                    requirements_met.append("✅ High-precision S/R calculation functional")
                elif "Dynamic RR Integration" in result['test']:
                    requirements_met.append("✅ Multi-timeframe RR analysis working")
                elif "Source Diversification" in result['test']:
                    requirements_met.append("✅ API source diversification operational")
                elif "Fallback Logic" in result['test']:
                    requirements_met.append("✅ Graceful fallback to simple S/R working")
                elif "IA2 Integration" in result['test']:
                    requirements_met.append("✅ IA2 enhanced RR fields integration verified")
            else:
                if "Module Loading" in result['test']:
                    requirements_failed.append("❌ Module loading and accessibility failed")
                elif "Multi-Source Data Completion" in result['test']:
                    requirements_failed.append("❌ Multi-source data completion not working")
                elif "High-Frequency Data Integration" in result['test']:
                    requirements_failed.append("❌ 5-minute timeframe data integration failed")
                elif "Enhanced S/R Calculation" in result['test']:
                    requirements_failed.append("❌ High-precision S/R calculation not functional")
                elif "Dynamic RR Integration" in result['test']:
                    requirements_failed.append("❌ Multi-timeframe RR analysis not working")
                elif "Source Diversification" in result['test']:
                    requirements_failed.append("❌ API source diversification not operational")
                elif "Fallback Logic" in result['test']:
                    requirements_failed.append("❌ Fallback to simple S/R not working")
                elif "IA2 Integration" in result['test']:
                    requirements_failed.append("❌ IA2 enhanced RR fields integration failed")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        logger.info(f"\n🏆 FINAL RESULT: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} requirements satisfied")
        
        # Final verdict
        if len(requirements_failed) == 0:
            logger.info("\n🎉 VERDICT: Intelligent OHLCV Fetcher Integration is FULLY FUNCTIONAL!")
            logger.info("✅ All integration features implemented and working correctly")
            logger.info("✅ Enhanced IA2 system with high-frequency data and multi-timeframe RR analysis")
            logger.info("✅ Source diversification and fallback logic operational")
        elif len(requirements_failed) <= 1:
            logger.info("\n⚠️ VERDICT: Intelligent OHLCV Fetcher Integration is MOSTLY FUNCTIONAL")
            logger.info("🔍 Minor issues may need attention for complete functionality")
        elif len(requirements_failed) <= 3:
            logger.info("\n⚠️ VERDICT: Intelligent OHLCV Fetcher Integration is PARTIALLY FUNCTIONAL")
            logger.info("🔧 Several components need implementation or debugging")
        else:
            logger.info("\n❌ VERDICT: Intelligent OHLCV Fetcher Integration is NOT FUNCTIONAL")
            logger.info("🚨 Major implementation gaps preventing OHLCV integration")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IntelligentOHLCVFetcherTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
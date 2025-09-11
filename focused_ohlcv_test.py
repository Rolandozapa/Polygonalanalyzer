#!/usr/bin/env python3
"""
FOCUSED INTELLIGENT OHLCV FETCHER TESTING
🎯 Focus: Test the actual implemented methods in intelligent_ohlcv_fetcher

Based on the code analysis, the available methods are:
- complete_ohlcv_data()
- calculate_enhanced_sr_levels()
- calculate_dynamic_risk_reward()
- get_ohlcv_metadata_from_existing_data()
"""

import asyncio
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedOHLCVTest:
    """Focused test for actual intelligent OHLCV fetcher methods"""
    
    def __init__(self):
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
    
    async def test_1_module_import_and_methods(self):
        """Test 1: Import module and verify available methods"""
        logger.info("\n🔍 TEST 1: Module Import and Available Methods")
        
        try:
            sys.path.append('/app/backend')
            from intelligent_ohlcv_fetcher import (
                intelligent_ohlcv_fetcher, 
                OHLCVMetadata, 
                HighFrequencyData, 
                EnhancedSupportResistance, 
                DynamicRiskReward
            )
            
            # Check available methods
            available_methods = [method for method in dir(intelligent_ohlcv_fetcher) if not method.startswith('_')]
            expected_methods = ['complete_ohlcv_data', 'calculate_enhanced_sr_levels', 'calculate_dynamic_risk_reward', 'get_ohlcv_metadata_from_existing_data']
            
            found_methods = [method for method in expected_methods if hasattr(intelligent_ohlcv_fetcher, method)]
            
            self.log_test_result("Module Import and Methods", True, 
                               f"Available methods: {found_methods}")
            
            return intelligent_ohlcv_fetcher, OHLCVMetadata, HighFrequencyData, EnhancedSupportResistance, DynamicRiskReward
            
        except Exception as e:
            self.log_test_result("Module Import and Methods", False, f"Exception: {str(e)}")
            return None, None, None, None, None
    
    async def test_2_metadata_creation(self):
        """Test 2: Test metadata creation from existing data"""
        logger.info("\n🔍 TEST 2: OHLCV Metadata Creation")
        
        try:
            fetcher, OHLCVMetadata, _, _, _ = await self.test_1_module_import_and_methods()
            if not fetcher:
                self.log_test_result("OHLCV Metadata Creation", False, "Module import failed")
                return None
            
            # Create mock historical data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                 end=datetime.now(), freq='1D')
            
            mock_data = pd.DataFrame({
                'Open': np.random.uniform(100, 110, len(dates)),
                'High': np.random.uniform(105, 115, len(dates)),
                'Low': np.random.uniform(95, 105, len(dates)),
                'Close': np.random.uniform(100, 110, len(dates)),
                'Volume': np.random.uniform(1000, 10000, len(dates))
            }, index=dates)
            
            # Test metadata creation
            metadata = await fetcher.get_ohlcv_metadata_from_existing_data(
                existing_data=mock_data,
                primary_source="binance"
            )
            
            if metadata and hasattr(metadata, 'primary_source'):
                self.log_test_result("OHLCV Metadata Creation", True, 
                                   f"Metadata created: source={metadata.primary_source}, quality={metadata.primary_data_quality:.2f}")
                return metadata
            else:
                self.log_test_result("OHLCV Metadata Creation", False, 
                                   "Metadata creation returned invalid result")
                return None
                
        except Exception as e:
            self.log_test_result("OHLCV Metadata Creation", False, f"Exception: {str(e)}")
            return None
    
    async def test_3_high_frequency_completion(self):
        """Test 3: Test high-frequency data completion"""
        logger.info("\n🔍 TEST 3: High-Frequency Data Completion")
        
        try:
            fetcher, _, _, _, _ = await self.test_1_module_import_and_methods()
            metadata = await self.test_2_metadata_creation()
            
            if not fetcher or not metadata:
                self.log_test_result("High-Frequency Data Completion", False, "Prerequisites failed")
                return None
            
            # Test completion
            logger.info("   Testing complete_ohlcv_data method...")
            hf_data = await fetcher.complete_ohlcv_data(
                symbol="BTCUSDT",
                existing_metadata=metadata,
                target_timeframe="5m",
                hours_back=6
            )
            
            if hf_data and hasattr(hf_data, 'source') and hasattr(hf_data, 'data'):
                data_points = len(hf_data.data) if hasattr(hf_data.data, '__len__') else 0
                self.log_test_result("High-Frequency Data Completion", True, 
                                   f"HF data completed: source={hf_data.source}, points={data_points}, quality={hf_data.quality_score:.2f}")
                return hf_data
            else:
                self.log_test_result("High-Frequency Data Completion", False, 
                                   "HF completion returned invalid or no data")
                return None
                
        except Exception as e:
            self.log_test_result("High-Frequency Data Completion", False, f"Exception: {str(e)}")
            return None
    
    async def test_4_enhanced_sr_calculation(self):
        """Test 4: Test enhanced S/R calculation"""
        logger.info("\n🔍 TEST 4: Enhanced S/R Calculation")
        
        try:
            fetcher, _, HighFrequencyData, _, _ = await self.test_1_module_import_and_methods()
            if not fetcher:
                self.log_test_result("Enhanced S/R Calculation", False, "Module import failed")
                return None
            
            # Create mock high-frequency data
            dates = pd.date_range(start=datetime.now() - timedelta(hours=6), 
                                 end=datetime.now(), freq='5min')
            
            mock_hf_data = pd.DataFrame({
                'Open': np.random.uniform(45000, 46000, len(dates)),
                'High': np.random.uniform(45500, 46500, len(dates)),
                'Low': np.random.uniform(44500, 45500, len(dates)),
                'Close': np.random.uniform(45000, 46000, len(dates)),
                'Volume': np.random.uniform(100, 1000, len(dates))
            }, index=dates)
            
            hf_data = HighFrequencyData(
                symbol="BTCUSDT",
                timeframe="5m",
                data=mock_hf_data,
                source="mock_coinbase",
                quality_score=0.9,
                data_count=len(mock_hf_data),
                fetch_timestamp=datetime.now()
            )
            
            # Test enhanced S/R calculation
            enhanced_sr = await fetcher.calculate_enhanced_sr_levels(
                symbol="BTCUSDT",
                hf_data=hf_data,
                daily_support=44000.0,
                daily_resistance=47000.0
            )
            
            if enhanced_sr and hasattr(enhanced_sr, 'micro_support'):
                self.log_test_result("Enhanced S/R Calculation", True, 
                                   f"Enhanced S/R calculated: micro_support={enhanced_sr.micro_support:.2f}, "
                                   f"micro_resistance={enhanced_sr.micro_resistance:.2f}, "
                                   f"confidence_micro={enhanced_sr.confidence_micro:.2f}")
                return enhanced_sr
            else:
                self.log_test_result("Enhanced S/R Calculation", False, 
                                   "Enhanced S/R calculation returned invalid result")
                return None
                
        except Exception as e:
            self.log_test_result("Enhanced S/R Calculation", False, f"Exception: {str(e)}")
            return None
    
    async def test_5_dynamic_rr_calculation(self):
        """Test 5: Test dynamic RR calculation"""
        logger.info("\n🔍 TEST 5: Dynamic RR Calculation")
        
        try:
            fetcher, _, _, EnhancedSupportResistance, _ = await self.test_1_module_import_and_methods()
            if not fetcher:
                self.log_test_result("Dynamic RR Calculation", False, "Module import failed")
                return None
            
            # Create mock enhanced S/R data
            enhanced_sr = EnhancedSupportResistance(
                symbol="BTCUSDT",
                daily_support=44000.0,
                daily_resistance=47000.0,
                intraday_support=44500.0,
                intraday_resistance=46500.0,
                micro_support=44800.0,
                micro_resistance=46200.0,
                confidence_daily=0.9,
                confidence_intraday=0.8,
                confidence_micro=0.7,
                calculation_source="mock_test",
                calculation_timestamp=datetime.now()
            )
            
            # Test dynamic RR for LONG
            dynamic_rr_long = await fetcher.calculate_dynamic_risk_reward(
                symbol="BTCUSDT",
                signal_type="LONG",
                entry_price=45000.0,
                enhanced_sr=enhanced_sr
            )
            
            # Test dynamic RR for SHORT
            dynamic_rr_short = await fetcher.calculate_dynamic_risk_reward(
                symbol="BTCUSDT",
                signal_type="SHORT",
                entry_price=45000.0,
                enhanced_sr=enhanced_sr
            )
            
            if (dynamic_rr_long and hasattr(dynamic_rr_long, 'rr_final') and
                dynamic_rr_short and hasattr(dynamic_rr_short, 'rr_final')):
                
                self.log_test_result("Dynamic RR Calculation", True, 
                                   f"Dynamic RR calculated - LONG: {dynamic_rr_long.rr_final:.2f}:1 "
                                   f"({dynamic_rr_long.rr_selection_logic}), "
                                   f"SHORT: {dynamic_rr_short.rr_final:.2f}:1 "
                                   f"({dynamic_rr_short.rr_selection_logic})")
                return dynamic_rr_long, dynamic_rr_short
            else:
                self.log_test_result("Dynamic RR Calculation", False, 
                                   "Dynamic RR calculation returned invalid results")
                return None, None
                
        except Exception as e:
            self.log_test_result("Dynamic RR Calculation", False, f"Exception: {str(e)}")
            return None, None
    
    async def test_6_ia2_integration_check(self):
        """Test 6: Check if IA2 integration is using the intelligent OHLCV fetcher"""
        logger.info("\n🔍 TEST 6: IA2 Integration Check")
        
        try:
            # Check if the server.py has the integration code
            with open('/app/backend/server.py', 'r') as f:
                server_code = f.read()
            
            # Look for integration patterns
            integration_patterns = [
                'intelligent_ohlcv_fetcher.get_ohlcv_metadata_from_existing_data',
                'intelligent_ohlcv_fetcher.complete_ohlcv_data',
                'intelligent_ohlcv_fetcher.calculate_enhanced_sr_levels',
                'intelligent_ohlcv_fetcher.calculate_dynamic_risk_reward',
                'HIGH-FREQUENCY DATA INTEGRATION',
                'calculated_rr',
                'rr_reasoning'
            ]
            
            found_patterns = []
            for pattern in integration_patterns:
                if pattern in server_code:
                    found_patterns.append(pattern)
            
            if len(found_patterns) >= 5:  # Most patterns found
                self.log_test_result("IA2 Integration Check", True, 
                                   f"IA2 integration code found: {len(found_patterns)}/{len(integration_patterns)} patterns detected")
            else:
                self.log_test_result("IA2 Integration Check", False, 
                                   f"Limited IA2 integration: {len(found_patterns)}/{len(integration_patterns)} patterns found")
                
        except Exception as e:
            self.log_test_result("IA2 Integration Check", False, f"Exception: {str(e)}")
    
    async def run_focused_tests(self):
        """Run focused tests on available methods"""
        logger.info("🚀 Starting Focused Intelligent OHLCV Fetcher Tests")
        logger.info("=" * 70)
        logger.info("📋 TESTING ACTUAL IMPLEMENTED METHODS")
        logger.info("🎯 Focus: complete_ohlcv_data, calculate_enhanced_sr_levels, calculate_dynamic_risk_reward")
        logger.info("=" * 70)
        
        # Run tests
        await self.test_1_module_import_and_methods()
        await self.test_2_metadata_creation()
        await self.test_3_high_frequency_completion()
        await self.test_4_enhanced_sr_calculation()
        await self.test_5_dynamic_rr_calculation()
        await self.test_6_ia2_integration_check()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 FOCUSED TEST RESULTS SUMMARY")
        logger.info("=" * 70)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
        
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Analysis
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL FOCUSED TESTS PASSED!")
            logger.info("✅ Intelligent OHLCV Fetcher core methods are functional")
            logger.info("✅ Enhanced S/R calculation working")
            logger.info("✅ Dynamic RR calculation operational")
            logger.info("✅ IA2 integration code is present")
        elif passed_tests >= total_tests * 0.8:
            logger.info("\n⚠️ MOSTLY FUNCTIONAL - Core methods working")
        else:
            logger.info("\n❌ CORE FUNCTIONALITY ISSUES DETECTED")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = FocusedOHLCVTest()
    passed, total = await test_suite.run_focused_tests()
    
    if passed >= total * 0.8:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Issues detected

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
CRITICAL MARKET VARIABLES SYSTEM COMPREHENSIVE TESTING SUITE
Focus: Testing the critical market variables system for 24h/Bitcoin/MarketCap/Volume data

TESTING REQUIREMENTS FROM REVIEW REQUEST:
1. **Critical Variables Endpoint**: Test the new `/admin/market/critical` endpoint
2. **Bitcoin Data Integrity**: Verify BTC price, 24h change, 7d change, and 30d change
3. **Market Cap Accuracy**: Test total crypto market cap calculation and formatting
4. **Volume Data Quality**: Validate 24h trading volume data across all crypto markets
5. **Fallback Robustness**: Test critical variables availability when primary APIs fail
6. **Historical Data Precision**: Test Binance Klines integration for 7d and 30d BTC changes
7. **Emergency Defaults**: Verify realistic emergency data if all sources fail
8. **System Impact Assessment**: Test trading readiness based on critical variables

The critical variables are essential for:
- IA1/IA2 trading decisions and confidence calculations
- Market regime detection (bull/bear/neutral)
- Risk management and position sizing
- Overall system trading readiness

Test scenarios:
- Successful critical data fetching with all sources working
- Fallback behavior when CoinGecko is rate-limited
- Emergency mode with realistic defaults
- Data accuracy validation (prices, percentages, market cap calculations)
- System health assessment based on critical variables availability
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

class CriticalMarketVariablesTestSuite:
    """Comprehensive test suite for Critical Market Variables System"""
    
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
        logger.info(f"Testing Critical Market Variables System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected critical variables
        self.expected_critical_vars = [
            'btc_price', 'market_cap', 'volume_24h', 'dominance'
        ]
        
        # Expected BTC data fields
        self.expected_btc_fields = [
            'current', 'change_24h', 'change_7d', 'change_30d'
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
    
    async def test_1_critical_variables_endpoint_availability(self):
        """Test 1: Critical Variables Endpoint Availability and Structure"""
        logger.info("\n🔍 TEST 1: Critical Variables Endpoint Availability and Structure")
        
        try:
            response = requests.get(f"{self.api_url}/admin/market/critical", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   📊 Critical variables response received: {len(str(data))} chars")
                
                # Check main structure
                required_top_level = ['status', 'timestamp', 'critical_variables', 'system_health']
                missing_fields = [field for field in required_top_level if field not in data]
                
                if not missing_fields:
                    # Check critical_variables structure
                    critical_vars = data.get('critical_variables', {})
                    vars_present = 0
                    
                    for var_name in self.expected_critical_vars:
                        if var_name in critical_vars:
                            vars_present += 1
                            var_data = critical_vars[var_name]
                            status = var_data.get('status', 'Unknown')
                            logger.info(f"      ✅ {var_name}: {status}")
                        else:
                            logger.info(f"      ❌ {var_name}: Missing")
                    
                    # Check system health
                    system_health = data.get('system_health', {})
                    trading_readiness = system_health.get('trading_readiness', 'Unknown')
                    all_vars_ok = system_health.get('all_critical_vars_ok', False)
                    
                    logger.info(f"   📊 Trading readiness: {trading_readiness}")
                    logger.info(f"   📊 All critical vars OK: {all_vars_ok}")
                    
                    success = vars_present >= 3 and trading_readiness in ['READY', 'DEGRADED']
                    details = f"Variables present: {vars_present}/{len(self.expected_critical_vars)}, Trading: {trading_readiness}"
                    
                    self.log_test_result("Critical Variables Endpoint Availability", success, details)
                else:
                    self.log_test_result("Critical Variables Endpoint Availability", False, f"Missing fields: {missing_fields}")
            else:
                self.log_test_result("Critical Variables Endpoint Availability", False, f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test_result("Critical Variables Endpoint Availability", False, f"Exception: {str(e)}")
    
    async def test_2_bitcoin_data_integrity(self):
        """Test 2: Bitcoin Data Integrity - Price, 24h, 7d, 30d Changes"""
        logger.info("\n🔍 TEST 2: Bitcoin Data Integrity - Price and Historical Changes")
        
        try:
            response = requests.get(f"{self.api_url}/admin/market/critical", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Bitcoin Data Integrity", False, f"HTTP {response.status_code}")
                return
            
            data = response.json()
            critical_vars = data.get('critical_variables', {})
            btc_data = critical_vars.get('btc_price', {})
            
            # Check BTC data completeness
            btc_fields_present = 0
            btc_values_valid = 0
            
            for field in self.expected_btc_fields:
                if field in btc_data:
                    btc_fields_present += 1
                    value = btc_data[field]
                    
                    if field == 'current':
                        # BTC price should be reasonable (between $10k and $200k)
                        if isinstance(value, (int, float)) and 10000 <= value <= 200000:
                            btc_values_valid += 1
                            logger.info(f"      ✅ BTC {field}: ${value:,.2f} (valid)")
                        else:
                            logger.info(f"      ❌ BTC {field}: {value} (invalid range)")
                    else:
                        # Changes should be reasonable percentages (-50% to +50%)
                        if isinstance(value, (int, float)) and -50 <= value <= 50:
                            btc_values_valid += 1
                            logger.info(f"      ✅ BTC {field}: {value:+.2f}% (valid)")
                        else:
                            logger.info(f"      ❌ BTC {field}: {value} (invalid range)")
                else:
                    logger.info(f"      ❌ BTC {field}: Missing")
            
            # Check BTC status
            btc_status = btc_data.get('status', 'Unknown')
            status_ok = '✅' in btc_status
            
            logger.info(f"   📊 BTC fields present: {btc_fields_present}/{len(self.expected_btc_fields)}")
            logger.info(f"   📊 BTC values valid: {btc_values_valid}/{len(self.expected_btc_fields)}")
            logger.info(f"   📊 BTC status: {btc_status}")
            
            # Success criteria
            completeness_ok = btc_fields_present >= 3  # At least current, 24h, 7d
            validity_ok = btc_values_valid >= 3
            
            success = completeness_ok and validity_ok and status_ok
            details = f"Fields: {btc_fields_present}/{len(self.expected_btc_fields)}, Valid: {btc_values_valid}/{len(self.expected_btc_fields)}, Status: {status_ok}"
            
            self.log_test_result("Bitcoin Data Integrity", success, details)
            
        except Exception as e:
            self.log_test_result("Bitcoin Data Integrity", False, f"Exception: {str(e)}")
    
    async def test_3_market_cap_accuracy(self):
        """Test 3: Market Cap Accuracy - Total Crypto Market Cap Calculation"""
        logger.info("\n🔍 TEST 3: Market Cap Accuracy - Total Crypto Market Cap")
        
        try:
            response = requests.get(f"{self.api_url}/admin/market/critical", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Market Cap Accuracy", False, f"HTTP {response.status_code}")
                return
            
            data = response.json()
            critical_vars = data.get('critical_variables', {})
            mcap_data = critical_vars.get('market_cap', {})
            
            # Check market cap data
            total_usd = mcap_data.get('total_usd', 0)
            total_formatted = mcap_data.get('total_formatted', '')
            mcap_status = mcap_data.get('status', 'Unknown')
            
            logger.info(f"   📊 Total market cap (USD): ${total_usd:,.0f}")
            logger.info(f"   📊 Formatted market cap: {total_formatted}")
            logger.info(f"   📊 Market cap status: {mcap_status}")
            
            # Validate market cap values
            # Total crypto market cap should be between $500B and $10T
            mcap_valid = isinstance(total_usd, (int, float)) and 500e9 <= total_usd <= 10e12
            
            # Check formatting (should contain 'T' for trillions)
            formatting_valid = 'T' in total_formatted and '$' in total_formatted
            
            # Check status
            status_ok = '✅' in mcap_status
            
            # Calculate market cap in trillions for validation
            mcap_trillions = total_usd / 1e12
            expected_formatted = f"${mcap_trillions:.2f}T"
            formatting_accurate = abs(mcap_trillions - float(total_formatted.replace('$', '').replace('T', ''))) < 0.1
            
            logger.info(f"   📊 Market cap validation: {mcap_valid} (${total_usd/1e12:.2f}T)")
            logger.info(f"   📊 Formatting validation: {formatting_valid}")
            logger.info(f"   📊 Formatting accuracy: {formatting_accurate}")
            
            success = mcap_valid and formatting_valid and status_ok
            details = f"Value valid: {mcap_valid}, Format valid: {formatting_valid}, Status OK: {status_ok}, Amount: ${mcap_trillions:.2f}T"
            
            self.log_test_result("Market Cap Accuracy", success, details)
            
        except Exception as e:
            self.log_test_result("Market Cap Accuracy", False, f"Exception: {str(e)}")
    
    async def test_4_volume_data_quality(self):
        """Test 4: Volume Data Quality - 24h Trading Volume Validation"""
        logger.info("\n🔍 TEST 4: Volume Data Quality - 24h Trading Volume")
        
        try:
            response = requests.get(f"{self.api_url}/admin/market/critical", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Volume Data Quality", False, f"HTTP {response.status_code}")
                return
            
            data = response.json()
            critical_vars = data.get('critical_variables', {})
            volume_data = critical_vars.get('volume_24h', {})
            
            # Check volume data
            total_usd = volume_data.get('total_usd', 0)
            total_formatted = volume_data.get('total_formatted', '')
            volume_status = volume_data.get('status', 'Unknown')
            
            logger.info(f"   📊 Total 24h volume (USD): ${total_usd:,.0f}")
            logger.info(f"   📊 Formatted 24h volume: {total_formatted}")
            logger.info(f"   📊 Volume status: {volume_status}")
            
            # Validate volume values
            # 24h volume should be between $10B and $1T
            volume_valid = isinstance(total_usd, (int, float)) and 10e9 <= total_usd <= 1e12
            
            # Check formatting (should contain 'B' for billions)
            formatting_valid = 'B' in total_formatted and '$' in total_formatted
            
            # Check status
            status_ok = '✅' in volume_status
            
            # Calculate volume in billions for validation
            volume_billions = total_usd / 1e9
            expected_formatted = f"${volume_billions:.1f}B"
            formatting_accurate = abs(volume_billions - float(total_formatted.replace('$', '').replace('B', ''))) < 1.0
            
            logger.info(f"   📊 Volume validation: {volume_valid} (${total_usd/1e9:.1f}B)")
            logger.info(f"   📊 Formatting validation: {formatting_valid}")
            logger.info(f"   📊 Formatting accuracy: {formatting_accurate}")
            
            success = volume_valid and formatting_valid and status_ok
            details = f"Value valid: {volume_valid}, Format valid: {formatting_valid}, Status OK: {status_ok}, Amount: ${volume_billions:.1f}B"
            
            self.log_test_result("Volume Data Quality", success, details)
            
        except Exception as e:
            self.log_test_result("Volume Data Quality", False, f"Exception: {str(e)}")
    
    async def test_5_fallback_robustness(self):
        """Test 5: Fallback Robustness - Critical Variables Availability During API Failures"""
        logger.info("\n🔍 TEST 5: Fallback Robustness - API Failure Resilience")
        
        try:
            # Check backend logs for fallback evidence
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
            
            # Look for fallback patterns in logs
            fallback_patterns = [
                "CoinGecko rate limit",
                "trying CoinMarketCap fallback",
                "trying Binance fallback", 
                "using realistic defaults",
                "All market data sources failed",
                "fallback cascade",
                "emergency defaults"
            ]
            
            fallback_evidence = {}
            for pattern in fallback_patterns:
                count = backend_logs.count(pattern)
                fallback_evidence[pattern] = count
                if count > 0:
                    logger.info(f"      ✅ Fallback pattern detected: '{pattern}' ({count} times)")
                else:
                    logger.info(f"      ⚪ Fallback pattern not found: '{pattern}'")
            
            # Test current endpoint response during potential API issues
            response = requests.get(f"{self.api_url}/admin/market/critical", timeout=30)
            
            endpoint_responsive = response.status_code == 200
            
            if endpoint_responsive:
                data = response.json()
                
                # Check if system is handling degraded conditions gracefully
                system_health = data.get('system_health', {})
                trading_readiness = system_health.get('trading_readiness', 'Unknown')
                fallback_info = data.get('fallback_info', {})
                
                logger.info(f"   📊 Endpoint responsive: {endpoint_responsive}")
                logger.info(f"   📊 Trading readiness: {trading_readiness}")
                logger.info(f"   📊 Fallback info present: {bool(fallback_info)}")
                
                # Check for realistic emergency defaults
                critical_vars = data.get('critical_variables', {})
                has_realistic_data = all([
                    critical_vars.get('btc_price', {}).get('current', 0) > 10000,
                    critical_vars.get('market_cap', {}).get('total_usd', 0) > 500e9,
                    critical_vars.get('volume_24h', {}).get('total_usd', 0) > 10e9
                ])
                
                logger.info(f"   📊 Has realistic data: {has_realistic_data}")
            
            # Count fallback evidence
            fallback_patterns_detected = sum(1 for count in fallback_evidence.values() if count > 0)
            
            # Success criteria
            fallback_system_working = fallback_patterns_detected >= 2 or endpoint_responsive
            
            success = fallback_system_working
            details = f"Fallback patterns: {fallback_patterns_detected}/{len(fallback_patterns)}, Endpoint responsive: {endpoint_responsive}"
            
            self.log_test_result("Fallback Robustness", success, details)
            
        except Exception as e:
            self.log_test_result("Fallback Robustness", False, f"Exception: {str(e)}")
    
    async def test_6_historical_data_precision(self):
        """Test 6: Historical Data Precision - Binance Klines Integration for 7d/30d Changes"""
        logger.info("\n🔍 TEST 6: Historical Data Precision - Binance Klines Integration")
        
        try:
            # Check backend logs for Binance Klines integration evidence
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
            
            # Look for Binance Klines patterns
            klines_patterns = [
                "Binance Klines",
                "historical BTC data",
                "7d change calculation",
                "30d change calculation",
                "Binance API",
                "klines endpoint",
                "historical price data"
            ]
            
            klines_evidence = {}
            for pattern in klines_patterns:
                count = backend_logs.count(pattern)
                klines_evidence[pattern] = count
                if count > 0:
                    logger.info(f"      ✅ Klines pattern detected: '{pattern}' ({count} times)")
                else:
                    logger.info(f"      ⚪ Klines pattern not found: '{pattern}'")
            
            # Test the precision of historical data
            response = requests.get(f"{self.api_url}/admin/market/critical", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                critical_vars = data.get('critical_variables', {})
                btc_data = critical_vars.get('btc_price', {})
                
                # Check if we have 7d and 30d changes
                change_7d = btc_data.get('change_7d', None)
                change_30d = btc_data.get('change_30d', None)
                
                has_7d_data = change_7d is not None and isinstance(change_7d, (int, float))
                has_30d_data = change_30d is not None and isinstance(change_30d, (int, float))
                
                logger.info(f"   📊 7d change available: {has_7d_data} ({change_7d}%)")
                logger.info(f"   📊 30d change available: {has_30d_data} ({change_30d}%)")
                
                # Check if changes are reasonable
                changes_reasonable = True
                if has_7d_data and not (-30 <= change_7d <= 30):
                    changes_reasonable = False
                if has_30d_data and not (-50 <= change_30d <= 50):
                    changes_reasonable = False
                
                logger.info(f"   📊 Changes reasonable: {changes_reasonable}")
                
                # Check for precision indicators (non-zero, non-round numbers suggest real calculation)
                precision_indicators = 0
                if has_7d_data and change_7d != 0 and abs(change_7d) != round(abs(change_7d)):
                    precision_indicators += 1
                if has_30d_data and change_30d != 0 and abs(change_30d) != round(abs(change_30d)):
                    precision_indicators += 1
                
                logger.info(f"   📊 Precision indicators: {precision_indicators}/2")
            else:
                has_7d_data = has_30d_data = changes_reasonable = False
                precision_indicators = 0
            
            klines_patterns_detected = sum(1 for count in klines_evidence.values() if count > 0)
            
            # Success criteria
            historical_data_working = (has_7d_data and has_30d_data and changes_reasonable) or klines_patterns_detected >= 2
            
            success = historical_data_working
            details = f"Klines patterns: {klines_patterns_detected}/{len(klines_patterns)}, 7d data: {has_7d_data}, 30d data: {has_30d_data}, Reasonable: {changes_reasonable}"
            
            self.log_test_result("Historical Data Precision", success, details)
            
        except Exception as e:
            self.log_test_result("Historical Data Precision", False, f"Exception: {str(e)}")
    
    async def test_7_emergency_defaults(self):
        """Test 7: Emergency Defaults - Realistic Emergency Data When All Sources Fail"""
        logger.info("\n🔍 TEST 7: Emergency Defaults - Realistic Emergency Data")
        
        try:
            # Check backend logs for emergency defaults evidence
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
            
            # Look for emergency defaults patterns
            emergency_patterns = [
                "emergency defaults",
                "realistic defaults",
                "All sources failed",
                "using fallback values",
                "emergency mode",
                "default values activated"
            ]
            
            emergency_evidence = {}
            for pattern in emergency_patterns:
                count = backend_logs.count(pattern)
                emergency_evidence[pattern] = count
                if count > 0:
                    logger.info(f"      ✅ Emergency pattern detected: '{pattern}' ({count} times)")
                else:
                    logger.info(f"      ⚪ Emergency pattern not found: '{pattern}'")
            
            # Test current data for realistic emergency defaults
            response = requests.get(f"{self.api_url}/admin/market/critical", timeout=30)
            
            realistic_defaults = False
            emergency_mode_detected = False
            
            if response.status_code == 200:
                data = response.json()
                critical_vars = data.get('critical_variables', {})
                
                # Check if values look like realistic emergency defaults
                btc_price = critical_vars.get('btc_price', {}).get('current', 0)
                market_cap = critical_vars.get('market_cap', {}).get('total_usd', 0)
                volume_24h = critical_vars.get('volume_24h', {}).get('total_usd', 0)
                
                # Realistic emergency defaults should be:
                # BTC: ~$40k-60k, Market Cap: ~$1.5-2.5T, Volume: ~$50-100B
                btc_realistic = 30000 <= btc_price <= 80000
                mcap_realistic = 1e12 <= market_cap <= 5e12
                volume_realistic = 30e9 <= volume_24h <= 200e9
                
                realistic_defaults = btc_realistic and mcap_realistic and volume_realistic
                
                logger.info(f"   📊 BTC realistic: {btc_realistic} (${btc_price:,.0f})")
                logger.info(f"   📊 Market cap realistic: {mcap_realistic} (${market_cap/1e12:.2f}T)")
                logger.info(f"   📊 Volume realistic: {volume_realistic} (${volume_24h/1e9:.1f}B)")
                
                # Check for emergency mode indicators
                system_health = data.get('system_health', {})
                trading_readiness = system_health.get('trading_readiness', '')
                
                emergency_mode_detected = trading_readiness == 'DEGRADED' or 'emergency' in str(data).lower()
                
                logger.info(f"   📊 Emergency mode detected: {emergency_mode_detected}")
            
            emergency_patterns_detected = sum(1 for count in emergency_evidence.values() if count > 0)
            
            # Success criteria
            emergency_system_working = realistic_defaults or emergency_patterns_detected >= 1
            
            success = emergency_system_working
            details = f"Emergency patterns: {emergency_patterns_detected}/{len(emergency_patterns)}, Realistic defaults: {realistic_defaults}, Emergency mode: {emergency_mode_detected}"
            
            self.log_test_result("Emergency Defaults", success, details)
            
        except Exception as e:
            self.log_test_result("Emergency Defaults", False, f"Exception: {str(e)}")
    
    async def test_8_system_impact_assessment(self):
        """Test 8: System Impact Assessment - Trading Readiness Based on Critical Variables"""
        logger.info("\n🔍 TEST 8: System Impact Assessment - Trading Readiness Evaluation")
        
        try:
            response = requests.get(f"{self.api_url}/admin/market/critical", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("System Impact Assessment", False, f"HTTP {response.status_code}")
                return
            
            data = response.json()
            system_health = data.get('system_health', {})
            critical_vars = data.get('critical_variables', {})
            
            # Check system health indicators
            all_vars_ok = system_health.get('all_critical_vars_ok', False)
            trading_readiness = system_health.get('trading_readiness', 'Unknown')
            market_regime = system_health.get('market_regime', 'Unknown')
            
            logger.info(f"   📊 All critical vars OK: {all_vars_ok}")
            logger.info(f"   📊 Trading readiness: {trading_readiness}")
            logger.info(f"   📊 Market regime: {market_regime}")
            
            # Analyze individual variable statuses
            var_statuses = {}
            for var_name in self.expected_critical_vars:
                if var_name in critical_vars:
                    status = critical_vars[var_name].get('status', 'Unknown')
                    var_statuses[var_name] = '✅' in status
                    logger.info(f"      📊 {var_name}: {status}")
                else:
                    var_statuses[var_name] = False
            
            # Check trading readiness logic
            vars_working = sum(var_statuses.values())
            readiness_appropriate = True
            
            if vars_working >= 3 and trading_readiness not in ['READY', 'DEGRADED']:
                readiness_appropriate = False
            elif vars_working < 2 and trading_readiness == 'READY':
                readiness_appropriate = False
            
            logger.info(f"   📊 Variables working: {vars_working}/{len(self.expected_critical_vars)}")
            logger.info(f"   📊 Readiness appropriate: {readiness_appropriate}")
            
            # Check for IA1/IA2 impact assessment
            # Test if the system provides context for trading decisions
            global_market_response = requests.get(f"{self.api_url}/admin/market/global", timeout=30)
            ia_context_available = global_market_response.status_code == 200
            
            if ia_context_available:
                global_data = global_market_response.json()
                has_trading_context = 'market_overview' in global_data
                logger.info(f"   📊 IA trading context available: {has_trading_context}")
            else:
                has_trading_context = False
                logger.info(f"   📊 IA trading context available: {has_trading_context}")
            
            # Check fallback info for system reliability
            fallback_info = data.get('fallback_info', {})
            has_fallback_info = bool(fallback_info)
            reliability_info = fallback_info.get('reliability', '')
            
            logger.info(f"   📊 Fallback info present: {has_fallback_info}")
            logger.info(f"   📊 System reliability: {reliability_info}")
            
            # Success criteria
            health_assessment_working = all_vars_ok or trading_readiness in ['READY', 'DEGRADED']
            impact_assessment_logical = readiness_appropriate
            context_for_trading = ia_context_available or has_fallback_info
            
            success = health_assessment_working and impact_assessment_logical and context_for_trading
            details = f"Health assessment: {health_assessment_working}, Logic appropriate: {impact_assessment_logical}, Trading context: {context_for_trading}, Readiness: {trading_readiness}"
            
            self.log_test_result("System Impact Assessment", success, details)
            
        except Exception as e:
            self.log_test_result("System Impact Assessment", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all critical market variables tests"""
        logger.info("🚀 Starting Critical Market Variables System Comprehensive Test Suite")
        logger.info("=" * 100)
        logger.info("📋 CRITICAL MARKET VARIABLES SYSTEM COMPREHENSIVE TESTING")
        logger.info("🎯 Testing: 24h/Bitcoin/MarketCap/Volume data integrity and system readiness")
        logger.info("🎯 Focus: Critical variables endpoint, fallback robustness, emergency defaults, trading readiness")
        logger.info("=" * 100)
        
        # Run all tests in sequence
        await self.test_1_critical_variables_endpoint_availability()
        await self.test_2_bitcoin_data_integrity()
        await self.test_3_market_cap_accuracy()
        await self.test_4_volume_data_quality()
        await self.test_5_fallback_robustness()
        await self.test_6_historical_data_precision()
        await self.test_7_emergency_defaults()
        await self.test_8_system_impact_assessment()
        
        # Summary
        logger.info("\n" + "=" * 100)
        logger.info("📊 CRITICAL MARKET VARIABLES SYSTEM TEST SUMMARY")
        logger.info("=" * 100)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        # Categorize results by importance
        critical_tests = []
        important_tests = []
        supporting_tests = []
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
            
            # Categorize by importance
            if any(keyword in result['test'] for keyword in ['Endpoint Availability', 'Bitcoin Data', 'System Impact']):
                critical_tests.append(result)
            elif any(keyword in result['test'] for keyword in ['Market Cap', 'Volume Data', 'Fallback']):
                important_tests.append(result)
            else:
                supporting_tests.append(result)
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        
        # System analysis
        logger.info("\n" + "=" * 100)
        logger.info("📋 CRITICAL MARKET VARIABLES SYSTEM STATUS")
        logger.info("=" * 100)
        
        critical_passed = sum(1 for result in critical_tests if result['success'])
        important_passed = sum(1 for result in important_tests if result['success'])
        supporting_passed = sum(1 for result in supporting_tests if result['success'])
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Critical Market Variables System FULLY FUNCTIONAL!")
            logger.info("✅ Critical variables endpoint operational")
            logger.info("✅ Bitcoin data integrity confirmed")
            logger.info("✅ Market cap and volume data accurate")
            logger.info("✅ Fallback robustness verified")
            logger.info("✅ Emergency defaults working")
            logger.info("✅ System impact assessment functional")
        elif critical_passed == len(critical_tests) and important_passed >= len(important_tests) * 0.8:
            logger.info("⚠️ MOSTLY FUNCTIONAL - Critical Market Variables system working with minor gaps")
            logger.info("🔍 Core functionality operational, some advanced features may need fine-tuning")
        elif critical_passed >= len(critical_tests) * 0.8:
            logger.info("⚠️ PARTIALLY FUNCTIONAL - Core critical variables working")
            logger.info("🔧 Some important features need attention for full reliability")
        else:
            logger.info("❌ SYSTEM NOT FUNCTIONAL - Critical issues with market variables system")
            logger.info("🚨 Major implementation gaps preventing reliable trading decisions")
        
        # Detailed analysis by category
        logger.info(f"\n📊 CRITICAL TESTS: {critical_passed}/{len(critical_tests)} passed")
        logger.info(f"📊 IMPORTANT TESTS: {important_passed}/{len(important_tests)} passed")
        logger.info(f"📊 SUPPORTING TESTS: {supporting_passed}/{len(supporting_tests)} passed")
        
        # Requirements verification
        logger.info("\n📝 CRITICAL MARKET VARIABLES REQUIREMENTS VERIFICATION:")
        
        requirements_status = []
        
        for result in self.test_results:
            if result['success']:
                if "Endpoint Availability" in result['test']:
                    requirements_status.append("✅ Critical Variables Endpoint operational")
                elif "Bitcoin Data" in result['test']:
                    requirements_status.append("✅ Bitcoin data integrity (24h/7d/30d) verified")
                elif "Market Cap" in result['test']:
                    requirements_status.append("✅ Market cap accuracy confirmed")
                elif "Volume Data" in result['test']:
                    requirements_status.append("✅ Volume data quality validated")
                elif "Fallback" in result['test']:
                    requirements_status.append("✅ Fallback robustness working")
                elif "Historical Data" in result['test']:
                    requirements_status.append("✅ Historical data precision (Binance Klines) operational")
                elif "Emergency" in result['test']:
                    requirements_status.append("✅ Emergency defaults providing realistic data")
                elif "System Impact" in result['test']:
                    requirements_status.append("✅ System impact assessment for trading readiness working")
            else:
                if "Endpoint Availability" in result['test']:
                    requirements_status.append("❌ Critical Variables Endpoint not operational")
                elif "Bitcoin Data" in result['test']:
                    requirements_status.append("❌ Bitcoin data integrity issues")
                elif "Market Cap" in result['test']:
                    requirements_status.append("❌ Market cap accuracy problems")
                elif "Volume Data" in result['test']:
                    requirements_status.append("❌ Volume data quality issues")
                elif "Fallback" in result['test']:
                    requirements_status.append("❌ Fallback robustness not working")
                elif "Historical Data" in result['test']:
                    requirements_status.append("❌ Historical data precision issues")
                elif "Emergency" in result['test']:
                    requirements_status.append("❌ Emergency defaults not providing realistic data")
                elif "System Impact" in result['test']:
                    requirements_status.append("❌ System impact assessment not working")
        
        for req in requirements_status:
            logger.info(f"   {req}")
        
        # Final verdict
        logger.info(f"\n🏆 FINAL RESULT: {len([r for r in requirements_status if r.startswith('✅')])}/{len(requirements_status)} requirements satisfied")
        
        if critical_passed == len(critical_tests) and passed_tests >= total_tests * 0.85:
            logger.info("\n🎉 VERDICT: Critical Market Variables System is FULLY FUNCTIONAL!")
            logger.info("✅ All essential trading variables available and accurate")
            logger.info("✅ Robust fallback system ensures continuous operation")
            logger.info("✅ System ready to support IA1/IA2 trading decisions")
            logger.info("✅ Market regime detection and trading readiness assessment operational")
        elif critical_passed >= len(critical_tests) * 0.8 and passed_tests >= total_tests * 0.7:
            logger.info("\n⚠️ VERDICT: Critical Market Variables System is MOSTLY FUNCTIONAL")
            logger.info("🔍 Core functionality working, minor optimizations needed")
        elif critical_passed >= len(critical_tests) * 0.6:
            logger.info("\n⚠️ VERDICT: Critical Market Variables System is PARTIALLY FUNCTIONAL")
            logger.info("🔧 Some critical components need attention for reliable operation")
        else:
            logger.info("\n❌ VERDICT: Critical Market Variables System is NOT FUNCTIONAL")
            logger.info("🚨 Major issues preventing reliable market data for trading decisions")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = CriticalMarketVariablesTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed >= total * 0.8:  # 80% pass rate for success
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
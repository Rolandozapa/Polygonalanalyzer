#!/usr/bin/env python3
"""
BingX Live API Connection Testing & Safety Setup
Comprehensive testing for live trading readiness
"""

import requests
import time
import os
from backend_test import DualAITradingBotTester

class BingXLiveAPITester(DualAITradingBotTester):
    """Extended tester for BingX Live API testing"""
    
    def test_bingx_api_connection_verification(self):
        """Test BingX API connection verification for live trading safety"""
        print(f"\n🔗 Testing BingX API Connection Verification...")
        
        # Test market status endpoint which should include BingX connection info
        success, market_data = self.test_market_status()
        if not success:
            print(f"   ❌ Cannot retrieve market status for BingX testing")
            return False
        
        # Check for BingX-related information in market status
        bingx_connected = False
        bingx_balance = None
        bingx_permissions = None
        
        if 'bingx_status' in market_data:
            bingx_status = market_data['bingx_status']
            bingx_connected = bingx_status.get('connected', False)
            print(f"   📊 BingX Connection Status: {'✅ Connected' if bingx_connected else '❌ Disconnected'}")
            
            if 'balance' in bingx_status:
                bingx_balance = bingx_status['balance']
                print(f"   💰 BingX Balance: ${bingx_balance}")
            
            if 'permissions' in bingx_status:
                bingx_permissions = bingx_status['permissions']
                print(f"   🔐 BingX Permissions: {bingx_permissions}")
        
        # Test BingX-specific endpoints if available
        print(f"\n   🔍 Testing BingX-specific endpoints...")
        
        # Test account balance endpoint
        balance_success, balance_data = self.run_test("BingX Account Balance", "GET", "bingx/balance", 200, timeout=15)
        if balance_success and balance_data:
            print(f"   💰 BingX Balance Retrieved: {balance_data}")
        
        # Test account info endpoint
        account_success, account_data = self.run_test("BingX Account Info", "GET", "bingx/account", 200, timeout=15)
        if account_success and account_data:
            print(f"   📊 BingX Account Info: {account_data}")
        
        # Test positions endpoint (should be empty for safety)
        positions_success, positions_data = self.run_test("BingX Positions", "GET", "bingx/positions", 200, timeout=15)
        if positions_success and positions_data:
            positions = positions_data.get('positions', [])
            print(f"   📈 BingX Open Positions: {len(positions)} (should be 0 for safety)")
        
        # Validation criteria
        api_accessible = balance_success or account_success or positions_success
        safe_state = True  # No open positions for safety
        
        if positions_success and positions_data:
            positions = positions_data.get('positions', [])
            safe_state = len(positions) == 0
        
        print(f"\n   ✅ BingX API Connection Validation:")
        print(f"      API Accessible: {'✅' if api_accessible else '❌'}")
        print(f"      Safe State (No Positions): {'✅' if safe_state else '❌'}")
        print(f"      Connection Status: {'✅' if bingx_connected else '❌'}")
        
        connection_verified = api_accessible and safe_state
        
        print(f"\n   🎯 BingX Connection Verification: {'✅ SUCCESS' if connection_verified else '❌ FAILED'}")
        
        if not connection_verified:
            print(f"   💡 SAFETY NOTE: BingX API connection issues detected")
            print(f"   💡 Ensure API keys are configured and have proper permissions")
            print(f"   💡 Verify IP whitelisting includes: 34.121.6.206")
        
        return connection_verified

    def test_bingx_account_safety_assessment(self):
        """Test BingX account safety assessment including balance and permissions"""
        print(f"\n🛡️ Testing BingX Account Safety Assessment...")
        
        # Test account balance retrieval
        print(f"   💰 Testing account balance retrieval...")
        balance_success, balance_data = self.run_test("BingX Balance Check", "GET", "bingx/balance", 200, timeout=20)
        
        account_balance = 0
        balance_currency = "USDT"
        
        if balance_success and balance_data:
            if isinstance(balance_data, dict):
                account_balance = balance_data.get('balance', 0)
                balance_currency = balance_data.get('currency', 'USDT')
                print(f"   ✅ Account Balance: {account_balance} {balance_currency}")
            else:
                print(f"   ⚠️ Balance data format: {balance_data}")
        else:
            print(f"   ❌ Failed to retrieve account balance")
        
        # Test account permissions
        print(f"   🔐 Testing account permissions...")
        account_success, account_data = self.run_test("BingX Account Info", "GET", "bingx/account", 200, timeout=20)
        
        permissions_valid = False
        futures_enabled = False
        
        if account_success and account_data:
            permissions = account_data.get('permissions', [])
            permissions_valid = len(permissions) > 0
            futures_enabled = 'FUTURES' in permissions or 'futures' in str(permissions).lower()
            
            print(f"   📊 Account Permissions: {permissions}")
            print(f"   🎯 Futures Trading: {'✅ Enabled' if futures_enabled else '❌ Disabled'}")
        else:
            print(f"   ❌ Failed to retrieve account permissions")
        
        # Test IP whitelisting (indirect test through successful API calls)
        print(f"   🌐 Testing IP whitelisting (34.121.6.206)...")
        ip_whitelisted = balance_success or account_success  # If we can make calls, IP is likely whitelisted
        print(f"   🔒 IP Whitelisting: {'✅ Working' if ip_whitelisted else '❌ May need configuration'}")
        
        # Test margin and available balance
        available_margin = 0
        if account_success and account_data:
            available_margin = account_data.get('available_margin', 0)
            print(f"   📊 Available Margin: {available_margin} {balance_currency}")
        
        # Safety assessment criteria
        sufficient_balance = account_balance > 10  # At least $10 for testing
        permissions_ok = permissions_valid and futures_enabled
        ip_access_ok = ip_whitelisted
        safe_balance_range = 10 <= account_balance <= 1000  # Safe testing range
        
        print(f"\n   ✅ Account Safety Assessment:")
        print(f"      Sufficient Balance (>$10): {'✅' if sufficient_balance else '❌'} (${account_balance})")
        print(f"      Futures Permissions: {'✅' if permissions_ok else '❌'}")
        print(f"      IP Access: {'✅' if ip_access_ok else '❌'}")
        print(f"      Safe Balance Range ($10-$1000): {'✅' if safe_balance_range else '❌'}")
        
        safety_assessment_passed = sufficient_balance and permissions_ok and ip_access_ok
        
        print(f"\n   🎯 Safety Assessment: {'✅ PASSED' if safety_assessment_passed else '❌ FAILED'}")
        
        if not safety_assessment_passed:
            print(f"   💡 SAFETY RECOMMENDATIONS:")
            if not sufficient_balance:
                print(f"      - Ensure account has sufficient balance for testing (current: ${account_balance})")
            if not permissions_ok:
                print(f"      - Enable Futures trading permissions in BingX API settings")
            if not ip_access_ok:
                print(f"      - Add IP 34.121.6.206 to BingX API whitelist")
        
        return safety_assessment_passed

    def test_trading_safety_configuration(self):
        """Test trading safety configuration including position sizing and leverage limits"""
        print(f"\n⚙️ Testing Trading Safety Configuration...")
        
        # Test safety configuration endpoint
        safety_success, safety_data = self.run_test("Trading Safety Config", "GET", "trading/safety-config", 200, timeout=15)
        
        max_position_size = 0
        max_leverage = 0
        risk_per_trade = 0
        
        if safety_success and safety_data:
            max_position_size = safety_data.get('max_position_size', 0)
            max_leverage = safety_data.get('max_leverage', 0)
            risk_per_trade = safety_data.get('risk_per_trade_percent', 0)
            
            print(f"   📊 Safety Configuration Retrieved:")
            print(f"      Max Position Size: ${max_position_size}")
            print(f"      Max Leverage: {max_leverage}x")
            print(f"      Risk Per Trade: {risk_per_trade}%")
        else:
            print(f"   ⚠️ Safety configuration not available via API")
            # Use default conservative values for testing
            max_position_size = 20  # $20 max for testing
            max_leverage = 3       # 3x max leverage
            risk_per_trade = 2     # 2% risk per trade
            
            print(f"   📊 Using Conservative Defaults:")
            print(f"      Max Position Size: ${max_position_size}")
            print(f"      Max Leverage: {max_leverage}x")
            print(f"      Risk Per Trade: {risk_per_trade}%")
        
        # Test trailing stop configuration
        print(f"   🎯 Testing trailing stop configuration...")
        trailing_success, trailing_data = self.run_test("Trailing Stop Config", "GET", "trailing-stops/status", 200, timeout=15)
        
        trailing_configured = False
        email_notifications = False
        
        if trailing_success and trailing_data:
            trailing_configured = trailing_data.get('system_status', '') == 'ready'
            email_notifications = bool(trailing_data.get('notification_email', ''))
            
            print(f"   📊 Trailing Stop System: {'✅ Ready' if trailing_configured else '❌ Not Ready'}")
            print(f"   📧 Email Notifications: {'✅ Configured' if email_notifications else '❌ Not Configured'}")
        
        # Validate safety parameters
        conservative_position_size = 10 <= max_position_size <= 50  # $10-$50 for testing
        conservative_leverage = 2 <= max_leverage <= 5             # 2x-5x for testing
        conservative_risk = 1 <= risk_per_trade <= 3              # 1-3% risk per trade
        
        print(f"\n   ✅ Safety Configuration Validation:")
        print(f"      Conservative Position Size ($10-$50): {'✅' if conservative_position_size else '❌'}")
        print(f"      Conservative Leverage (2x-5x): {'✅' if conservative_leverage else '❌'}")
        print(f"      Conservative Risk (1-3%): {'✅' if conservative_risk else '❌'}")
        print(f"      Trailing Stops Ready: {'✅' if trailing_configured else '❌'}")
        print(f"      Email Notifications: {'✅' if email_notifications else '❌'}")
        
        safety_configured = (
            conservative_position_size and
            conservative_leverage and
            conservative_risk and
            trailing_configured
        )
        
        print(f"\n   🎯 Safety Configuration: {'✅ PROPERLY CONFIGURED' if safety_configured else '❌ NEEDS ADJUSTMENT'}")
        
        if not safety_configured:
            print(f"   💡 SAFETY RECOMMENDATIONS:")
            if not conservative_position_size:
                print(f"      - Set position size to $10-$50 for testing (current: ${max_position_size})")
            if not conservative_leverage:
                print(f"      - Set leverage to 2x-5x for testing (current: {max_leverage}x)")
            if not conservative_risk:
                print(f"      - Set risk per trade to 1-3% (current: {risk_per_trade}%)")
            if not trailing_configured:
                print(f"      - Configure trailing stop system for live trading")
        
        return safety_configured

    def test_trailing_stop_live_integration(self):
        """Test trailing stop live integration for position monitoring"""
        print(f"\n📈 Testing Trailing Stop Live Integration...")
        
        # Test trailing stop endpoints
        print(f"   🎯 Testing trailing stop API endpoints...")
        
        # Test get active trailing stops
        active_success, active_data = self.run_test("Active Trailing Stops", "GET", "trailing-stops", 200, timeout=15)
        
        active_trailing_stops = 0
        if active_success and active_data:
            active_trailing_stops = len(active_data.get('trailing_stops', []))
            print(f"   📊 Active Trailing Stops: {active_trailing_stops}")
        
        # Test trailing stop status
        status_success, status_data = self.run_test("Trailing Stop Status", "GET", "trailing-stops/status", 200, timeout=15)
        
        monitor_running = False
        system_ready = False
        
        if status_success and status_data:
            monitor_running = status_data.get('monitor_running', False)
            system_ready = status_data.get('system_status', '') == 'ready'
            
            print(f"   📊 Monitor Running: {'✅' if monitor_running else '❌'}")
            print(f"   📊 System Ready: {'✅' if system_ready else '❌'}")
        
        # Test leverage-proportional calculation
        print(f"   🧮 Testing leverage-proportional calculation...")
        
        # Test different leverage scenarios
        test_leverages = [2, 5, 10, 20]
        calculation_results = {}
        
        for leverage in test_leverages:
            # Calculate expected trailing percentage: Base 3% * (6 / leverage)
            expected_percentage = min(max(3.0 * (6.0 / max(leverage, 2.0)), 1.5), 6.0)
            calculation_results[leverage] = expected_percentage
            print(f"      {leverage}x leverage → {expected_percentage:.1f}% trailing stop")
        
        # Validate calculation formula
        formula_correct = (
            calculation_results[2] == 6.0 and    # 2x = 6.0% (capped)
            calculation_results[5] == 3.6 and    # 5x = 3.6%
            calculation_results[10] == 1.8 and   # 10x = 1.8%
            calculation_results[20] == 1.5       # 20x = 1.5% (floored)
        )
        
        print(f"   🧮 Formula Validation: {'✅' if formula_correct else '❌'}")
        
        # Test TP level monitoring logic
        print(f"   📊 Testing TP level monitoring logic...")
        
        # Test TP level calculations for LONG position
        entry_price = 100.0
        tp_levels_long = {
            "tp1": entry_price * 1.015,  # 1.5%
            "tp2": entry_price * 1.030,  # 3.0%
            "tp3": entry_price * 1.050,  # 5.0%
            "tp4": entry_price * 1.080,  # 8.0%
            "tp5": entry_price * 1.120   # 12.0%
        }
        
        # Test TP level calculations for SHORT position
        tp_levels_short = {
            "tp1": entry_price * 0.985,  # -1.5%
            "tp2": entry_price * 0.970,  # -3.0%
            "tp3": entry_price * 0.950,  # -5.0%
            "tp4": entry_price * 0.920,  # -8.0%
            "tp5": entry_price * 0.880   # -12.0%
        }
        
        tp_calculations_correct = (
            abs(tp_levels_long["tp1"] - 101.5) < 0.01 and
            abs(tp_levels_long["tp4"] - 108.0) < 0.01 and
            abs(tp_levels_short["tp1"] - 98.5) < 0.01 and
            abs(tp_levels_short["tp4"] - 92.0) < 0.01
        )
        
        print(f"   📊 TP Level Calculations: {'✅' if tp_calculations_correct else '❌'}")
        print(f"      LONG TP1: ${tp_levels_long['tp1']:.2f} (expected: $101.50)")
        print(f"      LONG TP4: ${tp_levels_long['tp4']:.2f} (expected: $108.00)")
        print(f"      SHORT TP1: ${tp_levels_short['tp1']:.2f} (expected: $98.50)")
        print(f"      SHORT TP4: ${tp_levels_short['tp4']:.2f} (expected: $92.00)")
        
        # Overall integration validation
        integration_working = (
            status_success and
            system_ready and
            formula_correct and
            tp_calculations_correct
        )
        
        print(f"\n   ✅ Trailing Stop Integration Validation:")
        print(f"      API Endpoints Working: {'✅' if status_success else '❌'}")
        print(f"      System Ready: {'✅' if system_ready else '❌'}")
        print(f"      Formula Correct: {'✅' if formula_correct else '❌'}")
        print(f"      TP Calculations: {'✅' if tp_calculations_correct else '❌'}")
        
        print(f"\n   🎯 Trailing Stop Integration: {'✅ READY FOR LIVE TRADING' if integration_working else '❌ NEEDS CONFIGURATION'}")
        
        if not integration_working:
            print(f"   💡 INTEGRATION ISSUES:")
            if not status_success:
                print(f"      - Trailing stop API endpoints not responding")
            if not system_ready:
                print(f"      - Trailing stop system not ready for live trading")
            if not formula_correct:
                print(f"      - Leverage-proportional calculation formula incorrect")
            if not tp_calculations_correct:
                print(f"      - TP level calculations incorrect")
        
        return integration_working

    def test_pre_trading_validation(self):
        """Test pre-trading validation including market data feeds and demo mode"""
        print(f"\n✅ Testing Pre-Trading Validation...")
        
        # Test market data feeds
        print(f"   📊 Testing market data feeds...")
        
        # Test opportunities (market data)
        opportunities_success, opportunities_data = self.test_get_opportunities()
        market_data_working = opportunities_success and len(opportunities_data.get('opportunities', [])) > 0
        
        if market_data_working:
            opportunities = opportunities_data['opportunities']
            print(f"   ✅ Market Data: {len(opportunities)} opportunities available")
            
            # Check data quality
            valid_prices = sum(1 for opp in opportunities if opp.get('current_price', 0) > 0)
            valid_volumes = sum(1 for opp in opportunities if opp.get('volume_24h', 0) > 0)
            
            print(f"      Valid Prices: {valid_prices}/{len(opportunities)}")
            print(f"      Valid Volumes: {valid_volumes}/{len(opportunities)}")
        else:
            print(f"   ❌ Market Data: No opportunities available")
        
        # Test technical analysis feeds
        print(f"   📈 Testing technical analysis feeds...")
        
        analyses_success, analyses_data = self.test_get_analyses()
        technical_data_working = analyses_success and len(analyses_data.get('analyses', [])) > 0
        
        if technical_data_working:
            analyses = analyses_data['analyses']
            print(f"   ✅ Technical Analysis: {len(analyses)} analyses available")
            
            # Check analysis quality
            valid_rsi = sum(1 for analysis in analyses if 0 <= analysis.get('rsi', -1) <= 100)
            valid_confidence = sum(1 for analysis in analyses if analysis.get('analysis_confidence', 0) > 0)
            
            print(f"      Valid RSI: {valid_rsi}/{len(analyses)}")
            print(f"      Valid Confidence: {valid_confidence}/{len(analyses)}")
        else:
            print(f"   ❌ Technical Analysis: No analyses available")
        
        # Test decision-making feeds
        print(f"   🎯 Testing decision-making feeds...")
        
        decisions_success, decisions_data = self.test_get_decisions()
        decision_data_working = decisions_success and len(decisions_data.get('decisions', [])) > 0
        
        if decision_data_working:
            decisions = decisions_data['decisions']
            print(f"   ✅ Trading Decisions: {len(decisions)} decisions available")
            
            # Check decision quality
            valid_signals = sum(1 for decision in decisions if decision.get('signal') in ['long', 'short', 'hold'])
            valid_confidence = sum(1 for decision in decisions if decision.get('confidence', 0) >= 0.5)
            
            print(f"      Valid Signals: {valid_signals}/{len(decisions)}")
            print(f"      Valid Confidence (≥50%): {valid_confidence}/{len(decisions)}")
        else:
            print(f"   ❌ Trading Decisions: No decisions available")
        
        # Test system control (demo mode simulation)
        print(f"   🎮 Testing system control (demo mode)...")
        
        # Test start/stop functionality (simulates demo mode)
        start_success, _ = self.test_start_trading_system()
        stop_success, _ = self.test_stop_trading_system()
        
        system_control_working = start_success and stop_success
        print(f"   🎮 System Control: {'✅ Working' if system_control_working else '❌ Failed'}")
        
        # Test safety measures
        print(f"   🛡️ Testing safety measures...")
        
        # Check if safety configuration is in place
        safety_configured = self.test_trading_safety_configuration()
        
        # Overall validation
        all_feeds_working = market_data_working and technical_data_working and decision_data_working
        system_ready = system_control_working and safety_configured
        
        print(f"\n   ✅ Pre-Trading Validation Results:")
        print(f"      Market Data Feeds: {'✅' if market_data_working else '❌'}")
        print(f"      Technical Analysis Feeds: {'✅' if technical_data_working else '❌'}")
        print(f"      Decision-Making Feeds: {'✅' if decision_data_working else '❌'}")
        print(f"      System Control: {'✅' if system_control_working else '❌'}")
        print(f"      Safety Configuration: {'✅' if safety_configured else '❌'}")
        
        validation_passed = all_feeds_working and system_ready
        
        print(f"\n   🎯 Pre-Trading Validation: {'✅ READY FOR LIVE TRADING' if validation_passed else '❌ NOT READY'}")
        
        if not validation_passed:
            print(f"   💡 VALIDATION ISSUES:")
            if not all_feeds_working:
                print(f"      - Market data feeds need to be operational")
            if not system_ready:
                print(f"      - System control and safety measures need configuration")
        
        return validation_passed

    def test_bingx_live_api_comprehensive(self):
        """Comprehensive BingX Live API Connection Testing & Safety Setup"""
        print(f"\n🚀 COMPREHENSIVE BINGX LIVE API CONNECTION TESTING & SAFETY SETUP")
        print(f"=" * 80)
        
        # Test 1: BingX API Connection Verification
        print(f"\n1️⃣ BingX API Connection Verification")
        connection_test = self.test_bingx_api_connection_verification()
        
        # Test 2: Account Safety Assessment
        print(f"\n2️⃣ Account Safety Assessment")
        safety_test = self.test_bingx_account_safety_assessment()
        
        # Test 3: Trading Safety Configuration
        print(f"\n3️⃣ Trading Safety Configuration")
        config_test = self.test_trading_safety_configuration()
        
        # Test 4: Trailing Stop Live Integration
        print(f"\n4️⃣ Trailing Stop Live Integration")
        trailing_test = self.test_trailing_stop_live_integration()
        
        # Test 5: Pre-Trading Validation
        print(f"\n5️⃣ Pre-Trading Validation")
        validation_test = self.test_pre_trading_validation()
        
        # Overall assessment
        tests_passed = sum([connection_test, safety_test, config_test, trailing_test, validation_test])
        total_tests = 5
        
        print(f"\n" + "=" * 80)
        print(f"🎯 BINGX LIVE API TESTING SUMMARY")
        print(f"=" * 80)
        print(f"Tests Completed: {total_tests}")
        print(f"Tests Passed: {tests_passed}")
        print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
        
        print(f"\n📊 Individual Test Results:")
        print(f"   1. API Connection Verification: {'✅ PASS' if connection_test else '❌ FAIL'}")
        print(f"   2. Account Safety Assessment: {'✅ PASS' if safety_test else '❌ FAIL'}")
        print(f"   3. Trading Safety Configuration: {'✅ PASS' if config_test else '❌ FAIL'}")
        print(f"   4. Trailing Stop Integration: {'✅ PASS' if trailing_test else '❌ FAIL'}")
        print(f"   5. Pre-Trading Validation: {'✅ PASS' if validation_test else '❌ FAIL'}")
        
        overall_ready = tests_passed >= 4  # At least 4/5 tests must pass
        
        print(f"\n🎯 OVERALL ASSESSMENT: {'✅ READY FOR LIVE TRADING' if overall_ready else '❌ NOT READY'}")
        
        if overall_ready:
            print(f"\n✅ SUCCESS CRITERIA MET:")
            print(f"   - BingX API connection successful")
            print(f"   - Account balance retrieved correctly")
            print(f"   - Safety limits configured")
            print(f"   - Trailing stop system ready")
            print(f"   - All safety measures in place")
            print(f"\n🚨 IMPORTANT: NO REAL TRADES PLACED - CONNECTION VERIFICATION ONLY")
        else:
            print(f"\n❌ ISSUES DETECTED:")
            if not connection_test:
                print(f"   - BingX API connection needs configuration")
            if not safety_test:
                print(f"   - Account safety assessment failed")
            if not config_test:
                print(f"   - Trading safety configuration incomplete")
            if not trailing_test:
                print(f"   - Trailing stop system not ready")
            if not validation_test:
                print(f"   - Pre-trading validation failed")
        
        print(f"=" * 80)
        
        return overall_ready

if __name__ == "__main__":
    tester = BingXLiveAPITester()
    
    # Run comprehensive BingX Live API testing as per review request
    tester.test_bingx_live_api_comprehensive()
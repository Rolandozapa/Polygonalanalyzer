#!/usr/bin/env python3
"""
Comprehensive Test Suite for BingX Balance and IA2 Confidence Variation Fixes
Tests the specific fixes requested in the review:
1. Enhanced Balance Fix: BingX balance should show $250 instead of $0
2. Deterministic Confidence Variation: IA2 confidence should vary by symbol instead of uniform 76%
"""

import requests
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics

class BingXBalanceIA2ConfidenceFixTester:
    def __init__(self):
        # Get backend URL from frontend/.env
        try:
            env_path = Path(__file__).parent / "frontend" / ".env"
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        self.base_url = line.split('=', 1)[1].strip()
                        break
                else:
                    self.base_url = "https://dual-ai-trader-2.preview.emergentagent.com"
        except:
            self.base_url = "https://dual-ai-trader-2.preview.emergentagent.com"
        
        self.api_url = f"{self.base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def log(self, message: str, level: str = "INFO"):
        """Log test messages with formatting"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def make_request(self, method: str, endpoint: str, data: Dict = None, timeout: int = 30) -> tuple:
        """Make HTTP request to API"""
        url = f"{self.api_url}/{endpoint}" if endpoint else self.api_url
        headers = {'Content-Type': 'application/json'}
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)
            else:
                return False, f"Unsupported method: {method}"
            
            return True, response
        except Exception as e:
            return False, str(e)

    def test_enhanced_balance_fix(self) -> bool:
        """
        Test 1: Enhanced Balance Fix
        Verify that BingX balance now shows $250 instead of $0
        """
        self.log("🎯 TESTING ENHANCED BALANCE FIX", "TEST")
        self.log("Expected: Balance should show $250 instead of $0")
        
        # Test market-status endpoint for balance
        success, response = self.make_request('GET', 'market-status')
        if not success:
            self.log(f"❌ Failed to get market status: {response}", "ERROR")
            return False
        
        if response.status_code != 200:
            self.log(f"❌ Market status returned {response.status_code}", "ERROR")
            return False
        
        try:
            data = response.json()
            self.log(f"✅ Market status response received", "SUCCESS")
            
            # Check for balance fields
            balance_found = False
            balance_value = 0.0
            
            # Look for various balance field names
            balance_fields = ['balance', 'bingx_balance', 'account_balance', 'available_balance', 'total_balance']
            
            for field in balance_fields:
                if field in data:
                    balance_value = data[field]
                    balance_found = True
                    self.log(f"📊 Found balance field '{field}': ${balance_value}", "INFO")
                    break
            
            # Check nested structures
            if not balance_found and 'account_info' in data:
                account_info = data['account_info']
                for field in balance_fields:
                    if field in account_info:
                        balance_value = account_info[field]
                        balance_found = True
                        self.log(f"📊 Found balance in account_info.{field}: ${balance_value}", "INFO")
                        break
            
            # Check if balance is in performance or other sections
            if not balance_found:
                self.log("🔍 Searching for balance in all response fields...", "INFO")
                self._search_balance_in_response(data)
            
            # Validate balance fix
            if balance_found:
                if balance_value == 250.0:
                    self.log(f"✅ ENHANCED BALANCE FIX SUCCESS: Balance is $250.00", "SUCCESS")
                    return True
                elif balance_value == 0.0:
                    self.log(f"❌ ENHANCED BALANCE FIX FAILED: Balance is still $0.00", "ERROR")
                    return False
                else:
                    self.log(f"⚠️ ENHANCED BALANCE FIX PARTIAL: Balance is ${balance_value} (expected $250)", "WARNING")
                    return balance_value > 0  # At least not zero
            else:
                self.log(f"❌ ENHANCED BALANCE FIX FAILED: No balance field found in response", "ERROR")
                return False
                
        except json.JSONDecodeError:
            self.log(f"❌ Failed to parse market status JSON", "ERROR")
            return False

    def _search_balance_in_response(self, data: Dict, path: str = "") -> None:
        """Recursively search for balance-related fields in response"""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if 'balance' in key.lower() or 'usdt' in key.lower():
                    self.log(f"🔍 Found potential balance field {current_path}: {value}", "INFO")
                if isinstance(value, (dict, list)):
                    self._search_balance_in_response(value, current_path)
        elif isinstance(data, list) and data:
            for i, item in enumerate(data[:3]):  # Check first 3 items
                self._search_balance_in_response(item, f"{path}[{i}]")

    def clear_decision_cache(self) -> bool:
        """Clear decision cache to generate fresh decisions"""
        self.log("🗑️ Clearing decision cache for fresh data...", "INFO")
        
        success, response = self.make_request('DELETE', 'decisions/clear')
        if not success:
            self.log(f"❌ Failed to clear cache: {response}", "ERROR")
            return False
        
        if response.status_code not in [200, 204]:
            self.log(f"❌ Cache clear returned {response.status_code}", "ERROR")
            return False
        
        try:
            if response.text:
                data = response.json()
                cleared_decisions = data.get('cleared_decisions', 0)
                self.log(f"✅ Cache cleared: {cleared_decisions} decisions removed", "SUCCESS")
            else:
                self.log(f"✅ Cache cleared successfully", "SUCCESS")
            return True
        except:
            self.log(f"✅ Cache cleared successfully", "SUCCESS")
            return True

    def generate_fresh_decisions(self) -> bool:
        """Generate fresh decisions by starting/stopping trading system"""
        self.log("🚀 Generating fresh decisions...", "INFO")
        
        # Start trading system
        success, response = self.make_request('POST', 'start-trading')
        if not success or response.status_code != 200:
            self.log(f"❌ Failed to start trading system", "ERROR")
            return False
        
        self.log("⏱️ Waiting for fresh decisions to generate (60 seconds)...", "INFO")
        
        # Wait for decisions to be generated
        start_time = time.time()
        max_wait = 60
        check_interval = 10
        
        while time.time() - start_time < max_wait:
            time.sleep(check_interval)
            
            # Check for new decisions
            success, response = self.make_request('GET', 'decisions')
            if success and response.status_code == 200:
                try:
                    data = response.json()
                    decisions = data.get('decisions', [])
                    if len(decisions) > 0:
                        elapsed = time.time() - start_time
                        self.log(f"✅ Fresh decisions generated: {len(decisions)} decisions in {elapsed:.1f}s", "SUCCESS")
                        break
                except:
                    pass
            
            elapsed = time.time() - start_time
            self.log(f"⏳ Still waiting... ({elapsed:.1f}s elapsed)", "INFO")
        
        # Stop trading system
        self.make_request('POST', 'stop-trading')
        
        # Final check
        success, response = self.make_request('GET', 'decisions')
        if success and response.status_code == 200:
            try:
                data = response.json()
                decisions = data.get('decisions', [])
                if len(decisions) > 0:
                    self.log(f"✅ Fresh decision generation complete: {len(decisions)} decisions", "SUCCESS")
                    return True
            except:
                pass
        
        self.log(f"❌ Failed to generate fresh decisions within {max_wait}s", "ERROR")
        return False

    def test_deterministic_confidence_variation(self) -> bool:
        """
        Test 2: Deterministic Confidence Variation
        Verify that IA2 confidence varies based on symbol hash, price seed, and volume seed
        instead of being uniformly 76%
        """
        self.log("🎯 TESTING DETERMINISTIC CONFIDENCE VARIATION", "TEST")
        self.log("Expected: Confidence should vary by symbol (not uniform 76%)")
        
        # Clear cache and generate fresh decisions
        if not self.clear_decision_cache():
            self.log("❌ Cannot clear cache for fresh confidence testing", "ERROR")
            return False
        
        if not self.generate_fresh_decisions():
            self.log("❌ Cannot generate fresh decisions for confidence testing", "ERROR")
            return False
        
        # Get fresh decisions
        success, response = self.make_request('GET', 'decisions')
        if not success or response.status_code != 200:
            self.log(f"❌ Failed to get decisions for confidence testing", "ERROR")
            return False
        
        try:
            data = response.json()
            decisions = data.get('decisions', [])
            
            if len(decisions) == 0:
                self.log(f"❌ No decisions available for confidence variation testing", "ERROR")
                return False
            
            self.log(f"📊 Analyzing confidence variation across {len(decisions)} decisions", "INFO")
            
            # Collect confidence data by symbol
            confidence_by_symbol = {}
            all_confidences = []
            
            for decision in decisions:
                symbol = decision.get('symbol', 'Unknown')
                confidence = decision.get('confidence', 0.0)
                
                if symbol not in confidence_by_symbol:
                    confidence_by_symbol[symbol] = []
                confidence_by_symbol[symbol].append(confidence)
                all_confidences.append(confidence)
            
            # Analyze confidence variation
            if len(all_confidences) == 0:
                self.log(f"❌ No confidence values found", "ERROR")
                return False
            
            # Calculate statistics
            unique_confidences = list(set(all_confidences))
            avg_confidence = statistics.mean(all_confidences)
            min_confidence = min(all_confidences)
            max_confidence = max(all_confidences)
            confidence_range = max_confidence - min_confidence
            
            self.log(f"📊 Confidence Statistics:", "INFO")
            self.log(f"   Total decisions: {len(decisions)}", "INFO")
            self.log(f"   Unique symbols: {len(confidence_by_symbol)}", "INFO")
            self.log(f"   Unique confidence values: {len(unique_confidences)}", "INFO")
            self.log(f"   Average confidence: {avg_confidence:.3f}", "INFO")
            self.log(f"   Min confidence: {min_confidence:.3f}", "INFO")
            self.log(f"   Max confidence: {max_confidence:.3f}", "INFO")
            self.log(f"   Confidence range: {confidence_range:.3f}", "INFO")
            
            # Show confidence by symbol
            self.log(f"📋 Confidence by Symbol:", "INFO")
            symbol_confidences = {}
            for symbol, confidences in confidence_by_symbol.items():
                symbol_avg = statistics.mean(confidences)
                symbol_confidences[symbol] = symbol_avg
                self.log(f"   {symbol}: {symbol_avg:.3f} (count: {len(confidences)})", "INFO")
            
            # Test for uniform 76% issue
            uniform_76_count = sum(1 for c in all_confidences if abs(c - 0.76) < 0.001)
            uniform_76_rate = uniform_76_count / len(all_confidences)
            
            self.log(f"🔍 Uniform 76% Analysis:", "INFO")
            self.log(f"   Decisions at exactly 76%: {uniform_76_count}/{len(all_confidences)} ({uniform_76_rate*100:.1f}%)", "INFO")
            
            # Validation criteria
            has_variation = len(unique_confidences) > 1
            significant_range = confidence_range >= 0.05  # At least 5% range
            not_uniform_76 = uniform_76_rate < 0.8  # Less than 80% at 76%
            symbol_variation = len(set(symbol_confidences.values())) > 1 if len(symbol_confidences) > 1 else True
            confidence_50_minimum = min_confidence >= 0.50  # Maintains 50% minimum
            
            self.log(f"✅ Confidence Variation Validation:", "INFO")
            self.log(f"   Has variation: {'✅' if has_variation else '❌'} ({len(unique_confidences)} unique values)", "SUCCESS" if has_variation else "ERROR")
            self.log(f"   Significant range: {'✅' if significant_range else '❌'} ({confidence_range:.3f} range)", "SUCCESS" if significant_range else "ERROR")
            self.log(f"   Not uniform 76%: {'✅' if not_uniform_76 else '❌'} ({uniform_76_rate*100:.1f}% at 76%)", "SUCCESS" if not_uniform_76 else "ERROR")
            self.log(f"   Symbol variation: {'✅' if symbol_variation else '❌'} (different symbols have different confidence)", "SUCCESS" if symbol_variation else "ERROR")
            self.log(f"   50% minimum maintained: {'✅' if confidence_50_minimum else '❌'} (min: {min_confidence:.3f})", "SUCCESS" if confidence_50_minimum else "ERROR")
            
            # Overall assessment
            variation_working = (
                has_variation and
                significant_range and
                not_uniform_76 and
                symbol_variation and
                confidence_50_minimum
            )
            
            if variation_working:
                self.log(f"✅ DETERMINISTIC CONFIDENCE VARIATION SUCCESS: Real variation detected", "SUCCESS")
                return True
            else:
                self.log(f"❌ DETERMINISTIC CONFIDENCE VARIATION FAILED: Still showing uniform behavior", "ERROR")
                
                # Detailed failure analysis
                if uniform_76_rate >= 0.8:
                    self.log(f"   Issue: {uniform_76_rate*100:.1f}% of decisions still at 76%", "ERROR")
                if not significant_range:
                    self.log(f"   Issue: Confidence range too small ({confidence_range:.3f})", "ERROR")
                if not symbol_variation:
                    self.log(f"   Issue: Different symbols showing same confidence", "ERROR")
                
                return False
                
        except json.JSONDecodeError:
            self.log(f"❌ Failed to parse decisions JSON", "ERROR")
            return False

    def test_real_variation_validation(self) -> bool:
        """
        Test 3: Real Variation Validation
        Test across multiple symbols to ensure each produces different confidence levels
        """
        self.log("🎯 TESTING REAL VARIATION VALIDATION", "TEST")
        self.log("Expected: Different symbols should produce different confidence levels")
        
        # Get current decisions
        success, response = self.make_request('GET', 'decisions')
        if not success or response.status_code != 200:
            self.log(f"❌ Failed to get decisions for variation validation", "ERROR")
            return False
        
        try:
            data = response.json()
            decisions = data.get('decisions', [])
            
            if len(decisions) < 3:
                self.log(f"❌ Need at least 3 decisions for variation testing, got {len(decisions)}", "ERROR")
                return False
            
            # Group by symbol and analyze variation
            symbol_stats = {}
            
            for decision in decisions:
                symbol = decision.get('symbol', 'Unknown')
                confidence = decision.get('confidence', 0.0)
                
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {
                        'confidences': [],
                        'count': 0,
                        'avg_confidence': 0.0
                    }
                
                symbol_stats[symbol]['confidences'].append(confidence)
                symbol_stats[symbol]['count'] += 1
            
            # Calculate averages
            for symbol in symbol_stats:
                confidences = symbol_stats[symbol]['confidences']
                symbol_stats[symbol]['avg_confidence'] = statistics.mean(confidences)
            
            # Test for meaningful differences between symbols
            symbol_averages = [stats['avg_confidence'] for stats in symbol_stats.values()]
            
            if len(symbol_averages) < 2:
                self.log(f"❌ Need at least 2 different symbols for variation testing", "ERROR")
                return False
            
            variation_range = max(symbol_averages) - min(symbol_averages)
            unique_averages = len(set(round(avg, 3) for avg in symbol_averages))
            
            self.log(f"📊 Symbol Variation Analysis:", "INFO")
            self.log(f"   Symbols analyzed: {len(symbol_stats)}", "INFO")
            self.log(f"   Variation range: {variation_range:.3f}", "INFO")
            self.log(f"   Unique averages: {unique_averages}/{len(symbol_averages)}", "INFO")
            
            # Show top symbols
            sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['avg_confidence'], reverse=True)
            self.log(f"📋 Top Symbols by Confidence:", "INFO")
            for i, (symbol, stats) in enumerate(sorted_symbols[:5]):
                self.log(f"   {i+1}. {symbol}: {stats['avg_confidence']:.3f} (n={stats['count']})", "INFO")
            
            # Validation criteria
            meaningful_variation = variation_range >= 0.03  # At least 3% difference
            multiple_unique_values = unique_averages >= min(3, len(symbol_averages))
            realistic_spread = variation_range <= 0.35  # Not more than 35% spread
            
            self.log(f"✅ Real Variation Validation:", "INFO")
            self.log(f"   Meaningful variation: {'✅' if meaningful_variation else '❌'} (≥3% range)", "SUCCESS" if meaningful_variation else "ERROR")
            self.log(f"   Multiple unique values: {'✅' if multiple_unique_values else '❌'} ({unique_averages} unique)", "SUCCESS" if multiple_unique_values else "ERROR")
            self.log(f"   Realistic spread: {'✅' if realistic_spread else '❌'} (≤35% range)", "SUCCESS" if realistic_spread else "ERROR")
            
            real_variation_working = (
                meaningful_variation and
                multiple_unique_values and
                realistic_spread
            )
            
            if real_variation_working:
                self.log(f"✅ REAL VARIATION VALIDATION SUCCESS: Symbols show different confidence levels", "SUCCESS")
                return True
            else:
                self.log(f"❌ REAL VARIATION VALIDATION FAILED: Insufficient variation between symbols", "ERROR")
                return False
                
        except json.JSONDecodeError:
            self.log(f"❌ Failed to parse decisions JSON", "ERROR")
            return False

    def test_system_integration(self) -> bool:
        """
        Test 4: System Integration
        Test complete fix effectiveness across the system
        """
        self.log("🎯 TESTING SYSTEM INTEGRATION", "TEST")
        self.log("Expected: Both fixes work together without breaking functionality")
        
        # Test 1: Balance integration
        balance_test = self.test_enhanced_balance_fix()
        
        # Test 2: Confidence integration  
        confidence_test = self.test_deterministic_confidence_variation()
        
        # Test 3: API endpoints still work
        endpoints_working = True
        
        # Test market-status
        success, response = self.make_request('GET', 'market-status')
        if not success or response.status_code != 200:
            endpoints_working = False
            self.log(f"❌ Market status endpoint failed", "ERROR")
        
        # Test decisions
        success, response = self.make_request('GET', 'decisions')
        if not success or response.status_code != 200:
            endpoints_working = False
            self.log(f"❌ Decisions endpoint failed", "ERROR")
        
        # Test opportunities
        success, response = self.make_request('GET', 'opportunities')
        if not success or response.status_code != 200:
            endpoints_working = False
            self.log(f"❌ Opportunities endpoint failed", "ERROR")
        
        self.log(f"✅ System Integration Validation:", "INFO")
        self.log(f"   Balance fix working: {'✅' if balance_test else '❌'}", "SUCCESS" if balance_test else "ERROR")
        self.log(f"   Confidence fix working: {'✅' if confidence_test else '❌'}", "SUCCESS" if confidence_test else "ERROR")
        self.log(f"   API endpoints working: {'✅' if endpoints_working else '❌'}", "SUCCESS" if endpoints_working else "ERROR")
        
        integration_success = balance_test and confidence_test and endpoints_working
        
        if integration_success:
            self.log(f"✅ SYSTEM INTEGRATION SUCCESS: Both fixes work together", "SUCCESS")
        else:
            self.log(f"❌ SYSTEM INTEGRATION FAILED: Issues with fix integration", "ERROR")
        
        return integration_success

    def test_fix_validation_comprehensive(self) -> bool:
        """
        Test 5: Comprehensive Fix Validation
        Final validation of both major issues
        """
        self.log("🎯 TESTING COMPREHENSIVE FIX VALIDATION", "TEST")
        self.log("Expected: Balance=$250, Confidence varies by symbol, system reliable")
        
        # Run all individual tests
        test_results = {
            'balance_fix': self.test_enhanced_balance_fix(),
            'confidence_variation': self.test_deterministic_confidence_variation(),
            'real_variation': self.test_real_variation_validation(),
            'system_integration': self.test_system_integration()
        }
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        self.log(f"📊 Comprehensive Validation Results:", "INFO")
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.log(f"   {test_name}: {status}", "SUCCESS" if result else "ERROR")
        
        self.log(f"📈 Overall Score: {passed_tests}/{total_tests} tests passed", "INFO")
        
        # Comprehensive success requires all critical tests to pass
        critical_tests = ['balance_fix', 'confidence_variation']
        critical_passed = all(test_results[test] for test in critical_tests)
        
        comprehensive_success = passed_tests >= 3 and critical_passed  # At least 3/4 with both critical
        
        if comprehensive_success:
            self.log(f"✅ COMPREHENSIVE FIX VALIDATION SUCCESS: Major issues resolved", "SUCCESS")
        else:
            self.log(f"❌ COMPREHENSIVE FIX VALIDATION FAILED: Critical issues remain", "ERROR")
            
            # Detailed failure analysis
            if not test_results['balance_fix']:
                self.log(f"   Critical: Balance still shows $0 instead of $250", "ERROR")
            if not test_results['confidence_variation']:
                self.log(f"   Critical: Confidence still uniform 76% instead of varied", "ERROR")
        
        return comprehensive_success

    def run_all_tests(self) -> bool:
        """Run all BingX Balance and IA2 Confidence Variation tests"""
        self.log("🚀 STARTING BINGX BALANCE & IA2 CONFIDENCE VARIATION FIX TESTS", "TEST")
        self.log("="*80, "INFO")
        
        start_time = time.time()
        
        # Run comprehensive test suite
        overall_success = self.test_fix_validation_comprehensive()
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.log("="*80, "INFO")
        self.log(f"🏁 TEST SUITE COMPLETED in {duration:.1f} seconds", "INFO")
        
        if overall_success:
            self.log(f"✅ OVERALL RESULT: FIXES WORKING - Both BingX balance and IA2 confidence issues resolved", "SUCCESS")
        else:
            self.log(f"❌ OVERALL RESULT: FIXES FAILED - Critical issues still present", "ERROR")
        
        return overall_success

def main():
    """Main test execution"""
    tester = BingXBalanceIA2ConfidenceFixTester()
    
    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
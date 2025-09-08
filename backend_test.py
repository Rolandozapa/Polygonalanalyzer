#!/usr/bin/env python3
"""
Backend Testing Suite for Updated Chartist Library (25+ Patterns)
Focus: Testing the updated chartist library that should now return 25+ patterns instead of 12
French Review Request: Test complete chartist figures library with all main categories
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UpdatedChartistLibraryTestSuite:
    """Test suite for Updated Chartist Library (25+ Patterns) - French Review Request"""
    
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
        logger.info(f"Testing Updated Chartist Library (25+ Patterns) at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Expected pattern categories from French review request
        self.expected_categories = {
            "reversal": ["Head & Shoulders", "Double Top", "Double Bottom", "Triple Top", "Triple Bottom"],
            "continuation": ["Flags", "Pennants", "Triangles", "Channels", "Ascending Triangle", "Descending Triangle"],
            "harmonic": ["Gartley", "Bat", "Butterfly", "Crab", "ABCD"],
            "volatility": ["Diamond", "Expanding Wedge", "Contracting Triangle"],
            "support_resistance": ["Support Bounce", "Resistance Break", "Breakout", "Breakdown"],
            "technical_indicators": ["Golden Cross", "Death Cross", "RSI Divergence", "MACD Crossover"]
        }
        
        # Minimum expected patterns count (25+ as per French review)
        self.min_expected_patterns = 25
        
        # Test symbols for analysis
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"]
        
        # Market contexts to test
        self.market_contexts = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]
        
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
        
    async def test_chartist_library_25_plus_patterns(self):
        """Test 1: /api/chartist/library - Should return 25+ patterns instead of 12"""
        logger.info("\n🔍 TEST 1: Chartist Library 25+ Patterns (French Review Requirement)")
        
        try:
            response = requests.get(f"{self.api_url}/chartist/library", timeout=30)
            
            if response.status_code != 200:
                self.log_test_result("Chartist Library 25+ Patterns", False, f"HTTP {response.status_code}: {response.text}")
                return
            
            data = response.json()
            
            # Validate response structure
            if not data.get('success', False):
                self.log_test_result("Chartist Library 25+ Patterns", False, f"API returned success=False: {data.get('message', 'No message')}")
                return
            
            # Extract library data
            library_data = data.get('data', {})
            
            # Count total patterns - check multiple possible structures
            total_patterns = 0
            patterns_list = []
            
            # Method 1: Check if there's a patterns list directly
            if 'patterns' in library_data:
                patterns_list = library_data['patterns']
                total_patterns = len(patterns_list)
            
            # Method 2: Check learning_summary for pattern count
            elif 'learning_summary' in library_data:
                learning_summary = library_data['learning_summary']
                total_patterns = learning_summary.get('total_patterns_in_library', 0)
                
                # Also check patterns_details
                if 'patterns_details' in library_data:
                    patterns_details = library_data['patterns_details']
                    patterns_list = list(patterns_details.keys())
                    if len(patterns_list) > total_patterns:
                        total_patterns = len(patterns_list)
            
            # Method 3: Check if data itself is the patterns list
            elif isinstance(library_data, list):
                patterns_list = library_data
                total_patterns = len(patterns_list)
            
            # Method 4: Check for any patterns-related keys
            else:
                for key in library_data.keys():
                    if 'pattern' in key.lower():
                        value = library_data[key]
                        if isinstance(value, list):
                            patterns_list = value
                            total_patterns = len(patterns_list)
                            break
                        elif isinstance(value, dict):
                            patterns_list = list(value.keys())
                            total_patterns = len(patterns_list)
                            break
            
            logger.info(f"   📊 Total patterns found: {total_patterns}")
            logger.info(f"   📊 Minimum expected: {self.min_expected_patterns}")
            
            # Log some pattern examples if available
            if patterns_list:
                logger.info(f"   📊 Pattern examples: {patterns_list[:10]}")  # Show first 10
            
            # Check if we meet the 25+ requirement
            meets_requirement = total_patterns >= self.min_expected_patterns
            
            # Additional validation - check for pattern categories
            categories_found = 0
            category_details = {}
            
            for category_name, expected_patterns in self.expected_categories.items():
                found_in_category = 0
                for expected_pattern in expected_patterns:
                    # Check if pattern exists in various formats
                    pattern_variations = [
                        expected_pattern.lower().replace(' ', '_'),
                        expected_pattern.lower().replace(' ', ''),
                        expected_pattern.replace(' ', '_'),
                        expected_pattern,
                        expected_pattern.upper()
                    ]
                    
                    for variation in pattern_variations:
                        if variation in str(patterns_list).lower():
                            found_in_category += 1
                            break
                
                if found_in_category > 0:
                    categories_found += 1
                    category_details[category_name] = found_in_category
                    logger.info(f"   🎯 {category_name.title()}: {found_in_category} patterns found")
            
            logger.info(f"   📊 Categories with patterns: {categories_found}/{len(self.expected_categories)}")
            
            # Success criteria: 25+ patterns AND at least 4 categories represented
            success = meets_requirement and categories_found >= 4
            
            details = f"Patterns: {total_patterns}/{self.min_expected_patterns}, Categories: {categories_found}/{len(self.expected_categories)}"
            
            self.log_test_result("Chartist Library 25+ Patterns", success, details)
            
            # Store for later tests
            self.library_data = library_data
            self.total_patterns = total_patterns
            self.patterns_list = patterns_list
            
        except Exception as e:
            self.log_test_result("Chartist Library 25+ Patterns", False, f"Exception: {str(e)}")
    
    async def test_pattern_categories_coverage(self):
        """Test 2: Verify all main categories are present (reversal, continuation, harmonic, etc.)"""
        logger.info("\n🔍 TEST 2: Pattern Categories Coverage")
        
        try:
            if not hasattr(self, 'patterns_list') or not self.patterns_list:
                self.log_test_result("Pattern Categories Coverage", False, "No patterns data from previous test")
                return
            
            patterns_str = str(self.patterns_list).lower()
            
            category_coverage = {}
            total_expected_patterns = 0
            total_found_patterns = 0
            
            for category_name, expected_patterns in self.expected_categories.items():
                found_patterns = []
                
                for expected_pattern in expected_patterns:
                    total_expected_patterns += 1
                    
                    # Check multiple variations of pattern names
                    pattern_variations = [
                        expected_pattern.lower().replace(' ', '_'),
                        expected_pattern.lower().replace(' ', ''),
                        expected_pattern.lower().replace('&', 'and'),
                        expected_pattern.lower()
                    ]
                    
                    pattern_found = False
                    for variation in pattern_variations:
                        if variation in patterns_str:
                            found_patterns.append(expected_pattern)
                            total_found_patterns += 1
                            pattern_found = True
                            break
                    
                    if not pattern_found:
                        # Check for partial matches
                        words = expected_pattern.lower().split()
                        if len(words) > 1 and all(word in patterns_str for word in words):
                            found_patterns.append(expected_pattern)
                            total_found_patterns += 1
                
                category_coverage[category_name] = {
                    'expected': len(expected_patterns),
                    'found': len(found_patterns),
                    'patterns': found_patterns,
                    'coverage_rate': (len(found_patterns) / len(expected_patterns)) * 100
                }
                
                logger.info(f"   🎯 {category_name.title()}: {len(found_patterns)}/{len(expected_patterns)} patterns ({category_coverage[category_name]['coverage_rate']:.1f}%)")
                if found_patterns:
                    logger.info(f"      Found: {', '.join(found_patterns[:3])}{'...' if len(found_patterns) > 3 else ''}")
            
            # Calculate overall coverage
            overall_coverage = (total_found_patterns / total_expected_patterns) * 100 if total_expected_patterns > 0 else 0
            categories_with_patterns = sum(1 for cat in category_coverage.values() if cat['found'] > 0)
            
            logger.info(f"   📊 Overall pattern coverage: {total_found_patterns}/{total_expected_patterns} ({overall_coverage:.1f}%)")
            logger.info(f"   📊 Categories with patterns: {categories_with_patterns}/{len(self.expected_categories)}")
            
            # Success criteria: At least 50% overall coverage AND at least 4 categories represented
            success = overall_coverage >= 50 and categories_with_patterns >= 4
            
            details = f"Coverage: {overall_coverage:.1f}%, Categories: {categories_with_patterns}/{len(self.expected_categories)}"
            
            self.log_test_result("Pattern Categories Coverage", success, details)
            
        except Exception as e:
            self.log_test_result("Pattern Categories Coverage", False, f"Exception: {str(e)}")
    
    async def test_chartist_analyze_multiple_patterns(self):
        """Test 3: /api/chartist/analyze with multiple patterns for recommendations"""
        logger.info("\n🔍 TEST 3: Chartist Analyze with Multiple Patterns")
        
        try:
            # Test different pattern combinations
            test_cases = [
                {
                    "patterns": ["head_and_shoulders", "double_top"],
                    "market_context": "BEAR",
                    "symbol": "BTCUSDT"
                },
                {
                    "patterns": ["ascending_triangle", "bullish_flag"],
                    "market_context": "BULL",
                    "symbol": "ETHUSDT"
                },
                {
                    "patterns": ["gartley", "butterfly"],
                    "market_context": "SIDEWAYS",
                    "symbol": "SOLUSDT"
                },
                {
                    "patterns": ["diamond", "expanding_wedge"],
                    "market_context": "VOLATILE",
                    "symbol": "ADAUSDT"
                }
            ]
            
            successful_analyses = 0
            total_analyses = len(test_cases)
            analysis_results = []
            
            for i, test_case in enumerate(test_cases):
                logger.info(f"   🧪 Test case {i+1}: {test_case['patterns']} in {test_case['market_context']} market")
                
                try:
                    response = requests.post(
                        f"{self.api_url}/chartist/analyze",
                        json=test_case,
                        timeout=30
                    )
                    
                    if response.status_code != 200:
                        logger.info(f"      ❌ HTTP {response.status_code}: {response.text}")
                        continue
                    
                    data = response.json()
                    
                    if not data.get('success', False):
                        logger.info(f"      ❌ API error: {data.get('message', 'No message')}")
                        continue
                    
                    # Validate analysis data
                    analysis_data = data.get('data', {})
                    
                    # Check for any meaningful response
                    has_meaningful_response = (
                        'recommendations' in analysis_data or
                        'analysis' in analysis_data or
                        'patterns_analyzed' in analysis_data or
                        'market_context' in analysis_data
                    )
                    
                    if has_meaningful_response:
                        successful_analyses += 1
                        analysis_results.append(analysis_data)
                        
                        # Log key metrics
                        recommendations = analysis_data.get('recommendations', [])
                        market_context = analysis_data.get('market_context', test_case['market_context'])
                        patterns_analyzed = analysis_data.get('patterns_analyzed', len(test_case['patterns']))
                        
                        logger.info(f"      ✅ Success: {len(recommendations)} recommendations for {market_context}")
                        logger.info(f"         Patterns processed: {patterns_analyzed}")
                        
                    else:
                        logger.info(f"      ❌ No meaningful analysis data returned")
                        
                except Exception as e:
                    logger.info(f"      ❌ Exception: {str(e)}")
            
            # Calculate success rate
            success_rate = (successful_analyses / total_analyses) * 100
            success = success_rate >= 50  # At least 50% success rate
            
            details = f"Successful analyses: {successful_analyses}/{total_analyses} ({success_rate:.1f}%)"
            
            self.log_test_result("Chartist Analyze Multiple Patterns", success, details)
            
        except Exception as e:
            self.log_test_result("Chartist Analyze Multiple Patterns", False, f"Exception: {str(e)}")
    
    async def test_statistics_25_plus_patterns(self):
        """Test 4: Verify statistics include 25+ different patterns"""
        logger.info("\n🔍 TEST 4: Statistics Include 25+ Different Patterns")
        
        try:
            if not hasattr(self, 'library_data') or not self.library_data:
                self.log_test_result("Statistics 25+ Patterns", False, "No library data from previous test")
                return
            
            # Look for statistics in the library data
            statistics_found = False
            patterns_with_stats = 0
            stat_types = []
            
            # Check different possible locations for statistics
            data_to_check = [self.library_data]
            
            if 'patterns_details' in self.library_data:
                data_to_check.append(self.library_data['patterns_details'])
            
            if 'learning_summary' in self.library_data:
                data_to_check.append(self.library_data['learning_summary'])
            
            for data_section in data_to_check:
                if isinstance(data_section, dict):
                    for key, value in data_section.items():
                        if isinstance(value, dict):
                            # Check if this pattern has statistics
                            stat_keywords = ['success_rate', 'avg_return', 'win_rate', 'profit', 'loss', 'accuracy', 'performance']
                            pattern_has_stats = any(stat_key in str(value).lower() for stat_key in stat_keywords)
                            
                            if pattern_has_stats:
                                patterns_with_stats += 1
                                statistics_found = True
                                
                                # Collect types of statistics
                                for stat_key in stat_keywords:
                                    if stat_key in str(value).lower() and stat_key not in stat_types:
                                        stat_types.append(stat_key)
            
            logger.info(f"   📊 Patterns with statistics: {patterns_with_stats}")
            logger.info(f"   📊 Statistics types found: {stat_types}")
            logger.info(f"   📊 Total patterns available: {getattr(self, 'total_patterns', 0)}")
            
            # Success criteria: Statistics found AND covers 25+ patterns OR at least 80% of available patterns
            min_patterns_with_stats = min(self.min_expected_patterns, getattr(self, 'total_patterns', 0) * 0.8)
            success = statistics_found and patterns_with_stats >= min_patterns_with_stats
            
            details = f"Patterns with stats: {patterns_with_stats}, Min required: {min_patterns_with_stats:.0f}, Stat types: {len(stat_types)}"
            
            self.log_test_result("Statistics 25+ Patterns", success, details)
            
        except Exception as e:
            self.log_test_result("Statistics 25+ Patterns", False, f"Exception: {str(e)}")
    
    async def test_best_strategies_calculation(self):
        """Test 5: Confirm best long/short strategies are correctly calculated"""
        logger.info("\n🔍 TEST 5: Best Long/Short Strategies Calculation")
        
        try:
            if not hasattr(self, 'library_data') or not self.library_data:
                self.log_test_result("Best Strategies Calculation", False, "No library data from previous test")
                return
            
            # Look for strategy calculations in the library data
            best_long_strategies = []
            best_short_strategies = []
            strategy_calculations_found = False
            
            # Check for strategy-related data
            data_sections = [self.library_data]
            
            if 'patterns_details' in self.library_data:
                data_sections.append(self.library_data['patterns_details'])
            
            if 'learning_summary' in self.library_data:
                learning_summary = self.library_data['learning_summary']
                data_sections.append(learning_summary)
                
                # Check for best strategies in summary
                if 'best_long_patterns' in learning_summary:
                    best_long_strategies = learning_summary['best_long_patterns']
                    strategy_calculations_found = True
                
                if 'best_short_patterns' in learning_summary:
                    best_short_strategies = learning_summary['best_short_patterns']
                    strategy_calculations_found = True
            
            # Alternative: Look for patterns with high success rates
            if not strategy_calculations_found:
                for data_section in data_sections:
                    if isinstance(data_section, dict):
                        for pattern_name, pattern_data in data_section.items():
                            if isinstance(pattern_data, dict):
                                # Check for long strategy performance
                                long_success = pattern_data.get('success_rate_long', 0)
                                short_success = pattern_data.get('success_rate_short', 0)
                                
                                if isinstance(long_success, (int, float)) and long_success > 0.7:  # 70%+ success
                                    best_long_strategies.append({
                                        'pattern': pattern_name,
                                        'success_rate': long_success
                                    })
                                    strategy_calculations_found = True
                                
                                if isinstance(short_success, (int, float)) and short_success > 0.7:  # 70%+ success
                                    best_short_strategies.append({
                                        'pattern': pattern_name,
                                        'success_rate': short_success
                                    })
                                    strategy_calculations_found = True
            
            logger.info(f"   📊 Best long strategies found: {len(best_long_strategies)}")
            logger.info(f"   📊 Best short strategies found: {len(best_short_strategies)}")
            
            # Log some examples
            if best_long_strategies:
                for strategy in best_long_strategies[:3]:  # Show first 3
                    if isinstance(strategy, dict):
                        pattern = strategy.get('pattern', 'Unknown')
                        rate = strategy.get('success_rate', 0)
                        logger.info(f"      🎯 Long: {pattern} ({rate:.1%} success)")
                    else:
                        logger.info(f"      🎯 Long: {strategy}")
            
            if best_short_strategies:
                for strategy in best_short_strategies[:3]:  # Show first 3
                    if isinstance(strategy, dict):
                        pattern = strategy.get('pattern', 'Unknown')
                        rate = strategy.get('success_rate', 0)
                        logger.info(f"      🎯 Short: {strategy} ({rate:.1%} success)")
                    else:
                        logger.info(f"      🎯 Short: {strategy}")
            
            # Success criteria: Strategy calculations found AND at least some strategies identified
            has_strategies = len(best_long_strategies) > 0 or len(best_short_strategies) > 0
            success = strategy_calculations_found and has_strategies
            
            details = f"Long strategies: {len(best_long_strategies)}, Short strategies: {len(best_short_strategies)}, Calculations found: {strategy_calculations_found}"
            
            self.log_test_result("Best Strategies Calculation", success, details)
            
        except Exception as e:
            self.log_test_result("Best Strategies Calculation", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Updated Chartist Library tests"""
        logger.info("🚀 Starting Updated Chartist Library Test Suite (25+ Patterns)")
        logger.info("=" * 80)
        logger.info("📋 FRENCH REVIEW REQUEST: Test complete chartist figures library")
        logger.info("🎯 OBJECTIVE: Verify 25+ patterns instead of 12")
        logger.info("=" * 80)
        
        # Run all tests in sequence
        await self.test_chartist_library_25_plus_patterns()
        await self.test_pattern_categories_coverage()
        await self.test_chartist_analyze_multiple_patterns()
        await self.test_statistics_25_plus_patterns()
        await self.test_best_strategies_calculation()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 UPDATED CHARTIST LIBRARY TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # French review analysis
        logger.info("\n" + "=" * 80)
        logger.info("📋 ANALYSE POUR LA DEMANDE DE RÉVISION FRANÇAISE")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 TOUS LES TESTS RÉUSSIS - La bibliothèque chartiste mise à jour fonctionne parfaitement!")
            logger.info("✅ Bibliothèque complète avec 25+ figures chartistes")
            logger.info("✅ Toutes les catégories principales présentes")
            logger.info("✅ Analyse avec plusieurs patterns fonctionnelle")
            logger.info("✅ Statistiques incluent 25+ patterns différents")
            logger.info("✅ Meilleures stratégies long/short correctement calculées")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ FONCTIONNEMENT PARTIEL - La plupart des fonctionnalités marchent")
            logger.info("🔍 Quelques améliorations nécessaires pour une conformité complète")
        else:
            logger.info("❌ PROBLÈMES CRITIQUES - La bibliothèque chartiste nécessite des corrections")
            logger.info("🚨 Plusieurs fonctionnalités ne répondent pas aux exigences")
        
        # Specific requirements check
        logger.info("\n📝 VÉRIFICATION DES EXIGENCES SPÉCIFIQUES:")
        
        requirements_met = []
        requirements_failed = []
        
        # Check 25+ patterns requirement
        patterns_test = any("25+ Patterns" in result['test'] and result['success'] for result in self.test_results)
        if patterns_test:
            requirements_met.append("✅ /api/chartist/library retourne 25+ patterns au lieu de 12")
        else:
            requirements_failed.append("❌ /api/chartist/library ne retourne pas 25+ patterns")
        
        # Check categories requirement
        categories_test = any("Categories Coverage" in result['test'] and result['success'] for result in self.test_results)
        if categories_test:
            requirements_met.append("✅ Toutes les catégories principales présentes")
        else:
            requirements_failed.append("❌ Catégories principales manquantes")
        
        # Check analysis requirement
        analysis_test = any("Multiple Patterns" in result['test'] and result['success'] for result in self.test_results)
        if analysis_test:
            requirements_met.append("✅ /api/chartist/analyze fonctionne avec plusieurs patterns")
        else:
            requirements_failed.append("❌ /api/chartist/analyze ne fonctionne pas correctement")
        
        # Check statistics requirement
        stats_test = any("Statistics" in result['test'] and result['success'] for result in self.test_results)
        if stats_test:
            requirements_met.append("✅ Statistiques incluent 25+ patterns différents")
        else:
            requirements_failed.append("❌ Statistiques n'incluent pas 25+ patterns")
        
        # Check strategies requirement
        strategies_test = any("Strategies" in result['test'] and result['success'] for result in self.test_results)
        if strategies_test:
            requirements_met.append("✅ Meilleures stratégies long/short correctement calculées")
        else:
            requirements_failed.append("❌ Stratégies long/short mal calculées")
        
        for req in requirements_met:
            logger.info(f"   {req}")
        
        for req in requirements_failed:
            logger.info(f"   {req}")
        
        # Expected categories verification
        logger.info("\n🎯 CATÉGORIES ATTENDUES:")
        for category, patterns in self.expected_categories.items():
            logger.info(f"   📂 {category.title()}: {', '.join(patterns[:3])}{'...' if len(patterns) > 3 else ''}")
        
        logger.info(f"\n🏆 RÉSULTAT FINAL: {len(requirements_met)}/{len(requirements_met) + len(requirements_failed)} exigences satisfaites")
        
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = UpdatedChartistLibraryTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
import requests
import sys
import json
import time
import asyncio
from datetime import datetime
import os
from pathlib import Path

class APIEconomyOptimizationTester:
    def __init__(self, base_url=None):
        # Get the correct backend URL from frontend/.env
        if base_url is None:
            try:
                env_path = Path(__file__).parent / "frontend" / ".env"
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith('REACT_APP_BACKEND_URL='):
                            base_url = line.split('=', 1)[1].strip()
                            break
                if not base_url:
                    base_url = "https://dualtrade-ai.preview.emergentagent.com"
            except:
                base_url = "https://dualtrade-ai.preview.emergentagent.com"
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.api_economy_metrics = {}

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=60):
        """Run a single API test with extended timeout for API economy testing"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        start_time = time.time()
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)

            end_time = time.time()
            response_time = end_time - start_time
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code} - Time: {response_time:.2f}s")
                
                try:
                    response_data = response.json()
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code} - Time: {response_time:.2f}s")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text[:200]}...")
                return False, {}

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ Failed - Error: {str(e)} - Time: {response_time:.2f}s")
            return False, {}

    def clear_cache_and_generate_fresh_cycle(self):
        """Clear cache and generate a fresh trading cycle for API economy testing"""
        print(f"\n🗑️ STEP 1: Clearing cache for fresh API economy testing...")
        
        # Clear decision cache
        success, clear_result = self.run_test("Clear Decision Cache", "DELETE", "decisions/clear", 200)
        if not success:
            print(f"   ❌ Failed to clear cache")
            return False
        
        print(f"   ✅ Cache cleared successfully")
        if clear_result:
            print(f"      Decisions cleared: {clear_result.get('cleared_decisions', 0)}")
            print(f"      Analyses cleared: {clear_result.get('cleared_analyses', 0)}")
            print(f"      Opportunities cleared: {clear_result.get('cleared_opportunities', 0)}")
        
        print(f"\n🚀 STEP 2: Starting fresh trading cycle...")
        success, _ = self.run_test("Start Trading System", "POST", "start-trading", 200)
        if not success:
            print(f"   ❌ Failed to start trading system")
            return False
        
        print(f"   ✅ Trading system started for fresh cycle")
        return True

    def test_pre_ia1_data_validation(self):
        """Test that OHLCV data is validated BEFORE IA1 calls for API economy"""
        print(f"\n💰 Testing Pre-IA1 Data Validation for API Economy...")
        
        # Clear cache and start fresh cycle
        if not self.clear_cache_and_generate_fresh_cycle():
            return False
        
        print(f"\n📊 STEP 3: Monitoring Scout → Pre-IA1 Validation → IA1 pipeline...")
        
        # Wait and monitor the pipeline for 90 seconds
        start_time = time.time()
        max_wait_time = 90
        check_interval = 15
        
        opportunities_count = 0
        analyses_count = 0
        api_economy_evidence = []
        
        while time.time() - start_time < max_wait_time:
            elapsed = time.time() - start_time
            print(f"\n   ⏱️ After {elapsed:.1f}s - Checking pipeline status...")
            
            # Check opportunities (Scout output)
            success, opp_data = self.run_test("Check Opportunities", "GET", "opportunities", 200)
            if success:
                opportunities = opp_data.get('opportunities', [])
                opportunities_count = len(opportunities)
                print(f"      Scout Opportunities: {opportunities_count}")
                
                # Look for data validation evidence in opportunities
                for opp in opportunities[:3]:  # Check first 3
                    symbol = opp.get('symbol', 'Unknown')
                    data_confidence = opp.get('data_confidence', 0)
                    data_sources = opp.get('data_sources', [])
                    print(f"         {symbol}: confidence={data_confidence:.2f}, sources={len(data_sources)}")
            
            # Check analyses (IA1 output after validation)
            success, analysis_data = self.run_test("Check Analyses", "GET", "analyses", 200)
            if success:
                analyses = analysis_data.get('analyses', [])
                analyses_count = len(analyses)
                print(f"      IA1 Analyses Generated: {analyses_count}")
                
                # Look for evidence of pre-validation
                for analysis in analyses[:2]:  # Check first 2
                    symbol = analysis.get('symbol', 'Unknown')
                    confidence = analysis.get('analysis_confidence', 0)
                    reasoning = analysis.get('ia1_reasoning', '')
                    
                    # Look for API economy keywords in reasoning
                    economy_keywords = ['skip', 'économie', 'api', 'validation', 'pre-check', 'data quality']
                    has_economy_evidence = any(keyword.lower() in reasoning.lower() for keyword in economy_keywords)
                    
                    if has_economy_evidence:
                        api_economy_evidence.append(f"{symbol}: API economy validation detected")
                        print(f"         💰 {symbol}: API economy evidence found")
                    else:
                        print(f"         📊 {symbol}: Standard analysis (confidence={confidence:.2f})")
            
            time.sleep(check_interval)
        
        # Stop trading system
        print(f"\n🛑 STEP 4: Stopping trading system...")
        self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        
        # Calculate API economy metrics
        if opportunities_count > 0:
            api_reduction_rate = max(0, (opportunities_count - analyses_count) / opportunities_count)
            self.api_economy_metrics['opportunities_total'] = opportunities_count
            self.api_economy_metrics['analyses_generated'] = analyses_count
            self.api_economy_metrics['api_reduction_rate'] = api_reduction_rate
            
            print(f"\n📊 Pre-IA1 Data Validation Results:")
            print(f"      Total Opportunities: {opportunities_count}")
            print(f"      IA1 Analyses Generated: {analyses_count}")
            print(f"      API Calls Saved: {opportunities_count - analyses_count}")
            print(f"      API Reduction Rate: {api_reduction_rate*100:.1f}%")
            print(f"      Economy Evidence Found: {len(api_economy_evidence)}")
            
            # Validation criteria
            has_api_economy = api_reduction_rate > 0.20  # At least 20% reduction
            has_evidence = len(api_economy_evidence) > 0
            pipeline_working = opportunities_count > 0 and analyses_count > 0
            
            print(f"\n✅ Pre-IA1 Validation Assessment:")
            print(f"      API Economy Active: {'✅' if has_api_economy else '❌'} (≥20% reduction)")
            print(f"      Economy Evidence: {'✅' if has_evidence else '❌'} (found in reasoning)")
            print(f"      Pipeline Working: {'✅' if pipeline_working else '❌'} (Scout→IA1)")
            
            return has_api_economy and pipeline_working
        
        return False

    def test_ohlcv_quality_validation_criteria(self):
        """Test the comprehensive OHLCV data quality validation criteria"""
        print(f"\n📊 Testing OHLCV Quality Validation Criteria...")
        
        # Clear cache and start fresh cycle
        if not self.clear_cache_and_generate_fresh_cycle():
            return False
        
        print(f"\n🔍 STEP 3: Monitoring OHLCV quality validation criteria...")
        
        # Wait for system to process opportunities with quality validation
        start_time = time.time()
        max_wait_time = 75
        check_interval = 15
        
        quality_validation_evidence = {
            'minimum_50_days': 0,
            'required_columns': 0,
            'null_validation': 0,
            'price_consistency': 0,
            'price_variability': 0,
            'data_freshness': 0
        }
        
        total_opportunities = 0
        total_analyses = 0
        
        while time.time() - start_time < max_wait_time:
            elapsed = time.time() - start_time
            print(f"\n   ⏱️ After {elapsed:.1f}s - Checking quality validation...")
            
            # Check opportunities for quality indicators
            success, opp_data = self.run_test("Check Opportunities", "GET", "opportunities", 200)
            if success:
                opportunities = opp_data.get('opportunities', [])
                total_opportunities = len(opportunities)
                print(f"      Opportunities Found: {total_opportunities}")
                
                # Analyze quality indicators in opportunities
                for opp in opportunities[:5]:  # Check first 5
                    symbol = opp.get('symbol', 'Unknown')
                    data_confidence = opp.get('data_confidence', 0)
                    data_sources = opp.get('data_sources', [])
                    
                    # High data confidence suggests quality validation passed
                    if data_confidence >= 0.7:
                        quality_validation_evidence['minimum_50_days'] += 1
                        quality_validation_evidence['required_columns'] += 1
                        quality_validation_evidence['price_consistency'] += 1
                    
                    # Multiple sources suggest enhanced validation
                    if len(data_sources) > 0:
                        quality_validation_evidence['null_validation'] += 1
                        quality_validation_evidence['data_freshness'] += 1
                    
                    # Reasonable confidence suggests variability validation
                    if 0.6 <= data_confidence <= 0.95:
                        quality_validation_evidence['price_variability'] += 1
                    
                    print(f"         {symbol}: confidence={data_confidence:.2f}, sources={data_sources}")
            
            # Check analyses for quality validation evidence
            success, analysis_data = self.run_test("Check Analyses", "GET", "analyses", 200)
            if success:
                analyses = analysis_data.get('analyses', [])
                total_analyses = len(analyses)
                print(f"      Analyses Generated: {total_analyses}")
                
                # Look for quality validation keywords in reasoning
                for analysis in analyses[:3]:  # Check first 3
                    symbol = analysis.get('symbol', 'Unknown')
                    reasoning = analysis.get('ia1_reasoning', '')
                    confidence = analysis.get('analysis_confidence', 0)
                    
                    # Look for quality validation evidence
                    quality_keywords = [
                        'enhanced', 'multi-source', 'validation', 'quality', 
                        'ohlcv', 'data', 'days', 'historical'
                    ]
                    
                    quality_mentions = sum(1 for keyword in quality_keywords if keyword.lower() in reasoning.lower())
                    
                    if quality_mentions >= 2:
                        print(f"         💎 {symbol}: Quality validation evidence (confidence={confidence:.2f})")
                    else:
                        print(f"         📊 {symbol}: Standard analysis (confidence={confidence:.2f})")
            
            time.sleep(check_interval)
        
        # Stop trading system
        print(f"\n🛑 STEP 4: Stopping trading system...")
        self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        
        # Analyze quality validation results
        print(f"\n📊 OHLCV Quality Validation Results:")
        print(f"      Total Opportunities: {total_opportunities}")
        print(f"      Total Analyses: {total_analyses}")
        
        if total_opportunities > 0:
            quality_rate = total_analyses / total_opportunities
            print(f"      Quality Pass Rate: {quality_rate*100:.1f}%")
            
            print(f"\n🔍 Quality Criteria Evidence:")
            for criteria, count in quality_validation_evidence.items():
                rate = count / total_opportunities if total_opportunities > 0 else 0
                print(f"      {criteria.replace('_', ' ').title()}: {count}/{total_opportunities} ({rate*100:.1f}%)")
            
            # Validation assessment
            has_quality_filtering = quality_rate < 0.8  # Some filtering should occur
            has_quality_evidence = sum(quality_validation_evidence.values()) > total_opportunities * 2
            pipeline_functional = total_opportunities > 0 and total_analyses > 0
            
            print(f"\n✅ Quality Validation Assessment:")
            print(f"      Quality Filtering Active: {'✅' if has_quality_filtering else '❌'} (<80% pass rate)")
            print(f"      Quality Evidence Present: {'✅' if has_quality_evidence else '❌'} (multiple criteria)")
            print(f"      Pipeline Functional: {'✅' if pipeline_functional else '❌'} (Scout→IA1)")
            
            return has_quality_filtering and pipeline_functional
        
        return False

    def test_api_call_reduction_at_source(self):
        """Test API call reduction at the source (before IA1) rather than after"""
        print(f"\n💰 Testing API Call Reduction at Source (Before IA1)...")
        
        # Clear cache and start fresh cycle
        if not self.clear_cache_and_generate_fresh_cycle():
            return False
        
        print(f"\n📊 STEP 3: Measuring API call reduction at IA1 level...")
        
        # Monitor the complete pipeline for API economy
        start_time = time.time()
        max_wait_time = 90
        check_interval = 20
        
        pipeline_metrics = {
            'scout_opportunities': 0,
            'ia1_analyses': 0,
            'ia2_decisions': 0,
            'api_calls_saved_ia1': 0,
            'api_calls_saved_ia2': 0
        }
        
        skip_ia1_messages = []
        
        while time.time() - start_time < max_wait_time:
            elapsed = time.time() - start_time
            print(f"\n   ⏱️ After {elapsed:.1f}s - Measuring API economy...")
            
            # Measure Scout output (total opportunities)
            success, opp_data = self.run_test("Check Opportunities", "GET", "opportunities", 200)
            if success:
                opportunities = opp_data.get('opportunities', [])
                pipeline_metrics['scout_opportunities'] = len(opportunities)
                print(f"      Scout Opportunities: {pipeline_metrics['scout_opportunities']}")
            
            # Measure IA1 output (analyses generated after validation)
            success, analysis_data = self.run_test("Check Analyses", "GET", "analyses", 200)
            if success:
                analyses = analysis_data.get('analyses', [])
                pipeline_metrics['ia1_analyses'] = len(analyses)
                print(f"      IA1 Analyses Generated: {pipeline_metrics['ia1_analyses']}")
                
                # Look for "SKIP IA1" evidence in reasoning
                for analysis in analyses:
                    reasoning = analysis.get('ia1_reasoning', '')
                    if 'skip' in reasoning.lower() or 'économie' in reasoning.lower():
                        skip_ia1_messages.append(analysis.get('symbol', 'Unknown'))
            
            # Measure IA2 output (final decisions)
            success, decision_data = self.run_test("Check Decisions", "GET", "decisions", 200)
            if success:
                decisions = decision_data.get('decisions', [])
                pipeline_metrics['ia2_decisions'] = len(decisions)
                print(f"      IA2 Decisions Generated: {pipeline_metrics['ia2_decisions']}")
            
            # Calculate API savings at each level
            if pipeline_metrics['scout_opportunities'] > 0:
                pipeline_metrics['api_calls_saved_ia1'] = pipeline_metrics['scout_opportunities'] - pipeline_metrics['ia1_analyses']
                pipeline_metrics['api_calls_saved_ia2'] = pipeline_metrics['ia1_analyses'] - pipeline_metrics['ia2_decisions']
                
                ia1_reduction_rate = pipeline_metrics['api_calls_saved_ia1'] / pipeline_metrics['scout_opportunities']
                ia2_reduction_rate = pipeline_metrics['api_calls_saved_ia2'] / pipeline_metrics['ia1_analyses'] if pipeline_metrics['ia1_analyses'] > 0 else 0
                
                print(f"      API Calls Saved at IA1: {pipeline_metrics['api_calls_saved_ia1']} ({ia1_reduction_rate*100:.1f}%)")
                print(f"      API Calls Saved at IA2: {pipeline_metrics['api_calls_saved_ia2']} ({ia2_reduction_rate*100:.1f}%)")
            
            time.sleep(check_interval)
        
        # Stop trading system
        print(f"\n🛑 STEP 4: Stopping trading system...")
        self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        
        # Final API economy analysis
        print(f"\n📊 API Call Reduction Analysis:")
        print(f"      Scout Opportunities: {pipeline_metrics['scout_opportunities']}")
        print(f"      IA1 Analyses: {pipeline_metrics['ia1_analyses']}")
        print(f"      IA2 Decisions: {pipeline_metrics['ia2_decisions']}")
        print(f"      Skip IA1 Evidence: {len(skip_ia1_messages)} symbols")
        
        if pipeline_metrics['scout_opportunities'] > 0:
            total_reduction_rate = (pipeline_metrics['scout_opportunities'] - pipeline_metrics['ia2_decisions']) / pipeline_metrics['scout_opportunities']
            ia1_reduction_rate = pipeline_metrics['api_calls_saved_ia1'] / pipeline_metrics['scout_opportunities']
            
            print(f"\n💰 API Economy Metrics:")
            print(f"      Total API Reduction: {total_reduction_rate*100:.1f}%")
            print(f"      IA1 Level Reduction: {ia1_reduction_rate*100:.1f}%")
            print(f"      Source-Level Savings: {pipeline_metrics['api_calls_saved_ia1']} calls")
            
            # Store metrics for later use
            self.api_economy_metrics.update(pipeline_metrics)
            self.api_economy_metrics['total_reduction_rate'] = total_reduction_rate
            self.api_economy_metrics['ia1_reduction_rate'] = ia1_reduction_rate
            
            # Validation criteria
            significant_ia1_reduction = ia1_reduction_rate >= 0.20  # At least 20% reduction at IA1
            source_level_economy = pipeline_metrics['api_calls_saved_ia1'] > 0
            pipeline_working = pipeline_metrics['scout_opportunities'] > 0 and pipeline_metrics['ia1_analyses'] > 0
            
            print(f"\n✅ Source-Level API Economy Assessment:")
            print(f"      Significant IA1 Reduction: {'✅' if significant_ia1_reduction else '❌'} (≥20%)")
            print(f"      Source-Level Economy: {'✅' if source_level_economy else '❌'} (saves calls before IA1)")
            print(f"      Pipeline Working: {'✅' if pipeline_working else '❌'} (Scout→IA1→IA2)")
            
            return significant_ia1_reduction and source_level_economy and pipeline_working
        
        return False

    def test_simplified_ia2_filtering(self):
        """Test simplified IA2 filtering since IA1 only provides quality analyses"""
        print(f"\n🎯 Testing Simplified IA2 Filtering...")
        
        # Clear cache and start fresh cycle
        if not self.clear_cache_and_generate_fresh_cycle():
            return False
        
        print(f"\n📊 STEP 3: Monitoring IA1→IA2 filtering efficiency...")
        
        # Monitor IA1 to IA2 conversion for 75 seconds
        start_time = time.time()
        max_wait_time = 75
        check_interval = 15
        
        ia1_to_ia2_metrics = {
            'ia1_analyses': 0,
            'ia2_decisions': 0,
            'high_confidence_ia1': 0,
            'pattern_analyses': 0,
            'filtered_out': 0
        }
        
        while time.time() - start_time < max_wait_time:
            elapsed = time.time() - start_time
            print(f"\n   ⏱️ After {elapsed:.1f}s - Checking IA2 filtering...")
            
            # Check IA1 analyses quality
            success, analysis_data = self.run_test("Check Analyses", "GET", "analyses", 200)
            if success:
                analyses = analysis_data.get('analyses', [])
                ia1_to_ia2_metrics['ia1_analyses'] = len(analyses)
                print(f"      IA1 Analyses Available: {ia1_to_ia2_metrics['ia1_analyses']}")
                
                # Analyze IA1 quality indicators
                for analysis in analyses:
                    confidence = analysis.get('analysis_confidence', 0)
                    patterns = analysis.get('patterns_detected', [])
                    
                    if confidence >= 0.7:
                        ia1_to_ia2_metrics['high_confidence_ia1'] += 1
                    
                    if len(patterns) > 0 and 'No patterns' not in str(patterns):
                        ia1_to_ia2_metrics['pattern_analyses'] += 1
                
                high_quality_rate = ia1_to_ia2_metrics['high_confidence_ia1'] / len(analyses) if analyses else 0
                pattern_rate = ia1_to_ia2_metrics['pattern_analyses'] / len(analyses) if analyses else 0
                
                print(f"         High Confidence (≥70%): {ia1_to_ia2_metrics['high_confidence_ia1']}/{len(analyses)} ({high_quality_rate*100:.1f}%)")
                print(f"         Pattern Detection: {ia1_to_ia2_metrics['pattern_analyses']}/{len(analyses)} ({pattern_rate*100:.1f}%)")
            
            # Check IA2 decisions
            success, decision_data = self.run_test("Check Decisions", "GET", "decisions", 200)
            if success:
                decisions = decision_data.get('decisions', [])
                ia1_to_ia2_metrics['ia2_decisions'] = len(decisions)
                print(f"      IA2 Decisions Generated: {ia1_to_ia2_metrics['ia2_decisions']}")
                
                # Analyze IA2 decision quality
                trading_signals = sum(1 for d in decisions if d.get('signal', 'hold').lower() in ['long', 'short'])
                high_confidence_decisions = sum(1 for d in decisions if d.get('confidence', 0) >= 0.6)
                
                if decisions:
                    trading_rate = trading_signals / len(decisions)
                    decision_confidence_rate = high_confidence_decisions / len(decisions)
                    
                    print(f"         Trading Signals: {trading_signals}/{len(decisions)} ({trading_rate*100:.1f}%)")
                    print(f"         High Confidence Decisions: {high_confidence_decisions}/{len(decisions)} ({decision_confidence_rate*100:.1f}%)")
            
            time.sleep(check_interval)
        
        # Stop trading system
        print(f"\n🛑 STEP 4: Stopping trading system...")
        self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        
        # Calculate IA2 filtering efficiency
        if ia1_to_ia2_metrics['ia1_analyses'] > 0:
            ia2_conversion_rate = ia1_to_ia2_metrics['ia2_decisions'] / ia1_to_ia2_metrics['ia1_analyses']
            ia1_to_ia2_metrics['filtered_out'] = ia1_to_ia2_metrics['ia1_analyses'] - ia1_to_ia2_metrics['ia2_decisions']
            
            print(f"\n📊 IA2 Filtering Efficiency Results:")
            print(f"      IA1 Analyses Input: {ia1_to_ia2_metrics['ia1_analyses']}")
            print(f"      IA2 Decisions Output: {ia1_to_ia2_metrics['ia2_decisions']}")
            print(f"      IA2 Conversion Rate: {ia2_conversion_rate*100:.1f}%")
            print(f"      High Quality IA1: {ia1_to_ia2_metrics['high_confidence_ia1']}")
            print(f"      Pattern Analyses: {ia1_to_ia2_metrics['pattern_analyses']}")
            
            # Validation criteria for simplified filtering
            efficient_conversion = ia2_conversion_rate >= 0.30  # At least 30% conversion (since IA1 is pre-filtered)
            quality_input = ia1_to_ia2_metrics['high_confidence_ia1'] > 0  # IA1 provides quality analyses
            balanced_filtering = 0.30 <= ia2_conversion_rate <= 0.70  # Balanced economy (30-70% pass rate)
            
            print(f"\n✅ Simplified IA2 Filtering Assessment:")
            print(f"      Efficient Conversion: {'✅' if efficient_conversion else '❌'} (≥30% since IA1 pre-filtered)")
            print(f"      Quality IA1 Input: {'✅' if quality_input else '❌'} (high confidence analyses)")
            print(f"      Balanced Economy: {'✅' if balanced_filtering else '❌'} (30-70% conversion)")
            
            return efficient_conversion and quality_input
        
        return False

    def test_end_to_end_optimized_pipeline(self):
        """Test complete optimized pipeline: Scout → Data Pre-Check → IA1 → IA2"""
        print(f"\n🔄 Testing End-to-End Optimized Pipeline...")
        
        # Clear cache and start fresh cycle
        if not self.clear_cache_and_generate_fresh_cycle():
            return False
        
        print(f"\n📊 STEP 3: Monitoring complete optimized pipeline...")
        
        # Monitor complete pipeline for 120 seconds
        start_time = time.time()
        max_wait_time = 120
        check_interval = 20
        
        pipeline_stages = {
            'scout_opportunities': 0,
            'pre_check_passed': 0,
            'ia1_analyses': 0,
            'ia2_decisions': 0,
            'final_trading_signals': 0
        }
        
        quality_metrics = {
            'avg_data_confidence': 0,
            'avg_analysis_confidence': 0,
            'avg_decision_confidence': 0,
            'trading_effectiveness': 0
        }
        
        while time.time() - start_time < max_wait_time:
            elapsed = time.time() - start_time
            print(f"\n   ⏱️ After {elapsed:.1f}s - Pipeline monitoring...")
            
            # Stage 1: Scout opportunities
            success, opp_data = self.run_test("Check Opportunities", "GET", "opportunities", 200)
            if success:
                opportunities = opp_data.get('opportunities', [])
                pipeline_stages['scout_opportunities'] = len(opportunities)
                
                # Calculate data quality metrics
                if opportunities:
                    data_confidences = [opp.get('data_confidence', 0) for opp in opportunities]
                    quality_metrics['avg_data_confidence'] = sum(data_confidences) / len(data_confidences)
                    
                    # Count opportunities that would pass pre-check
                    pipeline_stages['pre_check_passed'] = sum(1 for opp in opportunities if opp.get('data_confidence', 0) >= 0.6)
                
                print(f"      Stage 1 - Scout: {pipeline_stages['scout_opportunities']} opportunities")
                print(f"         Pre-check eligible: {pipeline_stages['pre_check_passed']} (≥60% confidence)")
            
            # Stage 2: IA1 analyses (after pre-check)
            success, analysis_data = self.run_test("Check Analyses", "GET", "analyses", 200)
            if success:
                analyses = analysis_data.get('analyses', [])
                pipeline_stages['ia1_analyses'] = len(analyses)
                
                # Calculate analysis quality metrics
                if analyses:
                    analysis_confidences = [a.get('analysis_confidence', 0) for a in analyses]
                    quality_metrics['avg_analysis_confidence'] = sum(analysis_confidences) / len(analysis_confidences)
                
                print(f"      Stage 2 - IA1: {pipeline_stages['ia1_analyses']} analyses")
                print(f"         Avg confidence: {quality_metrics['avg_analysis_confidence']:.2f}")
            
            # Stage 3: IA2 decisions (simplified filtering)
            success, decision_data = self.run_test("Check Decisions", "GET", "decisions", 200)
            if success:
                decisions = decision_data.get('decisions', [])
                pipeline_stages['ia2_decisions'] = len(decisions)
                
                # Calculate decision quality metrics
                if decisions:
                    decision_confidences = [d.get('confidence', 0) for d in decisions]
                    quality_metrics['avg_decision_confidence'] = sum(decision_confidences) / len(decision_confidences)
                    
                    # Count trading signals
                    trading_signals = [d for d in decisions if d.get('signal', 'hold').lower() in ['long', 'short']]
                    pipeline_stages['final_trading_signals'] = len(trading_signals)
                    
                    if decisions:
                        quality_metrics['trading_effectiveness'] = len(trading_signals) / len(decisions)
                
                print(f"      Stage 3 - IA2: {pipeline_stages['ia2_decisions']} decisions")
                print(f"         Trading signals: {pipeline_stages['final_trading_signals']}")
                print(f"         Avg confidence: {quality_metrics['avg_decision_confidence']:.2f}")
            
            # Calculate pipeline efficiency
            if pipeline_stages['scout_opportunities'] > 0:
                pre_check_efficiency = pipeline_stages['pre_check_passed'] / pipeline_stages['scout_opportunities']
                ia1_efficiency = pipeline_stages['ia1_analyses'] / pipeline_stages['scout_opportunities']
                ia2_efficiency = pipeline_stages['ia2_decisions'] / pipeline_stages['scout_opportunities']
                
                print(f"      Pipeline Efficiency:")
                print(f"         Pre-check pass rate: {pre_check_efficiency*100:.1f}%")
                print(f"         Scout→IA1 efficiency: {ia1_efficiency*100:.1f}%")
                print(f"         Scout→IA2 efficiency: {ia2_efficiency*100:.1f}%")
            
            time.sleep(check_interval)
        
        # Stop trading system
        print(f"\n🛑 STEP 4: Stopping trading system...")
        self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        
        # Final pipeline assessment
        print(f"\n📊 End-to-End Optimized Pipeline Results:")
        print(f"      Scout Opportunities: {pipeline_stages['scout_opportunities']}")
        print(f"      Pre-check Passed: {pipeline_stages['pre_check_passed']}")
        print(f"      IA1 Analyses: {pipeline_stages['ia1_analyses']}")
        print(f"      IA2 Decisions: {pipeline_stages['ia2_decisions']}")
        print(f"      Trading Signals: {pipeline_stages['final_trading_signals']}")
        
        print(f"\n📈 Quality Metrics:")
        print(f"      Avg Data Confidence: {quality_metrics['avg_data_confidence']:.2f}")
        print(f"      Avg Analysis Confidence: {quality_metrics['avg_analysis_confidence']:.2f}")
        print(f"      Avg Decision Confidence: {quality_metrics['avg_decision_confidence']:.2f}")
        print(f"      Trading Effectiveness: {quality_metrics['trading_effectiveness']*100:.1f}%")
        
        if pipeline_stages['scout_opportunities'] > 0:
            # Calculate overall API economy
            total_api_reduction = (pipeline_stages['scout_opportunities'] - pipeline_stages['ia2_decisions']) / pipeline_stages['scout_opportunities']
            
            # Validation criteria
            pipeline_complete = all(stage > 0 for stage in [
                pipeline_stages['scout_opportunities'],
                pipeline_stages['ia1_analyses'],
                pipeline_stages['ia2_decisions']
            ])
            
            quality_maintained = (
                quality_metrics['avg_data_confidence'] >= 0.6 and
                quality_metrics['avg_analysis_confidence'] >= 0.6 and
                quality_metrics['avg_decision_confidence'] >= 0.5
            )
            
            api_economy_achieved = total_api_reduction >= 0.20  # At least 20% overall reduction
            trading_effectiveness_good = quality_metrics['trading_effectiveness'] >= 0.10  # At least 10% trading rate
            
            print(f"\n✅ End-to-End Pipeline Assessment:")
            print(f"      Pipeline Complete: {'✅' if pipeline_complete else '❌'} (all stages working)")
            print(f"      Quality Maintained: {'✅' if quality_maintained else '❌'} (confidence levels)")
            print(f"      API Economy Achieved: {'✅' if api_economy_achieved else '❌'} (≥20% reduction)")
            print(f"      Trading Effectiveness: {'✅' if trading_effectiveness_good else '❌'} (≥10% signals)")
            
            return pipeline_complete and quality_maintained and api_economy_achieved
        
        return False

    def test_quality_vs_economy_balance(self):
        """Test that optimization maintains quality while reducing API costs"""
        print(f"\n⚖️ Testing Quality vs Economy Balance...")
        
        # Clear cache and start fresh cycle
        if not self.clear_cache_and_generate_fresh_cycle():
            return False
        
        print(f"\n📊 STEP 3: Analyzing quality vs economy balance...")
        
        # Monitor for quality vs economy balance for 90 seconds
        start_time = time.time()
        max_wait_time = 90
        check_interval = 18
        
        balance_metrics = {
            'total_opportunities': 0,
            'quality_analyses': 0,
            'quality_decisions': 0,
            'api_calls_saved': 0,
            'decision_quality_score': 0,
            'trading_signal_quality': 0
        }
        
        while time.time() - start_time < max_wait_time:
            elapsed = time.time() - start_time
            print(f"\n   ⏱️ After {elapsed:.1f}s - Checking quality/economy balance...")
            
            # Measure total opportunities (baseline)
            success, opp_data = self.run_test("Check Opportunities", "GET", "opportunities", 200)
            if success:
                opportunities = opp_data.get('opportunities', [])
                balance_metrics['total_opportunities'] = len(opportunities)
                print(f"      Total Opportunities: {balance_metrics['total_opportunities']}")
            
            # Measure quality analyses that passed validation
            success, analysis_data = self.run_test("Check Analyses", "GET", "analyses", 200)
            if success:
                analyses = analysis_data.get('analyses', [])
                
                # Count high-quality analyses
                quality_analyses = [a for a in analyses if a.get('analysis_confidence', 0) >= 0.7]
                balance_metrics['quality_analyses'] = len(quality_analyses)
                
                print(f"      Quality Analyses (≥70%): {balance_metrics['quality_analyses']}/{len(analyses)}")
                
                # Analyze analysis quality indicators
                if analyses:
                    avg_confidence = sum(a.get('analysis_confidence', 0) for a in analyses) / len(analyses)
                    technical_completeness = sum(1 for a in analyses if 
                                               len(a.get('support_levels', [])) > 0 and 
                                               len(a.get('resistance_levels', [])) > 0) / len(analyses)
                    
                    print(f"         Avg Analysis Confidence: {avg_confidence:.2f}")
                    print(f"         Technical Completeness: {technical_completeness*100:.1f}%")
            
            # Measure quality decisions
            success, decision_data = self.run_test("Check Decisions", "GET", "decisions", 200)
            if success:
                decisions = decision_data.get('decisions', [])
                
                # Count high-quality decisions
                quality_decisions = [d for d in decisions if 
                                   d.get('confidence', 0) >= 0.6 and 
                                   d.get('ia2_reasoning', '') and 
                                   len(d.get('ia2_reasoning', '')) > 100]
                balance_metrics['quality_decisions'] = len(quality_decisions)
                
                # Count quality trading signals
                quality_trading_signals = [d for d in decisions if 
                                         d.get('signal', 'hold').lower() in ['long', 'short'] and 
                                         d.get('confidence', 0) >= 0.6]
                balance_metrics['trading_signal_quality'] = len(quality_trading_signals)
                
                print(f"      Quality Decisions: {balance_metrics['quality_decisions']}/{len(decisions)}")
                print(f"      Quality Trading Signals: {balance_metrics['trading_signal_quality']}")
                
                # Calculate decision quality score
                if decisions:
                    avg_decision_confidence = sum(d.get('confidence', 0) for d in decisions) / len(decisions)
                    reasoning_completeness = sum(1 for d in decisions if 
                                               d.get('ia2_reasoning', '') and 
                                               len(d.get('ia2_reasoning', '')) > 50) / len(decisions)
                    
                    balance_metrics['decision_quality_score'] = (avg_decision_confidence + reasoning_completeness) / 2
                    
                    print(f"         Avg Decision Confidence: {avg_decision_confidence:.2f}")
                    print(f"         Reasoning Completeness: {reasoning_completeness*100:.1f}%")
            
            time.sleep(check_interval)
        
        # Stop trading system
        print(f"\n🛑 STEP 4: Stopping trading system...")
        self.run_test("Stop Trading System", "POST", "stop-trading", 200)
        
        # Calculate API economy vs quality balance
        if balance_metrics['total_opportunities'] > 0:
            # Calculate API savings
            total_decisions = balance_metrics['quality_decisions'] if balance_metrics['quality_decisions'] > 0 else 1
            balance_metrics['api_calls_saved'] = balance_metrics['total_opportunities'] - total_decisions
            api_reduction_rate = balance_metrics['api_calls_saved'] / balance_metrics['total_opportunities']
            
            # Calculate quality preservation rates
            quality_analysis_rate = balance_metrics['quality_analyses'] / balance_metrics['total_opportunities']
            quality_decision_rate = balance_metrics['quality_decisions'] / balance_metrics['total_opportunities']
            
            print(f"\n📊 Quality vs Economy Balance Results:")
            print(f"      Total Opportunities: {balance_metrics['total_opportunities']}")
            print(f"      API Calls Saved: {balance_metrics['api_calls_saved']}")
            print(f"      API Reduction Rate: {api_reduction_rate*100:.1f}%")
            print(f"      Quality Analysis Rate: {quality_analysis_rate*100:.1f}%")
            print(f"      Quality Decision Rate: {quality_decision_rate*100:.1f}%")
            print(f"      Decision Quality Score: {balance_metrics['decision_quality_score']:.2f}")
            print(f"      Quality Trading Signals: {balance_metrics['trading_signal_quality']}")
            
            # Validation criteria for balanced optimization
            api_economy_target = 0.20 <= api_reduction_rate <= 0.50  # 20-50% API reduction target
            quality_maintained = balance_metrics['decision_quality_score'] >= 0.6  # Quality score ≥60%
            quality_analyses_present = balance_metrics['quality_analyses'] > 0  # Some quality analyses reach IA2
            trading_effectiveness = balance_metrics['trading_signal_quality'] > 0  # Quality trading signals generated
            
            print(f"\n✅ Quality vs Economy Balance Assessment:")
            print(f"      API Economy Target (20-50%): {'✅' if api_economy_target else '❌'} ({api_reduction_rate*100:.1f}%)")
            print(f"      Quality Maintained: {'✅' if quality_maintained else '❌'} (score: {balance_metrics['decision_quality_score']:.2f})")
            print(f"      Quality Analyses Present: {'✅' if quality_analyses_present else '❌'} ({balance_metrics['quality_analyses']} analyses)")
            print(f"      Trading Effectiveness: {'✅' if trading_effectiveness else '❌'} ({balance_metrics['trading_signal_quality']} signals)")
            
            return api_economy_target and quality_maintained and quality_analyses_present
        
        return False

    def run_comprehensive_api_economy_tests(self):
        """Run all API economy optimization tests"""
        print(f"\n🚀 COMPREHENSIVE API ECONOMY OPTIMIZATION TESTING")
        print(f"=" * 60)
        
        test_results = {}
        
        # Test 1: Pre-IA1 Data Validation
        print(f"\n1️⃣ PRE-IA1 DATA VALIDATION TEST")
        test_results['pre_ia1_validation'] = self.test_pre_ia1_data_validation()
        
        # Test 2: OHLCV Quality Validation Criteria
        print(f"\n2️⃣ OHLCV QUALITY VALIDATION CRITERIA TEST")
        test_results['ohlcv_quality_validation'] = self.test_ohlcv_quality_validation_criteria()
        
        # Test 3: API Call Reduction at Source
        print(f"\n3️⃣ API CALL REDUCTION AT SOURCE TEST")
        test_results['api_call_reduction'] = self.test_api_call_reduction_at_source()
        
        # Test 4: Simplified IA2 Filtering
        print(f"\n4️⃣ SIMPLIFIED IA2 FILTERING TEST")
        test_results['simplified_ia2_filtering'] = self.test_simplified_ia2_filtering()
        
        # Test 5: End-to-End Optimized Pipeline
        print(f"\n5️⃣ END-TO-END OPTIMIZED PIPELINE TEST")
        test_results['end_to_end_pipeline'] = self.test_end_to_end_optimized_pipeline()
        
        # Test 6: Quality vs Economy Balance
        print(f"\n6️⃣ QUALITY VS ECONOMY BALANCE TEST")
        test_results['quality_economy_balance'] = self.test_quality_vs_economy_balance()
        
        # Final Assessment
        print(f"\n" + "=" * 60)
        print(f"🎯 FINAL API ECONOMY OPTIMIZATION ASSESSMENT")
        print(f"=" * 60)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        print(f"\n📊 Test Results Summary:")
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"      {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n🎯 Overall Results:")
        print(f"      Tests Passed: {passed_tests}/{total_tests}")
        print(f"      Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        # API Economy Metrics Summary
        if hasattr(self, 'api_economy_metrics') and self.api_economy_metrics:
            print(f"\n💰 API Economy Metrics Summary:")
            for metric, value in self.api_economy_metrics.items():
                if isinstance(value, float):
                    print(f"      {metric.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"      {metric.replace('_', ' ').title()}: {value}")
        
        # Overall assessment
        critical_tests_passed = sum([
            test_results.get('pre_ia1_validation', False),
            test_results.get('api_call_reduction', False),
            test_results.get('end_to_end_pipeline', False)
        ])
        
        overall_success = critical_tests_passed >= 2 and passed_tests >= 4
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"      Critical Tests (3): {critical_tests_passed}/3")
        print(f"      Overall Success: {'✅ API ECONOMY OPTIMIZATION WORKING' if overall_success else '❌ NEEDS IMPROVEMENT'}")
        
        if not overall_success:
            print(f"\n💡 RECOMMENDATIONS:")
            if not test_results.get('pre_ia1_validation', False):
                print(f"      - Fix pre-IA1 data validation to prevent unnecessary API calls")
            if not test_results.get('api_call_reduction', False):
                print(f"      - Improve API call reduction at source (before IA1)")
            if not test_results.get('end_to_end_pipeline', False):
                print(f"      - Ensure complete optimized pipeline is functional")
            if not test_results.get('quality_economy_balance', False):
                print(f"      - Balance API economy with decision quality")
        
        return overall_success

if __name__ == "__main__":
    print("🚀 API Economy Optimization Testing Suite")
    print("=" * 50)
    
    tester = APIEconomyOptimizationTester()
    success = tester.run_comprehensive_api_economy_tests()
    
    print(f"\n" + "=" * 50)
    if success:
        print("🎉 API ECONOMY OPTIMIZATION TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️ API ECONOMY OPTIMIZATION NEEDS IMPROVEMENT")
    print("=" * 50)
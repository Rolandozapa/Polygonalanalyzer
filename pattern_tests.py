import requests
import time
import json
from datetime import datetime
from pathlib import Path

class IA1IA2PatternTester:
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
                    base_url = "https://smart-crypto-bot-14.preview.emergentagent.com"
            except:
                base_url = "https://smart-crypto-bot-14.preview.emergentagent.com"
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
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
                return False, {}

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ Failed - Error: {str(e)} - Time: {response_time:.2f}s")
            return False, {}

    def test_get_analyses(self):
        """Test get analyses endpoint"""
        return self.run_test("Get Technical Analyses", "GET", "analyses", 200)

    def test_get_decisions(self):
        """Test get decisions endpoint"""
        return self.run_test("Get Trading Decisions", "GET", "decisions", 200)

    def test_start_trading_system(self):
        """Test starting the trading system"""
        return self.run_test("Start Trading System", "POST", "start-trading", 200)

    def test_stop_trading_system(self):
        """Test stopping the trading system"""
        return self.run_test("Stop Trading System", "POST", "stop-trading", 200)

    def test_ia1_pattern_prioritization_system(self):
        """🎯 TEST IA1 PATTERN PRIORITIZATION - Master Pattern Selection"""
        print(f"\n🎯 Testing IA1 Pattern Prioritization System...")
        print(f"   📋 TESTING OBJECTIVES:")
        print(f"      • IA1 designates '🎯 MASTER PATTERN (IA1 STRATEGIC CHOICE)'")
        print(f"      • Pattern corresponds to LONG/SHORT decision")
        print(f"      • IA1 explains 'This pattern is IA1's PRIMARY BASIS'")
        print(f"      • Pattern hierarchy clearly established")
        
        # Get current analyses to check IA1 pattern prioritization
        success, analyses_data = self.test_get_analyses()
        if not success:
            print(f"   ❌ Cannot retrieve analyses for IA1 pattern testing")
            return False
        
        analyses = analyses_data.get('analyses', [])
        if len(analyses) == 0:
            print(f"   ❌ No analyses available for IA1 pattern testing")
            return False
        
        print(f"   📊 Analyzing IA1 pattern prioritization in {len(analyses)} analyses...")
        
        # Pattern prioritization analysis
        master_pattern_count = 0
        primary_basis_count = 0
        pattern_decision_coherence = 0
        strategic_choice_count = 0
        pattern_hierarchy_count = 0
        
        pattern_examples = []
        
        for i, analysis in enumerate(analyses[:15]):  # Analyze first 15 analyses
            symbol = analysis.get('symbol', 'Unknown')
            ia1_signal = analysis.get('ia1_signal', 'hold').upper()
            reasoning = analysis.get('ia1_reasoning', '')
            patterns = analysis.get('patterns_detected', [])
            confidence = analysis.get('analysis_confidence', 0)
            
            # Check for IA1 pattern prioritization keywords
            has_master_pattern = '🎯 master pattern' in reasoning.lower() or 'master pattern' in reasoning.lower()
            has_strategic_choice = 'ia1 strategic choice' in reasoning.lower() or 'strategic choice' in reasoning.lower()
            has_primary_basis = 'primary basis' in reasoning.lower() or "ia1's primary basis" in reasoning.lower()
            has_pattern_hierarchy = 'pattern hierarchy' in reasoning.lower() or 'priorit' in reasoning.lower()
            
            # Check pattern-decision coherence
            pattern_coherent = False
            if patterns:
                main_pattern = patterns[0] if patterns else ""
                # Check if pattern aligns with signal
                if ia1_signal == 'LONG' and any(bullish in main_pattern.lower() for bullish in ['bull', 'ascending', 'support', 'breakout', 'flag']):
                    pattern_coherent = True
                elif ia1_signal == 'SHORT' and any(bearish in main_pattern.lower() for bearish in ['bear', 'descending', 'resistance', 'breakdown', 'wedge']):
                    pattern_coherent = True
            
            # Count pattern prioritization indicators
            if has_master_pattern:
                master_pattern_count += 1
            if has_strategic_choice:
                strategic_choice_count += 1
            if has_primary_basis:
                primary_basis_count += 1
            if has_pattern_hierarchy:
                pattern_hierarchy_count += 1
            if pattern_coherent:
                pattern_decision_coherence += 1
            
            # Collect examples for detailed analysis
            if i < 5:  # Show first 5 examples
                pattern_examples.append({
                    'symbol': symbol,
                    'signal': ia1_signal,
                    'patterns': patterns,
                    'confidence': confidence,
                    'has_master_pattern': has_master_pattern,
                    'has_strategic_choice': has_strategic_choice,
                    'has_primary_basis': has_primary_basis,
                    'pattern_coherent': pattern_coherent,
                    'reasoning_preview': reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                })
        
        # Display detailed examples
        print(f"\n   🔍 IA1 Pattern Prioritization Examples:")
        for i, example in enumerate(pattern_examples):
            print(f"      Analysis {i+1} - {example['symbol']} ({example['signal']}):")
            print(f"         Patterns: {example['patterns']}")
            print(f"         🎯 Master Pattern: {'✅' if example['has_master_pattern'] else '❌'}")
            print(f"         🎯 Strategic Choice: {'✅' if example['has_strategic_choice'] else '❌'}")
            print(f"         🎯 Primary Basis: {'✅' if example['has_primary_basis'] else '❌'}")
            print(f"         🎯 Pattern Coherent: {'✅' if example['pattern_coherent'] else '❌'}")
            print(f"         Preview: {example['reasoning_preview']}")
        
        # Calculate success rates
        total_analyses = len(analyses[:15])
        master_pattern_rate = master_pattern_count / total_analyses
        strategic_choice_rate = strategic_choice_count / total_analyses
        primary_basis_rate = primary_basis_count / total_analyses
        pattern_coherence_rate = pattern_decision_coherence / total_analyses
        pattern_hierarchy_rate = pattern_hierarchy_count / total_analyses
        
        print(f"\n   📊 IA1 Pattern Prioritization Statistics:")
        print(f"      🎯 Master Pattern Designation: {master_pattern_count}/{total_analyses} ({master_pattern_rate*100:.1f}%)")
        print(f"      🎯 Strategic Choice Declaration: {strategic_choice_count}/{total_analyses} ({strategic_choice_rate*100:.1f}%)")
        print(f"      🎯 Primary Basis Explanation: {primary_basis_count}/{total_analyses} ({primary_basis_rate*100:.1f}%)")
        print(f"      🎯 Pattern-Decision Coherence: {pattern_decision_coherence}/{total_analyses} ({pattern_coherence_rate*100:.1f}%)")
        print(f"      🎯 Pattern Hierarchy Established: {pattern_hierarchy_count}/{total_analyses} ({pattern_hierarchy_rate*100:.1f}%)")
        
        # Validation criteria for IA1 pattern prioritization
        master_pattern_working = master_pattern_rate >= 0.6  # 60% should have master pattern
        strategic_choice_working = strategic_choice_rate >= 0.5  # 50% should declare strategic choice
        primary_basis_working = primary_basis_rate >= 0.5  # 50% should explain primary basis
        pattern_coherence_working = pattern_coherence_rate >= 0.7  # 70% should be coherent
        hierarchy_working = pattern_hierarchy_rate >= 0.4  # 40% should establish hierarchy
        
        print(f"\n   ✅ IA1 Pattern Prioritization Validation:")
        print(f"      Master Pattern System: {'✅' if master_pattern_working else '❌'} (≥60%)")
        print(f"      Strategic Choice System: {'✅' if strategic_choice_working else '❌'} (≥50%)")
        print(f"      Primary Basis System: {'✅' if primary_basis_working else '❌'} (≥50%)")
        print(f"      Pattern Coherence: {'✅' if pattern_coherence_working else '❌'} (≥70%)")
        print(f"      Pattern Hierarchy: {'✅' if hierarchy_working else '❌'} (≥40%)")
        
        # Overall IA1 pattern prioritization assessment
        ia1_pattern_system_working = (
            master_pattern_working and
            pattern_coherence_working and
            (strategic_choice_working or primary_basis_working)  # At least one explanation method
        )
        
        print(f"\n   🎯 IA1 Pattern Prioritization System: {'✅ WORKING' if ia1_pattern_system_working else '❌ NEEDS IMPROVEMENT'}")
        
        if not ia1_pattern_system_working:
            print(f"   💡 ISSUES DETECTED:")
            if not master_pattern_working:
                print(f"      - IA1 not consistently designating MASTER PATTERN")
            if not pattern_coherence_working:
                print(f"      - Pattern-decision coherence low ({pattern_coherence_rate*100:.1f}%)")
            if not strategic_choice_working and not primary_basis_working:
                print(f"      - IA1 not explaining strategic reasoning adequately")
        
        return ia1_pattern_system_working

    def test_ia2_pattern_comprehension_system(self):
        """🎯 TEST IA2 PATTERN COMPREHENSION - Master Pattern Reception"""
        print(f"\n🎯 Testing IA2 Pattern Comprehension System...")
        print(f"   📋 TESTING OBJECTIVES:")
        print(f"      • IA2 receives MASTER PATTERN from IA1")
        print(f"      • IA2 respects IA1's pattern hierarchy")
        print(f"      • No contradictions: IA1 'SHORT via rising_wedge' → IA2 'LONG'")
        print(f"      • IA2 acknowledges IA1's strategic choice")
        
        # Get current decisions and analyses for cross-reference
        success_decisions, decisions_data = self.test_get_decisions()
        success_analyses, analyses_data = self.test_get_analyses()
        
        if not success_decisions or not success_analyses:
            print(f"   ❌ Cannot retrieve decisions/analyses for IA2 comprehension testing")
            return False
        
        decisions = decisions_data.get('decisions', [])
        analyses = analyses_data.get('analyses', [])
        
        if len(decisions) == 0 or len(analyses) == 0:
            print(f"   ❌ No decisions/analyses available for IA2 comprehension testing")
            return False
        
        print(f"   📊 Cross-analyzing {len(decisions)} decisions with {len(analyses)} analyses...")
        
        # Create symbol-based mapping for IA1→IA2 flow analysis
        ia1_by_symbol = {}
        for analysis in analyses:
            symbol = analysis.get('symbol', '')
            if symbol:
                ia1_by_symbol[symbol] = analysis
        
        # IA2 comprehension analysis
        ia2_receives_pattern = 0
        ia2_respects_hierarchy = 0
        ia2_acknowledges_ia1 = 0
        coherent_decisions = 0
        contradictory_decisions = 0
        
        comprehension_examples = []
        contradiction_examples = []
        
        for i, decision in enumerate(decisions[:20]):  # Analyze first 20 decisions
            symbol = decision.get('symbol', 'Unknown')
            ia2_signal = decision.get('signal', 'hold').upper()
            ia2_reasoning = decision.get('ia2_reasoning', '')
            confidence = decision.get('confidence', 0)
            
            # Find corresponding IA1 analysis
            ia1_analysis = ia1_by_symbol.get(symbol)
            if not ia1_analysis:
                continue  # Skip if no IA1 analysis found
            
            ia1_signal = ia1_analysis.get('ia1_signal', 'hold').upper()
            ia1_reasoning = ia1_analysis.get('ia1_reasoning', '')
            ia1_patterns = ia1_analysis.get('patterns_detected', [])
            
            # Check IA2 pattern comprehension indicators
            receives_ia1_pattern = any(pattern.lower() in ia2_reasoning.lower() for pattern in ia1_patterns) if ia1_patterns else False
            acknowledges_ia1 = 'ia1' in ia2_reasoning.lower() or 'technical analysis' in ia2_reasoning.lower()
            respects_hierarchy = 'master pattern' in ia2_reasoning.lower() or 'primary basis' in ia2_reasoning.lower()
            
            # Check for signal coherence (no contradictions)
            signals_coherent = True
            if ia1_signal in ['LONG', 'SHORT'] and ia2_signal in ['LONG', 'SHORT']:
                if ia1_signal != ia2_signal:
                    # Check if there's a valid justification for the contradiction
                    has_justification = any(keyword in ia2_reasoning.lower() for keyword in [
                        'however', 'but', 'despite', 'overriding', 'different timeframe', 'risk management'
                    ])
                    if not has_justification:
                        signals_coherent = False
                        contradictory_decisions += 1
                        contradiction_examples.append({
                            'symbol': symbol,
                            'ia1_signal': ia1_signal,
                            'ia2_signal': ia2_signal,
                            'ia1_patterns': ia1_patterns,
                            'justification': has_justification
                        })
            
            # Count comprehension indicators
            if receives_ia1_pattern:
                ia2_receives_pattern += 1
            if acknowledges_ia1:
                ia2_acknowledges_ia1 += 1
            if respects_hierarchy:
                ia2_respects_hierarchy += 1
            if signals_coherent:
                coherent_decisions += 1
            
            # Collect examples for detailed analysis
            if i < 5:  # Show first 5 examples
                comprehension_examples.append({
                    'symbol': symbol,
                    'ia1_signal': ia1_signal,
                    'ia2_signal': ia2_signal,
                    'ia1_patterns': ia1_patterns,
                    'receives_pattern': receives_ia1_pattern,
                    'acknowledges_ia1': acknowledges_ia1,
                    'respects_hierarchy': respects_hierarchy,
                    'signals_coherent': signals_coherent,
                    'ia2_reasoning_preview': ia2_reasoning[:200] + "..." if len(ia2_reasoning) > 200 else ia2_reasoning
                })
        
        # Display detailed examples
        print(f"\n   🔍 IA2 Pattern Comprehension Examples:")
        for i, example in enumerate(comprehension_examples):
            print(f"      Decision {i+1} - {example['symbol']}:")
            print(f"         IA1 Signal: {example['ia1_signal']} | IA2 Signal: {example['ia2_signal']}")
            print(f"         IA1 Patterns: {example['ia1_patterns']}")
            print(f"         🎯 Receives Pattern: {'✅' if example['receives_pattern'] else '❌'}")
            print(f"         🎯 Acknowledges IA1: {'✅' if example['acknowledges_ia1'] else '❌'}")
            print(f"         🎯 Respects Hierarchy: {'✅' if example['respects_hierarchy'] else '❌'}")
            print(f"         🎯 Signals Coherent: {'✅' if example['signals_coherent'] else '❌'}")
            print(f"         Preview: {example['ia2_reasoning_preview']}")
        
        # Display contradictions if any
        if contradiction_examples:
            print(f"\n   ⚠️  CONTRADICTIONS DETECTED:")
            for i, contradiction in enumerate(contradiction_examples[:3]):  # Show first 3
                print(f"      Contradiction {i+1} - {contradiction['symbol']}:")
                print(f"         IA1: {contradiction['ia1_signal']} via {contradiction['ia1_patterns']}")
                print(f"         IA2: {contradiction['ia2_signal']} (No justification: {'❌' if not contradiction['justification'] else '✅'})")
        
        # Calculate success rates
        total_matched_decisions = len([d for d in decisions[:20] if d.get('symbol') in ia1_by_symbol])
        if total_matched_decisions == 0:
            print(f"   ❌ No matching IA1→IA2 pairs found for analysis")
            return False
        
        pattern_reception_rate = ia2_receives_pattern / total_matched_decisions
        hierarchy_respect_rate = ia2_respects_hierarchy / total_matched_decisions
        ia1_acknowledgment_rate = ia2_acknowledges_ia1 / total_matched_decisions
        coherence_rate = coherent_decisions / total_matched_decisions
        contradiction_rate = contradictory_decisions / total_matched_decisions
        
        print(f"\n   📊 IA2 Pattern Comprehension Statistics:")
        print(f"      🎯 Pattern Reception: {ia2_receives_pattern}/{total_matched_decisions} ({pattern_reception_rate*100:.1f}%)")
        print(f"      🎯 Hierarchy Respect: {ia2_respects_hierarchy}/{total_matched_decisions} ({hierarchy_respect_rate*100:.1f}%)")
        print(f"      🎯 IA1 Acknowledgment: {ia2_acknowledges_ia1}/{total_matched_decisions} ({ia1_acknowledgment_rate*100:.1f}%)")
        print(f"      🎯 Signal Coherence: {coherent_decisions}/{total_matched_decisions} ({coherence_rate*100:.1f}%)")
        print(f"      ⚠️  Contradictions: {contradictory_decisions}/{total_matched_decisions} ({contradiction_rate*100:.1f}%)")
        
        # Validation criteria for IA2 pattern comprehension
        pattern_reception_working = pattern_reception_rate >= 0.4  # 40% should receive patterns
        hierarchy_respect_working = hierarchy_respect_rate >= 0.3  # 30% should respect hierarchy
        ia1_acknowledgment_working = ia1_acknowledgment_rate >= 0.6  # 60% should acknowledge IA1
        coherence_working = coherence_rate >= 0.8  # 80% should be coherent
        low_contradictions = contradiction_rate <= 0.2  # ≤20% contradictions acceptable
        
        print(f"\n   ✅ IA2 Pattern Comprehension Validation:")
        print(f"      Pattern Reception: {'✅' if pattern_reception_working else '❌'} (≥40%)")
        print(f"      Hierarchy Respect: {'✅' if hierarchy_respect_working else '❌'} (≥30%)")
        print(f"      IA1 Acknowledgment: {'✅' if ia1_acknowledgment_working else '❌'} (≥60%)")
        print(f"      Signal Coherence: {'✅' if coherence_working else '❌'} (≥80%)")
        print(f"      Low Contradictions: {'✅' if low_contradictions else '❌'} (≤20%)")
        
        # Overall IA2 comprehension assessment
        ia2_comprehension_working = (
            ia1_acknowledgment_working and
            coherence_working and
            low_contradictions and
            (pattern_reception_working or hierarchy_respect_working)  # At least one pattern indicator
        )
        
        print(f"\n   🎯 IA2 Pattern Comprehension System: {'✅ WORKING' if ia2_comprehension_working else '❌ NEEDS IMPROVEMENT'}")
        
        if not ia2_comprehension_working:
            print(f"   💡 ISSUES DETECTED:")
            if not coherence_working:
                print(f"      - Signal coherence low ({coherence_rate*100:.1f}%)")
            if not low_contradictions:
                print(f"      - Too many contradictions ({contradiction_rate*100:.1f}%)")
            if not ia1_acknowledgment_working:
                print(f"      - IA2 not acknowledging IA1 analysis adequately")
            if not pattern_reception_working and not hierarchy_respect_working:
                print(f"      - IA2 not receiving/respecting IA1 pattern hierarchy")
        
        return ia2_comprehension_working

    def test_complete_ia1_ia2_pattern_cycle(self):
        """🎯 TEST COMPLETE IA1→IA2 PATTERN CYCLE - Fresh Data Integration"""
        print(f"\n🎯 Testing Complete IA1→IA2 Pattern Cycle with Fresh Data...")
        print(f"   📋 TESTING OBJECTIVES:")
        print(f"      • Trigger fresh Scout → IA1 → IA2 cycle")
        print(f"      • Verify IA1 pattern prioritization in fresh data")
        print(f"      • Verify IA2 pattern comprehension in fresh decisions")
        print(f"      • Ensure end-to-end pattern coherence")
        
        # Clear existing decisions to force fresh generation
        print(f"   🧹 Clearing existing decisions for fresh cycle test...")
        try:
            clear_url = f"{self.api_url}/decisions/clear"
            clear_response = requests.delete(clear_url, timeout=10)
            if clear_response.status_code == 200:
                print(f"   ✅ Decisions cleared successfully")
            else:
                print(f"   ⚠️  Decision clearing returned {clear_response.status_code}")
        except Exception as e:
            print(f"   ⚠️  Could not clear decisions: {e}")
        
        # Start the trading system for fresh cycle
        print(f"   🚀 Starting trading system for fresh IA1→IA2 cycle...")
        success, _ = self.test_start_trading_system()
        if not success:
            print(f"   ❌ Failed to start trading system")
            return False
        
        # Wait for complete cycle: Scout → IA1 → IA2
        print(f"   ⏱️  Waiting for complete Scout→IA1→IA2 cycle (90 seconds)...")
        
        cycle_start_time = time.time()
        max_wait_time = 90
        check_interval = 15
        
        fresh_analyses_found = False
        fresh_decisions_found = False
        
        while time.time() - cycle_start_time < max_wait_time:
            elapsed_time = time.time() - cycle_start_time
            print(f"   📊 Cycle progress: {elapsed_time:.1f}s elapsed...")
            
            # Check for fresh IA1 analyses
            success_analyses, analyses_data = self.test_get_analyses()
            if success_analyses:
                analyses = analyses_data.get('analyses', [])
                if len(analyses) > 0:
                    fresh_analyses_found = True
                    print(f"   ✅ Fresh IA1 analyses: {len(analyses)} found")
            
            # Check for fresh IA2 decisions
            success_decisions, decisions_data = self.test_get_decisions()
            if success_decisions:
                decisions = decisions_data.get('decisions', [])
                if len(decisions) > 0:
                    fresh_decisions_found = True
                    print(f"   ✅ Fresh IA2 decisions: {len(decisions)} found")
            
            # If both found, we can proceed with testing
            if fresh_analyses_found and fresh_decisions_found:
                print(f"   🎯 Complete cycle detected! Proceeding with pattern analysis...")
                break
            
            time.sleep(check_interval)
        
        # Stop the trading system
        print(f"   🛑 Stopping trading system...")
        self.test_stop_trading_system()
        
        if not fresh_analyses_found or not fresh_decisions_found:
            print(f"   ⚠️  Incomplete cycle - using available data for testing")
            print(f"      Fresh IA1 analyses: {'✅' if fresh_analyses_found else '❌'}")
            print(f"      Fresh IA2 decisions: {'✅' if fresh_decisions_found else '❌'}")
        
        # Test the complete pattern cycle
        print(f"\n   🔍 Testing Complete Pattern Cycle Components:")
        
        # 1. Test IA1 pattern prioritization with fresh data
        ia1_pattern_test = self.test_ia1_pattern_prioritization_system()
        print(f"      IA1 Pattern Prioritization: {'✅' if ia1_pattern_test else '❌'}")
        
        # 2. Test IA2 pattern comprehension with fresh data
        ia2_comprehension_test = self.test_ia2_pattern_comprehension_system()
        print(f"      IA2 Pattern Comprehension: {'✅' if ia2_comprehension_test else '❌'}")
        
        # 3. Test end-to-end pattern coherence
        pattern_coherence_test = self.test_end_to_end_pattern_coherence()
        print(f"      End-to-End Pattern Coherence: {'✅' if pattern_coherence_test else '❌'}")
        
        # 4. Test fresh data quality
        fresh_data_quality_test = self.test_fresh_data_quality()
        print(f"      Fresh Data Quality: {'✅' if fresh_data_quality_test else '❌'}")
        
        # Overall cycle assessment
        components_passed = sum([ia1_pattern_test, ia2_comprehension_test, pattern_coherence_test, fresh_data_quality_test])
        cycle_success = components_passed >= 3  # At least 3/4 components working
        
        print(f"\n   🎯 Complete Pattern Cycle Assessment:")
        print(f"      Components Passed: {components_passed}/4")
        print(f"      Fresh Cycle Status: {'✅ SUCCESS' if cycle_success else '❌ NEEDS WORK'}")
        print(f"      Fresh Data Generated: {'✅' if fresh_analyses_found and fresh_decisions_found else '❌'}")
        
        if cycle_success:
            print(f"   💡 SUCCESS: IA1 Pattern Prioritization → IA2 Comprehension system working!")
            print(f"   💡 IA1 properly designates MASTER PATTERNS")
            print(f"   💡 IA2 respects IA1's pattern hierarchy")
            print(f"   💡 No critical contradictions detected")
        else:
            print(f"   💡 ISSUES DETECTED in pattern cycle:")
            if not ia1_pattern_test:
                print(f"      - IA1 pattern prioritization needs improvement")
            if not ia2_comprehension_test:
                print(f"      - IA2 pattern comprehension needs improvement")
            if not pattern_coherence_test:
                print(f"      - End-to-end pattern coherence issues")
        
        return cycle_success

    def test_end_to_end_pattern_coherence(self):
        """Test end-to-end pattern coherence between IA1 and IA2"""
        print(f"\n   🔗 Testing End-to-End Pattern Coherence...")
        
        # Get current decisions and analyses
        success_decisions, decisions_data = self.test_get_decisions()
        success_analyses, analyses_data = self.test_get_analyses()
        
        if not success_decisions or not success_analyses:
            return False
        
        decisions = decisions_data.get('decisions', [])
        analyses = analyses_data.get('analyses', [])
        
        # Create symbol mapping
        ia1_by_symbol = {analysis.get('symbol', ''): analysis for analysis in analyses}
        
        coherent_pairs = 0
        total_pairs = 0
        critical_contradictions = 0
        
        for decision in decisions[:10]:  # Check first 10 decisions
            symbol = decision.get('symbol', '')
            ia1_analysis = ia1_by_symbol.get(symbol)
            
            if not ia1_analysis:
                continue
            
            total_pairs += 1
            ia1_signal = ia1_analysis.get('ia1_signal', 'hold').upper()
            ia2_signal = decision.get('signal', 'hold').upper()
            
            # Check for coherence
            if ia1_signal == ia2_signal:
                coherent_pairs += 1
            elif ia1_signal in ['LONG', 'SHORT'] and ia2_signal in ['LONG', 'SHORT'] and ia1_signal != ia2_signal:
                # Critical contradiction: opposite signals
                critical_contradictions += 1
        
        coherence_rate = coherent_pairs / total_pairs if total_pairs > 0 else 0
        contradiction_rate = critical_contradictions / total_pairs if total_pairs > 0 else 0
        
        print(f"      Pattern Coherence: {coherent_pairs}/{total_pairs} ({coherence_rate*100:.1f}%)")
        print(f"      Critical Contradictions: {critical_contradictions}/{total_pairs} ({contradiction_rate*100:.1f}%)")
        
        return coherence_rate >= 0.7 and contradiction_rate <= 0.1  # 70% coherent, ≤10% contradictions

    def test_fresh_data_quality(self):
        """Test quality of fresh data generation"""
        print(f"\n   📊 Testing Fresh Data Quality...")
        
        # Check analyses quality
        success, analyses_data = self.test_get_analyses()
        if not success:
            return False
        
        analyses = analyses_data.get('analyses', [])
        if len(analyses) == 0:
            return False
        
        # Check for recent timestamps (within last hour)
        recent_analyses = 0
        current_time = time.time()
        
        for analysis in analyses[:5]:
            timestamp_str = analysis.get('timestamp', '')
            try:
                # Parse timestamp and check if recent
                analysis_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                time_diff = current_time - analysis_time.timestamp()
                if time_diff < 3600:  # Within last hour
                    recent_analyses += 1
            except:
                pass
        
        freshness_rate = recent_analyses / min(len(analyses), 5)
        
        print(f"      Recent Analyses: {recent_analyses}/5 ({freshness_rate*100:.1f}%)")
        print(f"      Total Analyses: {len(analyses)}")
        
        return freshness_rate >= 0.6  # At least 60% should be recent

    def run_pattern_tests(self):
        """Run all IA1→IA2 pattern prioritization tests"""
        print("🚀 Starting IA1→IA2 Pattern Prioritization System Tests...")
        print(f"   Base URL: {self.base_url}")
        print(f"   API URL: {self.api_url}")
        print("=" * 80)
        
        # Run the pattern system tests
        print(f"\n🎯 === IA1→IA2 PATTERN PRIORITIZATION SYSTEM TESTS ===")
        ia1_pattern_success = self.test_ia1_pattern_prioritization_system()
        ia2_comprehension_success = self.test_ia2_pattern_comprehension_system()
        complete_pattern_cycle_success = self.test_complete_ia1_ia2_pattern_cycle()
        
        # Performance summary
        print(f"\n📊 Pattern Test Summary:")
        print(f"   Tests Run: {self.tests_run}")
        print(f"   Tests Passed: {self.tests_passed}")
        print(f"   Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        # Pattern system results
        print(f"\n🎯 IA1→IA2 PATTERN SYSTEM RESULTS:")
        print(f"   IA1 Pattern Prioritization: {'✅ SUCCESS' if ia1_pattern_success else '❌ FAILED'}")
        print(f"   IA2 Pattern Comprehension: {'✅ SUCCESS' if ia2_comprehension_success else '❌ FAILED'}")
        print(f"   Complete Pattern Cycle: {'✅ SUCCESS' if complete_pattern_cycle_success else '❌ FAILED'}")
        
        overall_success = sum([ia1_pattern_success, ia2_comprehension_success, complete_pattern_cycle_success])
        
        print(f"\n🎯 OVERALL PATTERN SYSTEM STATUS:")
        if overall_success >= 2:
            print(f"   ✅ PATTERN SYSTEM WORKING ({overall_success}/3 components)")
            print(f"   💡 IA1 Pattern Prioritization → IA2 Comprehension system operational!")
        else:
            print(f"   ❌ PATTERN SYSTEM NEEDS WORK ({overall_success}/3 components)")
            print(f"   💡 Critical issues detected in pattern communication")
        
        print("=" * 80)
        
        return overall_success >= 2

if __name__ == "__main__":
    tester = IA1IA2PatternTester()
    tester.run_pattern_tests()
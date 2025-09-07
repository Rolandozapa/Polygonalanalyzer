"""
IA1 Pattern Prioritization Tests for IA2
Tests the MASTER PATTERN system and complete reasoning transmission
"""

import requests
import json
import time
from pathlib import Path

class IA1PatternPrioritizationTester:
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
                    base_url = "https://aitra-platform.preview.emergentagent.com"
            except:
                base_url = "https://aitra-platform.preview.emergentagent.com"
        
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
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                try:
                    response_data = response.json()
                    return True, response_data
                except:
                    return True, {}
            else:
                return False, {}

        except Exception as e:
            return False, {}

    def test_get_analyses(self):
        """Test get analyses endpoint"""
        return self.run_test("Get Technical Analyses", "GET", "analyses", 200)

    def test_get_decisions(self):
        """Test get decisions endpoint"""
        return self.run_test("Get Trading Decisions", "GET", "decisions", 200)

    def test_ia1_master_pattern_prioritization(self):
        """🎯 TEST CRITIQUE - IA1 Pattern Prioritization pour IA2"""
        print(f"\n🎯 Testing IA1 MASTER PATTERN Prioritization System...")
        print(f"   📋 CRITICAL TESTING LOGIC:")
        print(f"      • IA1 must designate its 'MASTER PATTERN' as strategic choice")
        print(f"      • IA2 receives complete IA1 reasoning (no 500 char limit)")
        print(f"      • IA2 respects IA1's MASTER PATTERN hierarchy")
        print(f"      • No more contradictions: IA1 'SHORT via rising_wedge' → IA2 'LONG'")
        print(f"      • Test case: PYTHUSDT with rising_wedge pattern")
        
        # Get current analyses to check IA1 MASTER PATTERN designation
        success, analyses_data = self.test_get_analyses()
        if not success:
            print(f"   ❌ Cannot retrieve IA1 analyses for MASTER PATTERN testing")
            return False
        
        analyses = analyses_data.get('analyses', [])
        if len(analyses) == 0:
            print(f"   ❌ No IA1 analyses available for MASTER PATTERN testing")
            return False
        
        print(f"   📊 Analyzing MASTER PATTERN designation in {len(analyses)} IA1 analyses...")
        
        # Analyze IA1 analyses for MASTER PATTERN evidence
        master_pattern_count = 0
        strategic_choice_count = 0
        pattern_hierarchy_count = 0
        pythusdt_found = False
        pythusdt_analysis = None
        
        master_pattern_examples = []
        
        for i, analysis in enumerate(analyses[:20]):  # Analyze first 20 analyses
            symbol = analysis.get('symbol', 'Unknown')
            reasoning = analysis.get('ia1_reasoning', '')
            patterns = analysis.get('patterns_detected', [])
            ia1_signal = analysis.get('ia1_signal', 'hold')
            confidence = analysis.get('analysis_confidence', 0)
            
            # Check for MASTER PATTERN designation in reasoning
            has_master_pattern = '🎯 MASTER PATTERN' in reasoning or 'MASTER PATTERN' in reasoning
            has_strategic_choice = 'IA1 STRATEGIC CHOICE' in reasoning or 'STRATEGIC CHOICE' in reasoning
            has_pattern_hierarchy = 'PRIMARY BASIS' in reasoning or 'PATTERN HIERARCHY' in reasoning
            
            # Count MASTER PATTERN usage
            if has_master_pattern:
                master_pattern_count += 1
                master_pattern_examples.append({
                    'symbol': symbol,
                    'signal': ia1_signal,
                    'confidence': confidence,
                    'patterns': patterns,
                    'reasoning_preview': reasoning[:200]
                })
            
            if has_strategic_choice:
                strategic_choice_count += 1
            
            if has_pattern_hierarchy:
                pattern_hierarchy_count += 1
            
            # Special check for PYTHUSDT test case
            if symbol.upper() == 'PYTHUSDT':
                pythusdt_found = True
                pythusdt_analysis = {
                    'symbol': symbol,
                    'signal': ia1_signal,
                    'confidence': confidence,
                    'patterns': patterns,
                    'reasoning': reasoning,
                    'has_master_pattern': has_master_pattern,
                    'has_rising_wedge': 'rising_wedge' in str(patterns).lower() or 'rising wedge' in reasoning.lower()
                }
            
            # Show first few examples
            if i < 3:
                print(f"\n   IA1 Analysis {i+1} - {symbol}:")
                print(f"      Signal: {ia1_signal}")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Patterns: {patterns}")
                print(f"      MASTER PATTERN: {'✅' if has_master_pattern else '❌'}")
                print(f"      Strategic Choice: {'✅' if has_strategic_choice else '❌'}")
                print(f"      Reasoning Preview: {reasoning[:150]}...")
        
        # Get IA2 decisions to check pattern respect
        success, decisions_data = self.test_get_decisions()
        if not success:
            print(f"   ❌ Cannot retrieve IA2 decisions for pattern respect testing")
            return False
        
        decisions = decisions_data.get('decisions', [])
        
        # Analyze IA2 decisions for pattern hierarchy respect
        ia2_pattern_respect_count = 0
        ia2_master_pattern_mentions = 0
        pythusdt_ia2_decision = None
        
        for decision in decisions[:20]:
            symbol = decision.get('symbol', 'Unknown')
            signal = decision.get('signal', 'hold')
            reasoning = decision.get('ia2_reasoning', '')
            
            # Check if IA2 mentions MASTER PATTERN or pattern hierarchy
            mentions_master_pattern = 'MASTER PATTERN' in reasoning or 'master pattern' in reasoning.lower()
            mentions_pattern_hierarchy = 'pattern hierarchy' in reasoning.lower() or 'primary pattern' in reasoning.lower()
            mentions_ia1_pattern = 'ia1' in reasoning.lower() and 'pattern' in reasoning.lower()
            
            if mentions_master_pattern:
                ia2_master_pattern_mentions += 1
            
            if mentions_pattern_hierarchy or mentions_ia1_pattern:
                ia2_pattern_respect_count += 1
            
            # Check for PYTHUSDT IA2 decision
            if symbol.upper() == 'PYTHUSDT':
                pythusdt_ia2_decision = {
                    'symbol': symbol,
                    'signal': signal,
                    'reasoning': reasoning,
                    'mentions_master_pattern': mentions_master_pattern,
                    'mentions_pattern_hierarchy': mentions_pattern_hierarchy
                }
        
        # Calculate statistics
        total_analyses = len(analyses)
        total_decisions = len(decisions)
        master_pattern_rate = master_pattern_count / total_analyses if total_analyses > 0 else 0
        strategic_choice_rate = strategic_choice_count / total_analyses if total_analyses > 0 else 0
        pattern_hierarchy_rate = pattern_hierarchy_count / total_analyses if total_analyses > 0 else 0
        ia2_respect_rate = ia2_pattern_respect_count / total_decisions if total_decisions > 0 else 0
        
        print(f"\n   📊 IA1 MASTER PATTERN Analysis:")
        print(f"      Total IA1 Analyses: {total_analyses}")
        print(f"      MASTER PATTERN Designation: {master_pattern_count} ({master_pattern_rate*100:.1f}%)")
        print(f"      Strategic Choice Mentions: {strategic_choice_count} ({strategic_choice_rate*100:.1f}%)")
        print(f"      Pattern Hierarchy: {pattern_hierarchy_count} ({pattern_hierarchy_rate*100:.1f}%)")
        
        print(f"\n   📊 IA2 Pattern Respect Analysis:")
        print(f"      Total IA2 Decisions: {total_decisions}")
        print(f"      Pattern Respect: {ia2_pattern_respect_count} ({ia2_respect_rate*100:.1f}%)")
        print(f"      MASTER PATTERN Mentions: {ia2_master_pattern_mentions}")
        
        # Show MASTER PATTERN examples
        if master_pattern_examples:
            print(f"\n   ✅ IA1 MASTER PATTERN Examples:")
            for i, example in enumerate(master_pattern_examples[:3]):
                print(f"      {i+1}. {example['symbol']} → {example['signal'].upper()} (Confidence: {example['confidence']:.3f})")
                print(f"         Patterns: {example['patterns']}")
                print(f"         Preview: {example['reasoning_preview']}...")
        
        # Special PYTHUSDT test case analysis
        print(f"\n   🎯 PYTHUSDT Test Case Analysis:")
        if pythusdt_found and pythusdt_analysis:
            print(f"      ✅ PYTHUSDT IA1 Analysis Found:")
            print(f"         Signal: {pythusdt_analysis['signal']}")
            print(f"         Patterns: {pythusdt_analysis['patterns']}")
            print(f"         MASTER PATTERN: {'✅' if pythusdt_analysis['has_master_pattern'] else '❌'}")
            print(f"         Rising Wedge: {'✅' if pythusdt_analysis['has_rising_wedge'] else '❌'}")
            
            if pythusdt_ia2_decision:
                print(f"      ✅ PYTHUSDT IA2 Decision Found:")
                print(f"         IA2 Signal: {pythusdt_ia2_decision['signal']}")
                print(f"         Mentions MASTER PATTERN: {'✅' if pythusdt_ia2_decision['mentions_master_pattern'] else '❌'}")
                print(f"         Pattern Hierarchy Respect: {'✅' if pythusdt_ia2_decision['mentions_pattern_hierarchy'] else '❌'}")
                
                # Check for contradiction
                ia1_signal = pythusdt_analysis['signal'].lower()
                ia2_signal = pythusdt_ia2_decision['signal'].lower()
                contradiction = (ia1_signal == 'short' and ia2_signal == 'long') or (ia1_signal == 'long' and ia2_signal == 'short')
                
                print(f"         Contradiction Check: {'❌ CONTRADICTION' if contradiction else '✅ NO CONTRADICTION'}")
                print(f"         IA1: {ia1_signal.upper()} → IA2: {ia2_signal.upper()}")
            else:
                print(f"      ⚠️  PYTHUSDT IA2 Decision Not Found")
        else:
            print(f"      ⚠️  PYTHUSDT IA1 Analysis Not Found")
        
        # Validation criteria for MASTER PATTERN system
        ia1_designates_master = master_pattern_rate >= 0.30  # At least 30% should designate MASTER PATTERN
        strategic_choice_present = strategic_choice_rate >= 0.20  # At least 20% should mention strategic choice
        ia2_respects_patterns = ia2_respect_rate >= 0.25  # At least 25% should respect pattern hierarchy
        master_pattern_transmission = ia2_master_pattern_mentions > 0  # IA2 should mention MASTER PATTERN
        
        print(f"\n   ✅ MASTER PATTERN System Validation:")
        print(f"      IA1 Designates MASTER ≥30%: {'✅' if ia1_designates_master else '❌'} ({master_pattern_rate*100:.1f}%)")
        print(f"      Strategic Choice ≥20%: {'✅' if strategic_choice_present else '❌'} ({strategic_choice_rate*100:.1f}%)")
        print(f"      IA2 Respects Patterns ≥25%: {'✅' if ia2_respects_patterns else '❌'} ({ia2_respect_rate*100:.1f}%)")
        print(f"      MASTER PATTERN Transmission: {'✅' if master_pattern_transmission else '❌'} ({ia2_master_pattern_mentions} mentions)")
        
        master_pattern_system_working = (
            ia1_designates_master and
            strategic_choice_present and
            ia2_respects_patterns and
            master_pattern_transmission
        )
        
        print(f"\n   🎯 IA1 MASTER PATTERN Prioritization: {'✅ CRITICAL SUCCESS' if master_pattern_system_working else '❌ NEEDS IMPLEMENTATION'}")
        
        if not master_pattern_system_working:
            print(f"   💡 CRITICAL ISSUES IDENTIFIED:")
            if not ia1_designates_master:
                print(f"      - IA1 not designating MASTER PATTERN consistently ({master_pattern_rate*100:.1f}%)")
            if not strategic_choice_present:
                print(f"      - Missing 'IA1 STRATEGIC CHOICE' indicators ({strategic_choice_rate*100:.1f}%)")
            if not ia2_respects_patterns:
                print(f"      - IA2 not respecting pattern hierarchy ({ia2_respect_rate*100:.1f}%)")
            if not master_pattern_transmission:
                print(f"      - MASTER PATTERN not transmitted to IA2 ({ia2_master_pattern_mentions} mentions)")
        else:
            print(f"   🎉 SUCCESS: IA1 MASTER PATTERN system resolves IA2 pattern confusion!")
            print(f"   🎉 IA2 now understands which pattern is IA1's strategic basis!")
        
        return master_pattern_system_working

    def test_ia1_ia2_pattern_transmission_complete(self):
        """🔄 Test Complete IA1 Reasoning Transmission to IA2 (No 500 char limit)"""
        print(f"\n🔄 Testing Complete IA1 → IA2 Reasoning Transmission...")
        print(f"   📋 TRANSMISSION TESTING LOGIC:")
        print(f"      • IA2 receives COMPLETE IA1 reasoning (not truncated)")
        print(f"      • No more 500 character limitation")
        print(f"      • IA2 can access full pattern hierarchy context")
        print(f"      • Complete transmission enables better decision making")
        
        # Get IA1 analyses and IA2 decisions for transmission analysis
        success_ia1, analyses_data = self.test_get_analyses()
        success_ia2, decisions_data = self.test_get_decisions()
        
        if not success_ia1 or not success_ia2:
            print(f"   ❌ Cannot retrieve data for transmission testing")
            return False
        
        analyses = analyses_data.get('analyses', [])
        decisions = decisions_data.get('decisions', [])
        
        if len(analyses) == 0 or len(decisions) == 0:
            print(f"   ❌ Insufficient data for transmission testing")
            return False
        
        print(f"   📊 Analyzing transmission completeness...")
        print(f"      IA1 Analyses: {len(analyses)}")
        print(f"      IA2 Decisions: {len(decisions)}")
        
        # Analyze IA1 reasoning lengths
        ia1_reasoning_lengths = []
        ia1_detailed_analyses = 0
        
        for analysis in analyses[:10]:
            reasoning = analysis.get('ia1_reasoning', '')
            reasoning_length = len(reasoning)
            ia1_reasoning_lengths.append(reasoning_length)
            
            if reasoning_length > 200:  # Detailed analysis
                ia1_detailed_analyses += 1
        
        # Analyze IA2 reasoning for IA1 content references
        ia2_references_ia1 = 0
        ia2_detailed_reasoning = 0
        ia2_pattern_references = 0
        transmission_evidence = []
        
        for decision in decisions[:10]:
            symbol = decision.get('symbol', 'Unknown')
            ia2_reasoning = decision.get('ia2_reasoning', '')
            reasoning_length = len(ia2_reasoning)
            
            if reasoning_length > 300:  # Detailed IA2 reasoning
                ia2_detailed_reasoning += 1
            
            # Check for IA1 content references
            references_ia1 = ('ia1' in ia2_reasoning.lower() or 
                             'technical analysis' in ia2_reasoning.lower() or
                             'pattern' in ia2_reasoning.lower())
            
            if references_ia1:
                ia2_references_ia1 += 1
            
            # Check for pattern references (indicating transmission)
            pattern_keywords = ['rising_wedge', 'double_bottom', 'triangle', 'channel', 'support', 'resistance']
            has_pattern_ref = any(keyword in ia2_reasoning.lower() for keyword in pattern_keywords)
            
            if has_pattern_ref:
                ia2_pattern_references += 1
                transmission_evidence.append({
                    'symbol': symbol,
                    'reasoning_length': reasoning_length,
                    'pattern_references': [kw for kw in pattern_keywords if kw in ia2_reasoning.lower()],
                    'preview': ia2_reasoning[:150]
                })
        
        # Calculate transmission statistics
        avg_ia1_length = sum(ia1_reasoning_lengths) / len(ia1_reasoning_lengths) if ia1_reasoning_lengths else 0
        max_ia1_length = max(ia1_reasoning_lengths) if ia1_reasoning_lengths else 0
        min_ia1_length = min(ia1_reasoning_lengths) if ia1_reasoning_lengths else 0
        
        ia1_detailed_rate = ia1_detailed_analyses / len(analyses[:10]) if len(analyses) > 0 else 0
        ia2_reference_rate = ia2_references_ia1 / len(decisions[:10]) if len(decisions) > 0 else 0
        ia2_pattern_rate = ia2_pattern_references / len(decisions[:10]) if len(decisions) > 0 else 0
        
        print(f"\n   📊 IA1 Reasoning Analysis:")
        print(f"      Average Length: {avg_ia1_length:.0f} chars")
        print(f"      Max Length: {max_ia1_length} chars")
        print(f"      Min Length: {min_ia1_length} chars")
        print(f"      Detailed Analyses (>200 chars): {ia1_detailed_analyses}/10 ({ia1_detailed_rate*100:.1f}%)")
        
        print(f"\n   📊 IA2 Transmission Analysis:")
        print(f"      References IA1 Content: {ia2_references_ia1}/10 ({ia2_reference_rate*100:.1f}%)")
        print(f"      Pattern References: {ia2_pattern_references}/10 ({ia2_pattern_rate*100:.1f}%)")
        print(f"      Detailed IA2 Reasoning: {ia2_detailed_reasoning}/10")
        
        # Show transmission evidence
        if transmission_evidence:
            print(f"\n   ✅ Transmission Evidence Examples:")
            for i, evidence in enumerate(transmission_evidence[:3]):
                print(f"      {i+1}. {evidence['symbol']} ({evidence['reasoning_length']} chars):")
                print(f"         Pattern Refs: {evidence['pattern_references']}")
                print(f"         Preview: {evidence['preview']}...")
        
        # Validation criteria for complete transmission
        ia1_generates_detailed = ia1_detailed_rate >= 0.5  # At least 50% detailed IA1 analyses
        no_500_char_limit = max_ia1_length > 500  # IA1 can generate >500 chars
        ia2_receives_content = ia2_reference_rate >= 0.4  # At least 40% IA2 references IA1
        pattern_transmission = ia2_pattern_rate >= 0.3  # At least 30% pattern transmission
        complete_reasoning = avg_ia1_length >= 150  # Average IA1 reasoning substantial
        
        print(f"\n   ✅ Complete Transmission Validation:")
        print(f"      IA1 Detailed Generation ≥50%: {'✅' if ia1_generates_detailed else '❌'} ({ia1_detailed_rate*100:.1f}%)")
        print(f"      No 500 Char Limit: {'✅' if no_500_char_limit else '❌'} (Max: {max_ia1_length} chars)")
        print(f"      IA2 Receives Content ≥40%: {'✅' if ia2_receives_content else '❌'} ({ia2_reference_rate*100:.1f}%)")
        print(f"      Pattern Transmission ≥30%: {'✅' if pattern_transmission else '❌'} ({ia2_pattern_rate*100:.1f}%)")
        print(f"      Complete Reasoning: {'✅' if complete_reasoning else '❌'} (Avg: {avg_ia1_length:.0f} chars)")
        
        transmission_working = (
            ia1_generates_detailed and
            no_500_char_limit and
            ia2_receives_content and
            pattern_transmission and
            complete_reasoning
        )
        
        print(f"\n   🔄 Complete IA1 → IA2 Transmission: {'✅ SUCCESS' if transmission_working else '❌ NEEDS IMPROVEMENT'}")
        
        if not transmission_working:
            print(f"   💡 TRANSMISSION ISSUES:")
            if not ia1_generates_detailed:
                print(f"      - IA1 not generating detailed analyses ({ia1_detailed_rate*100:.1f}%)")
            if not no_500_char_limit:
                print(f"      - Possible 500 char limit still active (Max: {max_ia1_length})")
            if not ia2_receives_content:
                print(f"      - IA2 not receiving IA1 content ({ia2_reference_rate*100:.1f}%)")
            if not pattern_transmission:
                print(f"      - Pattern transmission insufficient ({ia2_pattern_rate*100:.1f}%)")
        
        return transmission_working

    def test_ia1_ia2_contradiction_resolution(self):
        """🚫 Test IA1-IA2 Contradiction Resolution System"""
        print(f"\n🚫 Testing IA1-IA2 Contradiction Resolution...")
        print(f"   📋 CONTRADICTION TESTING LOGIC:")
        print(f"      • Identify cases where IA1 recommends one direction")
        print(f"      • Check if IA2 contradicts with opposite direction")
        print(f"      • Validate MASTER PATTERN respect prevents contradictions")
        print(f"      • Test case: IA1 'SHORT via rising_wedge' should NOT → IA2 'LONG'")
        
        # Get IA1 analyses and IA2 decisions
        success_ia1, analyses_data = self.test_get_analyses()
        success_ia2, decisions_data = self.test_get_decisions()
        
        if not success_ia1 or not success_ia2:
            print(f"   ❌ Cannot retrieve data for contradiction testing")
            return False
        
        analyses = analyses_data.get('analyses', [])
        decisions = decisions_data.get('decisions', [])
        
        if len(analyses) == 0 or len(decisions) == 0:
            print(f"   ❌ Insufficient data for contradiction testing")
            return False
        
        print(f"   📊 Analyzing contradictions in {len(analyses)} IA1 analyses and {len(decisions)} IA2 decisions...")
        
        # Create symbol-based mapping for comparison
        ia1_signals = {}
        ia2_signals = {}
        
        for analysis in analyses:
            symbol = analysis.get('symbol', '')
            signal = analysis.get('ia1_signal', 'hold').lower()
            reasoning = analysis.get('ia1_reasoning', '')
            patterns = analysis.get('patterns_detected', [])
            
            ia1_signals[symbol] = {
                'signal': signal,
                'reasoning': reasoning,
                'patterns': patterns,
                'has_master_pattern': 'MASTER PATTERN' in reasoning
            }
        
        for decision in decisions:
            symbol = decision.get('symbol', '')
            signal = decision.get('signal', 'hold').lower()
            reasoning = decision.get('ia2_reasoning', '')
            
            ia2_signals[symbol] = {
                'signal': signal,
                'reasoning': reasoning,
                'mentions_ia1_pattern': 'ia1' in reasoning.lower() and 'pattern' in reasoning.lower()
            }
        
        # Find contradictions
        contradictions = []
        resolved_cases = []
        total_comparisons = 0
        
        for symbol in ia1_signals:
            if symbol in ia2_signals:
                total_comparisons += 1
                ia1_data = ia1_signals[symbol]
                ia2_data = ia2_signals[symbol]
                
                ia1_signal = ia1_data['signal']
                ia2_signal = ia2_data['signal']
                
                # Check for direct contradictions (opposite signals)
                is_contradiction = (
                    (ia1_signal == 'long' and ia2_signal == 'short') or
                    (ia1_signal == 'short' and ia2_signal == 'long')
                )
                
                if is_contradiction:
                    contradictions.append({
                        'symbol': symbol,
                        'ia1_signal': ia1_signal,
                        'ia2_signal': ia2_signal,
                        'ia1_has_master_pattern': ia1_data['has_master_pattern'],
                        'ia2_mentions_ia1_pattern': ia2_data['mentions_ia1_pattern'],
                        'ia1_patterns': ia1_data['patterns'],
                        'ia1_reasoning_preview': ia1_data['reasoning'][:150],
                        'ia2_reasoning_preview': ia2_data['reasoning'][:150]
                    })
                else:
                    # Cases where contradiction was avoided
                    if ia1_signal in ['long', 'short'] and ia2_signal == ia1_signal:
                        resolved_cases.append({
                            'symbol': symbol,
                            'agreed_signal': ia1_signal,
                            'ia1_has_master_pattern': ia1_data['has_master_pattern'],
                            'ia2_mentions_ia1_pattern': ia2_data['mentions_ia1_pattern']
                        })
        
        # Calculate contradiction statistics
        contradiction_count = len(contradictions)
        resolved_count = len(resolved_cases)
        contradiction_rate = (contradiction_count / total_comparisons * 100) if total_comparisons > 0 else 0
        resolution_rate = (resolved_count / total_comparisons * 100) if total_comparisons > 0 else 0
        
        print(f"\n   📊 Contradiction Analysis:")
        print(f"      Total Symbol Comparisons: {total_comparisons}")
        print(f"      Contradictions Found: {contradiction_count} ({contradiction_rate:.1f}%)")
        print(f"      Resolved Cases (Agreement): {resolved_count} ({resolution_rate:.1f}%)")
        
        # Show contradiction examples
        if contradictions:
            print(f"\n   ❌ CONTRADICTIONS FOUND:")
            for i, contradiction in enumerate(contradictions[:5]):
                print(f"      {i+1}. {contradiction['symbol']}:")
                print(f"         IA1: {contradiction['ia1_signal'].upper()} (MASTER PATTERN: {'✅' if contradiction['ia1_has_master_pattern'] else '❌'})")
                print(f"         IA2: {contradiction['ia2_signal'].upper()} (Mentions IA1 Pattern: {'✅' if contradiction['ia2_mentions_ia1_pattern'] else '❌'})")
                print(f"         IA1 Patterns: {contradiction['ia1_patterns']}")
                print(f"         IA1 Preview: {contradiction['ia1_reasoning_preview']}...")
                print(f"         IA2 Preview: {contradiction['ia2_reasoning_preview']}...")
        
        # Show resolved cases
        if resolved_cases:
            print(f"\n   ✅ RESOLVED CASES (No Contradictions):")
            for i, resolved in enumerate(resolved_cases[:3]):
                print(f"      {i+1}. {resolved['symbol']}: Both agree on {resolved['agreed_signal'].upper()}")
                print(f"         MASTER PATTERN: {'✅' if resolved['ia1_has_master_pattern'] else '❌'}")
                print(f"         IA2 Pattern Respect: {'✅' if resolved['ia2_mentions_ia1_pattern'] else '❌'}")
        
        # Special check for PYTHUSDT rising_wedge case
        pythusdt_contradiction_check = None
        for contradiction in contradictions:
            if contradiction['symbol'].upper() == 'PYTHUSDT':
                # Check if it's the specific rising_wedge → SHORT vs LONG case
                has_rising_wedge = any('rising' in str(pattern).lower() for pattern in contradiction['ia1_patterns'])
                if has_rising_wedge and contradiction['ia1_signal'] == 'short' and contradiction['ia2_signal'] == 'long':
                    pythusdt_contradiction_check = contradiction
                    break
        
        print(f"\n   🎯 PYTHUSDT Rising Wedge Test Case:")
        if pythusdt_contradiction_check:
            print(f"      ❌ CRITICAL CONTRADICTION FOUND:")
            print(f"         IA1: SHORT via rising_wedge")
            print(f"         IA2: LONG (ignoring MASTER PATTERN)")
            print(f"         This is the exact issue the fix should resolve!")
        else:
            print(f"      ✅ No PYTHUSDT rising_wedge contradiction found")
            print(f"      (Either case not present or successfully resolved)")
        
        # Validation criteria for contradiction resolution
        no_critical_contradictions = contradiction_count == 0  # Ideal: zero contradictions
        low_contradiction_rate = contradiction_rate <= 5.0  # Acceptable: ≤5% contradictions
        master_pattern_effectiveness = True  # Check if MASTER PATTERN helps
        
        # Check if MASTER PATTERN reduces contradictions
        contradictions_with_master = sum(1 for c in contradictions if c['ia1_has_master_pattern'])
        contradictions_without_master = contradiction_count - contradictions_with_master
        
        if contradiction_count > 0:
            master_pattern_helps = (contradictions_without_master > contradictions_with_master)
            master_pattern_effectiveness = master_pattern_helps
        
        print(f"\n   ✅ Contradiction Resolution Validation:")
        print(f"      No Critical Contradictions: {'✅' if no_critical_contradictions else '❌'} ({contradiction_count} found)")
        print(f"      Low Contradiction Rate ≤5%: {'✅' if low_contradiction_rate else '❌'} ({contradiction_rate:.1f}%)")
        print(f"      MASTER PATTERN Effectiveness: {'✅' if master_pattern_effectiveness else '❌'}")
        
        if contradiction_count > 0:
            print(f"      Contradictions with MASTER PATTERN: {contradictions_with_master}")
            print(f"      Contradictions without MASTER PATTERN: {contradictions_without_master}")
        
        # Overall assessment
        contradiction_resolution_working = (
            no_critical_contradictions or low_contradiction_rate
        ) and master_pattern_effectiveness
        
        print(f"\n   🚫 Contradiction Resolution System: {'✅ SUCCESS' if contradiction_resolution_working else '❌ NEEDS WORK'}")
        
        if not contradiction_resolution_working:
            print(f"   💡 CONTRADICTION ISSUES:")
            if not (no_critical_contradictions or low_contradiction_rate):
                print(f"      - Too many contradictions: {contradiction_count} ({contradiction_rate:.1f}%)")
            if not master_pattern_effectiveness:
                print(f"      - MASTER PATTERN not effectively preventing contradictions")
            if pythusdt_contradiction_check:
                print(f"      - CRITICAL: PYTHUSDT rising_wedge case still contradicting")
        else:
            print(f"   🎉 SUCCESS: IA1-IA2 contradiction resolution working!")
            print(f"   🎉 MASTER PATTERN system preventing signal conflicts!")
        
        return contradiction_resolution_working

    def run_pattern_prioritization_tests(self):
        """Run all IA1 Pattern Prioritization tests"""
        print("🎯 Starting IA1 Pattern Prioritization Tests for IA2")
        print("=" * 80)
        print(f"🔧 Testing CRITICAL FIX: IA1 MASTER PATTERN designation and transmission")
        print(f"🎯 Expected: IA1 designates MASTER PATTERN, IA2 respects hierarchy")
        print(f"🎯 Expected: Complete reasoning transmission (no 500 char limit)")
        print(f"🎯 Expected: No contradictions between IA1 and IA2 signals")
        print("=" * 80)
        
        # Test 1: IA1 MASTER PATTERN Prioritization
        print(f"\n1️⃣ IA1 MASTER PATTERN PRIORITIZATION TEST")
        master_pattern_test = self.test_ia1_master_pattern_prioritization()
        
        # Test 2: Complete IA1 → IA2 Reasoning Transmission
        print(f"\n2️⃣ COMPLETE IA1 → IA2 REASONING TRANSMISSION TEST")
        transmission_test = self.test_ia1_ia2_pattern_transmission_complete()
        
        # Test 3: IA1-IA2 Contradiction Resolution
        print(f"\n3️⃣ IA1-IA2 CONTRADICTION RESOLUTION TEST")
        contradiction_test = self.test_ia1_ia2_contradiction_resolution()
        
        # Results Summary
        print("\n" + "=" * 80)
        print("📊 IA1 PATTERN PRIORITIZATION TEST RESULTS")
        print("=" * 80)
        
        print(f"\n🔍 Test Results Summary:")
        print(f"   • IA1 MASTER PATTERN Prioritization: {'✅' if master_pattern_test else '❌'}")
        print(f"   • Complete IA1 → IA2 Transmission: {'✅' if transmission_test else '❌'}")
        print(f"   • IA1-IA2 Contradiction Resolution: {'✅' if contradiction_test else '❌'}")
        
        # Critical assessment
        critical_tests = [master_pattern_test, transmission_test, contradiction_test]
        critical_passed = sum(critical_tests)
        
        print(f"\n🎯 PATTERN PRIORITIZATION Assessment:")
        if critical_passed == 3:
            print(f"   ✅ IA1 PATTERN PRIORITIZATION SUCCESSFUL")
            print(f"   ✅ All critical components working properly")
            fix_status = "SUCCESS"
        elif critical_passed >= 2:
            print(f"   ⚠️  IA1 PATTERN PRIORITIZATION PARTIAL")
            print(f"   ⚠️  Most components working, minor issues detected")
            fix_status = "PARTIAL"
        else:
            print(f"   ❌ IA1 PATTERN PRIORITIZATION FAILED")
            print(f"   ❌ Critical issues detected - fix not working")
            fix_status = "FAILED"
        
        # Specific feedback on the critical fix
        print(f"\n📋 Critical Fix Status:")
        print(f"   • MASTER PATTERN Designation: {'✅' if master_pattern_test else '❌ CRITICAL ISSUE'}")
        print(f"   • Complete Reasoning Transmission: {'✅' if transmission_test else '❌ Still truncated'}")
        print(f"   • Contradiction Resolution: {'✅' if contradiction_test else '❌ Still conflicting'}")
        
        print(f"\n📋 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        return fix_status, {
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_run,
            "master_pattern_working": master_pattern_test,
            "transmission_working": transmission_test,
            "contradiction_resolution_working": contradiction_test
        }

if __name__ == "__main__":
    tester = IA1PatternPrioritizationTester()
    status, results = tester.run_pattern_prioritization_tests()
    print(f"\nFinal Status: {status}")
    print(f"Results: {results}")
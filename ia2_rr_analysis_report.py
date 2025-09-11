#!/usr/bin/env python3
"""
IA2 RR Calculation Analysis Report
Direct database analysis to validate the IA2 RR calculation fix

This script analyzes existing IA2 decisions in the database to check:
1. Presence of calculated_rr and rr_reasoning fields
2. RR calculation consistency with simple S/R formula
3. Elimination of fallback RR patterns
4. Quality of rr_reasoning explanations
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import subprocess
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA2RRAnalysisReport:
    """Generate comprehensive analysis report of IA2 RR calculation fix"""
    
    def __init__(self):
        self.analysis_results = {}
        self.ia2_decisions = []
        
    def run_mongo_query(self, query: str) -> List[Dict]:
        """Execute MongoDB query and return results"""
        try:
            cmd = f'mongosh myapp --eval "{query}" --quiet'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse the JSON output
                output = result.stdout.strip()
                if output and output != 'null':
                    try:
                        return json.loads(output)
                    except json.JSONDecodeError:
                        # Try to extract JSON from the output
                        json_match = re.search(r'\[.*\]', output, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group())
                        else:
                            logger.warning(f"Could not parse MongoDB output: {output[:200]}...")
                            return []
                return []
            else:
                logger.error(f"MongoDB query failed: {result.stderr}")
                return []
        except Exception as e:
            logger.error(f"Error executing MongoDB query: {e}")
            return []
    
    def analyze_ia2_decisions_overview(self):
        """Analyze overview of IA2 decisions"""
        logger.info("🔍 ANALYZING IA2 DECISIONS OVERVIEW")
        
        # Get total IA2 decisions
        query = 'db.trading_decisions.find({ia2_reasoning: {$exists: true}}).count()'
        total_ia2 = self.run_mongo_query(query)
        
        # Get recent IA2 decisions (last 7 days)
        seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
        query = f'db.trading_decisions.find({{ia2_reasoning: {{$exists: true}}, timestamp: {{$gte: ISODate("{seven_days_ago}")}}}}).count()'
        recent_ia2 = self.run_mongo_query(query)
        
        # Get IA2 decisions with calculated_rr field
        query = 'db.trading_decisions.find({ia2_reasoning: {$exists: true}, calculated_rr: {$exists: true}}).count()'
        with_calculated_rr = self.run_mongo_query(query)
        
        # Get IA2 decisions with rr_reasoning field
        query = 'db.trading_decisions.find({ia2_reasoning: {$exists: true}, rr_reasoning: {$exists: true}}).count()'
        with_rr_reasoning = self.run_mongo_query(query)
        
        self.analysis_results['overview'] = {
            'total_ia2_decisions': total_ia2 if isinstance(total_ia2, int) else 0,
            'recent_ia2_decisions': recent_ia2 if isinstance(recent_ia2, int) else 0,
            'with_calculated_rr': with_calculated_rr if isinstance(with_calculated_rr, int) else 0,
            'with_rr_reasoning': with_rr_reasoning if isinstance(with_rr_reasoning, int) else 0
        }
        
        logger.info(f"   📊 Total IA2 decisions: {self.analysis_results['overview']['total_ia2_decisions']}")
        logger.info(f"   📊 Recent IA2 decisions (7 days): {self.analysis_results['overview']['recent_ia2_decisions']}")
        logger.info(f"   📊 With calculated_rr field: {self.analysis_results['overview']['with_calculated_rr']}")
        logger.info(f"   📊 With rr_reasoning field: {self.analysis_results['overview']['with_rr_reasoning']}")
    
    def analyze_recent_ia2_decisions(self):
        """Analyze recent IA2 decisions in detail"""
        logger.info("\n🔍 ANALYZING RECENT IA2 DECISIONS")
        
        # Get recent IA2 decisions with all relevant fields
        query = '''db.trading_decisions.find({
            ia2_reasoning: {$exists: true},
            timestamp: {$gte: ISODate("2025-09-10T00:00:00.000Z")}
        }, {
            symbol: 1,
            signal: 1,
            confidence: 1,
            calculated_rr: 1,
            rr_reasoning: 1,
            risk_reward_ratio: 1,
            entry_price: 1,
            stop_loss: 1,
            take_profit_1: 1,
            timestamp: 1,
            ia2_reasoning: 1
        }).sort({timestamp: -1}).limit(10).toArray()'''
        
        decisions = self.run_mongo_query(query)
        self.ia2_decisions = decisions
        
        logger.info(f"   📊 Analyzing {len(decisions)} recent IA2 decisions")
        
        # Analyze each decision
        calculated_rr_analysis = []
        rr_reasoning_analysis = []
        formula_validation = []
        fallback_patterns = []
        
        fallback_keywords = [
            "1.00:1 (IA1 R:R unavailable)",
            "IA1 R:R unavailable", 
            "fallback",
            "default R:R",
            "unavailable",
            "deemed suboptimal"
        ]
        
        for decision in decisions:
            symbol = decision.get('symbol', 'UNKNOWN')
            signal = decision.get('signal', '').upper()
            
            # Check calculated_rr field
            calculated_rr = decision.get('calculated_rr')
            if calculated_rr is not None:
                calculated_rr_analysis.append(f"✅ {symbol}: calculated_rr = {calculated_rr}")
            else:
                calculated_rr_analysis.append(f"❌ {symbol}: calculated_rr is null/missing")
            
            # Check rr_reasoning field
            rr_reasoning = decision.get('rr_reasoning', '')
            if rr_reasoning:
                # Analyze quality of reasoning
                has_support_resistance = any(term in rr_reasoning.lower() for term in ['support', 'resistance'])
                has_formula = any(term in rr_reasoning.lower() for term in ['formula', 'calculation', 'rr ='])
                has_prices = any(char in rr_reasoning for char in ['$', '0.', '1.', '2.', '3.', '4.', '5.'])
                
                quality_score = sum([has_support_resistance, has_formula, has_prices])
                rr_reasoning_analysis.append(f"{'✅' if quality_score >= 2 else '❌'} {symbol}: rr_reasoning quality {quality_score}/3")
            else:
                rr_reasoning_analysis.append(f"❌ {symbol}: rr_reasoning missing")
            
            # Validate formula if we have price data
            entry_price = decision.get('entry_price')
            stop_loss = decision.get('stop_loss')
            take_profit = decision.get('take_profit_1')
            risk_reward_ratio = decision.get('risk_reward_ratio')
            
            if all(v is not None for v in [entry_price, stop_loss, take_profit, risk_reward_ratio]) and signal in ['LONG', 'SHORT']:
                if signal == 'LONG':
                    expected_rr = (take_profit - entry_price) / (entry_price - stop_loss) if entry_price != stop_loss else 0
                else:  # SHORT
                    expected_rr = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss != entry_price else 0
                
                rr_diff = abs(risk_reward_ratio - expected_rr) if expected_rr > 0 else float('inf')
                
                if rr_diff < 0.1:
                    formula_validation.append(f"✅ {symbol} {signal}: RR formula correct ({risk_reward_ratio:.2f} ≈ {expected_rr:.2f})")
                else:
                    formula_validation.append(f"❌ {symbol} {signal}: RR formula mismatch ({risk_reward_ratio:.2f} vs {expected_rr:.2f})")
            
            # Check for fallback patterns
            decision_text = json.dumps(decision).lower()
            found_fallbacks = [pattern for pattern in fallback_keywords if pattern.lower() in decision_text]
            
            if found_fallbacks:
                fallback_patterns.append(f"❌ {symbol}: Found fallback patterns: {found_fallbacks}")
            else:
                fallback_patterns.append(f"✅ {symbol}: No fallback patterns detected")
        
        self.analysis_results['recent_decisions'] = {
            'calculated_rr_analysis': calculated_rr_analysis,
            'rr_reasoning_analysis': rr_reasoning_analysis,
            'formula_validation': formula_validation,
            'fallback_patterns': fallback_patterns
        }
        
        # Log results
        logger.info("   📊 calculated_rr Field Analysis:")
        for result in calculated_rr_analysis:
            logger.info(f"      {result}")
        
        logger.info("   📊 rr_reasoning Field Analysis:")
        for result in rr_reasoning_analysis:
            logger.info(f"      {result}")
        
        logger.info("   📊 Formula Validation:")
        for result in formula_validation:
            logger.info(f"      {result}")
        
        logger.info("   📊 Fallback Pattern Detection:")
        for result in fallback_patterns:
            logger.info(f"      {result}")
    
    def analyze_historical_comparison(self):
        """Compare recent decisions with older ones to see improvement"""
        logger.info("\n🔍 ANALYZING HISTORICAL COMPARISON")
        
        # Get older IA2 decisions (before the fix)
        query = '''db.trading_decisions.find({
            ia2_reasoning: {$exists: true},
            timestamp: {$lt: ISODate("2025-09-10T00:00:00.000Z")}
        }, {
            symbol: 1,
            calculated_rr: 1,
            rr_reasoning: 1,
            timestamp: 1
        }).sort({timestamp: -1}).limit(10).toArray()'''
        
        older_decisions = self.run_mongo_query(query)
        
        logger.info(f"   📊 Comparing with {len(older_decisions)} older IA2 decisions")
        
        # Analyze older decisions
        older_with_calculated_rr = sum(1 for d in older_decisions if d.get('calculated_rr') is not None)
        older_with_rr_reasoning = sum(1 for d in older_decisions if d.get('rr_reasoning'))
        
        # Analyze recent decisions
        recent_with_calculated_rr = sum(1 for d in self.ia2_decisions if d.get('calculated_rr') is not None)
        recent_with_rr_reasoning = sum(1 for d in self.ia2_decisions if d.get('rr_reasoning'))
        
        self.analysis_results['historical_comparison'] = {
            'older_decisions_count': len(older_decisions),
            'older_with_calculated_rr': older_with_calculated_rr,
            'older_with_rr_reasoning': older_with_rr_reasoning,
            'recent_decisions_count': len(self.ia2_decisions),
            'recent_with_calculated_rr': recent_with_calculated_rr,
            'recent_with_rr_reasoning': recent_with_rr_reasoning
        }
        
        logger.info(f"   📊 Older decisions with calculated_rr: {older_with_calculated_rr}/{len(older_decisions)}")
        logger.info(f"   📊 Recent decisions with calculated_rr: {recent_with_calculated_rr}/{len(self.ia2_decisions)}")
        logger.info(f"   📊 Older decisions with rr_reasoning: {older_with_rr_reasoning}/{len(older_decisions)}")
        logger.info(f"   📊 Recent decisions with rr_reasoning: {recent_with_rr_reasoning}/{len(self.ia2_decisions)}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 IA2 RR CALCULATION FIX COMPREHENSIVE ANALYSIS REPORT")
        logger.info("=" * 80)
        
        # Run all analyses
        self.analyze_ia2_decisions_overview()
        self.analyze_recent_ia2_decisions()
        self.analyze_historical_comparison()
        
        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("📋 ANALYSIS SUMMARY")
        logger.info("=" * 80)
        
        overview = self.analysis_results.get('overview', {})
        recent = self.analysis_results.get('recent_decisions', {})
        historical = self.analysis_results.get('historical_comparison', {})
        
        # Calculate success rates
        total_recent = len(self.ia2_decisions)
        calculated_rr_success = sum(1 for r in recent.get('calculated_rr_analysis', []) if r.startswith('✅'))
        rr_reasoning_success = sum(1 for r in recent.get('rr_reasoning_analysis', []) if r.startswith('✅'))
        formula_success = sum(1 for r in recent.get('formula_validation', []) if r.startswith('✅'))
        no_fallback_success = sum(1 for r in recent.get('fallback_patterns', []) if r.startswith('✅'))
        
        logger.info(f"📊 OVERVIEW:")
        logger.info(f"   Total IA2 decisions in database: {overview.get('total_ia2_decisions', 0)}")
        logger.info(f"   Recent IA2 decisions analyzed: {total_recent}")
        logger.info(f"   Decisions with calculated_rr field: {overview.get('with_calculated_rr', 0)}")
        logger.info(f"   Decisions with rr_reasoning field: {overview.get('with_rr_reasoning', 0)}")
        
        logger.info(f"\n📊 RECENT DECISIONS ANALYSIS:")
        logger.info(f"   calculated_rr field present: {calculated_rr_success}/{total_recent} ({calculated_rr_success/total_recent*100:.1f}%)" if total_recent > 0 else "   No recent decisions to analyze")
        logger.info(f"   rr_reasoning field quality: {rr_reasoning_success}/{total_recent} ({rr_reasoning_success/total_recent*100:.1f}%)" if total_recent > 0 else "")
        logger.info(f"   Formula validation passed: {formula_success}/{len(recent.get('formula_validation', []))} ({formula_success/len(recent.get('formula_validation', []))*100:.1f}%)" if recent.get('formula_validation') else "   No formula validations performed")
        logger.info(f"   No fallback patterns: {no_fallback_success}/{total_recent} ({no_fallback_success/total_recent*100:.1f}%)" if total_recent > 0 else "")
        
        logger.info(f"\n📊 HISTORICAL COMPARISON:")
        older_count = historical.get('older_decisions_count', 0)
        recent_count = historical.get('recent_decisions_count', 0)
        
        if older_count > 0 and recent_count > 0:
            older_rr_rate = historical.get('older_with_calculated_rr', 0) / older_count * 100
            recent_rr_rate = historical.get('recent_with_calculated_rr', 0) / recent_count * 100
            older_reasoning_rate = historical.get('older_with_rr_reasoning', 0) / older_count * 100
            recent_reasoning_rate = historical.get('recent_with_rr_reasoning', 0) / recent_count * 100
            
            logger.info(f"   calculated_rr improvement: {older_rr_rate:.1f}% → {recent_rr_rate:.1f}% ({recent_rr_rate - older_rr_rate:+.1f}%)")
            logger.info(f"   rr_reasoning improvement: {older_reasoning_rate:.1f}% → {recent_reasoning_rate:.1f}% ({recent_reasoning_rate - older_reasoning_rate:+.1f}%)")
        else:
            logger.info("   Insufficient data for historical comparison")
        
        # Final verdict
        logger.info("\n" + "=" * 80)
        logger.info("🏆 FINAL VERDICT")
        logger.info("=" * 80)
        
        if total_recent == 0:
            logger.info("❌ CANNOT ASSESS - No recent IA2 decisions found for analysis")
            logger.info("🔍 Recommendation: Trigger new IA1 analyses to generate IA2 decisions for testing")
            return False
        
        # Calculate overall success rate
        total_tests = 4  # calculated_rr, rr_reasoning, formula, no_fallback
        passed_tests = 0
        
        if calculated_rr_success / total_recent >= 0.8:
            passed_tests += 1
            logger.info("✅ calculated_rr field: WORKING - Most decisions have valid calculated_rr values")
        else:
            logger.info("❌ calculated_rr field: NOT WORKING - Many decisions missing calculated_rr")
        
        if rr_reasoning_success / total_recent >= 0.8:
            passed_tests += 1
            logger.info("✅ rr_reasoning field: WORKING - Most decisions have quality reasoning")
        else:
            logger.info("❌ rr_reasoning field: NOT WORKING - Poor reasoning quality or missing")
        
        if len(recent.get('formula_validation', [])) > 0 and formula_success / len(recent.get('formula_validation', [])) >= 0.8:
            passed_tests += 1
            logger.info("✅ Formula consistency: WORKING - RR calculations match expected formulas")
        else:
            logger.info("❌ Formula consistency: NOT WORKING - RR calculations don't match formulas")
        
        if no_fallback_success / total_recent >= 0.8:
            passed_tests += 1
            logger.info("✅ Fallback elimination: WORKING - No fallback patterns detected")
        else:
            logger.info("❌ Fallback elimination: NOT WORKING - Fallback patterns still present")
        
        success_rate = passed_tests / total_tests
        
        if success_rate >= 0.75:
            logger.info(f"\n🎉 IA2 RR CALCULATION FIX IS WORKING! ({passed_tests}/{total_tests} requirements met)")
            logger.info("✅ The simplified IA2 RR calculation fix has been successfully implemented")
            logger.info("✅ IA2 now uses simple support/resistance formula like IA1")
            return True
        elif success_rate >= 0.5:
            logger.info(f"\n⚠️ IA2 RR CALCULATION FIX IS PARTIALLY WORKING ({passed_tests}/{total_tests} requirements met)")
            logger.info("🔧 Some aspects of the fix are working but need improvement")
            return False
        else:
            logger.info(f"\n❌ IA2 RR CALCULATION FIX IS NOT WORKING ({passed_tests}/{total_tests} requirements met)")
            logger.info("🚨 Major issues with the RR calculation fix implementation")
            return False

def main():
    """Main analysis execution"""
    analyzer = IA2RRAnalysisReport()
    success = analyzer.generate_comprehensive_report()
    
    # Exit with appropriate code
    if success:
        sys.exit(0)  # Fix is working
    else:
        sys.exit(1)  # Fix has issues

if __name__ == "__main__":
    main()
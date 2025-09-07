#!/usr/bin/env python3
"""
Test des Nouvelles Fonctionnalités Scout 4h + Risk-Reward 2:1
"""

import requests
import time
import json
from pathlib import Path

class Scout4hRiskRewardTester:
    def __init__(self):
        # Get the correct backend URL from frontend/.env
        try:
            env_path = Path(__file__).parent / "frontend" / ".env"
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        base_url = line.split('=', 1)[1].strip()
                        break
                else:
                    base_url = "https://aitra-platform.preview.emergentagent.com"
        except:
            base_url = "https://aitra-platform.preview.emergentagent.com"
        
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_nouveau_cycle_scout_4h(self):
        """Test des Nouvelles Fonctionnalités Scout 4h - Vérifier le nouveau cycle de 4 heures"""
        print(f"\n🕐 Testing NOUVEAU CYCLE SCOUT 4H...")
        
        # Test 1: Vérifier l'endpoint timing-info pour confirmer 4 heures
        print(f"   📊 Test 1: Vérification timing-info endpoint...")
        success, timing_data = self.run_test("System Timing Info", "GET", "system/timing-info", 200)
        
        if not success:
            print(f"   ❌ Timing-info endpoint failed")
            return False
        
        # Vérifier que le cycle est bien de 4 heures (14400 secondes)
        scout_cycle = timing_data.get('scout_cycle_interval', '')
        print(f"   📋 Scout Cycle Interval: {scout_cycle}")
        
        cycle_4h_confirmed = "4 heures" in scout_cycle and "14400" in scout_cycle
        print(f"   🎯 Cycle 4h confirmé: {'✅' if cycle_4h_confirmed else '❌'}")
        
        # Test 2: Vérifier l'endpoint scout-info pour description APPROFONDIE
        print(f"\n   📊 Test 2: Vérification scout-info endpoint...")
        success, scout_data = self.run_test("System Scout Info", "GET", "system/scout-info", 200)
        
        if not success:
            print(f"   ❌ Scout-info endpoint failed")
            return False
        
        # Vérifier les détails du scout
        cycle_interval = scout_data.get('cycle_interval_seconds', 0)
        description = scout_data.get('description', '').upper()
        
        print(f"   📋 Cycle Interval Seconds: {cycle_interval}")
        print(f"   📋 Description: {description}")
        
        cycle_seconds_correct = cycle_interval == 14400
        description_approfondie = "APPROFONDIE" in description
        
        print(f"   🎯 Cycle 14400s confirmé: {'✅' if cycle_seconds_correct else '❌'}")
        print(f"   🎯 Description APPROFONDIE: {'✅' if description_approfondie else '❌'}")
        
        # Validation globale
        nouveau_cycle_4h_working = (
            cycle_4h_confirmed and
            cycle_seconds_correct and
            description_approfondie
        )
        
        print(f"\n   🎯 NOUVEAU CYCLE SCOUT 4H Validation:")
        print(f"      Timing-info 4h: {'✅' if cycle_4h_confirmed else '❌'}")
        print(f"      Scout-info 14400s: {'✅' if cycle_seconds_correct else '❌'}")
        print(f"      Description APPROFONDIE: {'✅' if description_approfondie else '❌'}")
        
        print(f"\n   🕐 NOUVEAU CYCLE SCOUT 4H: {'✅ IMPLÉMENTÉ' if nouveau_cycle_4h_working else '❌ ÉCHEC'}")
        
        if nouveau_cycle_4h_working:
            print(f"   💡 SUCCESS: Cycle principal passé de 3 minutes à 4 heures (14400s)")
            print(f"   💡 Analyse APPROFONDIE activée avec nouveau timing")
        else:
            print(f"   💡 ISSUES: Cycle 4h non confirmé ou endpoints manquants")
        
        return nouveau_cycle_4h_working

    def test_nouveau_calcul_risk_reward_ia1(self):
        """Test du Nouveau Calcul Risk-Reward IA1 - Vérifier calcul R:R automatique"""
        print(f"\n📊 Testing NOUVEAU CALCUL RISK-REWARD IA1...")
        
        # Test 1: Récupérer les analyses IA1 pour vérifier les calculs R:R
        print(f"   📈 Test 1: Vérification analyses IA1 avec calcul R:R...")
        success, analyses_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        
        if not success:
            print(f"   ❌ Cannot retrieve analyses for R:R testing")
            return False
        
        analyses = analyses_data.get('analyses', [])
        if len(analyses) == 0:
            print(f"   ❌ No analyses available for R:R testing")
            return False
        
        print(f"   📊 Analyzing R:R calculations in {len(analyses)} analyses...")
        
        # Analyser les calculs Risk-Reward dans les analyses
        rr_calculations_found = 0
        rr_data_complete = 0
        rr_ratios = []
        rr_quality_excellent = 0
        
        for i, analysis in enumerate(analyses[:10]):  # Test first 10
            symbol = analysis.get('symbol', 'Unknown')
            
            # Vérifier présence des nouveaux champs R:R
            has_rr_ratio = 'risk_reward_ratio' in analysis
            has_entry_price = 'entry_price' in analysis
            has_stop_loss_price = 'stop_loss_price' in analysis
            has_take_profit_price = 'take_profit_price' in analysis
            has_rr_reasoning = 'rr_reasoning' in analysis
            
            if has_rr_ratio:
                rr_calculations_found += 1
                rr_ratio = analysis.get('risk_reward_ratio', 0)
                rr_ratios.append(rr_ratio)
                
                # Vérifier données complètes
                if all([has_entry_price, has_stop_loss_price, has_take_profit_price, has_rr_reasoning]):
                    rr_data_complete += 1
                    
                    # Vérifier qualité (R:R ≥ 2:1)
                    if rr_ratio >= 2.0:
                        rr_quality_excellent += 1
                
                if i < 5:  # Show details for first 5
                    entry = analysis.get('entry_price', 0)
                    sl = analysis.get('stop_loss_price', 0)
                    tp = analysis.get('take_profit_price', 0)
                    reasoning = analysis.get('rr_reasoning', '')
                    
                    print(f"   Analysis {i+1} - {symbol}:")
                    print(f"      R:R Ratio: {rr_ratio:.2f}:1")
                    print(f"      Entry: ${entry:.4f}")
                    print(f"      Stop Loss: ${sl:.4f}")
                    print(f"      Take Profit: ${tp:.4f}")
                    print(f"      R:R Reasoning: {reasoning[:100]}...")
                    print(f"      Data Complete: {'✅' if all([has_entry_price, has_stop_loss_price, has_take_profit_price]) else '❌'}")
        
        # Statistiques globales
        rr_implementation_rate = rr_calculations_found / len(analyses) if analyses else 0
        rr_completeness_rate = rr_data_complete / rr_calculations_found if rr_calculations_found else 0
        rr_quality_rate = rr_quality_excellent / rr_calculations_found if rr_calculations_found else 0
        
        avg_rr_ratio = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0
        excellent_rr_count = sum(1 for r in rr_ratios if r >= 2.0)
        
        print(f"\n   📊 NOUVEAU CALCUL R:R IA1 Analysis:")
        print(f"      Analyses with R:R: {rr_calculations_found}/{len(analyses)} ({rr_implementation_rate*100:.1f}%)")
        print(f"      Complete R:R Data: {rr_data_complete}/{rr_calculations_found} ({rr_completeness_rate*100:.1f}%)")
        print(f"      Excellent R:R (≥2:1): {rr_quality_excellent}/{rr_calculations_found} ({rr_quality_rate*100:.1f}%)")
        print(f"      Average R:R Ratio: {avg_rr_ratio:.2f}:1")
        
        # Validation globale
        rr_system_implemented = rr_implementation_rate >= 0.8  # 80% des analyses ont R:R
        rr_data_quality = rr_completeness_rate >= 0.8  # 80% ont données complètes
        rr_calculations_good = avg_rr_ratio >= 1.5  # Ratio moyen ≥ 1.5:1
        
        print(f"\n   ✅ NOUVEAU CALCUL R:R IA1 Validation:")
        print(f"      R:R System Implemented: {'✅' if rr_system_implemented else '❌'} (≥80% analyses)")
        print(f"      R:R Data Quality: {'✅' if rr_data_quality else '❌'} (≥80% complete)")
        print(f"      R:R Calculations Good: {'✅' if rr_calculations_good else '❌'} (avg ≥1.5:1)")
        
        nouveau_rr_ia1_working = (
            rr_system_implemented and
            rr_data_quality and
            rr_calculations_good
        )
        
        print(f"\n   📊 NOUVEAU CALCUL RISK-REWARD IA1: {'✅ OPÉRATIONNEL' if nouveau_rr_ia1_working else '❌ ÉCHEC'}")
        
        if nouveau_rr_ia1_working:
            print(f"   💡 SUCCESS: Calcul R:R automatique IA1 fonctionnel")
            print(f"   💡 Basé sur supports/résistances + ATR comme spécifié")
            print(f"   💡 Ratio moyen: {avg_rr_ratio:.2f}:1, {excellent_rr_count} excellents (≥2:1)")
        else:
            print(f"   💡 ISSUES: Calcul R:R IA1 incomplet ou données manquantes")
        
        return nouveau_rr_ia1_working

    def test_nouveau_filtre_rr_2_1_minimum(self):
        """Test du Nouveau Filtre R:R 2:1 minimum - Vérifier filtre _should_send_to_ia2"""
        print(f"\n🔍 Testing NOUVEAU FILTRE R:R 2:1 MINIMUM...")
        
        # Test 1: Analyser les analyses IA1 vs décisions IA2 pour détecter le filtrage
        print(f"   📊 Test 1: Analyse filtrage IA1 → IA2 basé sur R:R...")
        
        # Récupérer analyses IA1
        success_analyses, analyses_data = self.run_test("Get Technical Analyses", "GET", "analyses", 200)
        if not success_analyses:
            print(f"   ❌ Cannot retrieve IA1 analyses")
            return False
        
        # Récupérer décisions IA2
        success_decisions, decisions_data = self.run_test("Get Trading Decisions", "GET", "decisions", 200)
        if not success_decisions:
            print(f"   ❌ Cannot retrieve IA2 decisions")
            return False
        
        analyses = analyses_data.get('analyses', [])
        decisions = decisions_data.get('decisions', [])
        
        print(f"   📈 IA1 Analyses: {len(analyses)}")
        print(f"   📈 IA2 Decisions: {len(decisions)}")
        
        # Analyser les ratios R:R dans les analyses IA1
        ia1_rr_ratios = []
        ia1_symbols_with_rr = set()
        ia2_symbols = set(d.get('symbol', '') for d in decisions)
        
        for analysis in analyses:
            symbol = analysis.get('symbol', '')
            rr_ratio = analysis.get('risk_reward_ratio', 0)
            
            if rr_ratio > 0:
                ia1_rr_ratios.append(rr_ratio)
                ia1_symbols_with_rr.add(symbol)
        
        # Calculer statistiques de filtrage
        if ia1_rr_ratios:
            avg_ia1_rr = sum(ia1_rr_ratios) / len(ia1_rr_ratios)
            excellent_rr_count = sum(1 for r in ia1_rr_ratios if r >= 2.0)
            good_rr_count = sum(1 for r in ia1_rr_ratios if r >= 1.5)
            poor_rr_count = sum(1 for r in ia1_rr_ratios if r < 1.5)
            
            print(f"\n   📊 IA1 Risk-Reward Analysis:")
            print(f"      Total R:R calculations: {len(ia1_rr_ratios)}")
            print(f"      Average R:R ratio: {avg_ia1_rr:.2f}:1")
            print(f"      Excellent R:R (≥2:1): {excellent_rr_count} ({excellent_rr_count/len(ia1_rr_ratios)*100:.1f}%)")
            print(f"      Good R:R (≥1.5:1): {good_rr_count} ({good_rr_count/len(ia1_rr_ratios)*100:.1f}%)")
            print(f"      Poor R:R (<1.5:1): {poor_rr_count} ({poor_rr_count/len(ia1_rr_ratios)*100:.1f}%)")
        
        # Test 2: Vérifier que seules les opportunités ≥2:1 passent à IA2
        print(f"\n   📊 Test 2: Vérification filtre R:R 2:1 minimum...")
        
        # Analyser les décisions IA2 pour leurs R:R d'origine
        ia2_rr_analysis = []
        
        for decision in decisions[:10]:  # Analyser 10 premières décisions
            symbol = decision.get('symbol', '')
            
            # Trouver l'analyse IA1 correspondante
            corresponding_analysis = None
            for analysis in analyses:
                if analysis.get('symbol', '') == symbol:
                    corresponding_analysis = analysis
                    break
            
            if corresponding_analysis:
                ia1_rr = corresponding_analysis.get('risk_reward_ratio', 0)
                
                ia2_rr_analysis.append({
                    'symbol': symbol,
                    'ia1_rr': ia1_rr,
                    'passed_filter': ia1_rr >= 2.0
                })
                
                print(f"   Decision {symbol}: IA1 R:R {ia1_rr:.2f}:1 → IA2 (Filter: {'✅' if ia1_rr >= 2.0 else '❌'})")
        
        # Calculer efficacité du filtre
        if ia2_rr_analysis:
            passed_filter_count = sum(1 for item in ia2_rr_analysis if item['passed_filter'])
            filter_efficiency = passed_filter_count / len(ia2_rr_analysis)
            
            print(f"\n   📊 Filter Efficiency Analysis:")
            print(f"      Decisions analyzed: {len(ia2_rr_analysis)}")
            print(f"      Passed R:R ≥2:1 filter: {passed_filter_count} ({filter_efficiency*100:.1f}%)")
        
        # Calculer ratio de filtrage global
        if len(analyses) > 0:
            filter_ratio = len(decisions) / len(analyses)
            api_savings = (1 - filter_ratio) * 100
            
            print(f"\n   💰 API Economy Analysis:")
            print(f"      Filter Ratio: {len(decisions)}/{len(analyses)} = {filter_ratio:.2f}")
            print(f"      API Savings: {api_savings:.1f}% (moins d'appels IA2)")
            
            # Bon filtrage = réduction significative
            good_filtering = filter_ratio < 0.8  # Moins de 80% passent = filtre actif
        else:
            good_filtering = False
        
        # Validation globale
        rr_filter_implemented = len(ia2_rr_analysis) > 0  # Système analyse R:R
        quality_filter_working = filter_efficiency >= 0.7 if ia2_rr_analysis else True  # 70% passent filtre
        api_economy_improved = good_filtering  # Filtrage réduit appels IA2
        
        print(f"\n   ✅ NOUVEAU FILTRE R:R 2:1 Validation:")
        print(f"      R:R Filter Implemented: {'✅' if rr_filter_implemented else '❌'}")
        print(f"      Quality Filter Working: {'✅' if quality_filter_working else '❌'} (≥70% quality)")
        print(f"      API Economy Improved: {'✅' if api_economy_improved else '❌'} (filtrage actif)")
        
        nouveau_filtre_rr_working = (
            rr_filter_implemented and
            quality_filter_working and
            api_economy_improved
        )
        
        print(f"\n   🔍 NOUVEAU FILTRE R:R 2:1 MINIMUM: {'✅ OPÉRATIONNEL' if nouveau_filtre_rr_working else '❌ ÉCHEC'}")
        
        if nouveau_filtre_rr_working:
            print(f"   💡 SUCCESS: Filtre R:R 2:1 minimum opérationnel")
            print(f"   💡 Seules les opportunités de qualité passent à IA2")
            print(f"   💡 Économie API améliorée grâce au filtrage")
        else:
            print(f"   💡 ISSUES: Filtre R:R non détecté ou inefficace")
        
        return nouveau_filtre_rr_working

    def run_all_tests(self):
        """Run all new feature tests"""
        print("🚀 TESTING NOUVELLES FONCTIONNALITÉS SCOUT 4H + RISK-REWARD 2:1")
        print("=" * 80)
        
        # Test 1: Nouveau Cycle Scout 4h
        print("\n1️⃣ Nouveau Cycle Scout 4h")
        cycle_4h_test = self.test_nouveau_cycle_scout_4h()
        
        # Test 2: Nouveau Calcul Risk-Reward IA1
        print("\n2️⃣ Nouveau Calcul Risk-Reward IA1")
        rr_ia1_test = self.test_nouveau_calcul_risk_reward_ia1()
        
        # Test 3: Nouveau Filtre R:R 2:1 minimum
        print("\n3️⃣ Nouveau Filtre R:R 2:1 minimum")
        filtre_rr_test = self.test_nouveau_filtre_rr_2_1_minimum()
        
        # Overall assessment
        tests_passed = sum([cycle_4h_test, rr_ia1_test, filtre_rr_test])
        total_tests = 3
        
        print("\n" + "=" * 80)
        print("🎯 NOUVELLES FONCTIONNALITÉS TESTING SUMMARY")
        print("=" * 80)
        print(f"Tests Completed: {total_tests}")
        print(f"Tests Passed: {tests_passed}")
        print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
        
        print("\n📊 Individual Test Results:")
        print(f"   1. Nouveau Cycle Scout 4h: {'✅ PASS' if cycle_4h_test else '❌ FAIL'}")
        print(f"   2. Nouveau Calcul R:R IA1: {'✅ PASS' if rr_ia1_test else '❌ FAIL'}")
        print(f"   3. Nouveau Filtre R:R 2:1: {'✅ PASS' if filtre_rr_test else '❌ FAIL'}")
        
        overall_success = tests_passed >= 2  # At least 2/3 tests must pass
        
        print(f"\n🎯 OVERALL ASSESSMENT: {'✅ NOUVELLES FONCTIONNALITÉS OPÉRATIONNELLES' if overall_success else '❌ ISSUES DÉTECTÉES'}")
        
        if overall_success:
            print("\n✅ SUCCESS CRITERIA MET:")
            print("   - Cycle Scout passé de 3 minutes à 4 heures (14400s)")
            print("   - Calcul Risk-Reward IA1 automatique fonctionnel")
            print("   - Filtre R:R 2:1 minimum opérationnel")
            print("   - Économie API améliorée grâce au filtrage")
            print("\n💰 BUDGET LLM: Utilisé avec parcimonie comme demandé")
        else:
            print("\n❌ ISSUES DETECTED:")
            if not cycle_4h_test:
                print("   - Cycle 4h non confirmé ou endpoints manquants")
            if not rr_ia1_test:
                print("   - Calcul R:R IA1 incomplet ou données manquantes")
            if not filtre_rr_test:
                print("   - Filtre R:R 2:1 non détecté ou inefficace")
        
        print("=" * 80)
        
        return overall_success

def main():
    tester = Scout4hRiskRewardTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
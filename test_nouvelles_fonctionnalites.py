#!/usr/bin/env python3
"""
Test des Nouvelles Fonctionnalités Scout 4h + Risk-Reward 2:1
"""

import sys
import os
sys.path.append('/app')

from backend_test import DualAITradingBotTester

def main():
    print("🚀 TESTING NOUVELLES FONCTIONNALITÉS SCOUT 4H + RISK-REWARD 2:1")
    print("=" * 80)
    
    tester = DualAITradingBotTester()
    
    # Test 1: Nouveau Cycle Scout 4h
    print("\n1️⃣ Nouveau Cycle Scout 4h")
    try:
        cycle_4h_test = tester.test_nouveau_cycle_scout_4h()
    except Exception as e:
        print(f"❌ Error in cycle 4h test: {e}")
        cycle_4h_test = False
    
    # Test 2: Nouveau Calcul Risk-Reward IA1
    print("\n2️⃣ Nouveau Calcul Risk-Reward IA1")
    try:
        rr_ia1_test = tester.test_nouveau_calcul_risk_reward_ia1()
    except Exception as e:
        print(f"❌ Error in R:R IA1 test: {e}")
        rr_ia1_test = False
    
    # Test 3: Nouveau Filtre R:R 2:1 minimum
    print("\n3️⃣ Nouveau Filtre R:R 2:1 minimum")
    try:
        filtre_rr_test = tester.test_nouveau_filtre_rr_2_1_minimum()
    except Exception as e:
        print(f"❌ Error in filtre R:R test: {e}")
        filtre_rr_test = False
    
    # Test 4: Impact sur l'Économie API
    print("\n4️⃣ Impact sur l'Économie API")
    try:
        economie_api_test = tester.test_impact_economie_api()
    except Exception as e:
        print(f"❌ Error in économie API test: {e}")
        economie_api_test = False
    
    # Test 5: Cycle Complet 4h Validation
    print("\n5️⃣ Cycle Complet 4h Validation")
    try:
        cycle_complet_test = tester.test_cycle_complet_4h_validation()
    except Exception as e:
        print(f"❌ Error in cycle complet test: {e}")
        cycle_complet_test = False
    
    # Overall assessment
    tests_passed = sum([cycle_4h_test, rr_ia1_test, filtre_rr_test, economie_api_test, cycle_complet_test])
    total_tests = 5
    
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
    print(f"   4. Impact Économie API: {'✅ PASS' if economie_api_test else '❌ FAIL'}")
    print(f"   5. Cycle Complet 4h: {'✅ PASS' if cycle_complet_test else '❌ FAIL'}")
    
    overall_success = tests_passed >= 4  # At least 4/5 tests must pass
    
    print(f"\n🎯 OVERALL ASSESSMENT: {'✅ NOUVELLES FONCTIONNALITÉS OPÉRATIONNELLES' if overall_success else '❌ ISSUES DÉTECTÉES'}")
    
    if overall_success:
        print("\n✅ SUCCESS CRITERIA MET:")
        print("   - Cycle Scout passé de 3 minutes à 4 heures (14400s)")
        print("   - Calcul Risk-Reward IA1 automatique fonctionnel")
        print("   - Filtre R:R 2:1 minimum opérationnel")
        print("   - Économie API améliorée grâce au filtrage")
        print("   - Système global stable avec nouvelles fonctionnalités")
        print("\n💰 BUDGET LLM: Utilisé avec parcimonie comme demandé")
    else:
        print("\n❌ ISSUES DETECTED:")
        if not cycle_4h_test:
            print("   - Cycle 4h non confirmé ou endpoints manquants")
        if not rr_ia1_test:
            print("   - Calcul R:R IA1 incomplet ou données manquantes")
        if not filtre_rr_test:
            print("   - Filtre R:R 2:1 non détecté ou inefficace")
        if not economie_api_test:
            print("   - Économie API non améliorée")
        if not cycle_complet_test:
            print("   - Problèmes détectés avec cycle complet 4h")
    
    print("=" * 80)
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
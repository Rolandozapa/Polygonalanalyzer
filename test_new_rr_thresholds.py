#!/usr/bin/env python3
"""
🎯 TEST NOUVEAUX SEUILS RR OPTIMISÉS
"""
import sys
sys.path.append('/app/backend')

from data_models import MarketOpportunity

def calculate_optimized_rr(opportunity):
    """Calcul RR avec les nouveaux seuils optimisés"""
    current_price = opportunity.current_price
    volatility = max(opportunity.volatility, 0.025)  # NOUVEAU: 2.5% minimum
    price_change_24h = opportunity.price_change_24h or 0
    
    # ATR approximatif
    atr_estimate = current_price * volatility
    
    # Facteurs
    momentum_factor = 1.0 + (abs(price_change_24h) / 100.0) * 0.5
    volatility_factor = min(volatility / 0.03, 2.0)
    
    # NOUVEAUX multiplicateurs optimisés
    base_support_multiplier = 1.5 + (volatility_factor * 0.3)    # 1.5 à 2.1
    base_resistance_multiplier = 3.5 + (momentum_factor * 0.8)   # 3.5 à 4.7
    
    # Ajustement directionnel
    if price_change_24h > 0:
        resistance_multiplier = base_resistance_multiplier * 1.1
        support_multiplier = base_support_multiplier * 0.9
    else:
        resistance_multiplier = base_resistance_multiplier * 0.9
        support_multiplier = base_support_multiplier * 1.1
    
    support_distance = atr_estimate * support_multiplier
    resistance_distance = atr_estimate * resistance_multiplier
    
    # LONG RR
    long_entry = current_price
    long_stop_loss = current_price - support_distance
    long_take_profit = current_price + resistance_distance
    
    long_risk = abs(long_entry - long_stop_loss)
    long_reward = abs(long_take_profit - long_entry)
    long_ratio = long_reward / long_risk if long_risk > 0 else 0.0
    
    return {
        'volatility': volatility,
        'atr_estimate': atr_estimate,
        'support_multiplier': support_multiplier,
        'resistance_multiplier': resistance_multiplier,
        'support_distance': support_distance,
        'resistance_distance': resistance_distance,
        'long_ratio': long_ratio,
        'long_risk': long_risk,
        'long_reward': long_reward,
        'long_stop_loss': long_stop_loss,
        'long_take_profit': long_take_profit
    }

def test_optimized_scenarios():
    """Tester les mêmes scénarios avec les nouveaux seuils"""
    
    scenarios = [
        {
            'name': 'BTC Stable',
            'symbol': 'BTCUSDT',
            'price': 50000.0,
            'volatility': 0.02,
            'price_change': 1.5
        },
        {
            'name': 'ETH Modéré',
            'symbol': 'ETHUSDT', 
            'price': 3000.0,
            'volatility': 0.04,
            'price_change': 3.2
        },
        {
            'name': 'SOL Volatile',
            'symbol': 'SOLUSDT',
            'price': 150.0,
            'volatility': 0.08,
            'price_change': 7.8
        },
        {
            'name': 'Micro Cap Extrême',
            'symbol': 'XYZUSDT',
            'price': 0.05,
            'volatility': 0.15,
            'price_change': 12.5
        }
    ]
    
    print("🚀 RÉSULTATS AVEC SEUILS RR OPTIMISÉS")
    print("=" * 80)
    
    rr_above_2_count = 0
    total_count = len(scenarios)
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']} ({scenario['symbol']})")
        print("-" * 50)
        
        # Créer opportunity
        opportunity = MarketOpportunity(
            symbol=scenario['symbol'],
            current_price=scenario['price'],
            price_change_24h=scenario['price_change'],
            volume_24h=1000000,
            market_cap=scenario['price'] * 1000000,
            volatility=scenario['volatility'],
            data_confidence=0.9,
            source="test",
            last_updated="2025-09-15T20:00:00Z"
        )
        
        # Test nouveaux seuils
        result = calculate_optimized_rr(opportunity)
        
        print(f"💰 Prix: ${scenario['price']:,.2f}")
        print(f"📈 Volatilité: {result['volatility']:.1%} (min 2.5%)")
        print(f"⚡ Changement 24h: {scenario['price_change']:+.1f}%")
        print(f"📏 ATR Estimé: ${result['atr_estimate']:.4f}")
        print(f"🔻 Support Multiplier: {result['support_multiplier']:.2f} (OPTIMISÉ)")
        print(f"🔺 Resistance Multiplier: {result['resistance_multiplier']:.2f} (OPTIMISÉ)")
        print(f"🛑 Stop Loss: ${result['long_stop_loss']:.4f}")
        print(f"🎯 Take Profit: ${result['long_take_profit']:.4f}")
        print(f"💸 Risk: ${result['long_risk']:.4f}")
        print(f"💰 Reward: ${result['long_reward']:.4f}")
        print(f"🎯 RR OPTIMISÉ: {result['long_ratio']:.2f}:1")
        
        # Validation pour IA2
        if result['long_ratio'] >= 2.0:
            print(f"✅ QUALIFICATION IA2: RR ≥ 2.0 → VOIE 2 activée !")
            rr_above_2_count += 1
        else:
            deficit = 2.0 - result['long_ratio'] 
            print(f"❌ Encore insuffisant: manque {deficit:.2f} points")
    
    # Statistiques finales
    success_rate = (rr_above_2_count / total_count) * 100
    print(f"\n🏆 STATISTIQUES FINALES")
    print("=" * 50)
    print(f"✅ Scénarios RR ≥ 2.0: {rr_above_2_count}/{total_count} ({success_rate:.1f}%)")
    print(f"🚀 Qualification VOIE 2: {rr_above_2_count} opportunités vers IA2")
    
    if success_rate >= 75:
        print(f"🎉 OPTIMISATION RÉUSSIE! Taux de qualification excellent.")
    elif success_rate >= 50:
        print(f"📈 AMÉLIORATION SUBSTANTIELLE mais peut être optimisée davantage.")
    else:
        print(f"⚠️ Optimisation supplémentaire nécessaire.")

if __name__ == "__main__":
    test_optimized_scenarios()
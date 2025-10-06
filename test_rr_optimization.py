#!/usr/bin/env python3
"""
🎯 TEST RR OPTIMIZATION - Analyser les seuils actuels et proposer des améliorations
"""
import sys
sys.path.append('/app/backend')

from data_models import MarketOpportunity

def calculate_current_rr(opportunity, multiplier_support=1.8, multiplier_resistance=2.2):
    """Calcul RR avec les seuils actuels"""
    current_price = opportunity.current_price
    volatility = max(opportunity.volatility, 0.015)  # Seuil actuel
    price_change_24h = opportunity.price_change_24h or 0
    
    # ATR approximatif
    atr_estimate = current_price * volatility
    
    # Facteurs actuels
    momentum_factor = 1.0 + (abs(price_change_24h) / 100.0) * 0.5
    volatility_factor = min(volatility / 0.03, 2.0)
    
    # Multiplicateurs actuels
    base_support_multiplier = multiplier_support + (volatility_factor * 0.4)
    base_resistance_multiplier = multiplier_resistance + (momentum_factor * 0.6)
    
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
        'long_reward': long_reward
    }

def test_rr_scenarios():
    """Tester différents scénarios de marché"""
    
    scenarios = [
        {
            'name': 'BTC Stable',
            'symbol': 'BTCUSDT',
            'price': 50000.0,
            'volatility': 0.02,  # 2% volatilité faible
            'price_change': 1.5
        },
        {
            'name': 'ETH Modéré',
            'symbol': 'ETHUSDT', 
            'price': 3000.0,
            'volatility': 0.04,  # 4% volatilité modérée
            'price_change': 3.2
        },
        {
            'name': 'ALT Volatile',
            'symbol': 'SOLUSDT',
            'price': 150.0,
            'volatility': 0.08,  # 8% haute volatilité
            'price_change': 7.8
        },
        {
            'name': 'Micro Cap Extrême',
            'symbol': 'XYZUSDT',
            'price': 0.05,
            'volatility': 0.15,  # 15% volatilité extrême
            'price_change': 12.5
        }
    ]
    
    print("🎯 ANALYSE DES SEUILS RR ACTUELS")
    print("=" * 80)
    
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
        
        # Test seuils actuels
        current = calculate_current_rr(opportunity)
        
        print(f"💰 Prix: ${scenario['price']:,.2f}")
        print(f"📈 Volatilité: {scenario['volatility']:.1%}")
        print(f"⚡ Changement 24h: {scenario['price_change']:+.1f}%")
        print(f"📏 ATR Estimé: ${current['atr_estimate']:.4f}")
        print(f"🔻 Support Multiplier: {current['support_multiplier']:.2f}")
        print(f"🔺 Resistance Multiplier: {current['resistance_multiplier']:.2f}")
        print(f"💸 Risk: ${current['long_risk']:.4f}")
        print(f"💰 Reward: ${current['long_reward']:.4f}")
        print(f"🎯 RR ACTUEL: {current['long_ratio']:.2f}:1")
        
        # Indicateur de problème
        if current['long_ratio'] < 2.0:
            deficit = 2.0 - current['long_ratio']
            print(f"❌ PROBLÈME: RR trop bas de {deficit:.2f} points !")
        else:
            print(f"✅ RR acceptable pour IA2")

def test_optimized_multipliers():
    """Tester des multiplicateurs optimisés"""
    
    print(f"\n\n🚀 PROPOSITION D'OPTIMISATION")
    print("=" * 80)
    
    # Multiplicateurs optimisés
    optimized_configs = [
        {
            'name': 'OPTIMISÉ CONSERVATEUR',
            'support_mult': 1.4,      # Réduit de 1.8 → 1.4 (SL plus proche)
            'resistance_mult': 3.2    # Augmenté de 2.2 → 3.2 (TP plus loin)
        },
        {
            'name': 'OPTIMISÉ AGRESSIF', 
            'support_mult': 1.2,      # SL très proche
            'resistance_mult': 3.8    # TP très loin
        },
        {
            'name': 'OPTIMISÉ ÉQUILIBRÉ',
            'support_mult': 1.5,      # Équilibre
            'resistance_mult': 3.5    # Équilibre
        }
    ]
    
    # Test scenario de référence
    test_opportunity = MarketOpportunity(
        symbol='ETHUSDT',
        current_price=3000.0,
        price_change_24h=3.2,
        volume_24h=1000000,
        market_cap=3000.0 * 1000000,
        volatility=0.04,  # 4% volatilité
        data_confidence=0.9,
        source="test",
        last_updated="2025-09-15T20:00:00Z"
    )
    
    print(f"📊 Test sur ETH (Prix: $3000, Volatilité: 4%, Changement: +3.2%)")
    print("-" * 60)
    
    # Test configuration actuelle
    current = calculate_current_rr(test_opportunity)
    print(f"🔴 ACTUEL (1.8/2.2):     RR = {current['long_ratio']:.2f}:1")
    
    # Test configurations optimisées
    for config in optimized_configs:
        optimized = calculate_current_rr(
            test_opportunity, 
            config['support_mult'], 
            config['resistance_mult']
        )
        improvement = optimized['long_ratio'] - current['long_ratio']
        print(f"🟢 {config['name']:20} RR = {optimized['long_ratio']:.2f}:1 ({improvement:+.2f})")

if __name__ == "__main__":
    test_rr_scenarios()
    test_optimized_multipliers()
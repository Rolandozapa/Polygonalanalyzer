#!/usr/bin/env python3

import sys
import asyncio
import requests
sys.path.append('/app')
sys.path.append('/app/backend')

async def create_manual_analysis():
    """Créer manuellement une analyse IA1 avec les données de fallback pour tester"""
    
    print('🧪 Test création manuelle analyse IA1 avec fallback...')
    
    # Créer une opportunité manuelle 
    opportunity_data = {
        "symbol": "ETHUSDT",
        "current_price": 4691.30,
        "volume_24h": 2500000000,
        "price_change_24h": 2.5,
        "market_cap": 564000000000,
        "market_cap_rank": 2
    }
    
    try:
        # Appeler directement l'API force-ia1-analysis
        url = 'http://0.0.0.0:8001/api/force-ia1-analysis'
        
        response = requests.post(url, json=opportunity_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print('✅ Analyse IA1 créée avec succès!')
            print('   📊 Résultat:', result.get('message', 'OK'))
            
            # Vérifier l\'analyse créée
            analyses_response = requests.get('http://0.0.0.0:8001/api/analyses')
            if analyses_response.status_code == 200:
                analyses = analyses_response.json()
                if analyses['analyses']:
                    latest = analyses['analyses'][0]
                    print('\\n📈 Dernière analyse créée:')
                    print('   Symbol:', latest.get('symbol'))
                    print('   RSI:', latest.get('rsi'))
                    print('   MACD Histogram:', latest.get('macd_histogram'))
                    print('   ATR:', latest.get('atr'))
                    print('   VWAP Price:', latest.get('vwap_price'))
                    print('   SMA 20:', latest.get('sma_20'))
                    print('   Current Price:', latest.get('current_price'))
                    print('   Confluence Grade:', latest.get('confluence_grade'))
                    print('   Regime:', latest.get('regime'))
                    
        else:
            print('❌ Erreur API:', response.status_code, response.text)
            
    except Exception as e:
        print('❌ Erreur:', str(e))

if __name__ == "__main__":
    asyncio.run(create_manual_analysis())
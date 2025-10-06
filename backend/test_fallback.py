#!/usr/bin/env python3

import sys
import asyncio
sys.path.append('/app')
sys.path.append('/app/backend')

from enhanced_ohlcv_fetcher import enhanced_ohlcv_fetcher

async def test_multiple_symbols():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
    
    print('🧪 Test du système de fallback multi-sources...')
    print('=' * 60)
    
    success_count = 0
    total_count = len(symbols)
    
    for symbol in symbols:
        try:
            result = await enhanced_ohlcv_fetcher.get_enhanced_ohlcv_data(symbol)
            
            if result is not None and not result.empty:
                price_col = 'Close' if 'Close' in result.columns else 'close'
                price = result[price_col].iloc[-1]
                days = len(result)
                
                print('✅ {}: {} jours, Prix=${:.2f}'.format(symbol, days, price))
                
                # Vérifier si suffisant pour TALib
                if days >= 20:
                    print('   📊 Données suffisantes pour TALib: OUI')
                else:
                    print('   ⚠️  Données insuffisantes pour TALib: {} jours'.format(days))
                
                success_count += 1
            else:
                print('❌ {}: Aucune donnée récupérée'.format(symbol))
                
        except Exception as e:
            print('❌ {}: Erreur - {}'.format(symbol, str(e)))
    
    print('=' * 60)
    print('📊 Résultats: {}/{} symboles récupérés avec succès'.format(success_count, total_count))
    print('💯 Taux de succès du fallback: {:.1f}%'.format(success_count / total_count * 100))
    
    if success_count > 0:
        print('✅ Le système de fallback multi-sources fonctionne!')
    else:
        print('❌ Le système de fallback a échoué sur tous les symboles')

if __name__ == "__main__":
    asyncio.run(test_multiple_symbols())
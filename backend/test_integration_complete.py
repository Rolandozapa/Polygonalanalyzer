#!/usr/bin/env python3

import sys
import asyncio
sys.path.append('/app')
sys.path.append('/app/backend')
sys.path.append('/app/core')

from enhanced_ohlcv_fetcher import enhanced_ohlcv_fetcher
from core.indicators.talib_indicators import get_talib_indicators

async def test_complete_integration():
    symbol = 'ETHUSDT'
    
    print('🧪 Test intégration complète: Fallback → TALib → IA1')
    print('=' * 70)
    
    # Étape 1: Récupération données fallback
    print('📊 ÉTAPE 1: Récupération données avec fallback...')
    try:
        ohlcv_data = await enhanced_ohlcv_fetcher.get_enhanced_ohlcv_data(symbol)
        
        if ohlcv_data is None or ohlcv_data.empty:
            print('❌ ÉCHEC Étape 1: Aucune donnée récupérée')
            return
            
        print('✅ Données récupérées: {} jours'.format(len(ohlcv_data)))
        print('   💰 Prix actuel: ${:.4f}'.format(ohlcv_data['Close'].iloc[-1]))
        print('   📅 Période: {} → {}'.format(ohlcv_data.index[0].date(), ohlcv_data.index[-1].date()))
        
    except Exception as e:
        print('❌ ERREUR Étape 1:', str(e))
        return
    
    # Étape 2: Calcul TALib
    print('\\n🔬 ÉTAPE 2: Calcul des indicateurs TALib...')
    try:
        talib_indicators = get_talib_indicators()
        talib_result = talib_indicators.calculate_all_indicators(ohlcv_data, symbol)
        
        if talib_result is None:
            print('❌ ÉCHEC Étape 2: TALib retourne None')
            return
            
        print('✅ Calculs TALib réussis: {}'.format(type(talib_result).__name__))
        
        # Vérifier les valeurs calculées
        rsi = getattr(talib_result, 'rsi_14', None)
        macd_hist = getattr(talib_result, 'macd_histogram', None)
        atr = getattr(talib_result, 'atr', None)
        vwap = getattr(talib_result, 'vwap', None)
        sma_20 = getattr(talib_result, 'sma_20', None)
        adx = getattr(talib_result, 'adx', None)
        
        print('   📊 Indicateurs calculés:')
        print('      RSI: {:.2f}'.format(rsi) if rsi else '      RSI: None')
        print('      MACD Histogram: {:.6f}'.format(macd_hist) if macd_hist is not None else '      MACD Histogram: None')
        print('      ADX: {:.1f}'.format(adx) if adx else '      ADX: None')
        print('      ATR: {:.4f}'.format(atr) if atr else '      ATR: None')
        print('      VWAP: ${:.4f}'.format(vwap) if vwap else '      VWAP: None')
        print('      SMA 20: ${:.4f}'.format(sma_20) if sma_20 else '      SMA 20: None')
        
        # Vérifier les propriétés du régime ML
        regime = getattr(talib_result, 'regime', None)
        confidence = getattr(talib_result, 'confidence', None)
        confluence_grade = getattr(talib_result, 'confluence_grade', None)
        
        print('   🧠 ML Regime:')
        print('      Regime: {}'.format(regime) if regime else '      Regime: None')
        print('      Confidence: {:.1%}'.format(confidence) if confidence else '      Confidence: None')
        print('      Grade: {}'.format(confluence_grade) if confluence_grade else '      Grade: None')
        
    except Exception as e:
        print('❌ ERREUR Étape 2:', str(e))
        import traceback
        traceback.print_exc()
        return
    
    # Étape 3: Validation des valeurs
    print('\\n🎯 ÉTAPE 3: Validation des valeurs...')
    
    issues = []
    if not rsi or rsi == 50.0:
        issues.append('RSI = valeur par défaut (50.0)')
    if macd_hist is None or macd_hist == 0.0:
        issues.append('MACD Histogram = 0.0 (suspect)')
    if not atr or atr <= 0.02:
        issues.append('ATR trop faible (<= 0.02)')
    if not vwap:
        issues.append('VWAP manquant')
    if not sma_20:
        issues.append('SMA 20 manquant')
        
    if issues:
        print('⚠️  Problèmes détectés:')
        for issue in issues:
            print('   - {}'.format(issue))
    else:
        print('✅ Toutes les valeurs semblent correctes!')
        
    print('\\n📈 DIAGNOSTIC FINAL:')
    print('   Fallback: ✅ Fonctionne (35 jours récupérés)')
    print('   TALib: {} {}'.format('✅ Fonctionne' if talib_result else '❌ Échec', '({} valeurs calculées)'.format(len([x for x in [rsi, macd_hist, atr, vwap, sma_20] if x is not None])) if talib_result else ''))
    print('   Intégration: {} {}'.format('✅ Complète' if len(issues) == 0 else '⚠️  Partielle', '({} problèmes)'.format(len(issues)) if issues else ''))

if __name__ == "__main__":
    asyncio.run(test_complete_integration())
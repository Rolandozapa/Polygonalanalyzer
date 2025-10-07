#!/usr/bin/env python3
"""
Debug script pour analyser le problème MACD = 0
"""

import asyncio
import pandas as pd
import numpy as np
import talib
from enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher

async def debug_macd_calculation():
    """Teste le calcul MACD avec des données réelles"""
    
    print("🔍 DEBUG MACD - Analyse du problème")
    print("="*50)
    
    fetcher = EnhancedOHLCVFetcher()
    
    # Test avec ETHUSDT
    symbol = "ETHUSDT"
    print(f"\n📊 Test avec {symbol}")
    
    try:
        # Récupérer plus de données historiques
        df = await fetcher.fetch_ohlcv_data(symbol, days=60)  # Plus de données
        
        if df is None or len(df) < 26:
            print(f"❌ Données insuffisantes: {len(df) if df is not None else 0} bars")
            return
            
        print(f"✅ Données récupérées: {len(df)} bars")
        
        # Normaliser les données  
        if 'Close' in df.columns:
            close = df['Close'].values.astype(np.float64)
        elif 'close' in df.columns:
            close = df['close'].values.astype(np.float64)
        else:
            print(f"❌ Colonne 'close' manquante. Colonnes: {df.columns.tolist()}")
            return
            
        print(f"📈 Prix de clôture: Min={close.min():.2f}, Max={close.max():.2f}")
        print(f"📈 Derniers prix: {close[-5:].tolist()}")
        
        # Calcul MACD avec TALib
        print(f"\n🔬 Calcul MACD TALib...")
        macd_line, macd_signal_line, macd_histogram = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        print(f"📊 MACD Line: {macd_line[-5:]}")
        print(f"📊 MACD Signal: {macd_signal_line[-5:]}")  
        print(f"📊 MACD Histogram: {macd_histogram[-5:]}")
        
        # Valeurs finales
        final_line = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else 0.0
        final_signal = float(macd_signal_line[-1]) if not np.isnan(macd_signal_line[-1]) else 0.0
        final_histogram = float(macd_histogram[-1]) if not np.isnan(macd_histogram[-1]) else 0.0
        
        print(f"\n🎯 RÉSULTATS FINAUX:")
        print(f"   MACD Line: {final_line:.6f}")
        print(f"   MACD Signal: {final_signal:.6f}")
        print(f"   MACD Histogram: {final_histogram:.6f}")
        
        # Diagnostic
        if final_line == 0.0 and final_signal == 0.0 and final_histogram == 0.0:
            print(f"\n❌ PROBLÈME IDENTIFIÉ: Tous les MACD = 0")
            
            # Vérifier les EMAs manuellement
            ema_12 = talib.EMA(close, timeperiod=12)
            ema_26 = talib.EMA(close, timeperiod=26)
            
            print(f"🔍 EMA 12: {ema_12[-3:]}")
            print(f"🔍 EMA 26: {ema_26[-3:]}")
            
            manual_macd = ema_12[-1] - ema_26[-1]
            print(f"🔍 MACD Manuel (EMA12-EMA26): {manual_macd:.6f}")
            
            if abs(manual_macd) < 0.000001:
                print("✅ DIAGNOSTIC: Marché en consolidation parfaite (MACD légitime proche de 0)")
            else:
                print("❌ DIAGNOSTIC: Problème de calcul TALib")
                
        else:
            print(f"✅ MACD fonctionne correctement")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_macd_calculation())
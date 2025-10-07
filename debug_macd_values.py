#!/usr/bin/env python3
"""
Analyse détaillée des valeurs MACD pour identifier si elles sont normales
"""

import asyncio
import sys
import os
sys.path.append('/app/backend')

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

async def analyze_macd_values():
    """Analyse les valeurs MACD récentes pour plusieurs cryptos"""
    
    load_dotenv('/app/backend/.env')
    client = AsyncIOMotorClient(os.getenv('MONGO_URL'))
    db = client.myapp
    
    print("🔍 ANALYSE COMPARATIVE DES VALEURS MACD")
    print("="*60)
    
    # Analyser plusieurs cryptos récentes
    symbols = ['ETHUSDT', 'BTCUSDT', 'ADAUSDT']
    
    for symbol in symbols:
        analysis = await db.technical_analyses.find({"symbol": symbol}).sort("timestamp", -1).limit(1).to_list(1)
        
        if analysis:
            a = analysis[0]
            
            # Récupérer les valeurs
            current_price = a.get("current_price", 0)
            macd_line = a.get("macd_line", 0)
            macd_signal = a.get("macd_signal", 0)
            macd_histogram = a.get("macd_histogram", 0)
            
            print(f"\n📊 {symbol}:")
            print(f"   Prix actuel: ${current_price:.2f}")
            print(f"   MACD Line: {macd_line:.6f}")
            print(f"   MACD Signal: {macd_signal:.6f}")
            print(f"   MACD Histogram: {macd_histogram:.6f}")
            
            # Calculs de validation
            if current_price > 0:
                macd_percentage = (macd_line / current_price) * 100
                histogram_percentage = (macd_histogram / current_price) * 100
                
                print(f"   MACD/Prix: {macd_percentage:.4f}%")
                print(f"   Histogram/Prix: {histogram_percentage:.4f}%")
                
                # Diagnostic
                if abs(macd_percentage) > 2.0:
                    print(f"   🚨 ALERTE: MACD représente {macd_percentage:.2f}% du prix (anormalement élevé)")
                elif abs(macd_percentage) > 0.5:
                    print(f"   ⚠️ ATTENTION: MACD élevé à {macd_percentage:.2f}% du prix")
                else:
                    print(f"   ✅ Normal: MACD à {macd_percentage:.2f}% du prix")
                    
            # Vérifier cohérence interne MACD
            if abs(macd_line - macd_signal) != abs(macd_histogram):
                calculated_histogram = macd_line - macd_signal
                print(f"   ❌ INCOHÉRENCE: Histogram devrait être {calculated_histogram:.6f} mais vaut {macd_histogram:.6f}")
            else:
                print(f"   ✅ Cohérence MACD validée")
                
        else:
            print(f"\n❌ {symbol}: Pas de données récentes")
    
    # Références normales pour comparaison
    print(f"\n📚 RÉFÉRENCES NORMALES:")
    print(f"   MACD Bitcoin (~$125k): Généralement 50-500")
    print(f"   MACD Ethereum (~$4.7k): Généralement 2-50")  
    print(f"   MACD ADA (~$1): Généralement 0.001-0.05")
    print(f"   Règle générale: |MACD| < 1% du prix")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(analyze_macd_values())
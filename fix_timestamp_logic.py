#!/usr/bin/env python3
"""
🕒 FIX TIMESTAMP LOGIC - Résoudre les problèmes de timestamps et anti-doublon
"""
import sys
sys.path.append('/app/backend')

from datetime import datetime, timezone, timedelta
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_timestamp_problems():
    """Analyser les problèmes de timestamps"""
    logger.info("🔍 ANALYSE DES PROBLÈMES DE TIMESTAMPS")
    logger.info("=" * 60)
    
    # Problème 1: Timestamps identiques
    logger.info("❌ PROBLÈME 1: Toutes les opportunités ont le même timestamp")
    logger.info("   • Les opportunités sont créées en boucle rapide")
    logger.info("   • get_paris_time() retourne la même valeur")
    logger.info("   • Manque de variation temporelle realistic")
    
    # Problème 2: Logique anti-doublon
    logger.info("\n❌ PROBLÈME 2: Logique anti-doublon défaillante")  
    logger.info("   • paris_time_to_timestamp_filter() existe mais inefficace")
    logger.info("   • Pas de vérification de 4h dans IA1 analysis")
    logger.info("   • Tokens analysés plusieurs fois de suite")
    
    # Problème 3: Affichage frontend
    logger.info("\n❌ PROBLÈME 3: Timestamps invisibles dans vignettes")
    logger.info("   • Vignettes IA1/IA2 sans timestamps")
    logger.info("   • Impossible d'identifier les doublons visuellement")
    logger.info("   • Pas de tri chronologique")

def propose_solutions():
    """Proposer des solutions concrètes"""
    logger.info("\n\n🚀 SOLUTIONS PROPOSÉES")
    logger.info("=" * 60)
    
    logger.info("✅ SOLUTION 1: Timestamps échelonnés pour opportunités")
    logger.info("   • Ajouter décalages de secondes entre opportunités")
    logger.info("   • Chaque opportunité = timestamp unique")
    logger.info("   • Préserver l'ordre chronologique réaliste")
    
    logger.info("\n✅ SOLUTION 2: Anti-doublon robuste IA1")
    logger.info("   • Vérifier 'dernière analyse < 4h' avant analyse")
    logger.info("   • Utiliser paris_time_to_timestamp_filter(4)")
    logger.info("   • Ajouter logs pour tracking anti-doublon")
    
    logger.info("\n✅ SOLUTION 3: Affichage timestamps frontend")
    logger.info("   • Ajouter timestamps dans vignettes IA1/IA2")
    logger.info("   • Format: 'il y a 2h34min' ou timestamp exact")
    logger.info("   • Code couleur pour fraîcheur (vert=récent, rouge=ancien)")

def demonstrate_timestamp_spacing():
    """Démontrer l'échelonnement des timestamps"""
    logger.info("\n\n📊 DÉMONSTRATION: Timestamps échelonnés")
    logger.info("=" * 60)
    
    from data_models import get_paris_time
    
    # Méthode actuelle (problématique)
    logger.info("❌ MÉTHODE ACTUELLE (tous identiques):")
    current_method_timestamps = []
    for i in range(5):
        ts = get_paris_time()
        current_method_timestamps.append(ts)
        logger.info(f"   Opportunité {i+1}: {ts.strftime('%H:%M:%S.%f')}")
    
    # Méthode proposée (échelonnée)
    logger.info("\n✅ MÉTHODE PROPOSÉE (échelonnés):")
    import time
    
    base_time = get_paris_time()
    for i in range(5):
        # Décalage de 30 secondes entre chaque opportunité
        offset_seconds = i * 30
        ts = base_time + timedelta(seconds=offset_seconds)
        logger.info(f"   Opportunité {i+1}: {ts.strftime('%H:%M:%S')} (+{offset_seconds}s)")

def test_anti_doublon_logic():
    """Tester la logique anti-doublon"""
    logger.info("\n\n🔍 TEST LOGIQUE ANTI-DOUBLON")
    logger.info("=" * 60)
    
    from server import paris_time_to_timestamp_filter
    from data_models import get_paris_time
    
    current_time = get_paris_time()
    
    # Test différentes périodes
    periods = [1, 2, 4, 6, 12, 24]
    
    for hours in periods:
        filter_result = paris_time_to_timestamp_filter(hours_ago=hours)
        cutoff_time = current_time - timedelta(hours=hours)
        
        logger.info(f"🕒 Filtre {hours}h:")
        logger.info(f"   • Cutoff: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   • Filter: {filter_result}")

def demo_frontend_timestamp_formats():
    """Démontrer les formats de timestamps pour frontend"""
    logger.info("\n\n🎨 FORMATS FRONTEND PROPOSÉS")
    logger.info("=" * 60)
    
    from data_models import get_paris_time
    
    current_time = get_paris_time()
    
    # Différents âges d'analyses
    test_ages = [
        ("Analyse récente", 15),      # 15 minutes
        ("Analyse modérée", 120),     # 2 heures  
        ("Analyse ancienne", 300),    # 5 heures
        ("Analyse très ancienne", 1440)  # 24 heures
    ]
    
    for label, minutes_ago in test_ages:
        analysis_time = current_time - timedelta(minutes=minutes_ago)
        
        # Format 1: Relatif
        if minutes_ago < 60:
            relative = f"il y a {minutes_ago}min"
        elif minutes_ago < 1440:
            hours = minutes_ago // 60
            mins = minutes_ago % 60
            relative = f"il y a {hours}h{mins:02d}min"
        else:
            days = minutes_ago // 1440
            relative = f"il y a {days}j"
        
        # Format 2: Absolu
        absolute = analysis_time.strftime("%H:%M")
        
        # Format 3: Complet
        complete = analysis_time.strftime("%d/%m %H:%M")
        
        # Code couleur
        if minutes_ago < 30:
            color = "🟢 FRAIS"
        elif minutes_ago < 240:  # 4h
            color = "🟡 RÉCENT"
        elif minutes_ago < 1440:  # 24h
            color = "🟠 ANCIEN"
        else:
            color = "🔴 PÉRIMÉ"
        
        logger.info(f"{label}:")
        logger.info(f"   • Relatif: {relative}")
        logger.info(f"   • Absolu: {absolute}")
        logger.info(f"   • Complet: {complete}")
        logger.info(f"   • Status: {color}")

if __name__ == "__main__":
    analyze_timestamp_problems()
    propose_solutions()
    demonstrate_timestamp_spacing()
    test_anti_doublon_logic()
    demo_frontend_timestamp_formats()
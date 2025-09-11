#!/bin/bash
# 📊 ANALYSE DE LA STRUCTURE DU CODE POUR OPTIMISATIONS

echo "📊 ANALYSE DU REPOSITORY - OPTIMISATIONS POSSIBLES"
echo "=================================================="

echo
echo "1. 📏 TAILLE DES FICHIERS BACKEND:"
echo "   (Fichiers >50KB qui pourraient être refactorisés)"
find /app/backend -name "*.py" -size +50k -exec ls -lh {} \; | awk '{print "   📄 " $9 ": " $5}'

echo
echo "2. 🔍 MODULES POTENTIELLEMENT REDONDANTS:"
echo "   (Modules avec fonctions similaires)"

echo "   📌 OHLCV Fetchers:"
echo "   - enhanced_ohlcv_fetcher.py ($(wc -l < /app/backend/enhanced_ohlcv_fetcher.py) lignes)"
echo "   - intelligent_ohlcv_fetcher.py ($(wc -l < /app/backend/intelligent_ohlcv_fetcher.py) lignes)"
echo "   - market_data_service.py ($(wc -l < /app/backend/market_data_service.py) lignes)"

echo "   📌 AI Systems:"
find /app/backend -name "*ai*.py" -exec basename {} \; | while read file; do
    lines=$(wc -l < "/app/backend/$file")
    echo "   - $file ($lines lignes)"
done

echo
echo "3. 🧮 COMPLEXITÉ DES FONCTIONS:"
echo "   (server.py - Fonctions avec >100 lignes)"
awk '/^[[:space:]]*def |^[[:space:]]*async def / { 
    func_name = $0; 
    func_line = NR; 
    indent = match($0, /[^ ]/); 
} 
/^[[:space:]]*def |^[[:space:]]*async def / && indent > 0 && NR > func_line { 
    if (NR - func_line > 100) 
        print "   🔴 " func_name " (" (NR - func_line) " lignes)"; 
    func_name = $0; 
    func_line = NR; 
    indent = match($0, /[^ ]/); 
}' /app/backend/server.py

echo
echo "4. 📦 IMPORTS ET DÉPENDANCES:"
echo "   (Modules avec beaucoup d'imports)"
find /app/backend -name "*.py" -exec basename {} \; | while read file; do
    import_count=$(grep -c "^import\|^from.*import" "/app/backend/$file" 2>/dev/null || echo 0)
    if [ "$import_count" -gt 20 ]; then
        echo "   📄 $file: $import_count imports"
    fi
done

echo
echo "5. 🔗 COUPLAGE ENTRE MODULES:"
echo "   (Modules fréquemment importés)"
echo "   Analysing interdependencies..."

# Analyser quels modules sont le plus souvent importés
for py_file in /app/backend/*.py; do
    basename_file=$(basename "$py_file" .py)
    count=$(grep -l "$basename_file" /app/backend/*.py | wc -l)
    if [ "$count" -gt 3 ]; then
        echo "   🔗 $basename_file.py: importé dans $count fichiers"
    fi
done

echo
echo "6. 💾 UTILISATION MÉMOIRE POTENTIELLE:"
echo "   (Gros imports ou données en mémoire)"
grep -n "import pandas\|import numpy\|pd\.DataFrame\|np\.array" /app/backend/server.py | wc -l | while read pandas_usage; do
    if [ "$pandas_usage" -gt 10 ]; then
        echo "   ⚠️  Utilisation intensive de Pandas/Numpy détectée ($pandas_usage occurrences)"
    else
        echo "   ✅ Utilisation modérée de Pandas/Numpy ($pandas_usage occurrences)"
    fi
done

echo
echo "7. 🎯 RECOMMANDATIONS D'OPTIMISATION:"
echo "=================================="

echo "   📏 STRUCTURE:"
echo "   - server.py (592KB) → Considérer split en modules thématiques"
echo "   - Séparer logiques IA1, IA2, Market Analysis, Trading"

echo "   🔄 REFACTORING SUGGÉRÉ:"
echo "   - Fusionner enhanced_ohlcv_fetcher + intelligent_ohlcv_fetcher"
echo "   - Centraliser les AI systems dans un package"
echo "   - Créer un module trading_core séparé"

echo "   🚀 PERFORMANCE:"
echo "   - Lazy loading des gros modules"
echo "   - Cache plus agressif pour market data"
echo "   - Async partout où possible"

echo
echo "✅ ANALYSE TERMINÉE"
echo "=================="
echo "📊 Total fichiers Python: $(find /app/backend -name "*.py" | wc -l)"
echo "📏 Taille totale backend: $(du -sh /app/backend | cut -f1)"
echo "🎯 Priority: Refactoriser server.py et consolider OHLCV modules"
#!/bin/bash
# 🧹 SCRIPT DE NETTOYAGE INTELLIGENT DU REPOSITORY
# Clean repo from unnecessary files while preserving essential ones

echo "🧹 NETTOYAGE INTELLIGENT DU REPOSITORY"
echo "======================================"

# Compteurs
DELETED_FILES=0
SAVED_SPACE=0

# Fonction pour calculer la taille d'un fichier/dossier
get_size() {
    if [ -e "$1" ]; then
        du -sb "$1" 2>/dev/null | cut -f1
    else
        echo 0
    fi
}

# Fonction pour supprimer avec confirmation de taille
safe_delete() {
    local target="$1"
    local description="$2"
    
    if [ -e "$target" ]; then
        local size=$(get_size "$target")
        if [ "$size" -gt 0 ]; then
            echo "🗑️  $description: $(du -sh "$target" 2>/dev/null | cut -f1)"
            rm -rf "$target"
            DELETED_FILES=$((DELETED_FILES + 1))
            SAVED_SPACE=$((SAVED_SPACE + size))
        fi
    fi
}

echo
echo "1. 🧹 Suppression des fichiers de cache Python..."
safe_delete "/app/backend/__pycache__" "Cache Python backend"

echo
echo "2. 🧹 Suppression des caches Node.js..."
safe_delete "/app/frontend/node_modules/.cache" "Cache Node.js"

echo
echo "3. 🧹 Suppression des fichiers de log de test..."
for log_file in /app/*.log; do
    if [ -f "$log_file" ]; then
        # Garder seulement les logs système essentiels, supprimer les logs de test
        basename_log=$(basename "$log_file")
        case "$basename_log" in
            *test* | *TEST* )
                safe_delete "$log_file" "Log de test: $basename_log"
                ;;
            *)
                echo "⏭️  Préservé: $basename_log (log système)"
                ;;
        esac
    fi
done

echo
echo "4. 🧹 Suppression des anciens fichiers de test Python..."
# Supprimer les fichiers de test mais garder les modules de test essentiels
cd /app
for test_file in *test*.py *TEST*.py; do
    if [ -f "$test_file" ]; then
        case "$test_file" in
            # Préserver les modules de test essentiels
            "backend_test.py" | "test_result.md" )
                echo "⏭️  Préservé: $test_file (module de test essentiel)"
                ;;
            # Supprimer les autres fichiers de test
            *)
                safe_delete "$test_file" "Fichier de test: $test_file"
                ;;
        esac
    fi
done

echo
echo "5. 🧹 Suppression des fichiers temporaires divers..."
find /app -name "*.tmp" -o -name "*.bak" -o -name "*.old" -o -name "*~" | while read -r file; do
    if [ -f "$file" ]; then
        safe_delete "$file" "Fichier temporaire: $(basename "$file")"
    fi
done

echo
echo "6. 🧹 Nettoyage des node_modules si nécessaire..."
# Vérifier si node_modules est très volumineux (>500MB) et le nettoyer
if [ -d "/app/frontend/node_modules" ]; then
    node_modules_size=$(get_size "/app/frontend/node_modules")
    node_modules_size_mb=$((node_modules_size / 1024 / 1024))
    
    if [ "$node_modules_size_mb" -gt 500 ]; then
        echo "⚠️  node_modules est volumineux (${node_modules_size_mb}MB). Considérer yarn install --production"
    else
        echo "✅ node_modules de taille raisonnable (${node_modules_size_mb}MB)"
    fi
fi

echo
echo "7. 🧹 Suppression des logs système volumineux (>10MB)..."
for log_file in /var/log/supervisor/*.log; do
    if [ -f "$log_file" ]; then
        log_size=$(get_size "$log_file")
        log_size_mb=$((log_size / 1024 / 1024))
        
        if [ "$log_size_mb" -gt 10 ]; then
            echo "🗑️  Log volumineux: $(basename "$log_file") (${log_size_mb}MB)"
            # Garder seulement les 1000 dernières lignes
            tail -n 1000 "$log_file" > "${log_file}.tmp"
            mv "${log_file}.tmp" "$log_file"
            DELETED_FILES=$((DELETED_FILES + 1))
        fi
    fi
done

# Conversion de l'espace sauvé en unités lisibles
if [ "$SAVED_SPACE" -gt 1073741824 ]; then
    SAVED_SPACE_HUMAN="$(echo "scale=1; $SAVED_SPACE/1073741824" | bc)GB"
elif [ "$SAVED_SPACE" -gt 1048576 ]; then
    SAVED_SPACE_HUMAN="$(echo "scale=1; $SAVED_SPACE/1048576" | bc)MB"
else
    SAVED_SPACE_HUMAN="$(echo "scale=1; $SAVED_SPACE/1024" | bc)KB"
fi

echo
echo "✅ NETTOYAGE TERMINÉ"
echo "==================="
echo "📁 Fichiers supprimés: $DELETED_FILES"
echo "💾 Espace libéré: $SAVED_SPACE_HUMAN"
echo
echo "🚀 Repository nettoyé et optimisé!"
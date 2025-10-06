# 🎯 Prompts Management System

## Structure des Prompts

Ce répertoire contient tous les prompts externalisés du système de trading IA dual.

### Prompts Disponibles

- **`ia1_v6_advanced.json`** : Prompt IA1 v6.0 avec analyse technique avancée et détection de régime ML
- **`ia2_strategic.json`** : Prompt IA2 pour l'analyse stratégique et la validation des décisions
- **`prompt_manager.py`** : Gestionnaire centralisé des prompts

### Format des Prompts JSON

```json
{
  "name": "Nom du prompt",
  "version": "1.0",
  "description": "Description du prompt",
  "model": "gpt-4o",
  "temperature": 0.1,
  "max_tokens": 4000,
  "system_prompt": "Prompt système",
  "prompt_template": "Template avec {variables}",
  "required_variables": ["var1", "var2"]
}
```

### Utilisation

```python
from prompts.prompt_manager import prompt_manager

# Charger un prompt
config = prompt_manager.get_prompt_config('ia1_v6_advanced')
formatted_prompt = prompt_manager.format_prompt('ia1_v6_advanced', variables)
```

### Avantages

- ✅ **Prompts externalisés** : Facilement modifiables sans redémarrage
- ✅ **Versioning** : Suivi des versions des prompts  
- ✅ **Validation** : Vérification des variables requises
- ✅ **Cache** : Performance optimisée
- ✅ **Configuration** : Paramètres de modèle centralisés

### Modification des Prompts

1. Éditer le fichier JSON correspondant
2. Recharger avec `prompt_manager.reload_prompt('nom_prompt')`
3. Ou redémarrer l'application

### Variables Disponibles

Voir `required_variables` dans chaque fichier JSON pour la liste complète des variables disponibles pour chaque prompt.

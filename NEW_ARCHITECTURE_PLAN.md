# NOUVELLE ARCHITECTURE - TRADING BOT REFONTE COMPLÈTE

## 🎯 OBJECTIFS DE LA REFONTE

1. **Prompts externalisés** : Fichiers JSON séparés, facilement modifiables
2. **Indicateurs modulaires** : Système d'indicateurs techniques propre et extensible  
3. **Architecture claire** : Séparation des responsabilités
4. **Configuration externe** : Tous les paramètres configurables
5. **Code maintenable** : Structure organisée et documentée

## 📁 NOUVELLE STRUCTURE DU REPO

```
/app/
├── 📁 config/
│   ├── api_keys.json                    # Clés API centralisées
│   ├── trading_config.json              # Configuration trading
│   └── indicators_config.json           # Configuration indicateurs
│
├── 📁 prompts/
│   ├── ia1_v6.json                      # Prompt IA1 complet
│   ├── ia2_v6.json                      # Prompt IA2 complet  
│   └── templates/                       # Templates réutilisables
│       ├── reasoning_template.json
│       └── analysis_template.json
│
├── 📁 core/
│   ├── __init__.py
│   ├── 📁 indicators/                   # Système d'indicateurs modulaire
│   │   ├── __init__.py
│   │   ├── base_indicator.py           # Classe de base
│   │   ├── talib_indicators.py         # Indicateurs TALib
│   │   ├── custom_indicators.py        # Indicateurs personnalisés
│   │   ├── regime_detector.py          # Détection de régime
│   │   └── confluence_calculator.py    # Calcul confluence
│   │
│   ├── 📁 ai/                          # Système IA
│   │   ├── __init__.py
│   │   ├── prompt_manager.py           # Gestion des prompts
│   │   ├── ia1_engine.py               # Moteur IA1
│   │   ├── ia2_engine.py               # Moteur IA2
│   │   └── llm_client.py               # Client LLM unifié
│   │
│   ├── 📁 market/                      # Données de marché
│   │   ├── __init__.py
│   │   ├── data_fetcher.py             # Récupération données
│   │   ├── bingx_client.py             # Client BingX
│   │   ├── ohlcv_manager.py            # Gestion OHLCV
│   │   └── opportunity_scout.py        # Scout opportunités
│   │
│   └── 📁 analysis/                    # Système d'analyse
│       ├── __init__.py
│       ├── technical_analyzer.py       # Analyseur technique
│       ├── pattern_detector.py         # Détecteur de patterns
│       ├── risk_calculator.py          # Calculateur risque
│       └── position_sizer.py           # Taille de position
│
├── 📁 api/                             # API FastAPI
│   ├── __init__.py
│   ├── main.py                         # Point d'entrée FastAPI
│   ├── 📁 routes/                      # Routes API
│   │   ├── __init__.py
│   │   ├── opportunities.py
│   │   ├── analysis.py
│   │   ├── trading.py
│   │   └── admin.py
│   │
│   └── 📁 models/                      # Modèles Pydantic
│       ├── __init__.py
│       ├── market_models.py
│       ├── analysis_models.py
│       └── trading_models.py
│
├── 📁 frontend/                        # Interface React (inchangé)
│   ├── src/
│   ├── public/
│   └── package.json
│
├── 📁 tests/                           # Tests automatisés
│   ├── test_indicators.py
│   ├── test_ia_engines.py
│   └── test_api.py
│
├── 📁 scripts/                         # Scripts utilitaires
│   ├── migrate_data.py
│   ├── test_indicators.py
│   └── setup_environment.py
│
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## 🔧 MODULES CLÉS À CRÉER

### 1. Prompt Manager (core/ai/prompt_manager.py)
```python
class PromptManager:
    def load_prompt(self, name: str) -> Dict
    def format_prompt(self, template: str, variables: Dict) -> str
    def validate_variables(self, prompt: Dict, variables: Dict) -> bool
```

### 2. Indicators System (core/indicators/)
```python
class BaseIndicator:
    def calculate(self, data: pd.DataFrame) -> Dict
    def validate_data(self, data: pd.DataFrame) -> bool

class TALibIndicators(BaseIndicator):
    def calculate_rsi(self, data, period=14) -> float
    def calculate_macd(self, data) -> Tuple[float, float, float]
    def calculate_all(self, data) -> TechnicalAnalysis
```

### 3. IA Engines (core/ai/)
```python
class IA1Engine:
    def __init__(self, prompt_manager: PromptManager, indicators: TALibIndicators)
    def analyze(self, opportunity: Opportunity) -> AnalysisResult

class IA2Engine:
    def __init__(self, prompt_manager: PromptManager)
    def decide(self, ia1_result: AnalysisResult) -> TradingDecision
```

## 📋 PLAN DE MIGRATION

### Phase 1: Infrastructure de base
1. ✅ Créer la nouvelle structure de dossiers
2. ✅ Externaliser les prompts en JSON
3. ✅ Créer le PromptManager
4. ✅ Migrer les configurations

### Phase 2: Système d'indicateurs
1. ✅ Créer BaseIndicator et TALibIndicators  
2. ✅ Migrer le système de calcul d'indicateurs
3. ✅ Tester tous les indicateurs isolément
4. ✅ Créer le régime detector modulaire

### Phase 3: Moteurs IA
1. ✅ Créer IA1Engine et IA2Engine
2. ✅ Migrer la logique d'analyse
3. ✅ Tester les moteurs IA séparément
4. ✅ Intégrer avec le nouveau système

### Phase 4: API refactorée  
1. ✅ Créer les nouvelles routes FastAPI
2. ✅ Migrer les endpoints existants
3. ✅ Tester l'API complète
4. ✅ Déployer la nouvelle version

## 🎯 AVANTAGES DE LA NOUVELLE ARCHITECTURE

1. **Maintenabilité** : Code organisé et modulaire
2. **Flexibilité** : Prompts et config modifiables sans redéploiement  
3. **Testabilité** : Modules isolés facilement testables
4. **Évolutivité** : Ajout facile de nouveaux indicateurs/IA
5. **Lisibilité** : Structure claire et documentée
6. **Performance** : Optimisation possible par module

## 🚀 DÉMARRAGE DE LA REFONTE

Commence par créer la structure de base et migrer progressivement les fonctionnalités existantes.
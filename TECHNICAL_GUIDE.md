# 🔧 GUIDE TECHNIQUE DÉTAILLÉ - DUAL AI TRADING BOT

## 📋 IMPLÉMENTATION TECHNIQUE

### État Actuel du Système

**Architecture Legacy** (en cours de migration) :
- `/app/backend/server.py` - Système monolithique (579KB)
- Prompts hardcodés dans le code
- Configuration dispersée
- Indicateurs mélangés avec logique métier

**Nouvelle Architecture v6.0** (en développement) :
- Structure modulaire avec séparation claire
- Configuration et prompts externalisés
- Système d'indicateurs basé sur TALib
- Modules testables indépendamment

---

## 🔄 MIGRATION EN COURS

### Phase Actuelle : Foundation
✅ **Terminé** :
- Structure de dossiers créée (`/app/core/`, `/app/config/`, `/app/prompts/`)
- Configuration externalisée (API keys, trading config)
- Prompt IA1 externalisé (`/app/prompts/ia1_v6.json`)
- PromptManager fonctionnel (`/app/core/ai/prompt_manager.py`)
- Base indicators system (`/app/core/indicators/base_indicator.py`)

🚧 **En Cours** :
- Système TALib complet (`/app/core/indicators/talib_indicators.py`)
- Moteurs IA1/IA2 modulaires
- Migration progressive des fonctionnalités

⏳ **À Faire** :
- API FastAPI refactorisée  
- Tests automatisés
- Décommissionnement du legacy system

---

## 📊 DÉTAILS TECHNIQUES SYSTÈME ACTUEL

### Backend Monolithique (`/app/backend/server.py`)

**Points Critiques à Migrer** :

1. **IA1 Analysis** (lignes 2000-3000) :
```python
# Logique actuelle complexe à extraire
async def run_ia1_ultra_professional_analysis(self, opportunity: MarketOpportunity):
    # 500+ lignes de logique mélangée
    # Indicateurs, prompts, LLM calls, parsing
    # → À migrer vers core/ai/ia1_engine.py
```

2. **Indicateurs Techniques** (dispersés) :
```python
# Calculs RSI, MACD, ADX éparpillés dans server.py  
# → À migrer vers core/indicators/talib_indicators.py
```

3. **Régime Detection** (lignes 4000+) :
```python
# Logique régime mélangée avec confluence
# → À migrer vers core/indicators/regime_detector.py
```

### Données et APIs

**BingX Integration** :
- API Keys : Stockées dans `/app/backend/.env`
- Client : Code dispersé dans server.py
- → Migrer vers `/app/core/market/bingx_client.py`

**OHLCV Management** :
- Multiple fetchers : `enhanced_ohlcv_fetcher.py`, `intelligent_ohlcv_fetcher.py`
- Logique dupliquée et complexe
- → Consolider dans `/app/core/market/ohlcv_manager.py`

---

## 🔧 DÉTAILS D'IMPLÉMENTATION

### Système d'Indicateurs TALib

**Structure Recommandée** :
```python
# /app/core/indicators/talib_indicators.py
class TALibIndicators(BaseIndicator):
    def __init__(self, config: Dict = None):
        # Load from /app/config/trading_config.json
        
    def calculate_all(self, data: pd.DataFrame) -> TechnicalAnalysisComplete:
        # Tous les indicateurs en une fois
        # RSI, MACD, ADX, BB, Stochastic, MFI, ATR, VWAP
        
    def calculate_rsi(self, close: np.array, periods: List[int]) -> Dict:
        # Utilise talib.RSI avec multi-periods
        
    def calculate_regime(self, indicators: Dict) -> RegimeResult:
        # Détection des 10 régimes
```

**Configuration** :
```json
// /app/config/trading_config.json
{
  "indicators": {
    "rsi": {"periods": [9, 14, 21], "overbought": 70},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "adx": {"period": 14, "strong_threshold": 25}
  }
}
```

### Moteur IA1 

**Structure Recommandée** :
```python
# /app/core/ai/ia1_engine.py
class IA1Engine:
    def __init__(self, prompt_manager: PromptManager, indicators: TALibIndicators):
        # Injection des dépendances
        
    async def analyze(self, opportunity: MarketOpportunity) -> AnalysisResult:
        # 1. Fetch OHLCV data
        # 2. Calculate indicators
        # 3. Detect regime  
        # 4. Format prompt with variables
        # 5. Call LLM
        # 6. Parse response
        # 7. Return structured result
```

**Prompt Integration** :
```python
# Utilisation du nouveau système
variables = {
    "symbol": opportunity.symbol,
    "regime": regime_analysis.regime,
    "confidence": regime_analysis.confidence,
    "rsi": indicators.rsi_14,
    # ... toutes les variables TALib
}

prompt = self.prompt_manager.format_prompt("ia1_v6", variables)
response = await self.llm_client.call(prompt)
```

### LLM Client Unifié

**Structure Recommandée** :
```python
# /app/core/ai/llm_client.py
class UnifiedLLMClient:
    def __init__(self):
        # Support GPT-4o, Claude-3.7, Emergent LLM Key
        
    async def call(self, prompt: str, model: str = "gpt-4o") -> str:
        # Unified interface pour tous les LLM
        
    async def call_with_retry(self, prompt: str) -> str:
        # Retry logic avec fallback models
```

---

## 🗂️ STRUCTURE DONNÉES

### Modèles Pydantic Actuels

**À Conserver/Migrer** :
```python
# Dans server.py - à migrer vers /app/api/models/
class MarketOpportunity(BaseModel):
    symbol: str
    current_price: float
    volume_24h: float
    price_change_24h: float
    data_sources: List[str]
    
class TechnicalAnalysis(BaseModel):
    symbol: str
    rsi: float 
    macd_histogram: float
    confluence_grade: str
    regime: str
    # ... 50+ autres champs
```

**Nouveau Système Modulaire** :
```python
# /app/core/indicators/base_indicator.py (déjà créé)
@dataclass
class TechnicalAnalysisComplete:
    # Structure complète avec tous les indicateurs TALib
    
# /app/api/models/analysis_models.py (à créer)
class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: Optional[str] = "1h"
    
class AnalysisResponse(BaseModel):
    success: bool
    analysis: TechnicalAnalysisComplete
    escalated: bool
```

---

## 🔄 FLUX DE DONNÉES DÉTAILLÉ

### Pipeline Actuel (Legacy)
```
1. BingX API → trending_auto_updater.py
2. Opportunities → advanced_market_aggregator.py  
3. OHLCV → intelligent_ohlcv_fetcher.py
4. Technical Analysis → server.py (monolithique)
5. IA1 → server.py (mélangé)
6. IA2 → server.py (mélangé)
```

### Pipeline Nouveau (Target)
```
1. BingX API → core/market/bingx_client.py
2. Scout → core/market/opportunity_scout.py
3. OHLCV → core/market/ohlcv_manager.py
4. Indicators → core/indicators/talib_indicators.py
5. Regime → core/indicators/regime_detector.py
6. Confluence → core/indicators/confluence_calculator.py
7. IA1 → core/ai/ia1_engine.py
8. IA2 → core/ai/ia2_engine.py
9. API → api/routes/*.py
```

---

## 🧪 TESTS ET VALIDATION

### Tests Actuels
- Peu de tests automatisés
- Validation manuelle via endpoints
- Logs pour debug

### Tests Nouveau Système
```python
# /app/tests/test_indicators.py
def test_rsi_calculation():
    # Test RSI avec données connues
    
def test_regime_detection():
    # Test détection régime avec scenarios
    
# /app/tests/test_ia_engines.py  
def test_ia1_analysis():
    # Test IA1 avec mock data
    
# /app/tests/test_prompts.py
def test_prompt_formatting():
    # Test formatage prompts avec variables
```

**Scripts de Test** :
```bash
# /app/scripts/test_indicators.py
python scripts/test_indicators.py --indicator RSI --symbol BTCUSDT

# /app/scripts/test_system_integration.py  
python scripts/test_system_integration.py --full-pipeline
```

---

## 📊 PERFORMANCE ET MONITORING

### Métriques Actuelles
- CPU usage élevé (90%+)
- Memory leaks possibles  
- Temps de réponse variables

### Optimisations Prévues

**Indicateurs** :
- Cache intelligent des calculs TALib
- Parallélisation calculs multi-symboles
- Réutilisation données OHLCV

**IA Calls** :
- Connection pooling LLM
- Retry exponential backoff
- Prompt caching intelligent

**Base de Données** :
- Index MongoDB optimisés
- Pagination résultats
- Cleanup automatique anciens data

---

## 🚨 POINTS CRITIQUES À SURVEILLER

### Migration Risks

1. **Compatibilité Données** :
   - Structure TechnicalAnalysis change
   - Versionning nécessaire
   - Migration graduelle requise

2. **Performance Dégradation** :
   - Nouveau système peut être plus lent initialement
   - Monitoring CPU/Memory crucial
   - Rollback plan nécessaire

3. **Prompt Variables** :
   - Nouveaux prompts utilisent variables différentes
   - Validation extensive requise
   - Fallback vers ancien système

### Code Legacy à Surveiller

**Fichiers Critiques** :
- `/app/backend/server.py` (579KB - cœur du système)
- `/app/backend/trending_auto_updater.py` (scout system)
- `/app/backend/advanced_market_aggregator.py` (opportunities)

**APIs Externes** :
- BingX rate limits
- LLM API quotas  
- Backup data sources

---

## 🎯 ROADMAP TECHNIQUE

### Phase 1 : Foundation ✅
- Structure modulaire
- Configuration externe
- Prompts externalisés  
- Base indicators

### Phase 2 : Core Systems 🚧
- TALib indicators complets
- Régime detection
- Confluence calculator
- IA1/IA2 engines

### Phase 3 : API Migration ⏳
- FastAPI routes modulaires
- Modèles Pydantic v2
- Tests automatisés
- Documentation API

### Phase 4 : Production 🔮
- Monitoring complet
- Performance optimizations
- Backup systems
- Scaling preparations

---

## 💻 COMMANDES DÉVELOPPEUR

### Setup Développement
```bash
# Installation complète
cd /app
pip install -r requirements.txt
yarn install

# Tests
python -m pytest tests/
python scripts/test_indicators.py

# Linting  
flake8 core/
black core/

# Services
sudo supervisorctl restart backend
sudo supervisorctl restart frontend
```

### Debug Courant
```bash
# Logs temps réel
tail -f /var/log/supervisor/backend.*.log

# Test API
curl -X POST /api/force-ia1-analysis -d '{"symbol":"BTCUSDT"}'

# Test nouveaux modules
cd /app && python -c "from core.ai.prompt_manager import get_prompt_manager; pm = get_prompt_manager(); print(pm.list_prompts())"
```

### Migration Utils
```bash
# Backup avant migration
cp -r backend/ backend_backup/

# Test nouveau système
python scripts/compare_old_new_system.py

# Rollback si nécessaire
mv backend_backup/ backend/
sudo supervisorctl restart backend
```

---

*Guide technique pour Dual AI Trading Bot v6.0 - Migration Architecture Modulaire*
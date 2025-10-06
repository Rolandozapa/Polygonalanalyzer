import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """Gestionnaire centralisé des prompts externalisés"""
    
    def __init__(self, prompts_dir: str = "/app/prompts"):
        self.prompts_dir = prompts_dir
        self.prompts_cache = {}
        
    def load_prompt(self, prompt_name: str) -> Optional[Dict[str, Any]]:
        """Charge un prompt depuis un fichier JSON"""
        
        if prompt_name in self.prompts_cache:
            return self.prompts_cache[prompt_name]
            
        prompt_file = os.path.join(self.prompts_dir, f"{prompt_name}.json")
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
                
            self.prompts_cache[prompt_name] = prompt_data
            logger.info(f"✅ Prompt loaded: {prompt_name} v{prompt_data.get('version', '1.0')}")
            return prompt_data
            
        except FileNotFoundError:
            logger.error(f"❌ Prompt file not found: {prompt_file}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in prompt file {prompt_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error loading prompt {prompt_name}: {e}")
            return None
            
    def format_prompt(self, prompt_name: str, variables: Dict[str, Any]) -> Optional[str]:
        """Formate un prompt avec les variables fournies"""
        
        prompt_data = self.load_prompt(prompt_name)
        if not prompt_data:
            return None
            
        template = prompt_data.get('prompt_template', '')
        required_vars = prompt_data.get('required_variables', [])
        
        # Vérifier les variables requises
        missing_vars = [var for var in required_vars if var not in variables]
        if missing_vars:
            logger.warning(f"⚠️ Missing variables for {prompt_name}: {missing_vars}")
            # Ajouter des valeurs par défaut pour les variables manquantes
            for var in missing_vars:
                variables[var] = 'N/A'
        
        try:
            formatted_prompt = template.format(**variables)
            logger.info(f"✅ Prompt formatted: {prompt_name} ({len(formatted_prompt)} chars)")
            return formatted_prompt
            
        except KeyError as e:
            logger.error(f"❌ Missing variable in prompt {prompt_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error formatting prompt {prompt_name}: {e}")
            return None
            
    def get_prompt_config(self, prompt_name: str) -> Optional[Dict[str, Any]]:
        """Récupère la configuration d'un prompt (modèle, température, etc.)"""
        
        prompt_data = self.load_prompt(prompt_name)
        if not prompt_data:
            return None
            
        return {
            'model': prompt_data.get('model', 'gpt-4o'),
            'temperature': prompt_data.get('temperature', 0.1),
            'max_tokens': prompt_data.get('max_tokens', 4000),
            'system_prompt': prompt_data.get('system_prompt', '')
        }
        
    def reload_prompt(self, prompt_name: str) -> bool:
        """Recharge un prompt depuis le fichier (utile pour le développement)"""
        
        if prompt_name in self.prompts_cache:
            del self.prompts_cache[prompt_name]
            
        return self.load_prompt(prompt_name) is not None
        
    def list_available_prompts(self) -> list:
        """Liste tous les prompts disponibles"""
        
        try:
            files = os.listdir(self.prompts_dir)
            prompt_files = [f[:-5] for f in files if f.endswith('.json') and not f.startswith('.')]
            return prompt_files
        except Exception as e:
            logger.error(f"❌ Error listing prompts: {e}")
            return []

# Instance globale
prompt_manager = PromptManager()

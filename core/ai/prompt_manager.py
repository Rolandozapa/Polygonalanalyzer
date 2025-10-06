"""
Professional Prompt Manager for Trading Bot
Handles external prompt loading, formatting, and validation
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages external prompt files and formatting for AI engines
    """
    
    def __init__(self, prompts_dir: str = "/app/prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.loaded_prompts: Dict[str, Dict] = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all prompt files from the prompts directory"""
        try:
            for prompt_file in self.prompts_dir.glob("*.json"):
                prompt_name = prompt_file.stem
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                    self.loaded_prompts[prompt_name] = prompt_data
                    logger.info(f"✅ Loaded prompt: {prompt_name} v{prompt_data.get('version', 'unknown')}")
            
            logger.info(f"🚀 PromptManager initialized with {len(self.loaded_prompts)} prompts")
            
        except Exception as e:
            logger.error(f"❌ Error loading prompts: {e}")
            self.loaded_prompts = {}
    
    def get_prompt(self, name: str) -> Optional[Dict]:
        """Get a prompt by name"""
        if name not in self.loaded_prompts:
            logger.error(f"❌ Prompt not found: {name}")
            logger.info(f"   Available prompts: {list(self.loaded_prompts.keys())}")
            return None
        
        return self.loaded_prompts[name]
    
    def format_prompt(self, prompt_name: str, variables: Dict[str, Any]) -> Optional[str]:
        """
        Format a prompt with provided variables
        
        Args:
            prompt_name: Name of the prompt to format
            variables: Dictionary of variables to substitute
            
        Returns:
            Formatted prompt string or None if error
        """
        try:
            prompt_data = self.get_prompt(prompt_name)
            if not prompt_data:
                return None
            
            # Validate required variables
            if not self.validate_variables(prompt_name, variables):
                return None
            
            # Build the complete prompt
            formatted_sections = []
            
            main_prompt = prompt_data.get('main_prompt', {})
            
            # Add header
            if 'header' in main_prompt:
                formatted_sections.append(main_prompt['header'].format(**variables))
            
            # Add market context
            if 'market_context' in main_prompt:
                formatted_sections.append(main_prompt['market_context'].format(**variables))
            
            # Add regime analysis
            if 'regime_analysis' in main_prompt:
                formatted_sections.append(main_prompt['regime_analysis'].format(**variables))
            
            # Add technical data
            if 'technical_data' in main_prompt:
                formatted_sections.append(main_prompt['technical_data'].format(**variables))
            
            # Add confluence grade
            if 'confluence_grade' in main_prompt:
                formatted_sections.append(main_prompt['confluence_grade'].format(**variables))
            
            # Add decision framework
            if 'decision_framework' in main_prompt:
                formatted_sections.append(main_prompt['decision_framework'].format(**variables))
            
            # Add instructions
            if 'instructions' in main_prompt:
                formatted_sections.append(main_prompt['instructions'].format(**variables))
            
            # Add JSON format with formatted reasoning
            if 'json_format' in main_prompt:
                json_format = main_prompt['json_format'].copy()
                if 'reasoning' in json_format:
                    json_format['reasoning'] = json_format['reasoning'].format(**variables)
                formatted_sections.append("🚨 FORMAT DE RÉPONSE OBLIGATOIRE:")
                formatted_sections.append("```json")
                formatted_sections.append(json.dumps(json_format, indent=2, ensure_ascii=False))
                formatted_sections.append("```")
            
            # Join all sections
            formatted_prompt = "\n\n".join(formatted_sections)
            
            logger.info(f"✅ Prompt {prompt_name} formatted successfully with {len(variables)} variables")
            logger.debug(f"   Variables used: {list(variables.keys())}")
            
            return formatted_prompt
            
        except KeyError as e:
            logger.error(f"❌ Missing variable in prompt {prompt_name}: {e}")
            logger.info(f"   Available variables: {list(variables.keys())}")
            return None
        except Exception as e:
            logger.error(f"❌ Error formatting prompt {prompt_name}: {e}")
            return None
    
    def validate_variables(self, prompt_name: str, variables: Dict[str, Any]) -> bool:
        """
        Validate that all required variables are present
        
        Args:
            prompt_name: Name of the prompt
            variables: Variables to validate
            
        Returns:
            True if valid, False otherwise
        """
        prompt_data = self.get_prompt(prompt_name)
        if not prompt_data:
            return False
        
        validation_config = prompt_data.get('validation', {})
        required_vars = validation_config.get('required_variables', [])
        
        missing_vars = [var for var in required_vars if var not in variables]
        
        if missing_vars:
            logger.error(f"❌ Missing required variables for {prompt_name}: {missing_vars}")
            return False
        
        # Validate signal options if present
        if 'signal' in variables and 'signal_options' in validation_config:
            valid_signals = validation_config['signal_options']
            if variables['signal'] not in valid_signals:
                logger.warning(f"⚠️ Invalid signal '{variables['signal']}' for {prompt_name}")
        
        # Validate confidence range if present
        if 'confidence' in variables and 'confidence_range' in validation_config:
            conf_range = validation_config['confidence_range']
            if not (conf_range[0] <= variables['confidence'] <= conf_range[1]):
                logger.warning(f"⚠️ Confidence {variables['confidence']} outside range {conf_range}")
        
        logger.debug(f"✅ Variables validation passed for {prompt_name}")
        return True
    
    def get_prompt_info(self, prompt_name: str) -> Dict:
        """Get information about a prompt"""
        prompt_data = self.get_prompt(prompt_name)
        if not prompt_data:
            return {}
        
        return {
            'name': prompt_data.get('name'),
            'version': prompt_data.get('version'),
            'description': prompt_data.get('description'),
            'model': prompt_data.get('model'),
            'variables': prompt_data.get('variables', []),
            'required_variables': prompt_data.get('validation', {}).get('required_variables', [])
        }
    
    def list_prompts(self) -> List[str]:
        """List all available prompts"""
        return list(self.loaded_prompts.keys())
    
    def reload_prompts(self):
        """Reload all prompts from disk"""
        logger.info("🔄 Reloading prompts from disk...")
        self.loaded_prompts.clear()
        self._load_all_prompts()

# Global instance
prompt_manager = PromptManager()

def get_prompt_manager() -> PromptManager:
    """Get global prompt manager instance"""
    return prompt_manager
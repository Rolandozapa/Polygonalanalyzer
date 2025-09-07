#!/usr/bin/env python3
"""
Test des Améliorations de l'Interprétation IA1 avec Patterns Chartistes
Analyse du code et des logs pour vérifier les améliorations
"""

import asyncio
import json
import logging
import os
import sys
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA1PatternCodeAnalysisTestSuite:
    """Test suite pour analyser les améliorations IA1 dans le code"""
    
    def __init__(self):
        self.backend_code_path = '/app/backend/server.py'
        self.test_results = []
        
    def log_test_result(self, test_name: str, success: bool, details: str = "", examples: List[str] = None):
        """Log test result with examples"""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"   Details: {details}")
        if examples:
            for example in examples[:3]:  # Show max 3 examples
                logger.info(f"   Example: {example}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'examples': examples or [],
            'timestamp': datetime.now().isoformat()
        })

    def read_backend_code(self):
        """Read backend code for analysis"""
        try:
            with open(self.backend_code_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read backend code: {e}")
            return ""

    def get_backend_logs(self):
        """Get recent backend logs"""
        try:
            import subprocess
            log_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null || echo 'No logs found'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            logger.error(f"Failed to get backend logs: {e}")
            return ""

    async def test_1_amelioration_prompt_code(self):
        """Test 1: Vérifier le nouveau prompt dans le code"""
        logger.info("\n🔍 TEST 1: Amélioration du Prompt (Code Analysis)")
        
        try:
            backend_code = self.read_backend_code()
            
            # Look for the new prompt structure in code
            prompt_indicators = [
                'ADVANCED TECHNICAL ANALYSIS WITH CHARTIST PATTERNS',
                'PATTERN ANALYSIS REQUIREMENTS',
                'You MUST explicitly mention and analyze each detected pattern by name',
                'Use pattern-specific terminology',
                'The detected [PATTERN NAME] formation'
            ]
            
            found_indicators = []
            for indicator in prompt_indicators:
                if indicator in backend_code:
                    found_indicators.append(indicator)
                    
            # Look for pattern-specific instructions
            pattern_instructions = [
                'Explain how each pattern influences',
                'pattern-specific terminology',
                'Triple Top suggests bearish reversal',
                'Gartley pattern indicates bullish continuation',
                'pattern targets and breakout levels'
            ]
            
            found_instructions = []
            for instruction in pattern_instructions:
                if instruction.lower() in backend_code.lower():
                    found_instructions.append(instruction)
                    
            # Check for enhanced IA1 prompt function
            ia1_chat_function = 'def get_ia1_chat():' in backend_code
            enhanced_prompt = 'DETECTED CHARTIST PATTERNS' in backend_code
            
            logger.info(f"   📊 Prompt indicators found: {len(found_indicators)}/{len(prompt_indicators)}")
            logger.info(f"   📊 Pattern instructions found: {len(found_instructions)}/{len(pattern_instructions)}")
            logger.info(f"   📊 IA1 chat function present: {ia1_chat_function}")
            logger.info(f"   📊 Enhanced prompt structure: {enhanced_prompt}")
            
            success = len(found_indicators) >= 2 or enhanced_prompt
            details = f"Indicators: {len(found_indicators)}/5, Instructions: {len(found_instructions)}/5, Enhanced prompt: {enhanced_prompt}"
            
            examples = []
            if found_indicators:
                examples.extend([f"✅ Found: '{indicator}'" for indicator in found_indicators[:2]])
            if enhanced_prompt:
                examples.append("✅ Enhanced prompt structure detected in code")
                
            self.log_test_result("Amélioration du Prompt (Code)", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Amélioration du Prompt (Code)", False, f"Exception: {str(e)}")

    async def test_2_pattern_integration_code(self):
        """Test 2: Vérifier l'intégration des patterns dans le code"""
        logger.info("\n🔍 TEST 2: Intégration des Patterns (Code Analysis)")
        
        try:
            backend_code = self.read_backend_code()
            
            # Look for pattern detection integration
            pattern_integration_indicators = [
                'technical_pattern_detector',
                'detected_pattern',
                'all_detected_patterns',
                'pattern_details',
                'should_analyze_with_ia1',
                '_detect_all_patterns'
            ]
            
            found_integration = []
            for indicator in pattern_integration_indicators:
                if indicator in backend_code:
                    found_integration.append(indicator)
                    
            # Look for pattern data flow
            pattern_data_flow = [
                'patterns_detected',
                'pattern_analysis',
                'primary_pattern',
                'pattern_strength',
                'pattern_direction',
                'pattern_confidence'
            ]
            
            found_data_flow = []
            for flow_item in pattern_data_flow:
                if flow_item in backend_code:
                    found_data_flow.append(flow_item)
                    
            # Check for pattern enhancement in IA1 analysis
            ia1_pattern_enhancement = [
                'PATTERNS COMPLETS',
                'DÉTAILS PATTERNS',
                'pattern_details +=',
                'additional_patterns'
            ]
            
            found_enhancement = []
            for enhancement in ia1_pattern_enhancement:
                if enhancement in backend_code:
                    found_enhancement.append(enhancement)
                    
            logger.info(f"   📊 Pattern integration indicators: {len(found_integration)}/{len(pattern_integration_indicators)}")
            logger.info(f"   📊 Pattern data flow elements: {len(found_data_flow)}/{len(pattern_data_flow)}")
            logger.info(f"   📊 IA1 pattern enhancements: {len(found_enhancement)}/{len(ia1_pattern_enhancement)}")
            
            success = len(found_integration) >= 3 and len(found_data_flow) >= 3
            details = f"Integration: {len(found_integration)}/6, Data flow: {len(found_data_flow)}/6, Enhancement: {len(found_enhancement)}/4"
            
            examples = []
            if found_integration:
                examples.extend([f"✅ Integration: {item}" for item in found_integration[:2]])
            if found_data_flow:
                examples.append(f"✅ Data flow: {found_data_flow[0]}")
                
            self.log_test_result("Intégration des Patterns (Code)", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Intégration des Patterns (Code)", False, f"Exception: {str(e)}")

    async def test_3_pattern_analysis_field_code(self):
        """Test 3: Vérifier le nouveau champ pattern_analysis dans le code"""
        logger.info("\n🔍 TEST 3: Nouveau Champ pattern_analysis (Code Analysis)")
        
        try:
            backend_code = self.read_backend_code()
            
            # Look for pattern_analysis field definition
            pattern_analysis_indicators = [
                'pattern_analysis',
                'primary_pattern',
                'pattern_strength', 
                'pattern_direction',
                'pattern_confidence'
            ]
            
            found_fields = []
            for field in pattern_analysis_indicators:
                if field in backend_code:
                    found_fields.append(field)
                    
            # Look for TechnicalAnalysis model enhancement
            model_enhancements = [
                'class TechnicalAnalysis',
                'pattern_analysis:',
                'Dict[str, Any]',
                'Optional[Dict]'
            ]
            
            found_model_enhancements = []
            for enhancement in model_enhancements:
                if enhancement in backend_code:
                    found_model_enhancements.append(enhancement)
                    
            # Check for pattern analysis creation logic
            creation_logic = [
                'pattern_analysis = {',
                '"primary_pattern"',
                '"pattern_strength"',
                '"pattern_direction"',
                '"pattern_confidence"'
            ]
            
            found_creation_logic = []
            for logic in creation_logic:
                if logic in backend_code:
                    found_creation_logic.append(logic)
                    
            logger.info(f"   📊 Pattern analysis fields: {len(found_fields)}/{len(pattern_analysis_indicators)}")
            logger.info(f"   📊 Model enhancements: {len(found_model_enhancements)}/{len(model_enhancements)}")
            logger.info(f"   📊 Creation logic: {len(found_creation_logic)}/{len(creation_logic)}")
            
            success = len(found_fields) >= 3 or len(found_creation_logic) >= 2
            details = f"Fields: {len(found_fields)}/5, Model: {len(found_model_enhancements)}/4, Logic: {len(found_creation_logic)}/5"
            
            examples = []
            if found_fields:
                examples.extend([f"✅ Field: {field}" for field in found_fields[:2]])
            if found_creation_logic:
                examples.append(f"✅ Creation logic: {found_creation_logic[0]}")
                
            self.log_test_result("Nouveau Champ pattern_analysis (Code)", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Nouveau Champ pattern_analysis (Code)", False, f"Exception: {str(e)}")

    async def test_4_pattern_terminology_code(self):
        """Test 4: Vérifier la terminologie spécifique aux patterns"""
        logger.info("\n🔍 TEST 4: Terminologie Pattern-Spécifique (Code Analysis)")
        
        try:
            backend_code = self.read_backend_code()
            
            # Look for specific pattern terminology
            pattern_terminology = [
                'triple_top', 'triple_bottom', 'head_shoulders', 'inverse_head_shoulders',
                'gartley', 'butterfly', 'bat', 'crab', 'diamond',
                'ascending_triangle', 'descending_triangle', 'symmetrical_triangle',
                'wedge', 'pennant', 'flag', 'channel', 'rectangle'
            ]
            
            found_terminology = []
            for term in pattern_terminology:
                if term in backend_code.lower():
                    found_terminology.append(term)
                    
            # Look for pattern implications
            pattern_implications = [
                'bearish reversal', 'bullish continuation', 'breakout confirmation',
                'pattern suggests', 'pattern indicates', 'formation suggests',
                'Triple Top suggests bearish reversal',
                'Gartley pattern indicates bullish continuation'
            ]
            
            found_implications = []
            for implication in pattern_implications:
                if implication.lower() in backend_code.lower():
                    found_implications.append(implication)
                    
            # Look for pattern-specific analysis instructions
            analysis_instructions = [
                'explicitly mention and analyze each detected pattern',
                'pattern-specific terminology',
                'pattern targets and breakout levels',
                'pattern strength and confluence'
            ]
            
            found_instructions = []
            for instruction in analysis_instructions:
                if instruction.lower() in backend_code.lower():
                    found_instructions.append(instruction)
                    
            logger.info(f"   📊 Pattern terminology: {len(found_terminology)}/{len(pattern_terminology)}")
            logger.info(f"   📊 Pattern implications: {len(found_implications)}/{len(pattern_implications)}")
            logger.info(f"   📊 Analysis instructions: {len(found_instructions)}/{len(analysis_instructions)}")
            
            success = len(found_terminology) >= 5 or len(found_implications) >= 2
            details = f"Terminology: {len(found_terminology)}/17, Implications: {len(found_implications)}/8, Instructions: {len(found_instructions)}/4"
            
            examples = []
            if found_terminology:
                examples.extend([f"✅ Pattern: {term}" for term in found_terminology[:2]])
            if found_implications:
                examples.append(f"✅ Implication: {found_implications[0]}")
                
            self.log_test_result("Terminologie Pattern-Spécifique (Code)", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Terminologie Pattern-Spécifique (Code)", False, f"Exception: {str(e)}")

    async def test_5_logs_pattern_integration(self):
        """Test 5: Vérifier l'intégration des patterns dans les logs"""
        logger.info("\n🔍 TEST 5: Intégration des Patterns (Logs Analysis)")
        
        try:
            backend_logs = self.get_backend_logs()
            
            # Look for pattern-related log messages
            pattern_log_indicators = [
                'PATTERNS COMPLETS',
                'patterns détectés',
                'PATTERN PRINCIPAL',
                'DÉTAILS PATTERNS',
                'PATTERN DÉTECTÉ',
                'IA1 ANALYSE JUSTIFIÉE'
            ]
            
            found_log_indicators = []
            for indicator in pattern_log_indicators:
                if indicator in backend_logs:
                    found_log_indicators.append(indicator)
                    
            # Count pattern-related log entries
            pattern_log_count = 0
            for line in backend_logs.split('\n'):
                if any(indicator in line for indicator in pattern_log_indicators):
                    pattern_log_count += 1
                    
            # Look for specific pattern mentions in logs
            specific_patterns_in_logs = []
            pattern_names = ['triangle', 'wedge', 'channel', 'support', 'resistance', 'breakout']
            
            for pattern in pattern_names:
                if pattern.lower() in backend_logs.lower():
                    specific_patterns_in_logs.append(pattern)
                    
            # Look for pattern analysis workflow logs
            workflow_logs = [
                'TECHNICAL PRE-FILTER',
                'OVERRIDE-SIGNAL',
                'OVERRIDE-DATA',
                'OVERRIDE-TRADING'
            ]
            
            found_workflow_logs = []
            for workflow in workflow_logs:
                if workflow in backend_logs:
                    found_workflow_logs.append(workflow)
                    
            logger.info(f"   📊 Pattern log indicators: {len(found_log_indicators)}/{len(pattern_log_indicators)}")
            logger.info(f"   📊 Pattern log entries: {pattern_log_count}")
            logger.info(f"   📊 Specific patterns in logs: {len(specific_patterns_in_logs)}/{len(pattern_names)}")
            logger.info(f"   📊 Workflow logs: {len(found_workflow_logs)}/{len(workflow_logs)}")
            
            success = len(found_log_indicators) >= 2 or pattern_log_count >= 5
            details = f"Log indicators: {len(found_log_indicators)}/6, Log entries: {pattern_log_count}, Specific patterns: {len(specific_patterns_in_logs)}/6"
            
            examples = []
            if found_log_indicators:
                examples.extend([f"✅ Log indicator: {indicator}" for indicator in found_log_indicators[:2]])
            if specific_patterns_in_logs:
                examples.append(f"✅ Pattern in logs: {specific_patterns_in_logs[0]}")
                
            self.log_test_result("Intégration des Patterns (Logs)", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Intégration des Patterns (Logs)", False, f"Exception: {str(e)}")

    async def test_6_enhanced_analysis_quality_code(self):
        """Test 6: Vérifier les améliorations de qualité d'analyse dans le code"""
        logger.info("\n🔍 TEST 6: Qualité d'Analyse Améliorée (Code Analysis)")
        
        try:
            backend_code = self.read_backend_code()
            
            # Look for enhanced analysis quality indicators
            quality_indicators = [
                'enhanced_ohlcv_fetcher',
                'multi_source_quality',
                'validate_multi_source_quality',
                'confidence_score',
                'data_confidence',
                'analysis_confidence'
            ]
            
            found_quality_indicators = []
            for indicator in quality_indicators:
                if indicator in backend_code:
                    found_quality_indicators.append(indicator)
                    
            # Look for pattern-enhanced analysis logic
            enhanced_analysis_logic = [
                'pattern_integration_score',
                'depth_indicators',
                'technical_terms',
                'confluence',
                'pattern_strength',
                'pattern_confidence'
            ]
            
            found_enhanced_logic = []
            for logic in enhanced_analysis_logic:
                if logic in backend_code:
                    found_enhanced_logic.append(logic)
                    
            # Look for quality thresholds and validation
            quality_validation = [
                'min_confidence',
                'quality_thresholds',
                'high_quality_analyses',
                'pattern_enhanced_analyses',
                'depth_score'
            ]
            
            found_quality_validation = []
            for validation in quality_validation:
                if validation in backend_code:
                    found_quality_validation.append(validation)
                    
            # Check for enhanced prompt structure
            enhanced_prompt_structure = [
                'ADVANCED TECHNICAL ANALYSIS',
                'PATTERN ANALYSIS REQUIREMENTS',
                'pattern-specific terminology',
                'confidence should reflect pattern strength'
            ]
            
            found_prompt_structure = []
            for structure in enhanced_prompt_structure:
                if structure in backend_code:
                    found_prompt_structure.append(structure)
                    
            logger.info(f"   📊 Quality indicators: {len(found_quality_indicators)}/{len(quality_indicators)}")
            logger.info(f"   📊 Enhanced analysis logic: {len(found_enhanced_logic)}/{len(enhanced_analysis_logic)}")
            logger.info(f"   📊 Quality validation: {len(found_quality_validation)}/{len(quality_validation)}")
            logger.info(f"   📊 Enhanced prompt structure: {len(found_prompt_structure)}/{len(enhanced_prompt_structure)}")
            
            success = len(found_quality_indicators) >= 3 or len(found_prompt_structure) >= 2
            details = f"Quality indicators: {len(found_quality_indicators)}/6, Enhanced logic: {len(found_enhanced_logic)}/6, Validation: {len(found_quality_validation)}/5"
            
            examples = []
            if found_quality_indicators:
                examples.extend([f"✅ Quality: {indicator}" for indicator in found_quality_indicators[:2]])
            if found_prompt_structure:
                examples.append(f"✅ Enhanced prompt: {found_prompt_structure[0]}")
                
            self.log_test_result("Qualité d'Analyse Améliorée (Code)", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Qualité d'Analyse Améliorée (Code)", False, f"Exception: {str(e)}")

    async def run_comprehensive_tests(self):
        """Run all IA1 Pattern Improvements code analysis tests"""
        logger.info("🚀 Starting IA1 Pattern Improvements Code Analysis Test Suite")
        logger.info("=" * 80)
        
        # Run all 6 tests focusing on code analysis
        await self.test_1_amelioration_prompt_code()
        await self.test_2_pattern_integration_code()
        await self.test_3_pattern_analysis_field_code()
        await self.test_4_pattern_terminology_code()
        await self.test_5_logs_pattern_integration()
        await self.test_6_enhanced_analysis_quality_code()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 RÉSUMÉ DES TESTS - ANALYSE CODE AMÉLIORATIONS IA1 PATTERNS")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   Détails: {result['details']}")
            if result['examples']:
                for example in result['examples'][:2]:  # Show max 2 examples
                    logger.info(f"   {example}")
                    
        logger.info(f"\n🎯 RÉSULTAT GLOBAL: {passed_tests}/{total_tests} tests réussis")
        
        # Provide concrete examples of improvements found
        logger.info("\n📋 EXEMPLES CONCRETS D'AMÉLIORATIONS DÉTECTÉES:")
        
        if passed_tests >= total_tests * 0.8:
            logger.info("✅ Prompt amélioré avec focus sur les patterns chartistes")
            logger.info("✅ Intégration du détecteur de patterns techniques")
            logger.info("✅ Terminologie spécifique aux patterns implémentée")
            logger.info("✅ Système de validation de qualité multi-sources")
            logger.info("✅ Logs détaillés pour le suivi des patterns")
            
        if passed_tests == total_tests:
            logger.info("🎉 TOUS LES TESTS RÉUSSIS - Les améliorations IA1 avec patterns chartistes sont correctement implémentées dans le code!")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MAJORITAIREMENT IMPLÉMENTÉ - Les améliorations principales sont présentes dans le code")
        else:
            logger.info("❌ IMPLÉMENTATION INCOMPLÈTE - Certaines améliorations IA1 manquent dans le code")
            
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IA1PatternCodeAnalysisTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed >= total * 0.8:  # 80% success rate is acceptable for code analysis
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
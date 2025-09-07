#!/usr/bin/env python3
"""
Test des Améliorations de l'Interprétation IA1 avec Patterns Chartistes
Focus sur les 6 objectifs spécifiés dans la demande de révision
"""

import asyncio
import json
import logging
import os
import sys
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add backend to path
sys.path.append('/app/backend')

import requests
from motor.motor_asyncio import AsyncIOMotorClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA1PatternImprovementsTestSuite:
    """Test suite pour les améliorations IA1 avec patterns chartistes"""
    
    def __init__(self):
        # Get backend URL from frontend env
        try:
            with open('/app/frontend/.env', 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        backend_url = line.split('=')[1].strip()
                        break
                else:
                    backend_url = "http://localhost:8001"
        except Exception:
            backend_url = "http://localhost:8001"
        
        self.api_url = f"{backend_url}/api"
        logger.info(f"Testing backend at: {self.api_url}")
        
        # MongoDB connection for direct data access
        self.mongo_client = None
        self.db = None
        
        # Test results
        self.test_results = []
        
        # Patterns chartistes attendus
        self.expected_patterns = [
            'triangle', 'wedge', 'channel', 'support', 'resistance', 'breakout',
            'triple_top', 'triple_bottom', 'head_shoulders', 'inverse_head_shoulders',
            'gartley', 'butterfly', 'bat', 'crab', 'diamond', 'pennant', 'flag'
        ]
        
    async def setup_database(self):
        """Setup database connection"""
        try:
            # Get MongoDB URL from backend env
            mongo_url = "mongodb://localhost:27017"  # Default
            try:
                with open('/app/backend/.env', 'r') as f:
                    for line in f:
                        if line.startswith('MONGO_URL='):
                            mongo_url = line.split('=')[1].strip().strip('"')
                            break
            except Exception:
                pass
            
            self.mongo_client = AsyncIOMotorClient(mongo_url)
            self.db = self.mongo_client['myapp']
            logger.info("✅ Database connection established")
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            
    async def cleanup_database(self):
        """Cleanup database connection"""
        if self.mongo_client:
            self.mongo_client.close()
            
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
        
    def get_analyses_from_api(self):
        """Helper method to get analyses from API"""
        try:
            response = requests.get(f"{self.api_url}/analyses", timeout=30)
            if response.status_code != 200:
                return None, f"API error: {response.status_code}"
                
            data = response.json()
            
            # Handle API response format
            if isinstance(data, dict) and 'analyses' in data:
                analyses = data['analyses']
            else:
                analyses = data
                
            return analyses, None
        except Exception as e:
            return None, f"Exception: {str(e)}"

    async def test_1_amelioration_prompt(self):
        """Test 1: Amélioration du Prompt - Vérifier le nouveau prompt "ADVANCED TECHNICAL ANALYSIS WITH CHARTIST PATTERNS" """
        logger.info("\n🔍 TEST 1: Amélioration du Prompt")
        
        try:
            # Check backend logs for the new prompt structure
            import subprocess
            
            # Look for the new prompt in backend logs
            log_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -A5 -B5 'ADVANCED TECHNICAL ANALYSIS WITH CHARTIST PATTERNS' || echo 'No prompt found'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            prompt_logs = result.stdout
            has_new_prompt = 'ADVANCED TECHNICAL ANALYSIS WITH CHARTIST PATTERNS' in prompt_logs
            
            # Look for PATTERN ANALYSIS REQUIREMENTS
            requirements_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -A3 'PATTERN ANALYSIS REQUIREMENTS' || echo 'No requirements found'"
            requirements_result = subprocess.run(requirements_cmd, shell=True, capture_output=True, text=True)
            
            requirements_logs = requirements_result.stdout
            has_requirements = 'PATTERN ANALYSIS REQUIREMENTS' in requirements_logs
            
            # Check for specific pattern instructions
            pattern_instructions = [
                'You MUST explicitly mention and analyze each detected pattern by name',
                'Explain how each pattern influences your technical assessment',
                'Use pattern-specific terminology',
                'Integrate pattern targets and breakout levels'
            ]
            
            instructions_found = 0
            for instruction in pattern_instructions:
                if instruction in prompt_logs or instruction in requirements_logs:
                    instructions_found += 1
                    
            logger.info(f"   📊 New prompt found: {has_new_prompt}")
            logger.info(f"   📊 Pattern requirements found: {has_requirements}")
            logger.info(f"   📊 Pattern instructions found: {instructions_found}/{len(pattern_instructions)}")
            
            # Check if IA1 is receiving the enhanced prompt
            analyses, error = self.get_analyses_from_api()
            enhanced_analyses = 0
            
            if not error and analyses:
                for analysis in analyses:
                    reasoning = analysis.get('ia1_reasoning', '')
                    # Look for evidence of enhanced prompt usage
                    if any(keyword in reasoning.lower() for keyword in ['pattern', 'chartist', 'formation', 'breakout']):
                        enhanced_analyses += 1
                        
                logger.info(f"   📊 Analyses with pattern terminology: {enhanced_analyses}/{len(analyses)}")
            
            success = has_new_prompt and has_requirements and instructions_found >= 2
            details = f"New prompt: {has_new_prompt}, Requirements: {has_requirements}, Instructions: {instructions_found}/4, Enhanced analyses: {enhanced_analyses}"
            
            examples = []
            if has_new_prompt:
                examples.append("✅ 'ADVANCED TECHNICAL ANALYSIS WITH CHARTIST PATTERNS' prompt detected")
            if has_requirements:
                examples.append("✅ 'PATTERN ANALYSIS REQUIREMENTS' section found")
                
            self.log_test_result("Amélioration du Prompt", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Amélioration du Prompt", False, f"Exception: {str(e)}")

    async def test_2_mention_explicite_patterns(self):
        """Test 2: Mention Explicite des Patterns - Chercher des phrases commençant par "The detected [PATTERN NAME] formation..." """
        logger.info("\n🔍 TEST 2: Mention Explicite des Patterns")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Mention Explicite des Patterns", False, error)
                return
                
            if not analyses:
                self.log_test_result("Mention Explicite des Patterns", False, "No analyses found")
                return
                
            # Look for explicit pattern mentions
            explicit_pattern_mentions = []
            pattern_terminology_count = 0
            pattern_explanations = []
            
            # Patterns to look for
            pattern_phrases = [
                r"the detected \w+ formation",
                r"the \w+ pattern suggests",
                r"the \w+ pattern indicates",
                r"\w+ formation suggests",
                r"triple top suggests",
                r"gartley pattern indicates",
                r"head and shoulders",
                r"inverse head and shoulders"
            ]
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                reasoning = analysis.get('ia1_reasoning', '')
                patterns_detected = analysis.get('patterns_detected', [])
                
                # Check for explicit pattern mentions
                for pattern_phrase in pattern_phrases:
                    matches = re.findall(pattern_phrase, reasoning.lower())
                    if matches:
                        explicit_pattern_mentions.extend(matches)
                        pattern_terminology_count += len(matches)
                        
                # Check if patterns are mentioned by name
                pattern_names_mentioned = []
                for pattern in patterns_detected:
                    if pattern.lower() in reasoning.lower():
                        pattern_names_mentioned.append(pattern)
                        
                if pattern_names_mentioned:
                    pattern_explanations.append({
                        'symbol': symbol,
                        'patterns_mentioned': pattern_names_mentioned,
                        'reasoning_snippet': reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                    })
                    
                    logger.info(f"   🎯 {symbol}: Patterns mentioned by name - {pattern_names_mentioned}")
                    
            # Check for specific pattern implications
            implication_keywords = [
                'bearish reversal', 'bullish continuation', 'breakout confirmation',
                'support level', 'resistance level', 'price target', 'stop loss'
            ]
            
            implications_found = 0
            for analysis in analyses:
                reasoning = analysis.get('ia1_reasoning', '')
                for keyword in implication_keywords:
                    if keyword in reasoning.lower():
                        implications_found += 1
                        break
                        
            logger.info(f"   📊 Explicit pattern mentions: {len(explicit_pattern_mentions)}")
            logger.info(f"   📊 Pattern terminology usage: {pattern_terminology_count}")
            logger.info(f"   📊 Analyses with pattern explanations: {len(pattern_explanations)}")
            logger.info(f"   📊 Analyses with pattern implications: {implications_found}")
            
            success = len(explicit_pattern_mentions) > 0 or pattern_terminology_count > 0 or len(pattern_explanations) > 0
            details = f"Explicit mentions: {len(explicit_pattern_mentions)}, Terminology usage: {pattern_terminology_count}, Pattern explanations: {len(pattern_explanations)}, Implications: {implications_found}"
            
            examples = []
            if explicit_pattern_mentions:
                examples.extend([f"✅ Pattern phrase: '{mention}'" for mention in explicit_pattern_mentions[:2]])
            if pattern_explanations:
                best_explanation = pattern_explanations[0]
                examples.append(f"✅ {best_explanation['symbol']}: {best_explanation['patterns_mentioned']} - {best_explanation['reasoning_snippet'][:100]}...")
                
            self.log_test_result("Mention Explicite des Patterns", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Mention Explicite des Patterns", False, f"Exception: {str(e)}")

    async def test_3_nouveau_champ_pattern_analysis(self):
        """Test 3: Nouveau Champ pattern_analysis - Vérifier la présence du champ pattern_analysis avec sous-champs """
        logger.info("\n🔍 TEST 3: Nouveau Champ pattern_analysis")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Nouveau Champ pattern_analysis", False, error)
                return
                
            if not analyses:
                self.log_test_result("Nouveau Champ pattern_analysis", False, "No analyses found")
                return
                
            # Check for pattern_analysis field
            analyses_with_pattern_analysis = 0
            pattern_analysis_examples = []
            
            # Expected sub-fields
            expected_subfields = ['primary_pattern', 'pattern_strength', 'pattern_direction', 'pattern_confidence']
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                
                # Check if pattern_analysis field exists
                pattern_analysis = analysis.get('pattern_analysis')
                
                if pattern_analysis:
                    analyses_with_pattern_analysis += 1
                    
                    # Check sub-fields
                    subfields_present = []
                    for subfield in expected_subfields:
                        if subfield in pattern_analysis:
                            subfields_present.append(subfield)
                            
                    pattern_analysis_examples.append({
                        'symbol': symbol,
                        'subfields_present': subfields_present,
                        'pattern_analysis': pattern_analysis
                    })
                    
                    logger.info(f"   🎯 {symbol}: pattern_analysis field present with {len(subfields_present)}/4 subfields")
                    logger.info(f"      Subfields: {subfields_present}")
                    
                    # Show pattern analysis content
                    if 'primary_pattern' in pattern_analysis:
                        logger.info(f"      Primary pattern: {pattern_analysis.get('primary_pattern', 'N/A')}")
                    if 'pattern_strength' in pattern_analysis:
                        logger.info(f"      Pattern strength: {pattern_analysis.get('pattern_strength', 'N/A')}")
                        
            # Check database for pattern_analysis storage
            pattern_analysis_in_db = False
            if self.db is not None:
                try:
                    # Check if technical_analyses collection has pattern_analysis data
                    analyses_cursor = self.db.technical_analyses.find({'pattern_analysis': {'$exists': True}}).limit(5)
                    db_analyses = await analyses_cursor.to_list(length=5)
                    
                    pattern_analysis_in_db = len(db_analyses) > 0
                    logger.info(f"   📊 Database pattern_analysis entries: {len(db_analyses)}")
                    
                    if db_analyses:
                        sample_analysis = db_analyses[0]
                        sample_pattern_analysis = sample_analysis.get('pattern_analysis', {})
                        logger.info(f"   📝 Sample DB pattern_analysis: {sample_pattern_analysis}")
                        
                except Exception as e:
                    logger.debug(f"Database check failed: {e}")
                    
            # Check if the field enriches the analysis
            enrichment_indicators = 0
            for example in pattern_analysis_examples:
                pattern_analysis = example['pattern_analysis']
                if isinstance(pattern_analysis, dict) and len(pattern_analysis) > 0:
                    enrichment_indicators += 1
                    
            logger.info(f"   📊 Analyses with pattern_analysis: {analyses_with_pattern_analysis}/{len(analyses)}")
            logger.info(f"   📊 Pattern analysis examples: {len(pattern_analysis_examples)}")
            logger.info(f"   📊 Database storage: {pattern_analysis_in_db}")
            logger.info(f"   📊 Enrichment indicators: {enrichment_indicators}")
            
            success = analyses_with_pattern_analysis > 0 or pattern_analysis_in_db
            details = f"Analyses with pattern_analysis: {analyses_with_pattern_analysis}/{len(analyses)}, DB storage: {pattern_analysis_in_db}, Examples: {len(pattern_analysis_examples)}"
            
            examples = []
            if pattern_analysis_examples:
                best_example = pattern_analysis_examples[0]
                examples.append(f"✅ {best_example['symbol']}: {len(best_example['subfields_present'])}/4 subfields present")
                examples.append(f"   Subfields: {best_example['subfields_present']}")
                
            self.log_test_result("Nouveau Champ pattern_analysis", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Nouveau Champ pattern_analysis", False, f"Exception: {str(e)}")

    async def test_4_qualite_analyse_amelioree(self):
        """Test 4: Qualité d'Analyse Améliorée - Comparer la longueur et profondeur des analyses """
        logger.info("\n🔍 TEST 4: Qualité d'Analyse Améliorée")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Qualité d'Analyse Améliorée", False, error)
                return
                
            if not analyses:
                self.log_test_result("Qualité d'Analyse Améliorée", False, "No analyses found")
                return
                
            # Analyze quality metrics
            reasoning_lengths = []
            confidence_scores = []
            pattern_integration_scores = []
            depth_indicators = []
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                reasoning = analysis.get('ia1_reasoning', '')
                confidence = analysis.get('analysis_confidence', 0)
                patterns_detected = analysis.get('patterns_detected', [])
                
                # Length analysis
                reasoning_length = len(reasoning)
                reasoning_lengths.append(reasoning_length)
                
                # Confidence analysis
                confidence_scores.append(confidence)
                
                # Pattern integration score (how well patterns are integrated)
                pattern_mentions = sum(1 for pattern in patterns_detected if pattern.lower() in reasoning.lower())
                pattern_integration_score = pattern_mentions / max(len(patterns_detected), 1) if patterns_detected else 0
                pattern_integration_scores.append(pattern_integration_score)
                
                # Depth indicators (technical terms, analysis depth)
                depth_keywords = [
                    'support', 'resistance', 'breakout', 'reversal', 'continuation',
                    'fibonacci', 'bollinger', 'rsi', 'macd', 'volume', 'momentum',
                    'trend', 'consolidation', 'volatility', 'confluence'
                ]
                
                depth_score = sum(1 for keyword in depth_keywords if keyword in reasoning.lower())
                depth_indicators.append(depth_score)
                
                logger.info(f"   📊 {symbol}: Length={reasoning_length}, Confidence={confidence:.2f}, Pattern integration={pattern_integration_score:.2f}, Depth={depth_score}")
                
            # Calculate quality metrics
            avg_length = sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            avg_pattern_integration = sum(pattern_integration_scores) / len(pattern_integration_scores) if pattern_integration_scores else 0
            avg_depth = sum(depth_indicators) / len(depth_indicators) if depth_indicators else 0
            
            # Quality thresholds (based on expected improvements)
            quality_thresholds = {
                'min_length': 500,  # Minimum 500 characters for detailed analysis
                'min_confidence': 0.70,  # Minimum 70% confidence
                'min_pattern_integration': 0.5,  # At least 50% of patterns mentioned
                'min_depth': 5  # At least 5 technical terms
            }
            
            # Check quality improvements
            high_quality_analyses = 0
            for i, analysis in enumerate(analyses):
                if (reasoning_lengths[i] >= quality_thresholds['min_length'] and
                    confidence_scores[i] >= quality_thresholds['min_confidence'] and
                    depth_indicators[i] >= quality_thresholds['min_depth']):
                    high_quality_analyses += 1
                    
            # Check for pattern-enhanced analyses
            pattern_enhanced_analyses = sum(1 for score in pattern_integration_scores if score > 0)
            
            logger.info(f"   📊 Average reasoning length: {avg_length:.0f} characters")
            logger.info(f"   📊 Average confidence: {avg_confidence:.2f}")
            logger.info(f"   📊 Average pattern integration: {avg_pattern_integration:.2f}")
            logger.info(f"   📊 Average depth score: {avg_depth:.1f}")
            logger.info(f"   📊 High quality analyses: {high_quality_analyses}/{len(analyses)}")
            logger.info(f"   📊 Pattern-enhanced analyses: {pattern_enhanced_analyses}/{len(analyses)}")
            
            success = (avg_length >= quality_thresholds['min_length'] and 
                      avg_confidence >= quality_thresholds['min_confidence'] and
                      high_quality_analyses >= len(analyses) * 0.6)  # 60% should be high quality
            
            details = f"Avg length: {avg_length:.0f}, Avg confidence: {avg_confidence:.2f}, High quality: {high_quality_analyses}/{len(analyses)}, Pattern enhanced: {pattern_enhanced_analyses}/{len(analyses)}"
            
            examples = []
            if high_quality_analyses > 0:
                examples.append(f"✅ {high_quality_analyses} analyses meet high quality thresholds")
            if pattern_enhanced_analyses > 0:
                examples.append(f"✅ {pattern_enhanced_analyses} analyses show pattern integration")
            examples.append(f"✅ Average analysis length: {avg_length:.0f} characters (target: {quality_thresholds['min_length']}+)")
                
            self.log_test_result("Qualité d'Analyse Améliorée", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Qualité d'Analyse Améliorée", False, f"Exception: {str(e)}")

    async def test_5_integration_pattern_specifique(self):
        """Test 5: Intégration Pattern-Spécifique - Chercher des mentions spécifiques aux patterns """
        logger.info("\n🔍 TEST 5: Intégration Pattern-Spécifique")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Intégration Pattern-Spécifique", False, error)
                return
                
            if not analyses:
                self.log_test_result("Intégration Pattern-Spécifique", False, "No analyses found")
                return
                
            # Look for specific pattern mentions and their implications
            specific_pattern_mentions = []
            pattern_implications = []
            target_mentions = []
            
            # Specific patterns to look for
            specific_patterns = {
                'triple_top': ['triple top suggests bearish reversal', 'triple top indicates', 'triple top formation'],
                'gartley': ['gartley pattern indicates bullish continuation', 'gartley pattern suggests', 'gartley formation'],
                'head_shoulders': ['head and shoulders', 'head shoulders pattern', 'inverse head and shoulders'],
                'triangle': ['triangle breakout', 'triangle pattern', 'symmetrical triangle'],
                'wedge': ['wedge pattern', 'falling wedge', 'rising wedge'],
                'support_resistance': ['support level', 'resistance level', 'key support', 'key resistance']
            }
            
            # Pattern-based targets and levels
            target_keywords = [
                'price target', 'target level', 'breakout target', 'pattern target',
                'measured move', 'projection', 'fibonacci target', 'pattern completion'
            ]
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                reasoning = analysis.get('ia1_reasoning', '').lower()
                patterns_detected = analysis.get('patterns_detected', [])
                
                # Check for specific pattern mentions
                found_specific_patterns = []
                for pattern_type, pattern_phrases in specific_patterns.items():
                    for phrase in pattern_phrases:
                        if phrase in reasoning:
                            found_specific_patterns.append(f"{pattern_type}: {phrase}")
                            
                if found_specific_patterns:
                    specific_pattern_mentions.append({
                        'symbol': symbol,
                        'patterns': found_specific_patterns
                    })
                    
                # Check for pattern implications
                implication_phrases = [
                    'suggests bearish reversal', 'indicates bullish continuation',
                    'bearish reversal', 'bullish continuation', 'reversal signal',
                    'continuation pattern', 'breakout confirmation'
                ]
                
                found_implications = []
                for implication in implication_phrases:
                    if implication in reasoning:
                        found_implications.append(implication)
                        
                if found_implications:
                    pattern_implications.append({
                        'symbol': symbol,
                        'implications': found_implications
                    })
                    
                # Check for target mentions
                found_targets = []
                for target_keyword in target_keywords:
                    if target_keyword in reasoning:
                        found_targets.append(target_keyword)
                        
                if found_targets:
                    target_mentions.append({
                        'symbol': symbol,
                        'targets': found_targets
                    })
                    
                # Log findings for this symbol
                if found_specific_patterns or found_implications or found_targets:
                    logger.info(f"   🎯 {symbol}:")
                    if found_specific_patterns:
                        logger.info(f"      Specific patterns: {found_specific_patterns}")
                    if found_implications:
                        logger.info(f"      Implications: {found_implications}")
                    if found_targets:
                        logger.info(f"      Targets: {found_targets}")
                        
            # Check backend logs for pattern-specific analysis
            import subprocess
            log_cmd = "tail -n 500 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'triple top\\|gartley\\|head.*shoulders\\|bearish reversal\\|bullish continuation' || echo 'No specific patterns'"
            result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
            
            pattern_logs = result.stdout
            has_specific_pattern_logs = pattern_logs != 'No specific patterns' and len(pattern_logs.strip()) > 0
            
            logger.info(f"   📊 Specific pattern mentions: {len(specific_pattern_mentions)}")
            logger.info(f"   📊 Pattern implications: {len(pattern_implications)}")
            logger.info(f"   📊 Target mentions: {len(target_mentions)}")
            logger.info(f"   📊 Pattern-specific logs: {has_specific_pattern_logs}")
            
            success = (len(specific_pattern_mentions) > 0 or 
                      len(pattern_implications) > 0 or 
                      len(target_mentions) > 0)
            
            details = f"Specific patterns: {len(specific_pattern_mentions)}, Implications: {len(pattern_implications)}, Targets: {len(target_mentions)}, Logs: {has_specific_pattern_logs}"
            
            examples = []
            if specific_pattern_mentions:
                example = specific_pattern_mentions[0]
                examples.append(f"✅ {example['symbol']}: {example['patterns'][0] if example['patterns'] else 'N/A'}")
            if pattern_implications:
                example = pattern_implications[0]
                examples.append(f"✅ {example['symbol']}: {example['implications'][0] if example['implications'] else 'N/A'}")
            if target_mentions:
                example = target_mentions[0]
                examples.append(f"✅ {example['symbol']}: {example['targets'][0] if example['targets'] else 'N/A'}")
                
            self.log_test_result("Intégration Pattern-Spécifique", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Intégration Pattern-Spécifique", False, f"Exception: {str(e)}")

    async def test_6_impact_sur_decisions(self):
        """Test 6: Impact sur les Décisions - Vérifier si les patterns influencent les recommandations """
        logger.info("\n🔍 TEST 6: Impact sur les Décisions")
        
        try:
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Impact sur les Décisions", False, error)
                return
                
            if not analyses:
                self.log_test_result("Impact sur les Décisions", False, "No analyses found")
                return
                
            # Analyze correlation between patterns and decisions
            pattern_decision_correlations = []
            confidence_adjustments = []
            decision_justifications = []
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                ia1_signal = analysis.get('ia1_signal', 'hold')
                confidence = analysis.get('analysis_confidence', 0)
                reasoning = analysis.get('ia1_reasoning', '')
                
                # Check if patterns influence the decision
                pattern_influence_indicators = []
                
                # Look for pattern-based decision justification
                decision_keywords = [
                    'pattern suggests', 'pattern indicates', 'formation suggests',
                    'breakout confirms', 'pattern completion', 'pattern target',
                    'based on the pattern', 'pattern analysis shows'
                ]
                
                pattern_justifications = []
                for keyword in decision_keywords:
                    if keyword in reasoning.lower():
                        pattern_justifications.append(keyword)
                        
                if pattern_justifications:
                    decision_justifications.append({
                        'symbol': symbol,
                        'signal': ia1_signal,
                        'justifications': pattern_justifications,
                        'patterns': patterns_detected
                    })
                    
                # Check if confidence reflects pattern strength
                if patterns_detected:
                    # Assume patterns should increase confidence
                    pattern_count = len(patterns_detected)
                    expected_confidence_boost = pattern_count * 0.05  # 5% per pattern
                    
                    confidence_adjustments.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'pattern_count': pattern_count,
                        'expected_boost': expected_confidence_boost
                    })
                    
                # Check pattern-decision coherence
                if patterns_detected and ia1_signal != 'hold':
                    # Check if decision aligns with pattern implications
                    bullish_patterns = ['ascending_triangle', 'bullish_flag', 'cup_handle', 'inverse_head_shoulders']
                    bearish_patterns = ['descending_triangle', 'bearish_flag', 'head_shoulders', 'triple_top']
                    
                    pattern_direction = None
                    for pattern in patterns_detected:
                        pattern_lower = pattern.lower()
                        if any(bp in pattern_lower for bp in bullish_patterns):
                            pattern_direction = 'bullish'
                            break
                        elif any(bp in pattern_lower for bp in bearish_patterns):
                            pattern_direction = 'bearish'
                            break
                            
                    if pattern_direction:
                        decision_coherent = (
                            (pattern_direction == 'bullish' and ia1_signal == 'long') or
                            (pattern_direction == 'bearish' and ia1_signal == 'short')
                        )
                        
                        pattern_decision_correlations.append({
                            'symbol': symbol,
                            'pattern_direction': pattern_direction,
                            'ia1_signal': ia1_signal,
                            'coherent': decision_coherent
                        })
                        
                logger.info(f"   📊 {symbol}: Signal={ia1_signal}, Patterns={len(patterns_detected)}, Confidence={confidence:.2f}")
                if pattern_justifications:
                    logger.info(f"      Pattern justifications: {pattern_justifications}")
                    
            # Get IA2 decisions to check pattern influence propagation
            try:
                decisions_response = requests.get(f"{self.api_url}/decisions", timeout=30)
                if decisions_response.status_code == 200:
                    decisions_data = decisions_response.json()
                    decisions = decisions_data if isinstance(decisions_data, list) else decisions_data.get('decisions', [])
                    
                    pattern_influenced_decisions = 0
                    for decision in decisions:
                        reasoning = decision.get('ia2_reasoning', '')
                        if any(keyword in reasoning.lower() for keyword in ['pattern', 'formation', 'breakout']):
                            pattern_influenced_decisions += 1
                            
                    logger.info(f"   📊 IA2 decisions influenced by patterns: {pattern_influenced_decisions}/{len(decisions)}")
                else:
                    pattern_influenced_decisions = 0
                    decisions = []
            except Exception:
                pattern_influenced_decisions = 0
                decisions = []
                
            # Calculate coherence metrics
            coherent_decisions = sum(1 for corr in pattern_decision_correlations if corr['coherent'])
            total_correlations = len(pattern_decision_correlations)
            
            high_confidence_with_patterns = sum(1 for adj in confidence_adjustments if adj['confidence'] >= 0.8 and adj['pattern_count'] > 0)
            
            logger.info(f"   📊 Pattern-decision correlations: {total_correlations}")
            logger.info(f"   📊 Coherent decisions: {coherent_decisions}/{total_correlations}")
            logger.info(f"   📊 Decision justifications: {len(decision_justifications)}")
            logger.info(f"   📊 High confidence with patterns: {high_confidence_with_patterns}")
            
            success = (len(decision_justifications) > 0 or 
                      coherent_decisions >= total_correlations * 0.7 or  # 70% coherence
                      pattern_influenced_decisions > 0)
            
            details = f"Decision justifications: {len(decision_justifications)}, Coherent decisions: {coherent_decisions}/{total_correlations}, IA2 pattern influence: {pattern_influenced_decisions}"
            
            examples = []
            if decision_justifications:
                example = decision_justifications[0]
                examples.append(f"✅ {example['symbol']} ({example['signal']}): {example['justifications'][0] if example['justifications'] else 'N/A'}")
            if pattern_decision_correlations:
                coherent_example = next((corr for corr in pattern_decision_correlations if corr['coherent']), None)
                if coherent_example:
                    examples.append(f"✅ Coherent: {coherent_example['symbol']} - {coherent_example['pattern_direction']} pattern → {coherent_example['ia1_signal']} signal")
                    
            self.log_test_result("Impact sur les Décisions", success, details, examples)
            
        except Exception as e:
            self.log_test_result("Impact sur les Décisions", False, f"Exception: {str(e)}")

    async def run_comprehensive_tests(self):
        """Run all IA1 Pattern Improvements tests"""
        logger.info("🚀 Starting IA1 Pattern Improvements Test Suite")
        logger.info("=" * 80)
        
        await self.setup_database()
        
        # Run all 6 tests as specified in the review request
        await self.test_1_amelioration_prompt()
        await self.test_2_mention_explicite_patterns()
        await self.test_3_nouveau_champ_pattern_analysis()
        await self.test_4_qualite_analyse_amelioree()
        await self.test_5_integration_pattern_specifique()
        await self.test_6_impact_sur_decisions()
        
        await self.cleanup_database()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 RÉSUMÉ DES TESTS - AMÉLIORATIONS IA1 AVEC PATTERNS CHARTISTES")
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
        
        if passed_tests == total_tests:
            logger.info("🎉 TOUS LES TESTS RÉUSSIS - Les améliorations IA1 avec patterns chartistes fonctionnent correctement!")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MAJORITAIREMENT FONCTIONNEL - Quelques améliorations mineures détectées")
        else:
            logger.info("❌ PROBLÈMES CRITIQUES - Les améliorations IA1 nécessitent une attention")
            
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = IA1PatternImprovementsTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
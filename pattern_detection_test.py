#!/usr/bin/env python3
"""
Test Suite pour le Système de Détection de Patterns Chartistes
Focus: Tester les corrections du système de détection de patterns chartistes selon la demande française
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
import requests
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternDetectionTestSuite:
    """Suite de tests pour le système de détection de patterns chartistes"""
    
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
        logger.info(f"Testing Pattern Detection System at: {self.api_url}")
        
        # Test results
        self.test_results = []
        
        # Pattern detection specific data
        self.detected_patterns = []
        self.pattern_types_found = set()
        self.symbols_with_patterns = {}
        
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if details:
            logger.info(f"   Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
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
    
    def get_decisions_from_api(self):
        """Helper method to get decisions from API"""
        try:
            response = requests.get(f"{self.api_url}/decisions", timeout=30)
            if response.status_code != 200:
                return None, f"API error: {response.status_code}"
                
            data = response.json()
            if isinstance(data, dict) and 'decisions' in data:
                decisions = data['decisions']
            else:
                decisions = data
            return decisions, None
        except Exception as e:
            return None, f"Exception: {str(e)}"
    
    async def test_augmentation_nombre_patterns(self):
        """Test 1: Augmentation du Nombre de Patterns Détectés - Vérifier plus de 2 patterns par symbole et limite 5→10"""
        logger.info("\n🔍 TEST 1: Augmentation du Nombre de Patterns Détectés")
        
        try:
            # Get analyses to check pattern detection
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Augmentation Nombre Patterns", False, error)
                return
            
            if not analyses:
                self.log_test_result("Augmentation Nombre Patterns", False, "No analyses found")
                return
            
            pattern_stats = {
                'symbols_analyzed': 0,
                'symbols_with_patterns': 0,
                'symbols_with_2plus_patterns': 0,
                'symbols_with_5plus_patterns': 0,
                'symbols_with_10plus_patterns': 0,
                'total_patterns_detected': 0,
                'max_patterns_per_symbol': 0,
                'pattern_distribution': {}
            }
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                
                pattern_stats['symbols_analyzed'] += 1
                
                if patterns_detected:
                    pattern_count = len(patterns_detected)
                    pattern_stats['symbols_with_patterns'] += 1
                    pattern_stats['total_patterns_detected'] += pattern_count
                    pattern_stats['max_patterns_per_symbol'] = max(pattern_stats['max_patterns_per_symbol'], pattern_count)
                    
                    # Store pattern distribution
                    pattern_stats['pattern_distribution'][symbol] = {
                        'count': pattern_count,
                        'patterns': patterns_detected
                    }
                    
                    # Count symbols with different pattern thresholds
                    if pattern_count >= 2:
                        pattern_stats['symbols_with_2plus_patterns'] += 1
                    if pattern_count >= 5:
                        pattern_stats['symbols_with_5plus_patterns'] += 1
                    if pattern_count >= 10:
                        pattern_stats['symbols_with_10plus_patterns'] += 1
                    
                    logger.info(f"   📊 {symbol}: {pattern_count} patterns détectés - {patterns_detected}")
            
            # Check backend logs for pattern detection limits
            pattern_limit_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'pattern.*limit\\|max.*pattern\\|pattern.*10\\|pattern.*5' || echo 'No pattern limit logs'"
            result = subprocess.run(pattern_limit_cmd, shell=True, capture_output=True, text=True)
            
            limit_logs = []
            for line in result.stdout.split('\n'):
                if line.strip() and 'No pattern limit' not in line:
                    limit_logs.append(line.strip())
            
            # Calculate averages
            avg_patterns_per_symbol = 0
            if pattern_stats['symbols_with_patterns'] > 0:
                avg_patterns_per_symbol = pattern_stats['total_patterns_detected'] / pattern_stats['symbols_with_patterns']
            
            # Show top symbols with most patterns
            top_symbols = sorted(pattern_stats['pattern_distribution'].items(), 
                               key=lambda x: x[1]['count'], reverse=True)[:5]
            
            logger.info(f"   📊 Pattern Detection Statistics:")
            logger.info(f"      Symbols analyzed: {pattern_stats['symbols_analyzed']}")
            logger.info(f"      Symbols with patterns: {pattern_stats['symbols_with_patterns']}")
            logger.info(f"      Symbols with 2+ patterns: {pattern_stats['symbols_with_2plus_patterns']}")
            logger.info(f"      Symbols with 5+ patterns: {pattern_stats['symbols_with_5plus_patterns']}")
            logger.info(f"      Symbols with 10+ patterns: {pattern_stats['symbols_with_10plus_patterns']}")
            logger.info(f"      Total patterns detected: {pattern_stats['total_patterns_detected']}")
            logger.info(f"      Max patterns per symbol: {pattern_stats['max_patterns_per_symbol']}")
            logger.info(f"      Average patterns per symbol: {avg_patterns_per_symbol:.2f}")
            logger.info(f"      Pattern limit logs found: {len(limit_logs)}")
            
            if top_symbols:
                logger.info(f"   🏆 Top Symbols with Most Patterns:")
                for symbol, data in top_symbols:
                    logger.info(f"      {symbol}: {data['count']} patterns - {data['patterns'][:3]}...")
            
            # Success criteria: More than 2 patterns per symbol on average AND evidence of increased limits
            success = (avg_patterns_per_symbol > 2.0 and 
                      pattern_stats['symbols_with_2plus_patterns'] > 0 and
                      pattern_stats['max_patterns_per_symbol'] >= 5)
            
            details = f"Avg patterns/symbol: {avg_patterns_per_symbol:.2f}, Max: {pattern_stats['max_patterns_per_symbol']}, 2+ patterns: {pattern_stats['symbols_with_2plus_patterns']}/{pattern_stats['symbols_analyzed']}, 5+ patterns: {pattern_stats['symbols_with_5plus_patterns']}"
            
            self.log_test_result("Augmentation Nombre Patterns", success, details)
            
        except Exception as e:
            self.log_test_result("Augmentation Nombre Patterns", False, f"Exception: {str(e)}")
    
    async def test_nouvelles_methodes_integrees(self):
        """Test 2: Nouvelles Méthodes Intégrées - Vérifier les nouvelles méthodes de détection"""
        logger.info("\n🔍 TEST 2: Nouvelles Méthodes Intégrées")
        
        try:
            # Check backend logs for new pattern detection methods
            new_methods = {
                '_detect_wedge_patterns_simple': 0,
                '_detect_flag_patterns_simple': 0,
                '_detect_triple_patterns': 0,
                '_detect_pennant_patterns': 0,
                '_detect_rectangle_consolidation': 0,
                '_detect_rounding_patterns': 0,
                'Rising Wedge': 0,
                'Falling Wedge': 0,
                'Flag Bullish': 0,
                'Flag Bearish': 0,
                'Triple Top': 0,
                'Triple Bottom': 0,
                'Pennant': 0,
                'Rectangle': 0,
                'Rounding': 0
            }
            
            # Search for method calls in logs
            for method in new_methods.keys():
                if method.startswith('_detect_'):
                    log_cmd = f"tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i '{method}' || echo 'No {method} logs'"
                else:
                    log_cmd = f"tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i '{method}' || echo 'No {method} logs'"
                
                result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True)
                
                matches = [line.strip() for line in result.stdout.split('\n') 
                          if line.strip() and f'No {method}' not in line and method.lower() in line.lower()]
                new_methods[method] = len(matches)
                
                if matches:
                    logger.info(f"   🎯 {method}: {len(matches)} détections")
                    logger.info(f"      Sample: {matches[-1][:120]}...")
            
            # Check analyses for new pattern types
            analyses, error = self.get_analyses_from_api()
            pattern_types_in_analyses = {}
            
            if not error and analyses:
                for analysis in analyses:
                    patterns_detected = analysis.get('patterns_detected', [])
                    symbol = analysis.get('symbol', 'UNKNOWN')
                    
                    for pattern in patterns_detected:
                        pattern_lower = pattern.lower()
                        
                        # Check for new pattern types
                        for new_pattern_type in ['wedge', 'flag', 'triple', 'pennant', 'rectangle', 'rounding']:
                            if new_pattern_type in pattern_lower:
                                if new_pattern_type not in pattern_types_in_analyses:
                                    pattern_types_in_analyses[new_pattern_type] = []
                                pattern_types_in_analyses[new_pattern_type].append({
                                    'symbol': symbol,
                                    'pattern': pattern
                                })
            
            # Check for specific new pattern implementations in code
            code_implementation_cmd = "find /app/backend -name '*.py' -exec grep -l '_detect_wedge_patterns_simple\\|_detect_flag_patterns_simple\\|_detect_triple_patterns\\|_detect_pennant_patterns\\|_detect_rectangle_consolidation\\|_detect_rounding_patterns' {} \\; 2>/dev/null || echo 'No implementation files'"
            code_result = subprocess.run(code_implementation_cmd, shell=True, capture_output=True, text=True)
            
            implementation_files = []
            for line in code_result.stdout.split('\n'):
                if line.strip() and 'No implementation' not in line and line.endswith('.py'):
                    implementation_files.append(line.strip())
            
            logger.info(f"   📊 New Methods Detection Summary:")
            logger.info(f"      Implementation files found: {len(implementation_files)}")
            
            for method, count in new_methods.items():
                logger.info(f"      {method}: {count} instances")
            
            if pattern_types_in_analyses:
                logger.info(f"   📊 New Pattern Types in Analyses:")
                for pattern_type, instances in pattern_types_in_analyses.items():
                    logger.info(f"      {pattern_type}: {len(instances)} instances")
                    for instance in instances[:2]:  # Show first 2 examples
                        logger.info(f"         {instance['symbol']}: {instance['pattern']}")
            
            # Success criteria: Evidence of new methods being called AND new pattern types detected
            method_calls_detected = sum(1 for count in new_methods.values() if count > 0)
            pattern_types_detected = len(pattern_types_in_analyses)
            
            success = (method_calls_detected >= 3 and  # At least 3 new methods active
                      pattern_types_detected >= 2 and  # At least 2 new pattern types found
                      len(implementation_files) > 0)    # Implementation files exist
            
            details = f"New methods active: {method_calls_detected}/15, New pattern types: {pattern_types_detected}, Implementation files: {len(implementation_files)}"
            
            self.log_test_result("Nouvelles Méthodes Intégrées", success, details)
            
        except Exception as e:
            self.log_test_result("Nouvelles Méthodes Intégrées", False, f"Exception: {str(e)}")
    
    async def test_filtre_deduplication_ameliore(self):
        """Test 3: Filtre de Déduplication Amélioré - Vérifier que les patterns similaires mais différents sont gardés"""
        logger.info("\n🔍 TEST 3: Filtre de Déduplication Amélioré")
        
        try:
            # Check backend logs for deduplication logic
            dedup_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'dedup\\|duplicate\\|similar.*pattern\\|pattern.*filter' || echo 'No deduplication logs'"
            result = subprocess.run(dedup_cmd, shell=True, capture_output=True, text=True)
            
            dedup_logs = []
            for line in result.stdout.split('\n'):
                if line.strip() and 'No deduplication' not in line:
                    dedup_logs.append(line.strip())
            
            # Analyze patterns for diversity and similarity
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Filtre Déduplication Amélioré", False, error)
                return
            
            pattern_diversity_analysis = {
                'total_symbols': 0,
                'symbols_with_diverse_patterns': 0,
                'exact_duplicates_found': 0,
                'similar_patterns_kept': 0,
                'pattern_type_diversity': {},
                'duplicate_examples': [],
                'diversity_examples': []
            }
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                
                if patterns_detected:
                    pattern_diversity_analysis['total_symbols'] += 1
                    
                    # Check for exact duplicates within same symbol
                    unique_patterns = set(patterns_detected)
                    if len(unique_patterns) < len(patterns_detected):
                        exact_duplicates = len(patterns_detected) - len(unique_patterns)
                        pattern_diversity_analysis['exact_duplicates_found'] += exact_duplicates
                        pattern_diversity_analysis['duplicate_examples'].append({
                            'symbol': symbol,
                            'total_patterns': len(patterns_detected),
                            'unique_patterns': len(unique_patterns),
                            'duplicates': exact_duplicates
                        })
                    
                    # Check for pattern type diversity (similar but different patterns)
                    pattern_types = {}
                    for pattern in patterns_detected:
                        # Extract base pattern type (e.g., "Triangle" from "Ascending Triangle")
                        base_type = pattern.split()[0] if pattern else 'Unknown'
                        if base_type not in pattern_types:
                            pattern_types[base_type] = []
                        pattern_types[base_type].append(pattern)
                    
                    # Count symbols with diverse pattern types
                    diverse_types = sum(1 for patterns in pattern_types.values() if len(patterns) > 1)
                    if diverse_types > 0:
                        pattern_diversity_analysis['symbols_with_diverse_patterns'] += 1
                        pattern_diversity_analysis['similar_patterns_kept'] += diverse_types
                        pattern_diversity_analysis['diversity_examples'].append({
                            'symbol': symbol,
                            'pattern_types': pattern_types,
                            'diverse_types': diverse_types
                        })
                    
                    # Track overall pattern type diversity
                    for base_type, patterns in pattern_types.items():
                        if base_type not in pattern_diversity_analysis['pattern_type_diversity']:
                            pattern_diversity_analysis['pattern_type_diversity'][base_type] = 0
                        pattern_diversity_analysis['pattern_type_diversity'][base_type] += len(patterns)
            
            # Show examples of diversity and deduplication
            logger.info(f"   📊 Deduplication Analysis:")
            logger.info(f"      Deduplication logs found: {len(dedup_logs)}")
            logger.info(f"      Symbols analyzed: {pattern_diversity_analysis['total_symbols']}")
            logger.info(f"      Exact duplicates found: {pattern_diversity_analysis['exact_duplicates_found']}")
            logger.info(f"      Symbols with diverse patterns: {pattern_diversity_analysis['symbols_with_diverse_patterns']}")
            logger.info(f"      Similar patterns kept: {pattern_diversity_analysis['similar_patterns_kept']}")
            
            if pattern_diversity_analysis['diversity_examples']:
                logger.info(f"   📝 Diversity Examples (Similar but Different Patterns Kept):")
                for example in pattern_diversity_analysis['diversity_examples'][:3]:
                    logger.info(f"      {example['symbol']}: {example['diverse_types']} diverse types")
                    for pattern_type, patterns in example['pattern_types'].items():
                        if len(patterns) > 1:
                            logger.info(f"         {pattern_type}: {patterns}")
            
            if pattern_diversity_analysis['duplicate_examples']:
                logger.info(f"   ⚠️ Duplicate Examples (Should be Avoided):")
                for example in pattern_diversity_analysis['duplicate_examples'][:2]:
                    logger.info(f"      {example['symbol']}: {example['duplicates']} duplicates out of {example['total_patterns']} patterns")
            
            # Pattern type diversity summary
            logger.info(f"   📊 Pattern Type Diversity:")
            for pattern_type, count in sorted(pattern_diversity_analysis['pattern_type_diversity'].items()):
                logger.info(f"      {pattern_type}: {count} instances")
            
            # Success criteria: Low exact duplicates AND high diversity AND evidence of deduplication logic
            duplicate_rate = 0
            if pattern_diversity_analysis['total_symbols'] > 0:
                duplicate_rate = pattern_diversity_analysis['exact_duplicates_found'] / pattern_diversity_analysis['total_symbols']
            
            diversity_rate = 0
            if pattern_diversity_analysis['total_symbols'] > 0:
                diversity_rate = pattern_diversity_analysis['symbols_with_diverse_patterns'] / pattern_diversity_analysis['total_symbols']
            
            success = (duplicate_rate < 0.1 and  # Less than 10% duplicate rate
                      diversity_rate > 0.3 and  # More than 30% symbols have diverse patterns
                      len(pattern_diversity_analysis['pattern_type_diversity']) >= 5)  # At least 5 different pattern types
            
            details = f"Duplicate rate: {duplicate_rate:.2%}, Diversity rate: {diversity_rate:.2%}, Pattern types: {len(pattern_diversity_analysis['pattern_type_diversity'])}, Dedup logs: {len(dedup_logs)}"
            
            self.log_test_result("Filtre Déduplication Amélioré", success, details)
            
        except Exception as e:
            self.log_test_result("Filtre Déduplication Amélioré", False, f"Exception: {str(e)}")
    
    async def test_qualite_detections(self):
        """Test 4: Qualité des Détections - Vérifier confidences 0.7-0.9 et directions cohérentes"""
        logger.info("\n🔍 TEST 4: Qualité des Détections")
        
        try:
            # Get analyses to check pattern quality
            analyses, error = self.get_analyses_from_api()
            
            if error:
                self.log_test_result("Qualité des Détections", False, error)
                return
            
            quality_analysis = {
                'total_analyses': 0,
                'analyses_with_patterns': 0,
                'confidence_in_range': 0,
                'confidence_below_07': 0,
                'confidence_above_09': 0,
                'coherent_directions': 0,
                'incoherent_directions': 0,
                'confidence_distribution': [],
                'direction_analysis': {},
                'quality_examples': []
            }
            
            for analysis in analyses:
                symbol = analysis.get('symbol', 'UNKNOWN')
                patterns_detected = analysis.get('patterns_detected', [])
                confidence = analysis.get('analysis_confidence', 0)
                ia1_signal = analysis.get('ia1_signal', 'hold')
                
                quality_analysis['total_analyses'] += 1
                
                if patterns_detected:
                    quality_analysis['analyses_with_patterns'] += 1
                    quality_analysis['confidence_distribution'].append(confidence)
                    
                    # Check confidence range (0.7-0.9)
                    if 0.7 <= confidence <= 0.9:
                        quality_analysis['confidence_in_range'] += 1
                    elif confidence < 0.7:
                        quality_analysis['confidence_below_07'] += 1
                    elif confidence > 0.9:
                        quality_analysis['confidence_above_09'] += 1
                    
                    # Check direction coherence
                    # Analyze if patterns suggest same direction as IA1 signal
                    bullish_patterns = sum(1 for p in patterns_detected if any(word in p.lower() for word in ['bullish', 'ascending', 'rising', 'bull', 'upward']))
                    bearish_patterns = sum(1 for p in patterns_detected if any(word in p.lower() for word in ['bearish', 'descending', 'falling', 'bear', 'downward']))
                    
                    pattern_direction = 'neutral'
                    if bullish_patterns > bearish_patterns:
                        pattern_direction = 'bullish'
                    elif bearish_patterns > bullish_patterns:
                        pattern_direction = 'bearish'
                    
                    # Check coherence with IA1 signal
                    is_coherent = False
                    if ia1_signal == 'long' and pattern_direction == 'bullish':
                        is_coherent = True
                    elif ia1_signal == 'short' and pattern_direction == 'bearish':
                        is_coherent = True
                    elif ia1_signal == 'hold' and pattern_direction == 'neutral':
                        is_coherent = True
                    elif ia1_signal == 'hold':  # HOLD can be coherent with any pattern direction
                        is_coherent = True
                    
                    if is_coherent:
                        quality_analysis['coherent_directions'] += 1
                    else:
                        quality_analysis['incoherent_directions'] += 1
                    
                    # Store direction analysis
                    quality_analysis['direction_analysis'][symbol] = {
                        'ia1_signal': ia1_signal,
                        'pattern_direction': pattern_direction,
                        'bullish_patterns': bullish_patterns,
                        'bearish_patterns': bearish_patterns,
                        'coherent': is_coherent,
                        'confidence': confidence
                    }
                    
                    # Store quality examples
                    quality_analysis['quality_examples'].append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'patterns_count': len(patterns_detected),
                        'patterns': patterns_detected[:3],  # First 3 patterns
                        'ia1_signal': ia1_signal,
                        'pattern_direction': pattern_direction,
                        'coherent': is_coherent
                    })
            
            # Calculate statistics
            if quality_analysis['confidence_distribution']:
                import numpy as np
                confidences = quality_analysis['confidence_distribution']
                avg_confidence = np.mean(confidences)
                min_confidence = min(confidences)
                max_confidence = max(confidences)
                std_confidence = np.std(confidences)
            else:
                avg_confidence = min_confidence = max_confidence = std_confidence = 0
            
            # Calculate rates
            confidence_in_range_rate = 0
            coherence_rate = 0
            
            if quality_analysis['analyses_with_patterns'] > 0:
                confidence_in_range_rate = quality_analysis['confidence_in_range'] / quality_analysis['analyses_with_patterns']
                total_direction_analyses = quality_analysis['coherent_directions'] + quality_analysis['incoherent_directions']
                if total_direction_analyses > 0:
                    coherence_rate = quality_analysis['coherent_directions'] / total_direction_analyses
            
            logger.info(f"   📊 Quality Analysis:")
            logger.info(f"      Total analyses: {quality_analysis['total_analyses']}")
            logger.info(f"      Analyses with patterns: {quality_analysis['analyses_with_patterns']}")
            logger.info(f"      Confidence in range (0.7-0.9): {quality_analysis['confidence_in_range']} ({confidence_in_range_rate:.1%})")
            logger.info(f"      Confidence below 0.7: {quality_analysis['confidence_below_07']}")
            logger.info(f"      Confidence above 0.9: {quality_analysis['confidence_above_09']}")
            logger.info(f"      Coherent directions: {quality_analysis['coherent_directions']} ({coherence_rate:.1%})")
            logger.info(f"      Incoherent directions: {quality_analysis['incoherent_directions']}")
            
            logger.info(f"   📊 Confidence Statistics:")
            logger.info(f"      Average: {avg_confidence:.3f}")
            logger.info(f"      Range: {min_confidence:.3f} - {max_confidence:.3f}")
            logger.info(f"      Standard deviation: {std_confidence:.3f}")
            
            # Show quality examples
            if quality_analysis['quality_examples']:
                logger.info(f"   📝 Quality Examples:")
                for example in sorted(quality_analysis['quality_examples'], key=lambda x: x['confidence'], reverse=True)[:5]:
                    coherent_str = "✅" if example['coherent'] else "❌"
                    logger.info(f"      {example['symbol']}: conf={example['confidence']:.2f}, {example['patterns_count']} patterns, {example['ia1_signal']}→{example['pattern_direction']} {coherent_str}")
                    logger.info(f"         Patterns: {example['patterns']}")
            
            # Success criteria: High confidence in range AND high coherence rate
            success = (confidence_in_range_rate >= 0.7 and  # At least 70% in 0.7-0.9 range
                      coherence_rate >= 0.8 and           # At least 80% coherent directions
                      avg_confidence >= 0.75)             # Average confidence >= 0.75
            
            details = f"Confidence in range: {confidence_in_range_rate:.1%}, Coherence rate: {coherence_rate:.1%}, Avg confidence: {avg_confidence:.3f}"
            
            self.log_test_result("Qualité des Détections", success, details)
            
        except Exception as e:
            self.log_test_result("Qualité des Détections", False, f"Exception: {str(e)}")
    
    async def test_impact_sur_ia1(self):
        """Test 5: Impact sur IA1 - Vérifier que IA1 reçoit plus de patterns et analyses plus riches"""
        logger.info("\n🔍 TEST 5: Impact sur IA1")
        
        try:
            # Get analyses and decisions to check IA1 impact
            analyses, analyses_error = self.get_analyses_from_api()
            decisions, decisions_error = self.get_decisions_from_api()
            
            if analyses_error:
                self.log_test_result("Impact sur IA1", False, analyses_error)
                return
            
            ia1_impact_analysis = {
                'total_analyses': 0,
                'analyses_with_patterns': 0,
                'rich_analyses': 0,
                'pattern_mentions_in_reasoning': 0,
                'detailed_reasoning_count': 0,
                'pattern_integration_score': 0,
                'reasoning_examples': [],
                'pattern_richness_examples': []
            }
            
            # Analyze IA1 analyses for pattern richness
            if analyses:
                for analysis in analyses:
                    symbol = analysis.get('symbol', 'UNKNOWN')
                    patterns_detected = analysis.get('patterns_detected', [])
                    ia1_reasoning = analysis.get('ia1_reasoning', '')
                    
                    ia1_impact_analysis['total_analyses'] += 1
                    
                    if patterns_detected:
                        ia1_impact_analysis['analyses_with_patterns'] += 1
                        
                        # Check if reasoning mentions patterns
                        pattern_mentions = 0
                        for pattern in patterns_detected:
                            if pattern.lower() in ia1_reasoning.lower():
                                pattern_mentions += 1
                        
                        if pattern_mentions > 0:
                            ia1_impact_analysis['pattern_mentions_in_reasoning'] += 1
                            ia1_impact_analysis['pattern_integration_score'] += pattern_mentions
                        
                        # Check for detailed reasoning (>500 chars indicates rich analysis)
                        if len(ia1_reasoning) > 500:
                            ia1_impact_analysis['detailed_reasoning_count'] += 1
                        
                        # Check for rich analysis (multiple patterns + detailed reasoning)
                        if len(patterns_detected) >= 2 and len(ia1_reasoning) > 300:
                            ia1_impact_analysis['rich_analyses'] += 1
                            ia1_impact_analysis['pattern_richness_examples'].append({
                                'symbol': symbol,
                                'patterns_count': len(patterns_detected),
                                'patterns': patterns_detected,
                                'reasoning_length': len(ia1_reasoning),
                                'pattern_mentions': pattern_mentions
                            })
                        
                        # Store reasoning examples
                        if pattern_mentions > 0:
                            ia1_impact_analysis['reasoning_examples'].append({
                                'symbol': symbol,
                                'patterns': patterns_detected,
                                'reasoning_snippet': ia1_reasoning[:200],
                                'pattern_mentions': pattern_mentions
                            })
            
            # Check backend logs for IA1 pattern integration
            ia1_pattern_cmd = "tail -n 1000 /var/log/supervisor/backend.*.log 2>/dev/null | grep -i 'ia1.*pattern\\|pattern.*ia1\\|chartist.*pattern\\|detected.*pattern' || echo 'No IA1 pattern logs'"
            result = subprocess.run(ia1_pattern_cmd, shell=True, capture_output=True, text=True)
            
            ia1_pattern_logs = []
            for line in result.stdout.split('\n'):
                if line.strip() and 'No IA1 pattern' not in line:
                    ia1_pattern_logs.append(line.strip())
            
            # Check decisions for pattern-influenced reasoning
            pattern_influenced_decisions = 0
            if not decisions_error and decisions:
                for decision in decisions:
                    ia2_reasoning = decision.get('ia2_reasoning', '')
                    if any(word in ia2_reasoning.lower() for word in ['pattern', 'triangle', 'wedge', 'flag', 'pennant', 'rectangle']):
                        pattern_influenced_decisions += 1
            
            # Calculate rates
            pattern_integration_rate = 0
            rich_analysis_rate = 0
            
            if ia1_impact_analysis['analyses_with_patterns'] > 0:
                pattern_integration_rate = ia1_impact_analysis['pattern_mentions_in_reasoning'] / ia1_impact_analysis['analyses_with_patterns']
                rich_analysis_rate = ia1_impact_analysis['rich_analyses'] / ia1_impact_analysis['analyses_with_patterns']
            
            avg_pattern_integration = 0
            if ia1_impact_analysis['analyses_with_patterns'] > 0:
                avg_pattern_integration = ia1_impact_analysis['pattern_integration_score'] / ia1_impact_analysis['analyses_with_patterns']
            
            logger.info(f"   📊 IA1 Impact Analysis:")
            logger.info(f"      Total IA1 analyses: {ia1_impact_analysis['total_analyses']}")
            logger.info(f"      Analyses with patterns: {ia1_impact_analysis['analyses_with_patterns']}")
            logger.info(f"      Pattern mentions in reasoning: {ia1_impact_analysis['pattern_mentions_in_reasoning']} ({pattern_integration_rate:.1%})")
            logger.info(f"      Rich analyses (2+ patterns + detailed): {ia1_impact_analysis['rich_analyses']} ({rich_analysis_rate:.1%})")
            logger.info(f"      Detailed reasoning (>500 chars): {ia1_impact_analysis['detailed_reasoning_count']}")
            logger.info(f"      Average pattern integration score: {avg_pattern_integration:.2f}")
            logger.info(f"      IA1 pattern logs found: {len(ia1_pattern_logs)}")
            logger.info(f"      Pattern-influenced IA2 decisions: {pattern_influenced_decisions}")
            
            # Show richness examples
            if ia1_impact_analysis['pattern_richness_examples']:
                logger.info(f"   📝 Pattern Richness Examples:")
                for example in sorted(ia1_impact_analysis['pattern_richness_examples'], 
                                    key=lambda x: x['patterns_count'], reverse=True)[:3]:
                    logger.info(f"      {example['symbol']}: {example['patterns_count']} patterns, {example['reasoning_length']} chars, {example['pattern_mentions']} mentions")
                    logger.info(f"         Patterns: {example['patterns'][:3]}")
            
            if ia1_impact_analysis['reasoning_examples']:
                logger.info(f"   📝 Pattern Integration Examples:")
                for example in ia1_impact_analysis['reasoning_examples'][:2]:
                    logger.info(f"      {example['symbol']} ({example['pattern_mentions']} mentions): {example['reasoning_snippet']}...")
            
            # Success criteria: High pattern integration AND rich analyses AND evidence in logs
            success = (pattern_integration_rate >= 0.6 and      # At least 60% mention patterns in reasoning
                      rich_analysis_rate >= 0.4 and            # At least 40% are rich analyses
                      avg_pattern_integration >= 1.5 and       # Average 1.5+ pattern mentions per analysis
                      len(ia1_pattern_logs) > 0)               # Evidence in logs
            
            details = f"Pattern integration: {pattern_integration_rate:.1%}, Rich analyses: {rich_analysis_rate:.1%}, Avg integration: {avg_pattern_integration:.2f}, Pattern logs: {len(ia1_pattern_logs)}"
            
            self.log_test_result("Impact sur IA1", success, details)
            
        except Exception as e:
            self.log_test_result("Impact sur IA1", False, f"Exception: {str(e)}")
    
    async def run_comprehensive_tests(self):
        """Run all Pattern Detection System tests"""
        logger.info("🚀 Starting Pattern Detection System Test Suite")
        logger.info("=" * 80)
        
        # Run all tests
        await self.test_augmentation_nombre_patterns()
        await self.test_nouvelles_methodes_integrees()
        await self.test_filtre_deduplication_ameliore()
        await self.test_qualite_detections()
        await self.test_impact_sur_ia1()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 PATTERN DETECTION SYSTEM TEST SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result['success'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {result['test']}")
            if result['details']:
                logger.info(f"   {result['details']}")
                
        logger.info(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
        
        # Specific analysis for French review request
        logger.info("\n" + "=" * 80)
        logger.info("📋 ANALYSE POUR LA DEMANDE FRANÇAISE")
        logger.info("=" * 80)
        
        if passed_tests == total_tests:
            logger.info("🎉 TOUS LES TESTS RÉUSSIS - Le système de détection de patterns chartistes fonctionne correctement!")
            logger.info("✅ Augmentation du nombre de patterns détectés confirmée")
            logger.info("✅ Nouvelles méthodes de détection intégrées et fonctionnelles")
            logger.info("✅ Filtre de déduplication amélioré opérationnel")
            logger.info("✅ Qualité des détections dans les normes (0.7-0.9 confidence)")
            logger.info("✅ Impact positif sur IA1 avec analyses plus riches")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ MAJORITAIREMENT FONCTIONNEL - Quelques problèmes détectés")
            logger.info("🔍 Réviser les tests échoués pour des problèmes spécifiques")
        else:
            logger.info("❌ PROBLÈMES CRITIQUES - Le système de patterns nécessite attention")
            logger.info("🚨 Problèmes majeurs avec la détection ou l'intégration des patterns")
            
        # Recommendations based on test results
        logger.info("\n📝 RECOMMANDATIONS:")
        
        failed_tests = [result for result in self.test_results if not result['success']]
        if failed_tests:
            for failed_test in failed_tests:
                logger.info(f"❌ {failed_test['test']}: {failed_test['details']}")
        else:
            logger.info("✅ Aucun problème critique trouvé avec le système de détection de patterns")
            logger.info("✅ Le système détecte maintenant plus de patterns par symbole")
            logger.info("✅ Les nouvelles méthodes (wedge, flag, triple, pennant, etc.) sont actives")
            logger.info("✅ La déduplication évite les doublons tout en gardant la diversité")
            logger.info("✅ La qualité des détections respecte les critères de confidence")
            logger.info("✅ IA1 bénéficie d'analyses plus riches grâce aux patterns détectés")
            
        return passed_tests, total_tests

async def main():
    """Main test execution"""
    test_suite = PatternDetectionTestSuite()
    passed, total = await test_suite.run_comprehensive_tests()
    
    # Exit with appropriate code
    if passed == total:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())
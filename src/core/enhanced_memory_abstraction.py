#!/usr/bin/env python3
"""
Enhanced Memory Abstraction System

This module enhances the 4-Phase Memory Optimization system to abstract patterns
into semantic concepts, enabling powerful analogical reasoning and knowledge transfer.

Key Features:
- Semantic Concept Extraction: Convert raw patterns into abstract concepts
- Analogical Reasoning: Find similar patterns across different contexts
- Knowledge Transfer: Apply learned concepts to new situations
- Concept Hierarchy: Build hierarchical relationships between concepts
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import pickle
import json

logger = logging.getLogger(__name__)


@dataclass
class SemanticConcept:
    """Represents an abstract semantic concept."""
    concept_id: str
    name: str
    description: str
    abstract_pattern: Dict[str, Any]
    concrete_examples: List[Dict[str, Any]]
    confidence: float
    usage_count: int
    last_used: float


@dataclass
class ConceptRelationship:
    """Represents a relationship between concepts."""
    concept_a: str
    concept_b: str
    relationship_type: str
    strength: float
    context: Dict[str, Any]


class SemanticConceptExtractor:
    """Extracts semantic concepts from raw patterns."""
    
    def __init__(self):
        self.concept_templates = self._initialize_concept_templates()
        self.extraction_threshold = 0.7
    
    def _initialize_concept_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize templates for common semantic concepts."""
        return {
            'barrier': {
                'description': 'An obstacle that blocks movement or progress',
                'pattern_features': ['static', 'blocking', 'solid'],
                'action_implications': ['avoid', 'circumvent', 'remove']
            },
            'key': {
                'description': 'An object that unlocks or activates something',
                'pattern_features': ['portable', 'interactive', 'unlocking'],
                'action_implications': ['collect', 'use', 'activate']
            },
            'door': {
                'description': 'An entrance or exit that can be opened/closed',
                'pattern_features': ['movable', 'passage', 'interactive'],
                'action_implications': ['open', 'close', 'pass_through']
            },
            'path': {
                'description': 'A route or way to move from one place to another',
                'pattern_features': ['linear', 'navigable', 'connecting'],
                'action_implications': ['follow', 'navigate', 'explore']
            },
            'container': {
                'description': 'An object that holds or stores other objects',
                'pattern_features': ['enclosed', 'storage', 'interactive'],
                'action_implications': ['open', 'store', 'retrieve']
            },
            'switch': {
                'description': 'A control that activates or deactivates something',
                'pattern_features': ['interactive', 'binary_state', 'control'],
                'action_implications': ['toggle', 'activate', 'control']
            }
        }
    
    def extract_concepts(self, patterns: List[Dict[str, Any]]) -> List[SemanticConcept]:
        """Extract semantic concepts from patterns."""
        concepts = []
        
        for pattern in patterns:
            # Try to match pattern to concept templates
            matched_concept = self._match_pattern_to_concept(pattern)
            if matched_concept:
                concepts.append(matched_concept)
        
        return concepts
    
    def _match_pattern_to_concept(self, pattern: Dict[str, Any]) -> Optional[SemanticConcept]:
        """Match a pattern to a semantic concept."""
        best_match = None
        best_score = 0.0
        
        for concept_name, template in self.concept_templates.items():
            score = self._calculate_concept_match_score(pattern, template)
            if score > self.extraction_threshold and score > best_score:
                best_score = score
                best_match = self._create_concept_from_template(concept_name, template, pattern, score)
        
        return best_match
    
    def _calculate_concept_match_score(self, pattern: Dict[str, Any], template: Dict[str, Any]) -> float:
        """Calculate how well a pattern matches a concept template."""
        score = 0.0
        total_features = 0
        
        # Check pattern features
        pattern_features = pattern.get('features', [])
        template_features = template.get('pattern_features', [])
        
        for feature in template_features:
            total_features += 1
            if feature in pattern_features:
                score += 1.0
            elif self._is_similar_feature(feature, pattern_features):
                score += 0.5
        
        if total_features > 0:
            score = score / total_features
        
        # Boost score for action implications
        action_implications = template.get('action_implications', [])
        pattern_actions = pattern.get('suggested_actions', [])
        
        if action_implications and pattern_actions:
            action_match = len(set(action_implications) & set(pattern_actions)) / len(action_implications)
            score = (score + action_match) / 2.0
        
        return score
    
    def _is_similar_feature(self, feature: str, pattern_features: List[str]) -> bool:
        """Check if a feature is similar to any pattern features."""
        similarity_map = {
            'static': ['stationary', 'fixed', 'immobile'],
            'blocking': ['obstacle', 'barrier', 'wall'],
            'solid': ['hard', 'rigid', 'firm'],
            'portable': ['movable', 'carryable', 'transportable'],
            'interactive': ['clickable', 'usable', 'manipulable'],
            'unlocking': ['key', 'access', 'permission'],
            'movable': ['portable', 'sliding', 'rotating'],
            'passage': ['opening', 'entrance', 'exit'],
            'linear': ['straight', 'direct', 'path'],
            'navigable': ['passable', 'traversable', 'accessible'],
            'connecting': ['linking', 'joining', 'bridging'],
            'enclosed': ['contained', 'bounded', 'walled'],
            'storage': ['container', 'holder', 'receptacle'],
            'binary_state': ['toggle', 'switch', 'on_off'],
            'control': ['command', 'direct', 'manage']
        }
        
        similar_features = similarity_map.get(feature, [])
        return any(sim_feature in pattern_features for sim_feature in similar_features)
    
    def _create_concept_from_template(self, concept_name: str, template: Dict[str, Any], 
                                    pattern: Dict[str, Any], score: float) -> SemanticConcept:
        """Create a semantic concept from a template and pattern."""
        return SemanticConcept(
            concept_id=f"{concept_name}_{int(time.time() * 1000)}",
            name=concept_name,
            description=template['description'],
            abstract_pattern=self._create_abstract_pattern(pattern),
            concrete_examples=[pattern],
            confidence=score,
            usage_count=1,
            last_used=time.time()
        )
    
    def _create_abstract_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Create an abstract pattern from a concrete pattern."""
        return {
            'type': pattern.get('type', 'unknown'),
            'features': pattern.get('features', []),
            'suggested_actions': pattern.get('suggested_actions', []),
            'spatial_properties': pattern.get('spatial_properties', {}),
            'interaction_properties': pattern.get('interaction_properties', {}),
            'abstraction_level': 'high'
        }


class AnalogicalReasoningEngine:
    """Finds similar patterns across different contexts for analogical reasoning."""
    
    def __init__(self):
        self.similarity_threshold = 0.6
        self.analogical_mappings = {}
    
    def find_analogies(self, target_concept: SemanticConcept, 
                      concept_database: List[SemanticConcept]) -> List[Tuple[SemanticConcept, float]]:
        """Find analogies for a target concept."""
        analogies = []
        
        for concept in concept_database:
            if concept.concept_id != target_concept.concept_id:
                similarity = self._calculate_concept_similarity(target_concept, concept)
                if similarity > self.similarity_threshold:
                    analogies.append((concept, similarity))
        
        # Sort by similarity
        analogies.sort(key=lambda x: x[1], reverse=True)
        return analogies
    
    def _calculate_concept_similarity(self, concept_a: SemanticConcept, concept_b: SemanticConcept) -> float:
        """Calculate similarity between two concepts."""
        similarity = 0.0
        
        # Compare abstract patterns
        pattern_similarity = self._compare_abstract_patterns(
            concept_a.abstract_pattern, concept_b.abstract_pattern
        )
        similarity += pattern_similarity * 0.4
        
        # Compare features
        feature_similarity = self._compare_features(
            concept_a.abstract_pattern.get('features', []),
            concept_b.abstract_pattern.get('features', [])
        )
        similarity += feature_similarity * 0.3
        
        # Compare suggested actions
        action_similarity = self._compare_actions(
            concept_a.abstract_pattern.get('suggested_actions', []),
            concept_b.abstract_pattern.get('suggested_actions', [])
        )
        similarity += action_similarity * 0.3
        
        return similarity
    
    def _compare_abstract_patterns(self, pattern_a: Dict[str, Any], pattern_b: Dict[str, Any]) -> float:
        """Compare two abstract patterns."""
        if pattern_a.get('type') == pattern_b.get('type'):
            return 1.0
        else:
            return 0.0
    
    def _compare_features(self, features_a: List[str], features_b: List[str]) -> float:
        """Compare feature lists."""
        if not features_a and not features_b:
            return 1.0
        if not features_a or not features_b:
            return 0.0
        
        common_features = set(features_a) & set(features_b)
        total_features = set(features_a) | set(features_b)
        
        return len(common_features) / len(total_features)
    
    def _compare_actions(self, actions_a: List[str], actions_b: List[str]) -> float:
        """Compare action lists."""
        if not actions_a and not actions_b:
            return 1.0
        if not actions_a or not actions_b:
            return 0.0
        
        common_actions = set(actions_a) & set(actions_b)
        total_actions = set(actions_a) | set(actions_b)
        
        return len(common_actions) / len(total_actions)
    
    def suggest_analogical_actions(self, target_concept: SemanticConcept, 
                                 analogies: List[Tuple[SemanticConcept, float]]) -> List[str]:
        """Suggest actions based on analogical reasoning."""
        suggested_actions = []
        
        for concept, similarity in analogies:
            # Get actions from similar concept
            concept_actions = concept.abstract_pattern.get('suggested_actions', [])
            
            # Weight actions by similarity
            for action in concept_actions:
                weighted_action = {
                    'action': action,
                    'weight': similarity,
                    'source_concept': concept.name
                }
                suggested_actions.append(weighted_action)
        
        # Sort by weight and return top actions
        suggested_actions.sort(key=lambda x: x['weight'], reverse=True)
        return [action['action'] for action in suggested_actions[:5]]


class KnowledgeTransferEngine:
    """Transfers knowledge between different contexts and games."""
    
    def __init__(self):
        self.transfer_threshold = 0.5
        self.transfer_history = []
    
    def transfer_knowledge(self, source_concept: SemanticConcept, 
                          target_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transfer knowledge from source concept to target context."""
        
        # Check if transfer is appropriate
        if not self._is_transfer_appropriate(source_concept, target_context):
            return None
        
        # Create transferred knowledge
        transferred_knowledge = {
            'source_concept': source_concept.name,
            'transferred_actions': source_concept.abstract_pattern.get('suggested_actions', []),
            'transferred_features': source_concept.abstract_pattern.get('features', []),
            'confidence': source_concept.confidence * 0.8,  # Reduce confidence for transfer
            'context_adaptations': self._adapt_to_context(source_concept, target_context),
            'transfer_timestamp': time.time()
        }
        
        # Record transfer
        self.transfer_history.append({
            'source_concept': source_concept.name,
            'target_context': target_context.get('game_id', 'unknown'),
            'timestamp': time.time(),
            'success': True
        })
        
        return transferred_knowledge
    
    def _is_transfer_appropriate(self, source_concept: SemanticConcept, target_context: Dict[str, Any]) -> bool:
        """Check if knowledge transfer is appropriate."""
        # Check if concept has been used successfully before
        if source_concept.usage_count < 2:
            return False
        
        # Check if target context is similar enough
        target_features = target_context.get('features', [])
        concept_features = source_concept.abstract_pattern.get('features', [])
        
        if not target_features or not concept_features:
            return False
        
        # Calculate feature overlap
        common_features = set(target_features) & set(concept_features)
        feature_overlap = len(common_features) / len(set(target_features) | set(concept_features))
        
        return feature_overlap > self.transfer_threshold
    
    def _adapt_to_context(self, source_concept: SemanticConcept, target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt transferred knowledge to target context."""
        adaptations = {}
        
        # Adapt actions to available actions in target context
        available_actions = target_context.get('available_actions', [])
        source_actions = source_concept.abstract_pattern.get('suggested_actions', [])
        
        adapted_actions = []
        for action in source_actions:
            if action in available_actions:
                adapted_actions.append(action)
            else:
                # Find similar action
                similar_action = self._find_similar_action(action, available_actions)
                if similar_action:
                    adapted_actions.append(similar_action)
        
        adaptations['adapted_actions'] = adapted_actions
        
        # Adapt spatial properties
        target_spatial = target_context.get('spatial_properties', {})
        source_spatial = source_concept.abstract_pattern.get('spatial_properties', {})
        
        if target_spatial and source_spatial:
            adaptations['spatial_adaptations'] = self._adapt_spatial_properties(source_spatial, target_spatial)
        
        return adaptations
    
    def _find_similar_action(self, action: str, available_actions: List[int]) -> Optional[int]:
        """Find a similar action in available actions."""
        action_mapping = {
            'move_up': 1,
            'move_down': 2,
            'move_left': 3,
            'move_right': 4,
            'interact': 5,
            'coordinate': 6,
            'undo': 7
        }
        
        # Try direct mapping first
        if action in action_mapping and action_mapping[action] in available_actions:
            return action_mapping[action]
        
        # Try partial matches
        for action_name, action_id in action_mapping.items():
            if action_id in available_actions:
                if any(word in action.lower() for word in action_name.split('_')):
                    return action_id
        
        return None
    
    def _adapt_spatial_properties(self, source_spatial: Dict[str, Any], target_spatial: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt spatial properties from source to target context."""
        adaptations = {}
        
        # Scale coordinates if needed
        if 'coordinates' in source_spatial and 'grid_size' in target_spatial:
            source_coords = source_spatial['coordinates']
            target_size = target_spatial['grid_size']
            source_size = source_spatial.get('grid_size', (64, 64))
            
            if source_size != target_size:
                scale_x = target_size[0] / source_size[0]
                scale_y = target_size[1] / source_size[1]
                
                adapted_coords = []
                for coord in source_coords:
                    adapted_coord = (
                        int(coord[0] * scale_x),
                        int(coord[1] * scale_y)
                    )
                    adapted_coords.append(adapted_coord)
                
                adaptations['scaled_coordinates'] = adapted_coords
        
        return adaptations


class EnhancedMemoryAbstractionSystem:
    """
    Main Enhanced Memory Abstraction System.
    
    This system enhances the 4-Phase Memory Optimization to abstract patterns
    into semantic concepts, enabling powerful analogical reasoning and knowledge transfer.
    """
    
    def __init__(self):
        self.concept_extractor = SemanticConceptExtractor()
        self.analogical_reasoning = AnalogicalReasoningEngine()
        self.knowledge_transfer = KnowledgeTransferEngine()
        
        # Concept database
        self.concept_database = []
        self.concept_relationships = []
        
        logger.info("Enhanced Memory Abstraction System initialized")
    
    def process_patterns(self, patterns: List[Dict[str, Any]]) -> List[SemanticConcept]:
        """Process patterns and extract semantic concepts."""
        concepts = self.concept_extractor.extract_concepts(patterns)
        
        # Add to concept database
        for concept in concepts:
            self._add_concept_to_database(concept)
        
        return concepts
    
    def find_analogies(self, target_concept: SemanticConcept) -> List[Tuple[SemanticConcept, float]]:
        """Find analogies for a target concept."""
        return self.analogical_reasoning.find_analogies(target_concept, self.concept_database)
    
    def transfer_knowledge(self, source_concept: SemanticConcept, target_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transfer knowledge from source concept to target context."""
        return self.knowledge_transfer.transfer_knowledge(source_concept, target_context)
    
    def suggest_actions_for_context(self, context: Dict[str, Any]) -> List[str]:
        """Suggest actions based on analogical reasoning for a given context."""
        # Find relevant concepts for the context
        relevant_concepts = self._find_relevant_concepts(context)
        
        if not relevant_concepts:
            return []
        
        # Find analogies for the most relevant concept
        best_concept = max(relevant_concepts, key=lambda x: x[1])
        analogies = self.find_analogies(best_concept[0])
        
        # Suggest actions based on analogies
        suggested_actions = self.analogical_reasoning.suggest_analogical_actions(best_concept[0], analogies)
        
        return suggested_actions
    
    def _add_concept_to_database(self, concept: SemanticConcept):
        """Add a concept to the database, merging with existing concepts if similar."""
        # Check for existing similar concepts
        for existing_concept in self.concept_database:
            if existing_concept.name == concept.name:
                # Merge with existing concept
                self._merge_concepts(existing_concept, concept)
                return
        
        # Add new concept
        self.concept_database.append(concept)
    
    def _merge_concepts(self, existing_concept: SemanticConcept, new_concept: SemanticConcept):
        """Merge a new concept with an existing one."""
        # Add concrete examples
        existing_concept.concrete_examples.extend(new_concept.concrete_examples)
        
        # Update confidence (weighted average)
        total_usage = existing_concept.usage_count + new_concept.usage_count
        existing_concept.confidence = (
            (existing_concept.confidence * existing_concept.usage_count + 
             new_concept.confidence * new_concept.usage_count) / total_usage
        )
        
        # Update usage count
        existing_concept.usage_count = total_usage
        existing_concept.last_used = max(existing_concept.last_used, new_concept.last_used)
    
    def _find_relevant_concepts(self, context: Dict[str, Any]) -> List[Tuple[SemanticConcept, float]]:
        """Find concepts relevant to the given context."""
        relevant_concepts = []
        
        context_features = context.get('features', [])
        
        for concept in self.concept_database:
            concept_features = concept.abstract_pattern.get('features', [])
            
            if context_features and concept_features:
                # Calculate relevance based on feature overlap
                common_features = set(context_features) & set(concept_features)
                relevance = len(common_features) / len(set(context_features) | set(concept_features))
                
                if relevance > 0.3:  # Minimum relevance threshold
                    relevant_concepts.append((concept, relevance))
        
        # Sort by relevance
        relevant_concepts.sort(key=lambda x: x[1], reverse=True)
        return relevant_concepts
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
        return {
            'total_concepts': len(self.concept_database),
            'concept_types': list(set(concept.name for concept in self.concept_database)),
            'total_relationships': len(self.concept_relationships),
            'transfer_history': len(self.knowledge_transfer.transfer_history),
            'recent_transfers': len([t for t in self.knowledge_transfer.transfer_history if time.time() - t['timestamp'] < 3600])
        }

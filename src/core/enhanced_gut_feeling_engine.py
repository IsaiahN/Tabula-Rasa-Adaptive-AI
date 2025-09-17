#!/usr/bin/env python3
"""
Enhanced Gut Feeling Engine - Conscious Architecture Enhancement

Implements explicit pattern matching and intuitive action selection inspired by
biological consciousness research. Provides "gut feeling" actions based on
pattern similarity and associative memory.

Key Features:
- Fast pattern matching for intuitive action suggestions
- Associative memory integration for gut feelings
- Confidence-based gut feeling weighting
- Integration with existing memory and learning systems
- Explicit reasoning for intuitive decisions
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import hashlib

logger = logging.getLogger(__name__)

class GutFeelingType(Enum):
    """Types of gut feelings based on pattern matching."""
    PATTERN_SIMILARITY = "pattern_similarity"
    SEMANTIC_ASSOCIATION = "semantic_association"
    TEMPORAL_PATTERN = "temporal_pattern"
    SPATIAL_PATTERN = "spatial_pattern"
    SUCCESS_PATTERN = "success_pattern"

class IntuitionConfidence(Enum):
    """Confidence levels for intuitive decisions."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

@dataclass
class GutFeeling:
    """Represents a gut feeling action suggestion."""
    action: int
    confidence: float
    gut_feeling_type: GutFeelingType
    reasoning: str
    pattern_id: Optional[str] = None
    similarity_score: float = 0.0
    success_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class PatternMatch:
    """Result of pattern matching analysis."""
    pattern_id: str
    similarity_score: float
    pattern_type: str
    success_rate: float
    context_similarity: float
    action_suggestion: int
    reasoning: str

class EnhancedGutFeelingEngine:
    """
    Enhanced Gut Feeling Engine for intuitive action selection.
    
    This system provides fast, pattern-based action suggestions that feel
    intuitive and are based on past successful experiences.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.6,
                 max_patterns: int = 1000,
                 confidence_decay: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.max_patterns = max_patterns
        self.confidence_decay = confidence_decay
        
        # Pattern storage
        self.patterns = {}  # pattern_id -> pattern_data
        self.pattern_index = defaultdict(list)  # pattern_type -> pattern_ids
        self.action_patterns = defaultdict(list)  # action -> pattern_ids
        
        # Performance tracking
        self.gut_feeling_history = []
        self.success_tracking = defaultdict(list)
        
        # Integration components
        self.memory_manager = None
        self.pattern_learner = None
        self.coordinate_manager = None
        
        logger.info("Enhanced Gut Feeling Engine initialized")
    
    def integrate_components(self, 
                           memory_manager=None, 
                           pattern_learner=None, 
                           coordinate_manager=None):
        """Integrate with existing Tabula Rasa components."""
        self.memory_manager = memory_manager
        self.pattern_learner = pattern_learner
        self.coordinate_manager = coordinate_manager
        logger.info("Enhanced Gut Feeling Engine integrated with existing components")
    
    def get_gut_feelings(self, 
                        current_state: Dict[str, Any],
                        available_actions: List[int],
                        context: Dict[str, Any]) -> List[GutFeeling]:
        """Get gut feeling action suggestions for current state."""
        gut_feelings = []
        
        # 1. Pattern similarity matching
        pattern_matches = self._find_similar_patterns(current_state, context)
        
        # 2. Generate gut feelings from patterns
        for match in pattern_matches:
            if match.similarity_score >= self.similarity_threshold:
                gut_feeling = self._create_gut_feeling_from_match(match, available_actions)
                if gut_feeling:
                    gut_feelings.append(gut_feeling)
        
        # 3. Semantic association gut feelings
        semantic_gut_feelings = self._get_semantic_gut_feelings(current_state, available_actions, context)
        gut_feelings.extend(semantic_gut_feelings)
        
        # 4. Temporal pattern gut feelings
        temporal_gut_feelings = self._get_temporal_gut_feelings(current_state, available_actions, context)
        gut_feelings.extend(temporal_gut_feelings)
        
        # 5. Sort by confidence and filter by available actions
        gut_feelings = [gf for gf in gut_feelings if gf.action in available_actions]
        gut_feelings.sort(key=lambda x: x.confidence, reverse=True)
        
        # 6. Apply confidence decay based on time
        gut_feelings = self._apply_confidence_decay(gut_feelings)
        
        logger.debug(f"Generated {len(gut_feelings)} gut feelings for {len(available_actions)} actions")
        
        return gut_feelings[:5]  # Return top 5 gut feelings
    
    def learn_from_outcome(self, 
                          gut_feeling: GutFeeling, 
                          outcome: Dict[str, Any],
                          context: Dict[str, Any]):
        """Learn from the outcome of a gut feeling action."""
        success = outcome.get('success', False)
        performance_score = outcome.get('performance_score', 0.0)
        
        # Update success tracking
        self.success_tracking[gut_feeling.action].append({
            'success': success,
            'performance_score': performance_score,
            'timestamp': time.time(),
            'context': context
        })
        
        # Update pattern success rate if pattern exists
        if gut_feeling.pattern_id and gut_feeling.pattern_id in self.patterns:
            pattern = self.patterns[gut_feeling.pattern_id]
            pattern['success_count'] += 1 if success else 0
            pattern['total_attempts'] += 1
            pattern['success_rate'] = pattern['success_count'] / pattern['total_attempts']
        
        # Store gut feeling outcome
        self.gut_feeling_history.append({
            'gut_feeling': gut_feeling,
            'outcome': outcome,
            'context': context,
            'timestamp': time.time()
        })
        
        logger.debug(f"Learned from gut feeling outcome: success={success}, "
                    f"performance={performance_score:.3f}")
    
    def add_pattern(self, 
                   pattern_data: Dict[str, Any],
                   pattern_type: str,
                   success_rate: float = 0.0) -> str:
        """Add a new pattern to the gut feeling engine."""
        pattern_id = self._generate_pattern_id(pattern_data)
        
        pattern = {
            'pattern_id': pattern_id,
            'pattern_data': pattern_data,
            'pattern_type': pattern_type,
            'success_rate': success_rate,
            'success_count': 0,
            'total_attempts': 0,
            'created_at': time.time(),
            'last_used': time.time()
        }
        
        self.patterns[pattern_id] = pattern
        self.pattern_index[pattern_type].append(pattern_id)
        
        # Index by suggested action
        suggested_action = pattern_data.get('suggested_action')
        if suggested_action is not None:
            self.action_patterns[suggested_action].append(pattern_id)
        
        # Limit total patterns
        if len(self.patterns) > self.max_patterns:
            self._prune_old_patterns()
        
        logger.debug(f"Added pattern {pattern_id} of type {pattern_type}")
        return pattern_id
    
    def _find_similar_patterns(self, 
                              current_state: Dict[str, Any], 
                              context: Dict[str, Any]) -> List[PatternMatch]:
        """Find patterns similar to current state."""
        matches = []
        
        for pattern_id, pattern in self.patterns.items():
            similarity_score = self._calculate_pattern_similarity(
                current_state, pattern['pattern_data'], context
            )
            
            if similarity_score > 0.3:  # Minimum similarity threshold
                match = PatternMatch(
                    pattern_id=pattern_id,
                    similarity_score=similarity_score,
                    pattern_type=pattern['pattern_type'],
                    success_rate=pattern['success_rate'],
                    context_similarity=self._calculate_context_similarity(context, pattern['pattern_data']),
                    action_suggestion=pattern['pattern_data'].get('suggested_action', 0),
                    reasoning=f"Similar to pattern {pattern_id[:8]}..."
                )
                matches.append(match)
        
        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:10]  # Return top 10 matches
    
    def _calculate_pattern_similarity(self, 
                                    current_state: Dict[str, Any],
                                    pattern_data: Dict[str, Any],
                                    context: Dict[str, Any]) -> float:
        """Calculate similarity between current state and pattern."""
        similarity_scores = []
        
        # State similarity
        if 'state' in pattern_data:
            state_similarity = self._calculate_dict_similarity(
                current_state, pattern_data['state']
            )
            similarity_scores.append(state_similarity)
        
        # Context similarity
        if 'context' in pattern_data:
            context_similarity = self._calculate_dict_similarity(
                context, pattern_data['context']
            )
            similarity_scores.append(context_similarity)
        
        # Frame similarity (if available)
        if 'frame_features' in current_state and 'frame_features' in pattern_data:
            frame_similarity = self._calculate_frame_similarity(
                current_state['frame_features'],
                pattern_data['frame_features']
            )
            similarity_scores.append(frame_similarity)
        
        # Return average similarity
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_dict_similarity(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> float:
        """Calculate similarity between two dictionaries."""
        if not dict1 or not dict2:
            return 0.0
        
        common_keys = set(dict1.keys()) & set(dict2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = dict1[key], dict2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    similarity = 1.0 if val1 == val2 else 0.0
                else:
                    similarity = 1.0 - abs(val1 - val2) / max_val
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple)
                similarity = 1.0 if val1 == val2 else 0.0
            elif isinstance(val1, list) and isinstance(val2, list):
                # List similarity
                if len(val1) == len(val2):
                    similarities_list = [1.0 if v1 == v2 else 0.0 for v1, v2 in zip(val1, val2)]
                    similarity = np.mean(similarities_list)
                else:
                    similarity = 0.0
            else:
                similarity = 1.0 if val1 == val2 else 0.0
            
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _calculate_frame_similarity(self, frame1: Any, frame2: Any) -> float:
        """Calculate similarity between frame features."""
        # Simplified frame similarity calculation
        if hasattr(frame1, 'shape') and hasattr(frame2, 'shape'):
            if frame1.shape == frame2.shape:
                # Calculate normalized difference
                diff = np.abs(frame1 - frame2)
                max_diff = np.max(diff)
                return 1.0 - (max_diff / 255.0) if max_diff > 0 else 1.0
        
        return 0.0
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between contexts."""
        return self._calculate_dict_similarity(context1, context2)
    
    def _create_gut_feeling_from_match(self, 
                                     match: PatternMatch, 
                                     available_actions: List[int]) -> Optional[GutFeeling]:
        """Create a gut feeling from a pattern match."""
        if match.action_suggestion not in available_actions:
            return None
        
        # Calculate confidence based on similarity and success rate
        confidence = (0.6 * match.similarity_score + 
                     0.4 * match.success_rate)
        
        # Determine gut feeling type
        gut_feeling_type = self._determine_gut_feeling_type(match)
        
        # Generate reasoning
        reasoning = self._generate_gut_feeling_reasoning(match)
        
        return GutFeeling(
            action=match.action_suggestion,
            confidence=confidence,
            gut_feeling_type=gut_feeling_type,
            reasoning=reasoning,
            pattern_id=match.pattern_id,
            similarity_score=match.similarity_score,
            success_rate=match.success_rate
        )
    
    def _get_semantic_gut_feelings(self, 
                                  current_state: Dict[str, Any],
                                  available_actions: List[int],
                                  context: Dict[str, Any]) -> List[GutFeeling]:
        """Get gut feelings based on semantic associations."""
        gut_feelings = []
        
        # Look for semantic patterns in current state
        semantic_patterns = self._find_semantic_patterns(current_state, context)
        
        for pattern in semantic_patterns:
            if pattern['suggested_action'] in available_actions:
                gut_feeling = GutFeeling(
                    action=pattern['suggested_action'],
                    confidence=pattern['confidence'],
                    gut_feeling_type=GutFeelingType.SEMANTIC_ASSOCIATION,
                    reasoning=f"Semantic association: {pattern['reasoning']}",
                    pattern_id=pattern.get('pattern_id')
                )
                gut_feelings.append(gut_feeling)
        
        return gut_feelings
    
    def _get_temporal_gut_feelings(self, 
                                  current_state: Dict[str, Any],
                                  available_actions: List[int],
                                  context: Dict[str, Any]) -> List[GutFeeling]:
        """Get gut feelings based on temporal patterns."""
        gut_feelings = []
        
        # Look for temporal patterns in recent history
        if len(self.gut_feeling_history) > 3:
            recent_actions = [entry['gut_feeling'].action 
                            for entry in self.gut_feeling_history[-5:]]
            
            # Find actions that often follow current sequence
            for action in available_actions:
                if action in recent_actions:
                    # Calculate temporal confidence
                    action_frequency = recent_actions.count(action) / len(recent_actions)
                    confidence = min(0.8, action_frequency * 0.5)
                    
                    if confidence > 0.3:
                        gut_feeling = GutFeeling(
                            action=action,
                            confidence=confidence,
                            gut_feeling_type=GutFeelingType.TEMPORAL_PATTERN,
                            reasoning=f"Temporal pattern: frequently follows recent actions"
                        )
                        gut_feelings.append(gut_feeling)
        
        return gut_feelings
    
    def _determine_gut_feeling_type(self, match: PatternMatch) -> GutFeelingType:
        """Determine the type of gut feeling based on pattern match."""
        if match.pattern_type == 'spatial':
            return GutFeelingType.SPATIAL_PATTERN
        elif match.pattern_type == 'temporal':
            return GutFeelingType.TEMPORAL_PATTERN
        elif match.pattern_type == 'success':
            return GutFeelingType.SUCCESS_PATTERN
        else:
            return GutFeelingType.PATTERN_SIMILARITY
    
    def _generate_gut_feeling_reasoning(self, match: PatternMatch) -> str:
        """Generate human-readable reasoning for gut feeling."""
        if match.success_rate > 0.8:
            return f"Strong gut feeling: similar successful pattern (success rate: {match.success_rate:.1%})"
        elif match.similarity_score > 0.8:
            return f"Intuitive match: very similar to past experience (similarity: {match.similarity_score:.1%})"
        else:
            return f"Gut feeling: reminds me of pattern {match.pattern_id[:8]} (similarity: {match.similarity_score:.1%})"
    
    def _find_semantic_patterns(self, 
                              current_state: Dict[str, Any], 
                              context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find semantic patterns in current state."""
        patterns = []
        
        # Look for semantic keywords in state
        semantic_keywords = ['button', 'portal', 'object', 'pattern', 'shape', 'color']
        
        for keyword in semantic_keywords:
            if keyword in str(current_state).lower():
                # Find patterns associated with this keyword
                for pattern_id, pattern in self.patterns.items():
                    if keyword in str(pattern['pattern_data']).lower():
                        patterns.append({
                            'suggested_action': pattern['pattern_data'].get('suggested_action', 0),
                            'confidence': pattern['success_rate'] * 0.8,
                            'reasoning': f"Semantic keyword '{keyword}' detected",
                            'pattern_id': pattern_id
                        })
        
        return patterns
    
    def _apply_confidence_decay(self, gut_feelings: List[GutFeeling]) -> List[GutFeeling]:
        """Apply confidence decay based on time since pattern creation."""
        current_time = time.time()
        
        for gut_feeling in gut_feelings:
            if gut_feeling.pattern_id and gut_feeling.pattern_id in self.patterns:
                pattern = self.patterns[gut_feeling.pattern_id]
                age_hours = (current_time - pattern['created_at']) / 3600
                decay_factor = self.confidence_decay ** age_hours
                gut_feeling.confidence *= decay_factor
        
        return gut_feelings
    
    def _generate_pattern_id(self, pattern_data: Dict[str, Any]) -> str:
        """Generate a unique pattern ID."""
        content = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _prune_old_patterns(self):
        """Remove old, low-performing patterns."""
        if len(self.patterns) <= self.max_patterns:
            return
        
        # Sort patterns by effectiveness (success_rate * recency)
        current_time = time.time()
        pattern_scores = []
        
        for pattern_id, pattern in self.patterns.items():
            age_hours = (current_time - pattern['created_at']) / 3600
            recency_factor = self.confidence_decay ** age_hours
            effectiveness = pattern['success_rate'] * recency_factor
            pattern_scores.append((pattern_id, effectiveness))
        
        # Remove lowest scoring patterns
        pattern_scores.sort(key=lambda x: x[1])
        patterns_to_remove = pattern_scores[:len(self.patterns) - self.max_patterns]
        
        for pattern_id, _ in patterns_to_remove:
            self._remove_pattern(pattern_id)
    
    def _remove_pattern(self, pattern_id: str):
        """Remove a pattern from all indices."""
        if pattern_id not in self.patterns:
            return
        
        pattern = self.patterns[pattern_id]
        
        # Remove from type index
        if pattern['pattern_type'] in self.pattern_index:
            if pattern_id in self.pattern_index[pattern['pattern_type']]:
                self.pattern_index[pattern['pattern_type']].remove(pattern_id)
        
        # Remove from action index
        suggested_action = pattern['pattern_data'].get('suggested_action')
        if suggested_action is not None and suggested_action in self.action_patterns:
            if pattern_id in self.action_patterns[suggested_action]:
                self.action_patterns[suggested_action].remove(pattern_id)
        
        # Remove from main patterns
        del self.patterns[pattern_id]
    
    def get_gut_feeling_metrics(self) -> Dict[str, Any]:
        """Get metrics about gut feeling performance."""
        if not self.gut_feeling_history:
            return {
                'total_gut_feelings': 0, 
                'success_rate': 0.0,
                'total_patterns': len(self.patterns),
                'pattern_types': list(self.pattern_index.keys()),
                'action_coverage': len(self.action_patterns),
                'average_confidence': 0.0
            }
        
        successful_gut_feelings = sum(1 for entry in self.gut_feeling_history 
                                    if entry['outcome'].get('success', False))
        total_gut_feelings = len(self.gut_feeling_history)
        success_rate = successful_gut_feelings / total_gut_feelings if total_gut_feelings > 0 else 0.0
        
        # Calculate average confidence from recent gut feelings
        recent_gut_feelings = []
        for entry in self.gut_feeling_history[-10:]:
            if 'gut_feeling' in entry and hasattr(entry['gut_feeling'], 'confidence'):
                recent_gut_feelings.append(entry['gut_feeling'].confidence)
        
        average_confidence = np.mean(recent_gut_feelings) if recent_gut_feelings else 0.0
        
        return {
            'total_gut_feelings': total_gut_feelings,
            'success_rate': success_rate,
            'total_patterns': len(self.patterns),
            'pattern_types': list(self.pattern_index.keys()),
            'action_coverage': len(self.action_patterns),
            'average_confidence': average_confidence
        }

# Factory function for easy integration
def create_enhanced_gut_feeling_engine(**kwargs) -> EnhancedGutFeelingEngine:
    """Create a configured enhanced gut feeling engine."""
    return EnhancedGutFeelingEngine(**kwargs)

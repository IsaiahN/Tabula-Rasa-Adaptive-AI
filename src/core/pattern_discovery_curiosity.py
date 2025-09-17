#!/usr/bin/env python3
"""
Pattern Discovery Curiosity - Advanced Cognitive Architecture Enhancement

Implements intellectual curiosity via pattern discovery based on Nagashima et al.
research, using compression-based intrinsic rewards and utility learning for
enhanced exploration and learning.

Key Features:
- Pattern-based intrinsic rewards using compression algorithms
- Utility learning with ACT-R-like production compilation
- Multi-level curiosity (sensory + intellectual)
- Integration with existing curiosity and pattern matching systems
- Compression-based confidence scoring for gut feeling engine

This system rewards the agent for discovering compressible patterns in the
environment, leading to more intelligent exploration and learning.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import gzip
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hashlib

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of patterns that can be discovered."""
    SYMMETRY = "symmetry"
    SEQUENCE = "sequence"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    SPATIAL_PATTERN = "spatial_pattern"
    TEMPORAL_PATTERN = "temporal_pattern"
    HIERARCHICAL_STRUCTURE = "hierarchical_structure"
    RECURRING_THEME = "recurring_theme"

class CuriosityLevel(Enum):
    """Levels of curiosity for different types of exploration."""
    SENSORY = "sensory"  # Novelty-seeking
    INTELLECTUAL = "intellectual"  # Pattern discovery
    META_COGNITIVE = "meta_cognitive"  # Understanding understanding

@dataclass
class DiscoveredPattern:
    """Represents a pattern discovered in the environment."""
    pattern_type: PatternType
    pattern_data: np.ndarray
    compression_ratio: float
    confidence: float
    utility_score: float
    discovery_time: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    pattern_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])

@dataclass
class CuriosityEvent:
    """Represents a curiosity-driven event or discovery."""
    event_type: str
    intensity: float
    learning_potential: float
    pattern_discovered: Optional[DiscoveredPattern] = None
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

class CompressionRewarder:
    """Rewards system for discovering compressible patterns using various compression algorithms."""
    
    def __init__(self, 
                 compression_methods: List[str] = None,
                 min_compression_ratio: float = 0.1,
                 max_compression_ratio: float = 0.9):
        
        self.compression_methods = compression_methods or ['gzip', 'pca', 'clustering']
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        self.compression_history = deque(maxlen=1000)
        
    def compute_compression_reward(self, 
                                 data: np.ndarray, 
                                 pattern_type: PatternType) -> Tuple[float, Dict[str, Any]]:
        """Compute compression-based reward for discovered patterns."""
        
        rewards = {}
        total_reward = 0.0
        
        # GZIP compression
        if 'gzip' in self.compression_methods:
            gzip_ratio = self._gzip_compression_ratio(data)
            rewards['gzip_ratio'] = gzip_ratio
            if self.min_compression_ratio <= gzip_ratio <= self.max_compression_ratio:
                total_reward += gzip_ratio * 0.4
        
        # PCA compression
        if 'pca' in self.compression_methods:
            pca_ratio = self._pca_compression_ratio(data)
            rewards['pca_ratio'] = pca_ratio
            if self.min_compression_ratio <= pca_ratio <= self.max_compression_ratio:
                total_reward += pca_ratio * 0.3
        
        # Clustering compression
        if 'clustering' in self.compression_methods:
            cluster_ratio = self._clustering_compression_ratio(data)
            rewards['cluster_ratio'] = cluster_ratio
            if self.min_compression_ratio <= cluster_ratio <= self.max_compression_ratio:
                total_reward += cluster_ratio * 0.3
        
        # Pattern type bonus
        pattern_bonus = self._get_pattern_type_bonus(pattern_type)
        rewards['pattern_bonus'] = pattern_bonus
        total_reward += pattern_bonus
        
        # Store compression history
        self.compression_history.append({
            'timestamp': time.time(),
            'pattern_type': pattern_type.value,
            'rewards': rewards,
            'total_reward': total_reward
        })
        
        return total_reward, rewards
    
    def _gzip_compression_ratio(self, data: np.ndarray) -> float:
        """Compute GZIP compression ratio."""
        try:
            # Convert to bytes
            data_bytes = data.tobytes()
            compressed = gzip.compress(data_bytes)
            return len(compressed) / len(data_bytes)
        except Exception:
            return 1.0
    
    def _pca_compression_ratio(self, data: np.ndarray) -> float:
        """Compute PCA compression ratio."""
        try:
            if data.shape[0] < 2 or data.shape[1] < 2:
                return 1.0
            
            # Fit PCA with 95% variance retention
            pca = PCA(n_components=0.95)
            pca.fit(data)
            
            # Compression ratio based on explained variance
            explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
            return 1.0 - explained_variance_ratio
        except Exception:
            return 1.0
    
    def _clustering_compression_ratio(self, data: np.ndarray) -> float:
        """Compute clustering compression ratio."""
        try:
            if data.shape[0] < 4:
                return 1.0
            
            # Use K-means clustering
            n_clusters = min(4, data.shape[0] // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(data)
            
            # Compression ratio based on cluster quality
            inertia = kmeans.inertia_
            max_inertia = np.sum(np.var(data, axis=0)) * data.shape[0]
            return inertia / max_inertia if max_inertia > 0 else 1.0
        except Exception:
            return 1.0
    
    def _get_pattern_type_bonus(self, pattern_type: PatternType) -> float:
        """Get bonus reward based on pattern type complexity."""
        bonuses = {
            PatternType.SYMMETRY: 0.1,
            PatternType.SEQUENCE: 0.15,
            PatternType.CAUSAL_RELATIONSHIP: 0.25,
            PatternType.SPATIAL_PATTERN: 0.12,
            PatternType.TEMPORAL_PATTERN: 0.18,
            PatternType.HIERARCHICAL_STRUCTURE: 0.3,
            PatternType.RECURRING_THEME: 0.2
        }
        return bonuses.get(pattern_type, 0.1)

class PatternClassifier:
    """Classifies patterns in data using various detection algorithms."""
    
    def __init__(self):
        self.pattern_detectors = {
            PatternType.SYMMETRY: self._detect_symmetry,
            PatternType.SEQUENCE: self._detect_sequence,
            PatternType.CAUSAL_RELATIONSHIP: self._detect_causal_relationship,
            PatternType.SPATIAL_PATTERN: self._detect_spatial_pattern,
            PatternType.TEMPORAL_PATTERN: self._detect_temporal_pattern,
            PatternType.HIERARCHICAL_STRUCTURE: self._detect_hierarchical_structure,
            PatternType.RECURRING_THEME: self._detect_recurring_theme
        }
    
    def detect_patterns(self, data: np.ndarray, context: Dict[str, Any] = None) -> List[DiscoveredPattern]:
        """Detect patterns in the given data."""
        
        patterns = []
        context = context or {}
        
        for pattern_type, detector in self.pattern_detectors.items():
            try:
                pattern_data, confidence = detector(data, context)
                if confidence > 0.5:  # Threshold for pattern detection
                    patterns.append(DiscoveredPattern(
                        pattern_type=pattern_type,
                        pattern_data=pattern_data,
                        compression_ratio=0.0,  # Will be computed later
                        confidence=confidence,
                        utility_score=0.0,  # Will be computed later
                        context=context
                    ))
            except Exception as e:
                logger.warning(f"Pattern detection failed for {pattern_type}: {e}")
        
        return patterns
    
    def _detect_symmetry(self, data: np.ndarray, context: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Detect symmetry patterns."""
        if len(data.shape) != 2:
            return data, 0.0
        
        # Check for horizontal symmetry
        if data.shape[0] > 1:
            top_half = data[:data.shape[0]//2]
            bottom_half = data[data.shape[0]//2:]
            if bottom_half.shape[0] == top_half.shape[0]:
                horizontal_symmetry = 1.0 - np.mean(np.abs(top_half - np.flipud(bottom_half)))
            else:
                horizontal_symmetry = 0.0
        else:
            horizontal_symmetry = 0.0
        
        # Check for vertical symmetry
        if data.shape[1] > 1:
            left_half = data[:, :data.shape[1]//2]
            right_half = data[:, data.shape[1]//2:]
            if right_half.shape[1] == left_half.shape[1]:
                vertical_symmetry = 1.0 - np.mean(np.abs(left_half - np.fliplr(right_half)))
            else:
                vertical_symmetry = 0.0
        else:
            vertical_symmetry = 0.0
        
        confidence = max(horizontal_symmetry, vertical_symmetry)
        return data, confidence
    
    def _detect_sequence(self, data: np.ndarray, context: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Detect sequence patterns."""
        if len(data.shape) != 1:
            return data, 0.0
        
        # Check for arithmetic sequences
        if len(data) > 2:
            diffs = np.diff(data)
            if len(set(diffs)) == 1:  # All differences are the same
                return data, 0.9
        
        # Check for geometric sequences
        if len(data) > 2 and np.all(data != 0):
            ratios = data[1:] / data[:-1]
            if len(set(ratios)) == 1:  # All ratios are the same
                return data, 0.8
        
        return data, 0.0
    
    def _detect_causal_relationship(self, data: np.ndarray, context: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Detect causal relationships (simplified)."""
        # This is a placeholder - real causal detection would be much more complex
        if 'action_sequence' in context and 'outcome_sequence' in context:
            actions = context['action_sequence']
            outcomes = context['outcome_sequence']
            if len(actions) == len(outcomes) and len(actions) > 1:
                # Simple correlation-based causal detection
                correlation = np.corrcoef(actions, outcomes)[0, 1]
                if not np.isnan(correlation):
                    return data, abs(correlation)
        
        return data, 0.0
    
    def _detect_spatial_pattern(self, data: np.ndarray, context: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Detect spatial patterns."""
        if len(data.shape) != 2:
            return data, 0.0
        
        # Check for regular grid patterns
        if data.shape[0] > 2 and data.shape[1] > 2:
            # Look for repeating sub-patterns
            pattern_size = min(data.shape) // 2
            if pattern_size > 1:
                sub_patterns = []
                for i in range(0, data.shape[0] - pattern_size + 1, pattern_size):
                    for j in range(0, data.shape[1] - pattern_size + 1, pattern_size):
                        sub_patterns.append(data[i:i+pattern_size, j:j+pattern_size])
                
                if len(sub_patterns) > 1:
                    # Check similarity between sub-patterns
                    similarities = []
                    for i in range(len(sub_patterns)):
                        for j in range(i + 1, len(sub_patterns)):
                            sim = 1.0 - np.mean(np.abs(sub_patterns[i] - sub_patterns[j]))
                            similarities.append(sim)
                    
                    if similarities:
                        confidence = np.mean(similarities)
                        return data, confidence
        
        return data, 0.0
    
    def _detect_temporal_pattern(self, data: np.ndarray, context: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Detect temporal patterns."""
        if 'temporal_sequence' in context:
            sequence = context['temporal_sequence']
            if len(sequence) > 3:
                # Look for periodic patterns
                for period in range(2, len(sequence) // 2 + 1):
                    if len(sequence) % period == 0:
                        periods = [sequence[i:i+period] for i in range(0, len(sequence), period)]
                        if len(periods) > 1:
                            similarities = []
                            for i in range(len(periods)):
                                for j in range(i + 1, len(periods)):
                                    sim = 1.0 - np.mean(np.abs(np.array(periods[i]) - np.array(periods[j])))
                                    similarities.append(sim)
                            
                            if similarities and np.mean(similarities) > 0.7:
                                return data, np.mean(similarities)
        
        return data, 0.0
    
    def _detect_hierarchical_structure(self, data: np.ndarray, context: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Detect hierarchical structures."""
        # This is a placeholder - real hierarchical detection would be complex
        if len(data.shape) == 2 and data.shape[0] > 2 and data.shape[1] > 2:
            # Simple hierarchical detection based on nested patterns
            center = data.shape[0] // 2, data.shape[1] // 2
            inner_region = data[center[0]-1:center[0]+2, center[1]-1:center[1]+2]
            outer_region = data
            
            if inner_region.shape[0] == 3 and inner_region.shape[1] == 3:
                # Check if inner region has different characteristics
                inner_mean = np.mean(inner_region)
                outer_mean = np.mean(outer_region)
                if abs(inner_mean - outer_mean) > 0.1:
                    return data, 0.6
        
        return data, 0.0
    
    def _detect_recurring_theme(self, data: np.ndarray, context: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Detect recurring themes."""
        if 'theme_history' in context:
            themes = context['theme_history']
            if len(themes) > 1:
                # Check for recurring patterns in themes
                current_theme = str(data)
                theme_counts = defaultdict(int)
                for theme in themes:
                    theme_counts[theme] += 1
                
                if current_theme in theme_counts and theme_counts[current_theme] > 1:
                    return data, min(0.9, theme_counts[current_theme] * 0.3)
        
        return data, 0.0

class UtilityLearner:
    """ACT-R-like utility learning for pattern-matching strategies."""
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 decay_rate: float = 0.95,
                 max_productions: int = 1000):
        
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.max_productions = max_productions
        
        # Production rules: pattern -> action -> utility
        self.productions = {}
        self.production_history = deque(maxlen=max_productions)
        
    def learn_from_pattern(self, 
                          pattern: DiscoveredPattern,
                          action_taken: int,
                          outcome: Dict[str, Any],
                          context: Dict[str, Any]):
        """Learn utility from pattern-action-outcome triple."""
        
        pattern_key = f"{pattern.pattern_type.value}_{pattern.pattern_id}"
        
        if pattern_key not in self.productions:
            self.productions[pattern_key] = {}
        
        if action_taken not in self.productions[pattern_key]:
            self.productions[pattern_key][action_taken] = {
                'utility': 0.5,  # Initial utility
                'success_count': 0,
                'total_count': 0,
                'last_updated': time.time()
            }
        
        production = self.productions[pattern_key][action_taken]
        
        # Update utility based on outcome
        success = outcome.get('success', False)
        reward = outcome.get('reward', 0.0)
        
        production['total_count'] += 1
        if success:
            production['success_count'] += 1
        
        # Compute utility update
        utility_update = self.learning_rate * (reward - production['utility'])
        production['utility'] += utility_update
        production['last_updated'] = time.time()
        
        # Store in history
        self.production_history.append({
            'pattern_key': pattern_key,
            'action': action_taken,
            'outcome': outcome,
            'utility': production['utility'],
            'timestamp': time.time()
        })
    
    def get_action_utility(self, pattern: DiscoveredPattern, action: int) -> float:
        """Get utility of an action for a given pattern."""
        pattern_key = f"{pattern.pattern_type.value}_{pattern.pattern_id}"
        
        if pattern_key in self.productions and action in self.productions[pattern_key]:
            return self.productions[pattern_key][action]['utility']
        
        return 0.5  # Default utility
    
    def decay_utilities(self):
        """Decay utilities over time (forgetting)."""
        current_time = time.time()
        
        for pattern_key, actions in self.productions.items():
            for action, production in actions.items():
                time_since_update = current_time - production['last_updated']
                if time_since_update > 3600:  # 1 hour
                    production['utility'] *= self.decay_rate
                    production['last_updated'] = current_time

class PatternDiscoveryCuriosity:
    """
    Main Pattern Discovery Curiosity system.
    
    Integrates compression-based rewards, pattern detection, and utility learning
    to drive intelligent exploration and learning.
    """
    
    def __init__(self,
                 compression_methods: List[str] = None,
                 learning_rate: float = 0.1,
                 curiosity_decay: float = 0.99,
                 max_patterns: int = 1000):
        
        # Initialize components
        self.compression_rewarder = CompressionRewarder(compression_methods)
        self.pattern_classifier = PatternClassifier()
        self.utility_learner = UtilityLearner(learning_rate)
        
        # Configuration
        self.curiosity_decay = curiosity_decay
        self.max_patterns = max_patterns
        
        # State tracking
        self.discovered_patterns = deque(maxlen=max_patterns)
        self.curiosity_events = deque(maxlen=1000)
        self.curiosity_levels = {
            CuriosityLevel.SENSORY: 0.5,
            CuriosityLevel.INTELLECTUAL: 0.5,
            CuriosityLevel.META_COGNITIVE: 0.3
        }
        
        # Integration components
        self.enhanced_curiosity_system = None
        self.gut_feeling_engine = None
        self.tree_based_architect = None
        
        logger.info("PatternDiscoveryCuriosity initialized")
    
    def integrate_components(self,
                           enhanced_curiosity_system=None,
                           gut_feeling_engine=None,
                           tree_based_architect=None):
        """Integrate with existing Tabula Rasa components."""
        self.enhanced_curiosity_system = enhanced_curiosity_system
        self.gut_feeling_engine = gut_feeling_engine
        self.tree_based_architect = tree_based_architect
        logger.info("PatternDiscoveryCuriosity integrated with existing components")
    
    def process_observation(self, 
                          observation: np.ndarray,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process an observation for pattern discovery and curiosity generation."""
        
        context = context or {}
        
        # Detect patterns in observation
        patterns = self.pattern_classifier.detect_patterns(observation, context)
        
        # Process each discovered pattern
        pattern_rewards = {}
        total_curiosity_boost = 0.0
        
        for pattern in patterns:
            # Compute compression rewards
            compression_reward, compression_details = self.compression_rewarder.compute_compression_reward(
                pattern.pattern_data, pattern.pattern_type
            )
            
            # Update pattern with compression info
            pattern.compression_ratio = compression_details.get('gzip_ratio', 0.0)
            pattern.utility_score = compression_reward
            
            # Store pattern
            self.discovered_patterns.append(pattern)
            
            # Generate curiosity event
            curiosity_event = CuriosityEvent(
                event_type=f"pattern_discovered_{pattern.pattern_type.value}",
                intensity=compression_reward,
                learning_potential=pattern.confidence * compression_reward,
                pattern_discovered=pattern,
                context=context
            )
            self.curiosity_events.append(curiosity_event)
            
            # Update curiosity levels
            self._update_curiosity_levels(pattern, compression_reward)
            
            # Store pattern rewards
            pattern_rewards[pattern.pattern_id] = {
                'pattern_type': pattern.pattern_type.value,
                'compression_reward': compression_reward,
                'confidence': pattern.confidence,
                'utility_score': pattern.utility_score
            }
            
            total_curiosity_boost += compression_reward
        
        # Update utility learner
        self.utility_learner.decay_utilities()
        
        # Generate intrinsic rewards
        intrinsic_rewards = self._generate_intrinsic_rewards(observation, patterns, context)
        
        return {
            'patterns_discovered': len(patterns),
            'pattern_rewards': pattern_rewards,
            'total_curiosity_boost': total_curiosity_boost,
            'intrinsic_rewards': intrinsic_rewards,
            'curiosity_levels': self.curiosity_levels.copy(),
            'recent_patterns': [p.pattern_type.value for p in list(self.discovered_patterns)[-5:]]
        }
    
    def learn_from_action_outcome(self,
                                pattern: DiscoveredPattern,
                                action_taken: int,
                                outcome: Dict[str, Any],
                                context: Dict[str, Any]):
        """Learn from pattern-action-outcome experience."""
        
        self.utility_learner.learn_from_pattern(pattern, action_taken, outcome, context)
        
        # Update pattern utility based on outcome
        pattern.utility_score = self.utility_learner.get_action_utility(pattern, action_taken)
    
    def get_pattern_confidence(self, pattern_data: np.ndarray, pattern_type: PatternType) -> float:
        """Get confidence score for pattern matching (for gut feeling engine)."""
        
        # Detect patterns in data
        patterns = self.pattern_classifier.detect_patterns(pattern_data)
        
        # Find matching pattern type
        matching_patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        if matching_patterns:
            # Return highest confidence pattern
            best_pattern = max(matching_patterns, key=lambda p: p.confidence)
            return best_pattern.confidence * best_pattern.utility_score
        
        return 0.0
    
    def _update_curiosity_levels(self, pattern: DiscoveredPattern, reward: float):
        """Update curiosity levels based on pattern discovery."""
        
        # Intellectual curiosity (pattern discovery)
        self.curiosity_levels[CuriosityLevel.INTELLECTUAL] = min(1.0, 
            self.curiosity_levels[CuriosityLevel.INTELLECTUAL] + reward * 0.1)
        
        # Meta-cognitive curiosity (understanding patterns)
        if pattern.pattern_type in [PatternType.CAUSAL_RELATIONSHIP, PatternType.HIERARCHICAL_STRUCTURE]:
            self.curiosity_levels[CuriosityLevel.META_COGNITIVE] = min(1.0,
                self.curiosity_levels[CuriosityLevel.META_COGNITIVE] + reward * 0.05)
        
        # Decay all curiosity levels
        for level in self.curiosity_levels:
            self.curiosity_levels[level] *= self.curiosity_decay
    
    def _generate_intrinsic_rewards(self, 
                                  observation: np.ndarray,
                                  patterns: List[DiscoveredPattern],
                                  context: Dict[str, Any]) -> Dict[str, float]:
        """Generate intrinsic rewards based on pattern discovery."""
        
        rewards = {}
        
        # Pattern discovery reward
        if patterns:
            total_pattern_reward = sum(p.utility_score for p in patterns)
            rewards['pattern_discovery'] = total_pattern_reward * 0.1
        
        # Compression reward
        if patterns:
            avg_compression = np.mean([p.compression_ratio for p in patterns])
            rewards['compression'] = avg_compression * 0.05
        
        # Curiosity level rewards
        rewards['intellectual_curiosity'] = self.curiosity_levels[CuriosityLevel.INTELLECTUAL] * 0.02
        rewards['meta_cognitive_curiosity'] = self.curiosity_levels[CuriosityLevel.META_COGNITIVE] * 0.01
        
        # Novelty reward (based on pattern diversity)
        if len(patterns) > 1:
            pattern_types = set(p.pattern_type for p in patterns)
            diversity_reward = len(pattern_types) / len(PatternType) * 0.03
            rewards['pattern_diversity'] = diversity_reward
        
        return rewards
    
    def get_curiosity_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics about pattern discovery curiosity."""
        
        return {
            'curiosity_levels': self.curiosity_levels.copy(),
            'total_patterns_discovered': len(self.discovered_patterns),
            'pattern_type_counts': {
                pattern_type.value: sum(1 for p in self.discovered_patterns if p.pattern_type == pattern_type)
                for pattern_type in PatternType
            },
            'recent_curiosity_events': len([e for e in self.curiosity_events if time.time() - e.timestamp < 300]),
            'avg_pattern_confidence': np.mean([p.confidence for p in list(self.discovered_patterns)[-100:]]) if self.discovered_patterns else 0.0,
            'avg_utility_score': np.mean([p.utility_score for p in list(self.discovered_patterns)[-100:]]) if self.discovered_patterns else 0.0,
            'compression_rewarder_stats': {
                'total_compressions': len(self.compression_rewarder.compression_history),
                'avg_gzip_ratio': np.mean([h['rewards'].get('gzip_ratio', 0) for h in self.compression_rewarder.compression_history]) if self.compression_rewarder.compression_history else 0.0
            },
            'utility_learner_stats': {
                'total_productions': len(self.utility_learner.productions),
                'total_learning_events': len(self.utility_learner.production_history)
            }
        }

# Factory function for easy integration
def create_pattern_discovery_curiosity(**kwargs) -> PatternDiscoveryCuriosity:
    """Create a configured pattern discovery curiosity system."""
    return PatternDiscoveryCuriosity(**kwargs)

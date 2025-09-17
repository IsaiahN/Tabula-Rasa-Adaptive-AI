#!/usr/bin/env python3
"""
Enhanced Architectural Systems - Advanced Cognitive Architecture Enhancement

Enhances existing Tree-Based Director, Tree-Based Architect, and Implicit Memory Manager
with new capabilities from the high-priority enhancements.

Key Features:
- Enhanced Tree-Based Director with self-prior and curiosity guidance
- Enhanced Tree-Based Architect with intrinsic motivation integration
- Enhanced Implicit Memory Manager with motivational significance clustering
- Integration with Self-Prior Mechanism and Pattern Discovery Curiosity
- Improved goal decomposition and architectural evolution

This module builds on the existing solid foundation to add advanced capabilities
without requiring major architectural changes.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)

class EnhancementType(Enum):
    """Types of architectural enhancements."""
    SELF_PRIOR_GUIDANCE = "self_prior_guidance"
    PATTERN_DISCOVERY_CURIOSITY = "pattern_discovery_curiosity"
    INTRINSIC_MOTIVATION = "intrinsic_motivation"
    MOTIVATIONAL_CLUSTERING = "motivational_clustering"

@dataclass
class EnhancedGoal:
    """Enhanced goal with additional metadata from self-prior and curiosity."""
    original_goal: Any
    self_prior_alignment: float
    curiosity_drive: float
    pattern_relevance: float
    intrinsic_motivation: float
    priority_score: float
    reasoning: str
    enhancement_type: EnhancementType

@dataclass
class ArchitecturalEnhancement:
    """Represents an architectural enhancement with motivation tracking."""
    enhancement_type: EnhancementType
    motivation_score: float
    success_rate: float
    complexity_score: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

class EnhancedTreeBasedDirector:
    """
    Enhanced Tree-Based Director with self-prior and curiosity guidance.
    
    Extends the existing Tree-Based Director to incorporate self-prior alignment
    and pattern discovery curiosity into goal decomposition and reasoning.
    """
    
    def __init__(self, 
                 original_director,
                 self_prior_manager=None,
                 pattern_discovery_curiosity=None,
                 enhancement_weight: float = 0.3):
        
        self.original_director = original_director
        self.self_prior_manager = self_prior_manager
        self.pattern_discovery_curiosity = pattern_discovery_curiosity
        self.enhancement_weight = enhancement_weight
        
        # Enhancement tracking
        self.enhancement_history = deque(maxlen=1000)
        self.goal_enhancements = deque(maxlen=500)
        
        # Integration state
        self.last_self_prior_state = None
        self.last_curiosity_state = None
        
        logger.info("EnhancedTreeBasedDirector initialized with self-prior and curiosity guidance")
    
    def enhanced_goal_decomposition(self, 
                                  original_goal: Any,
                                  context: Dict[str, Any]) -> List[EnhancedGoal]:
        """Enhanced goal decomposition with self-prior and curiosity guidance."""
        
        # Get original goal decomposition
        original_subgoals = self.original_director.decompose_goal(original_goal, context)
        
        # Enhance each subgoal
        enhanced_subgoals = []
        
        for subgoal in original_subgoals:
            enhanced_goal = self._enhance_goal(subgoal, context)
            enhanced_subgoals.append(enhanced_goal)
        
        # Sort by priority score
        enhanced_subgoals.sort(key=lambda g: g.priority_score, reverse=True)
        
        # Store enhancement
        self.goal_enhancements.append({
            'original_goal': original_goal,
            'enhanced_subgoals': enhanced_subgoals,
            'timestamp': time.time(),
            'context': context
        })
        
        return enhanced_subgoals
    
    def _enhance_goal(self, subgoal: Any, context: Dict[str, Any]) -> EnhancedGoal:
        """Enhance a single goal with self-prior and curiosity information."""
        
        # Initialize enhancement scores
        self_prior_alignment = 0.5
        curiosity_drive = 0.5
        pattern_relevance = 0.5
        intrinsic_motivation = 0.5
        
        # Get self-prior alignment if available
        if self.self_prior_manager:
            try:
                self_prior_metrics = self.self_prior_manager.get_self_prior_metrics()
                self_prior_alignment = self_prior_metrics.get('self_awareness_score', 0.5)
                
                # Check if goal aligns with active intrinsic goals
                active_goals = self_prior_metrics.get('active_goals_count', 0)
                if active_goals > 0:
                    self_prior_alignment = min(1.0, self_prior_alignment + 0.2)
            except Exception as e:
                logger.warning(f"Self-prior integration failed: {e}")
        
        # Get curiosity drive if available
        if self.pattern_discovery_curiosity:
            try:
                curiosity_metrics = self.pattern_discovery_curiosity.get_curiosity_metrics()
                curiosity_drive = curiosity_metrics.get('curiosity_levels', {}).get('intellectual', 0.5)
                
                # Check pattern relevance
                if 'observation' in context:
                    observation = context['observation']
                    pattern_relevance = self._compute_pattern_relevance(observation, subgoal)
            except Exception as e:
                logger.warning(f"Pattern discovery curiosity integration failed: {e}")
        
        # Compute intrinsic motivation
        intrinsic_motivation = (self_prior_alignment + curiosity_drive + pattern_relevance) / 3.0
        
        # Compute priority score
        priority_score = (
            getattr(subgoal, 'priority', 0.5) * 0.4 +
            self_prior_alignment * 0.2 +
            curiosity_drive * 0.2 +
            pattern_relevance * 0.1 +
            intrinsic_motivation * 0.1
        )
        
        # Generate reasoning
        reasoning = self._generate_enhancement_reasoning(
            subgoal, self_prior_alignment, curiosity_drive, pattern_relevance, intrinsic_motivation
        )
        
        return EnhancedGoal(
            original_goal=subgoal,
            self_prior_alignment=self_prior_alignment,
            curiosity_drive=curiosity_drive,
            pattern_relevance=pattern_relevance,
            intrinsic_motivation=intrinsic_motivation,
            priority_score=priority_score,
            reasoning=reasoning,
            enhancement_type=EnhancementType.SELF_PRIOR_GUIDANCE
        )
    
    def _compute_pattern_relevance(self, observation: np.ndarray, subgoal: Any) -> float:
        """Compute how relevant the observation is to the subgoal based on patterns."""
        
        if not self.pattern_discovery_curiosity:
            return 0.5
        
        try:
            # Process observation for pattern discovery
            pattern_result = self.pattern_discovery_curiosity.process_observation(observation)
            
            # Get pattern relevance based on discovered patterns
            patterns_discovered = pattern_result.get('patterns_discovered', 0)
            total_curiosity_boost = pattern_result.get('total_curiosity_boost', 0.0)
            
            # Higher pattern discovery = higher relevance
            relevance = min(1.0, patterns_discovered * 0.1 + total_curiosity_boost * 0.5)
            
            return relevance
        except Exception as e:
            logger.warning(f"Pattern relevance computation failed: {e}")
            return 0.5
    
    def _generate_enhancement_reasoning(self, 
                                      subgoal: Any,
                                      self_prior_alignment: float,
                                      curiosity_drive: float,
                                      pattern_relevance: float,
                                      intrinsic_motivation: float) -> str:
        """Generate reasoning for goal enhancement."""
        
        reasoning_parts = []
        
        if self_prior_alignment > 0.7:
            reasoning_parts.append("high self-prior alignment")
        elif self_prior_alignment < 0.3:
            reasoning_parts.append("low self-prior alignment")
        
        if curiosity_drive > 0.7:
            reasoning_parts.append("high curiosity drive")
        elif curiosity_drive < 0.3:
            reasoning_parts.append("low curiosity drive")
        
        if pattern_relevance > 0.7:
            reasoning_parts.append("high pattern relevance")
        elif pattern_relevance < 0.3:
            reasoning_parts.append("low pattern relevance")
        
        if intrinsic_motivation > 0.7:
            reasoning_parts.append("high intrinsic motivation")
        elif intrinsic_motivation < 0.3:
            reasoning_parts.append("low intrinsic motivation")
        
        if reasoning_parts:
            return f"Enhanced goal: {', '.join(reasoning_parts)}"
        else:
            return "Enhanced goal: moderate enhancement scores"
    
    def get_enhancement_metrics(self) -> Dict[str, Any]:
        """Get metrics about goal enhancement performance."""
        
        if not self.goal_enhancements:
            return {
                'total_enhanced_goals': 0,
                'avg_self_prior_alignment': 0.0,
                'avg_curiosity_drive': 0.0,
                'avg_pattern_relevance': 0.0,
                'avg_intrinsic_motivation': 0.0,
                'avg_priority_score': 0.0
            }
        
        recent_enhancements = list(self.goal_enhancements)[-100:]  # Last 100
        
        all_goals = []
        for enhancement in recent_enhancements:
            all_goals.extend(enhancement['enhanced_subgoals'])
        
        if not all_goals:
            return {
                'total_enhanced_goals': 0,
                'avg_self_prior_alignment': 0.0,
                'avg_curiosity_drive': 0.0,
                'avg_pattern_relevance': 0.0,
                'avg_intrinsic_motivation': 0.0,
                'avg_priority_score': 0.0
            }
        
        return {
            'total_enhanced_goals': len(all_goals),
            'avg_self_prior_alignment': np.mean([g.self_prior_alignment for g in all_goals]),
            'avg_curiosity_drive': np.mean([g.curiosity_drive for g in all_goals]),
            'avg_pattern_relevance': np.mean([g.pattern_relevance for g in all_goals]),
            'avg_intrinsic_motivation': np.mean([g.intrinsic_motivation for g in all_goals]),
            'avg_priority_score': np.mean([g.priority_score for g in all_goals])
        }

class EnhancedTreeBasedArchitect:
    """
    Enhanced Tree-Based Architect with intrinsic motivation integration.
    
    Extends the existing Tree-Based Architect to incorporate intrinsic motivation
    and pattern discovery into architectural evolution decisions.
    """
    
    def __init__(self, 
                 original_architect,
                 self_prior_manager=None,
                 pattern_discovery_curiosity=None,
                 motivation_weight: float = 0.2):
        
        self.original_architect = original_architect
        self.self_prior_manager = self_prior_manager
        self.pattern_discovery_curiosity = pattern_discovery_curiosity
        self.motivation_weight = motivation_weight
        
        # Enhancement tracking
        self.architectural_enhancements = deque(maxlen=1000)
        self.motivation_history = deque(maxlen=500)
        
        logger.info("EnhancedTreeBasedArchitect initialized with intrinsic motivation integration")
    
    def enhanced_architectural_evolution(self, 
                                       current_architecture: Any,
                                       performance_metrics: Dict[str, Any],
                                       context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Enhanced architectural evolution with intrinsic motivation."""
        
        # Get original architectural evolution
        evolved_architecture, original_metadata = self.original_architect.evolve_architecture(
            current_architecture, performance_metrics, context
        )
        
        # Compute intrinsic motivation factors
        intrinsic_factors = self._compute_intrinsic_factors(context)
        
        # Enhance architectural decision based on intrinsic motivation
        enhanced_architecture, enhancement_metadata = self._apply_intrinsic_enhancements(
            evolved_architecture, intrinsic_factors, context
        )
        
        # Combine metadata
        combined_metadata = {
            **original_metadata,
            'intrinsic_factors': intrinsic_factors,
            'enhancement_metadata': enhancement_metadata,
            'motivation_weight': self.motivation_weight
        }
        
        # Store enhancement
        self.architectural_enhancements.append({
            'original_architecture': current_architecture,
            'enhanced_architecture': enhanced_architecture,
            'intrinsic_factors': intrinsic_factors,
            'timestamp': time.time(),
            'context': context
        })
        
        return enhanced_architecture, combined_metadata
    
    def _compute_intrinsic_factors(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Compute intrinsic motivation factors for architectural evolution."""
        
        factors = {
            'self_prior_alignment': 0.5,
            'curiosity_drive': 0.5,
            'pattern_discovery_urge': 0.5,
            'exploration_motivation': 0.5,
            'learning_drive': 0.5
        }
        
        # Get self-prior alignment
        if self.self_prior_manager:
            try:
                self_prior_metrics = self.self_prior_manager.get_self_prior_metrics()
                factors['self_prior_alignment'] = self_prior_metrics.get('self_awareness_score', 0.5)
            except Exception as e:
                logger.warning(f"Self-prior factor computation failed: {e}")
        
        # Get curiosity and pattern discovery factors
        if self.pattern_discovery_curiosity:
            try:
                curiosity_metrics = self.pattern_discovery_curiosity.get_curiosity_metrics()
                curiosity_levels = curiosity_metrics.get('curiosity_levels', {})
                
                factors['curiosity_drive'] = curiosity_levels.get('intellectual', 0.5)
                factors['pattern_discovery_urge'] = curiosity_levels.get('meta_cognitive', 0.5)
                
                # Exploration motivation based on pattern discovery
                patterns_discovered = curiosity_metrics.get('total_patterns_discovered', 0)
                factors['exploration_motivation'] = min(1.0, patterns_discovered * 0.01)
                
            except Exception as e:
                logger.warning(f"Curiosity factor computation failed: {e}")
        
        # Learning drive from context
        factors['learning_drive'] = context.get('learning_progress', 0.5)
        
        return factors
    
    def _apply_intrinsic_enhancements(self, 
                                    architecture: Any,
                                    intrinsic_factors: Dict[str, float],
                                    context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Apply intrinsic motivation enhancements to architecture."""
        
        # This is a placeholder - in a real implementation, this would modify
        # the architecture based on intrinsic factors
        
        enhancement_metadata = {
            'intrinsic_modifications': [],
            'motivation_impact': 0.0,
            'enhancement_confidence': 0.0
        }
        
        # Compute overall motivation impact
        motivation_impact = np.mean(list(intrinsic_factors.values()))
        enhancement_metadata['motivation_impact'] = motivation_impact
        
        # Apply modifications based on intrinsic factors
        modifications = []
        
        # High self-prior alignment -> more conservative evolution
        if intrinsic_factors['self_prior_alignment'] > 0.7:
            modifications.append("conservative_evolution")
            enhancement_metadata['intrinsic_modifications'].append("conservative_evolution")
        
        # High curiosity drive -> more exploratory evolution
        if intrinsic_factors['curiosity_drive'] > 0.7:
            modifications.append("exploratory_evolution")
            enhancement_metadata['intrinsic_modifications'].append("exploratory_evolution")
        
        # High pattern discovery urge -> more pattern-focused evolution
        if intrinsic_factors['pattern_discovery_urge'] > 0.7:
            modifications.append("pattern_focused_evolution")
            enhancement_metadata['intrinsic_modifications'].append("pattern_focused_evolution")
        
        # High exploration motivation -> more diverse evolution
        if intrinsic_factors['exploration_motivation'] > 0.7:
            modifications.append("diverse_evolution")
            enhancement_metadata['intrinsic_modifications'].append("diverse_evolution")
        
        # High learning drive -> more aggressive evolution
        if intrinsic_factors['learning_drive'] > 0.7:
            modifications.append("aggressive_evolution")
            enhancement_metadata['intrinsic_modifications'].append("aggressive_evolution")
        
        enhancement_metadata['enhancement_confidence'] = len(modifications) / 5.0
        
        return architecture, enhancement_metadata
    
    def get_enhancement_metrics(self) -> Dict[str, Any]:
        """Get metrics about architectural enhancement performance."""
        
        if not self.architectural_enhancements:
            return {
                'total_enhancements': 0,
                'avg_motivation_impact': 0.0,
                'avg_enhancement_confidence': 0.0,
                'common_modifications': []
            }
        
        recent_enhancements = list(self.architectural_enhancements)[-100:]  # Last 100
        
        # Compute average metrics
        avg_motivation_impact = np.mean([
            e['intrinsic_factors'].get('self_prior_alignment', 0.5) for e in recent_enhancements
        ])
        
        # Count modification types
        modification_counts = {}
        for enhancement in recent_enhancements:
            metadata = enhancement.get('enhancement_metadata', {})
            modifications = metadata.get('intrinsic_modifications', [])
            for mod in modifications:
                modification_counts[mod] = modification_counts.get(mod, 0) + 1
        
        return {
            'total_enhancements': len(recent_enhancements),
            'avg_motivation_impact': avg_motivation_impact,
            'avg_enhancement_confidence': np.mean([
                e.get('enhancement_metadata', {}).get('enhancement_confidence', 0.0) 
                for e in recent_enhancements
            ]),
            'common_modifications': sorted(modification_counts.items(), key=lambda x: x[1], reverse=True)
        }

class EnhancedImplicitMemoryManager:
    """
    Enhanced Implicit Memory Manager with motivational significance clustering.
    
    Extends the existing Implicit Memory Manager to cluster memories based on
    motivational significance and integrate with self-prior and curiosity systems.
    """
    
    def __init__(self, 
                 original_memory_manager,
                 self_prior_manager=None,
                 pattern_discovery_curiosity=None,
                 motivational_weight: float = 0.3):
        
        self.original_memory_manager = original_memory_manager
        self.self_prior_manager = self_prior_manager
        self.pattern_discovery_curiosity = pattern_discovery_curiosity
        self.motivational_weight = motivational_weight
        
        # Motivational clustering
        self.motivational_clusters = {}
        self.motivational_significance_scores = {}
        
        logger.info("EnhancedImplicitMemoryManager initialized with motivational significance clustering")
    
    def enhanced_memory_storage(self, 
                              memory_data: Any,
                              context: Dict[str, Any]) -> str:
        """Enhanced memory storage with motivational significance clustering."""
        
        # Store in original memory manager
        memory_id = self.original_memory_manager.store_memory(memory_data, context)
        
        # Compute motivational significance
        motivational_significance = self._compute_motivational_significance(memory_data, context)
        
        # Store motivational significance
        self.motivational_significance_scores[memory_id] = motivational_significance
        
        # Update motivational clusters
        self._update_motivational_clusters(memory_id, motivational_significance, context)
        
        return memory_id
    
    def _compute_motivational_significance(self, 
                                         memory_data: Any,
                                         context: Dict[str, Any]) -> Dict[str, float]:
        """Compute motivational significance of a memory."""
        
        significance = {
            'self_prior_relevance': 0.5,
            'curiosity_value': 0.5,
            'pattern_importance': 0.5,
            'learning_potential': 0.5,
            'overall_significance': 0.5
        }
        
        # Self-prior relevance
        if self.self_prior_manager:
            try:
                self_prior_metrics = self.self_prior_manager.get_self_prior_metrics()
                significance['self_prior_relevance'] = self_prior_metrics.get('self_awareness_score', 0.5)
            except Exception as e:
                logger.warning(f"Self-prior significance computation failed: {e}")
        
        # Curiosity value
        if self.pattern_discovery_curiosity:
            try:
                curiosity_metrics = self.pattern_discovery_curiosity.get_curiosity_metrics()
                significance['curiosity_value'] = curiosity_metrics.get('curiosity_levels', {}).get('intellectual', 0.5)
                
                # Pattern importance based on pattern discovery
                if 'observation' in context:
                    observation = context['observation']
                    pattern_result = self.pattern_discovery_curiosity.process_observation(observation)
                    significance['pattern_importance'] = pattern_result.get('total_curiosity_boost', 0.5)
            except Exception as e:
                logger.warning(f"Curiosity significance computation failed: {e}")
        
        # Learning potential from context
        significance['learning_potential'] = context.get('learning_progress', 0.5)
        
        # Overall significance
        significance['overall_significance'] = np.mean(list(significance.values()))
        
        return significance
    
    def _update_motivational_clusters(self, 
                                    memory_id: str,
                                    motivational_significance: Dict[str, float],
                                    context: Dict[str, Any]):
        """Update motivational clusters based on memory significance."""
        
        # Create cluster key based on significance pattern
        cluster_key = self._get_cluster_key(motivational_significance)
        
        if cluster_key not in self.motivational_clusters:
            self.motivational_clusters[cluster_key] = {
                'memory_ids': [],
                'avg_significance': {},
                'cluster_type': self._classify_cluster_type(motivational_significance),
                'created_at': time.time()
            }
        
        # Add memory to cluster
        self.motivational_clusters[cluster_key]['memory_ids'].append(memory_id)
        
        # Update average significance
        cluster = self.motivational_clusters[cluster_key]
        for key, value in motivational_significance.items():
            if key not in cluster['avg_significance']:
                cluster['avg_significance'][key] = value
            else:
                # Update running average
                current_avg = cluster['avg_significance'][key]
                count = len(cluster['memory_ids'])
                cluster['avg_significance'][key] = (current_avg * (count - 1) + value) / count
    
    def _get_cluster_key(self, motivational_significance: Dict[str, float]) -> str:
        """Get cluster key based on motivational significance pattern."""
        
        # Create a key based on significance levels
        high_significance = [k for k, v in motivational_significance.items() if v > 0.7]
        medium_significance = [k for k, v in motivational_significance.items() if 0.4 <= v <= 0.7]
        low_significance = [k for k, v in motivational_significance.items() if v < 0.4]
        
        return f"high:{len(high_significance)}_med:{len(medium_significance)}_low:{len(low_significance)}"
    
    def _classify_cluster_type(self, motivational_significance: Dict[str, float]) -> str:
        """Classify the type of motivational cluster."""
        
        if motivational_significance['self_prior_relevance'] > 0.7:
            return "self_prior_focused"
        elif motivational_significance['curiosity_value'] > 0.7:
            return "curiosity_focused"
        elif motivational_significance['pattern_importance'] > 0.7:
            return "pattern_focused"
        elif motivational_significance['learning_potential'] > 0.7:
            return "learning_focused"
        else:
            return "balanced"
    
    def get_motivational_clusters(self) -> Dict[str, Any]:
        """Get information about motivational clusters."""
        
        return {
            'total_clusters': len(self.motivational_clusters),
            'cluster_types': {
                cluster_type: sum(1 for c in self.motivational_clusters.values() if c['cluster_type'] == cluster_type)
                for cluster_type in ['self_prior_focused', 'curiosity_focused', 'pattern_focused', 'learning_focused', 'balanced']
            },
            'avg_cluster_size': np.mean([len(c['memory_ids']) for c in self.motivational_clusters.values()]) if self.motivational_clusters else 0.0,
            'cluster_details': {
                cluster_key: {
                    'size': len(cluster['memory_ids']),
                    'type': cluster['cluster_type'],
                    'avg_significance': cluster['avg_significance']
                }
                for cluster_key, cluster in self.motivational_clusters.items()
            }
        }

# Factory functions for easy integration
def create_enhanced_tree_based_director(original_director, **kwargs) -> EnhancedTreeBasedDirector:
    """Create an enhanced tree-based director."""
    return EnhancedTreeBasedDirector(original_director, **kwargs)

def create_enhanced_tree_based_architect(original_architect, **kwargs) -> EnhancedTreeBasedArchitect:
    """Create an enhanced tree-based architect."""
    return EnhancedTreeBasedArchitect(original_architect, **kwargs)

def create_enhanced_implicit_memory_manager(original_memory_manager, **kwargs) -> EnhancedImplicitMemoryManager:
    """Create an enhanced implicit memory manager."""
    return EnhancedImplicitMemoryManager(original_memory_manager, **kwargs)

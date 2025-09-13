#!/usr/bin/env python3
"""
Cohesive Integration System

This module integrates all the enhanced subsystems into a cohesive, hypothesis-driven
learning system that transforms random exploration into directed experimentation.

Key Integration Points:
- Architect Priors + Governor Hypothesis Management
- Enhanced Curiosity + Memory Abstraction
- Knowledge Transfer + Cross-Session Learning
- Unified Decision Making and Action Selection
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import time

from .architect_priors import ArchitectPriorsSystem, SpatialStructure, ObjectPotential, CausalPrediction
from .governor_hypothesis_manager import GovernorHypothesisManager, SimulationHypothesis
from .simulation_models import SimulationContext
from .enhanced_curiosity_system import EnhancedCuriositySystem
from .enhanced_memory_abstraction import EnhancedMemoryAbstractionSystem, SemanticConcept

logger = logging.getLogger(__name__)


@dataclass
class CohesiveSystemState:
    """Represents the current state of the cohesive system."""
    curiosity_level: float
    boredom_level: float
    learning_acceleration: float
    active_hypotheses: int
    semantic_concepts: int
    knowledge_transfers: int
    strategy_switch_needed: bool
    system_health: float


class CohesiveIntegrationSystem:
    """
    Main Cohesive Integration System.
    
    This system integrates all enhanced subsystems to create a cohesive,
    hypothesis-driven learning system that moves away from brute-force
    exploration towards intelligent, curiosity-guided learning.
    """
    
    def __init__(self):
        # Initialize all subsystems
        self.architect_priors = ArchitectPriorsSystem()
        self.governor_hypothesis_manager = GovernorHypothesisManager(self.architect_priors)
        self.curiosity_system = EnhancedCuriositySystem()
        self.memory_abstraction = EnhancedMemoryAbstractionSystem()
        
        # Integration state
        self.current_state = CohesiveSystemState(
            curiosity_level=0.5,
            boredom_level=0.0,
            learning_acceleration=1.0,
            active_hypotheses=0,
            semantic_concepts=0,
            knowledge_transfers=0,
            strategy_switch_needed=False,
            system_health=1.0
        )
        
        # Decision history for learning
        self.decision_history = []
        self.learning_metrics = {
            'total_decisions': 0,
            'successful_predictions': 0,
            'curiosity_events': 0,
            'knowledge_transfers': 0,
            'strategy_switches': 0
        }
        
        logger.info("Cohesive Integration System initialized with all subsystems")
    
    def process_environment_update(self, 
                                 frame: np.ndarray,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Process an environment update and generate cohesive response."""
        
        # Step 1: Analyze priors
        spatial_priors = self.architect_priors.analyze_spatial_structure(frame)
        object_priors = self.architect_priors.analyze_object_potential(frame)
        causal_priors = self.architect_priors.predict_action_effects(6, context)
        
        # Step 2: Generate hypotheses based on priors
        simulation_context = self._create_simulation_context(context, spatial_priors, object_priors)
        hypotheses = self.governor_hypothesis_manager.generate_and_prioritize_hypotheses(
            frame, simulation_context
        )
        
        # Step 3: Process curiosity and boredom
        curiosity_response = self.curiosity_system.process_environment_update(
            causal_priors.__dict__, context, context, context
        )
        
        # Step 4: Extract semantic concepts from patterns
        patterns = self._extract_patterns_from_priors(spatial_priors, object_priors)
        concepts = self.memory_abstraction.process_patterns(patterns)
        
        # Step 5: Find analogies and transfer knowledge
        knowledge_transfers = self._perform_knowledge_transfer(concepts, context)
        
        # Step 6: Select next action based on integrated decision
        selected_action = self._select_cohesive_action(hypotheses, context, curiosity_response)
        
        # Step 7: Update system state
        self._update_system_state(curiosity_response, len(hypotheses), len(concepts), knowledge_transfers)
        
        # Step 8: Record decision for learning
        self._record_decision(selected_action, context, curiosity_response)
        
        return {
            'selected_action': selected_action,
            'system_state': self.current_state.__dict__,
            'hypotheses_generated': len(hypotheses),
            'concepts_extracted': len(concepts),
            'knowledge_transfers': knowledge_transfers,
            'curiosity_level': curiosity_response['curiosity_level'],
            'boredom_level': curiosity_response['boredom_level'],
            'learning_acceleration': curiosity_response.get('learning_rate', 1.0)
        }
    
    def _create_simulation_context(self, 
                                 context: Dict[str, Any],
                                 spatial_priors: SpatialStructure,
                                 object_priors: ObjectPotential) -> SimulationContext:
        """Create simulation context from environment context and priors."""
        return SimulationContext(
            current_state=context,
            available_actions=context.get('available_actions', [1, 2, 3, 4, 5, 6, 7]),
            energy_level=context.get('energy_level', 100.0),
            learning_drive=context.get('learning_drive', 0.5),
            frame_analysis={
                'spatial_insights': spatial_priors.__dict__,
                'object_insights': object_priors.__dict__
            },
            memory_patterns=context.get('memory_state', {})
        )
    
    def _extract_patterns_from_priors(self, 
                                    spatial_priors: SpatialStructure,
                                    object_priors: ObjectPotential) -> List[Dict[str, Any]]:
        """Extract patterns from prior analysis for concept extraction."""
        patterns = []
        
        # Extract spatial patterns
        for symmetry in spatial_priors.symmetries:
            patterns.append({
                'type': 'symmetry',
                'features': ['symmetric', 'balanced', 'structured'],
                'suggested_actions': ['coordinate', 'explore'],
                'spatial_properties': {'axis': symmetry.get('axis'), 'strength': symmetry.get('strength')},
                'interaction_properties': {'requires_precision': True}
            })
        
        for pattern in spatial_priors.patterns:
            patterns.append({
                'type': 'pattern',
                'features': ['repetitive', 'structured', 'organized'],
                'suggested_actions': ['coordinate', 'follow'],
                'spatial_properties': {'spacing': pattern.get('spacing'), 'strength': pattern.get('strength')},
                'interaction_properties': {'requires_pattern_recognition': True}
            })
        
        for path in spatial_priors.paths:
            patterns.append({
                'type': 'path',
                'features': ['navigable', 'connecting', 'linear'],
                'suggested_actions': ['follow', 'navigate', 'explore'],
                'spatial_properties': {'length': path.get('length'), 'linearity': path.get('linearity')},
                'interaction_properties': {'requires_movement': True}
            })
        
        # Extract object patterns
        for obj in object_priors.movable_objects:
            patterns.append({
                'type': 'movable_object',
                'features': ['portable', 'interactive', 'manipulable'],
                'suggested_actions': ['interact', 'move', 'use'],
                'spatial_properties': {'center': obj.get('center'), 'area': obj.get('area')},
                'interaction_properties': {'requires_interaction': True}
            })
        
        for combo in object_priors.combinable_objects:
            patterns.append({
                'type': 'combinable_objects',
                'features': ['combinable', 'interactive', 'related'],
                'suggested_actions': ['combine', 'interact', 'activate'],
                'spatial_properties': {'distance': combo.get('distance')},
                'interaction_properties': {'requires_combination': True}
            })
        
        return patterns
    
    def _perform_knowledge_transfer(self, 
                                  concepts: List[SemanticConcept],
                                  context: Dict[str, Any]) -> int:
        """Perform knowledge transfer using extracted concepts."""
        transfers = 0
        
        for concept in concepts:
            # Try to transfer knowledge to current context
            transferred_knowledge = self.memory_abstraction.transfer_knowledge(concept, context)
            if transferred_knowledge:
                transfers += 1
                self.learning_metrics['knowledge_transfers'] += 1
        
        return transfers
    
    def _select_cohesive_action(self, 
                              hypotheses: List[SimulationHypothesis],
                              context: Dict[str, Any],
                              curiosity_response: Dict[str, Any]) -> Optional[Tuple[int, Optional[Tuple[int, int]]]]:
        """Select action based on integrated decision making."""
        
        if not hypotheses:
            return None
        
        # High curiosity: Select most promising hypothesis
        if curiosity_response['curiosity_level'] > 0.8:
            best_hypothesis = max(hypotheses, key=lambda h: h.priority * h.learning_potential)
            if best_hypothesis.action_sequence:
                return best_hypothesis.action_sequence[0]
        
        # Boredom: Switch strategy
        if curiosity_response['boredom_level'] > 0.7 or curiosity_response.get('strategy_switch_needed', False):
            # Select hypothesis with different type
            recent_types = [h.hypothesis_type for h in self.decision_history[-5:]]
            for hypothesis in hypotheses:
                if hypothesis.hypothesis_type not in recent_types:
                    if hypothesis.action_sequence:
                        return hypothesis.action_sequence[0]
            self.learning_metrics['strategy_switches'] += 1
        
        # Normal selection: Use governor's selection
        # Create a minimal simulation context for hypothesis selection
        minimal_context = SimulationContext(
            current_state=context,
            available_actions=context.get('available_actions', [1, 2, 3, 4, 5, 6, 7]),
            energy_level=context.get('energy_level', 100.0),
            learning_drive=context.get('learning_drive', 0.5)
        )
        selected_hypothesis = self.governor_hypothesis_manager.select_next_experiment(minimal_context)
        
        if selected_hypothesis and selected_hypothesis.action_sequence:
            return selected_hypothesis.action_sequence[0]
        
        # Fallback: Select highest priority hypothesis
        if hypotheses:
            best_hypothesis = max(hypotheses, key=lambda h: h.priority)
            if best_hypothesis.action_sequence:
                return best_hypothesis.action_sequence[0]
        
        return None
    
    def _update_system_state(self, 
                           curiosity_response: Dict[str, Any],
                           hypotheses_count: int,
                           concepts_count: int,
                           transfers_count: int):
        """Update the cohesive system state."""
        self.current_state.curiosity_level = curiosity_response['curiosity_level']
        self.current_state.boredom_level = curiosity_response['boredom_level']
        self.current_state.learning_acceleration = curiosity_response.get('learning_rate', 1.0)
        self.current_state.active_hypotheses = hypotheses_count
        self.current_state.semantic_concepts = concepts_count
        self.current_state.knowledge_transfers = transfers_count
        self.current_state.strategy_switch_needed = curiosity_response.get('strategy_switch_needed', False)
        
        # Calculate system health
        self.current_state.system_health = self._calculate_system_health()
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health."""
        health_factors = []
        
        # Curiosity level (higher is better, but not too high)
        curiosity_health = 1.0 - abs(self.current_state.curiosity_level - 0.7)
        health_factors.append(curiosity_health)
        
        # Boredom level (lower is better)
        boredom_health = 1.0 - self.current_state.boredom_level
        health_factors.append(boredom_health)
        
        # Learning acceleration (moderate is best)
        acceleration_health = 1.0 - abs(self.current_state.learning_acceleration - 1.5) / 2.0
        health_factors.append(acceleration_health)
        
        # Active hypotheses (some is good)
        hypotheses_health = min(1.0, self.current_state.active_hypotheses / 10.0)
        health_factors.append(hypotheses_health)
        
        # Semantic concepts (more is better)
        concepts_health = min(1.0, self.current_state.semantic_concepts / 20.0)
        health_factors.append(concepts_health)
        
        return sum(health_factors) / len(health_factors)
    
    def _record_decision(self, 
                        selected_action: Optional[Tuple[int, Optional[Tuple[int, int]]]],
                        context: Dict[str, Any],
                        curiosity_response: Dict[str, Any]):
        """Record decision for learning and analysis."""
        decision = {
            'timestamp': time.time(),
            'action': selected_action,
            'context': context,
            'curiosity_level': curiosity_response['curiosity_level'],
            'boredom_level': curiosity_response['boredom_level'],
            'learning_acceleration': curiosity_response.get('learning_rate', 1.0)
        }
        
        self.decision_history.append(decision)
        self.learning_metrics['total_decisions'] += 1
        
        # Keep only recent decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'cohesive_state': self.current_state.__dict__,
            'learning_metrics': self.learning_metrics,
            'architect_priors_status': 'active',
            'governor_hypothesis_status': self.governor_hypothesis_manager.get_hypothesis_pool_status(),
            'curiosity_system_status': self.curiosity_system.get_system_status(),
            'memory_abstraction_status': self.memory_abstraction.get_system_status(),
            'recent_decisions': len(self.decision_history),
            'system_health': self.current_state.system_health
        }
    
    def force_strategy_switch(self):
        """Force a strategy switch (for testing or manual intervention)."""
        self.curiosity_system.force_boredom_reset()
        self.current_state.strategy_switch_needed = True
        logger.info("Strategy switch forced")
    
    def force_curiosity_boost(self, intensity: float = 0.5):
        """Force a curiosity boost (for testing or manual intervention)."""
        self.curiosity_system.force_curiosity_boost(intensity)
        self.current_state.curiosity_level = min(1.0, self.current_state.curiosity_level + intensity)
        logger.info(f"Curiosity boost forced: {intensity:.2f}")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning process."""
        if not self.decision_history:
            return {'insights': 'No decision history available'}
        
        recent_decisions = self.decision_history[-100:]  # Last 100 decisions
        
        # Calculate insights
        avg_curiosity = np.mean([d['curiosity_level'] for d in recent_decisions])
        avg_boredom = np.mean([d['boredom_level'] for d in recent_decisions])
        avg_acceleration = np.mean([d['learning_acceleration'] for d in recent_decisions])
        
        # Action distribution
        actions = [d['action'][0] if d['action'] else None for d in recent_decisions]
        action_counts = {}
        for action in actions:
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'average_curiosity': avg_curiosity,
            'average_boredom': avg_boredom,
            'average_learning_acceleration': avg_acceleration,
            'action_distribution': action_counts,
            'total_decisions': len(self.decision_history),
            'recent_decisions': len(recent_decisions),
            'system_health': self.current_state.system_health
        }

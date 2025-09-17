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
from .dual_pathway_processor import DualPathwayProcessor, CognitiveMode, ModeSwitchTrigger
from .enhanced_gut_feeling_engine import EnhancedGutFeelingEngine, GutFeeling, GutFeelingType
# Import symbiosis protocol components (avoid circular imports)
try:
    from .recursive_self_improvement import RecursiveSelfImprovementSystem, ImprovementCycleStatus, TriggerType
    from .governor_session_reporter import GovernorSessionReporter, SessionStatus
except ImportError:
    # Fallback for when symbiosis protocol is not available
    RecursiveSelfImprovementSystem = None
    ImprovementCycleStatus = None
    TriggerType = None
    GovernorSessionReporter = None
    SessionStatus = None

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
    # Symbiosis protocol state
    improvement_cycles_completed: int
    directives_executed: int
    performance_improvement: float
    evolution_health: float
    # Conscious architecture enhancements
    current_cognitive_mode: str = "TPN"
    gut_feeling_confidence: float = 0.0
    consciousness_score: float = 0.0


class CohesiveIntegrationSystem:
    """
    Main Cohesive Integration System.
    
    This system integrates all enhanced subsystems to create a cohesive,
    hypothesis-driven learning system that moves away from brute-force
    exploration towards intelligent, curiosity-guided learning.
    """
    
    def __init__(self, enable_symbiosis_protocol: bool = True, enable_conscious_architecture: bool = True):
        # Initialize all subsystems
        self.architect_priors = ArchitectPriorsSystem()
        self.governor_hypothesis_manager = GovernorHypothesisManager(self.architect_priors)
        self.curiosity_system = EnhancedCuriositySystem()
        self.memory_abstraction = EnhancedMemoryAbstractionSystem()
        
        # Initialize conscious architecture components
        self.enable_conscious_architecture = enable_conscious_architecture
        if enable_conscious_architecture:
            self.dual_pathway_processor = DualPathwayProcessor()
            self.gut_feeling_engine = EnhancedGutFeelingEngine()
            logger.info("Conscious architecture components initialized")
        else:
            self.dual_pathway_processor = None
            self.gut_feeling_engine = None
        
        # Initialize symbiosis protocol
        self.enable_symbiosis_protocol = enable_symbiosis_protocol and RecursiveSelfImprovementSystem is not None
        if self.enable_symbiosis_protocol:
            self.recursive_improvement = RecursiveSelfImprovementSystem()
        else:
            self.recursive_improvement = None
        
        # Integration state
        self.current_state = CohesiveSystemState(
            curiosity_level=0.5,
            boredom_level=0.0,
            learning_acceleration=1.0,
            active_hypotheses=0,
            semantic_concepts=0,
            knowledge_transfers=0,
            strategy_switch_needed=False,
            system_health=1.0,
            improvement_cycles_completed=0,
            directives_executed=0,
            performance_improvement=0.0,
            evolution_health=1.0
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
        
        logger.info("Cohesive Integration System initialized with all subsystems" + 
                   (" and Symbiosis Protocol" if enable_symbiosis_protocol else ""))
    
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
        
        # Step 6: Conscious Architecture Processing
        conscious_processing = self._process_conscious_architecture(
            frame, context, hypotheses, curiosity_response
        )
        
        # Step 7: Select next action based on integrated decision
        selected_action = self._select_cohesive_action(
            hypotheses, context, curiosity_response, conscious_processing
        )
        
        # Step 8: Update system state
        self._update_system_state(curiosity_response, len(hypotheses), len(concepts), knowledge_transfers, conscious_processing)
        
        # Step 8: Record decision for learning
        self._record_decision(selected_action, context, curiosity_response)
        
        return {
            'selected_action': selected_action,
            'conscious_processing': conscious_processing,
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
                           transfers_count: int,
                           conscious_processing: Optional[Dict[str, Any]] = None):
        """Update the cohesive system state."""
        self.current_state.curiosity_level = curiosity_response['curiosity_level']
        self.current_state.boredom_level = curiosity_response['boredom_level']
        self.current_state.learning_acceleration = curiosity_response.get('learning_rate', 1.0)
        self.current_state.active_hypotheses = hypotheses_count
        self.current_state.semantic_concepts = concepts_count
        self.current_state.knowledge_transfers = transfers_count
        self.current_state.strategy_switch_needed = curiosity_response.get('strategy_switch_needed', False)
        
        # Update conscious architecture state
        if conscious_processing and conscious_processing.get('enabled', False):
            consciousness_metrics = conscious_processing.get('consciousness_metrics', {})
            if consciousness_metrics:
                self.current_state.current_cognitive_mode = consciousness_metrics.get('current_mode', 'TPN')
                self.current_state.consciousness_score = consciousness_metrics.get('consciousness_score', 0.0)
            
            gut_feeling_metrics = conscious_processing.get('gut_feeling_metrics', {})
            if gut_feeling_metrics:
                self.current_state.gut_feeling_confidence = gut_feeling_metrics.get('average_confidence', 0.0)
        
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
    
    # ===== SYMBIOSIS PROTOCOL INTEGRATION =====
    
    def start_symbiosis_session(self, 
                               session_id: str, 
                               objectives: List[Dict[str, Any]],
                               previous_session_id: Optional[str] = None) -> None:
        """Start a new session with symbiosis protocol tracking."""
        if not self.enable_symbiosis_protocol or not self.recursive_improvement:
            logger.warning("Symbiosis protocol not enabled")
            return
        
        self.recursive_improvement.start_session(session_id, objectives, previous_session_id)
        logger.info(f"Started symbiosis session {session_id}")
    
    def log_symbiosis_decision(self, 
                              decision_type: str,
                              decision_data: Dict[str, Any],
                              result: Dict[str, Any],
                              success: bool,
                              confidence: float,
                              energy_cost: float = 0.0,
                              learning_gain: float = 0.0) -> None:
        """Log a decision for symbiosis protocol tracking."""
        if not self.enable_symbiosis_protocol or not self.recursive_improvement:
            return
        
        self.recursive_improvement.log_decision(
            decision_type, decision_data, result, success, 
            confidence, energy_cost, learning_gain
        )
    
    def log_symbiosis_performance(self, 
                                 performance_metrics: Dict[str, float],
                                 system_state: Dict[str, Any]) -> None:
        """Log performance metrics for symbiosis protocol tracking."""
        if not self.enable_symbiosis_protocol or not self.recursive_improvement:
            return
        
        self.recursive_improvement.log_performance_snapshot(performance_metrics, system_state)
        
        # Update cohesive system state with symbiosis metrics
        if self.recursive_improvement:
            symbiosis_status = self.recursive_improvement.get_system_status()
            self.current_state.improvement_cycles_completed = symbiosis_status['evolution_metrics']['total_cycles']
            self.current_state.directives_executed = symbiosis_status['evolution_metrics']['total_directives_executed']
            self.current_state.performance_improvement = symbiosis_status['evolution_metrics']['cumulative_performance_improvement']
            self.current_state.evolution_health = symbiosis_status['system_health']
    
    def end_symbiosis_session(self, 
                             session_status: SessionStatus = SessionStatus.COMPLETED,
                             next_session_recommendations: List[str] = None) -> Optional[Any]:
        """End the current symbiosis session and generate final report."""
        if not self.enable_symbiosis_protocol or not self.recursive_improvement:
            return None
        
        final_report = self.recursive_improvement.end_session(session_status, next_session_recommendations)
        logger.info(f"Ended symbiosis session with status: {session_status.value}")
        return final_report
    
    def force_symbiosis_improvement_cycle(self, trigger_reason: str = "manual") -> Optional[Any]:
        """Force an improvement cycle to run."""
        if not self.enable_symbiosis_protocol or not self.recursive_improvement:
            logger.warning("Symbiosis protocol not enabled")
            return None
        
        cycle = self.recursive_improvement.force_improvement_cycle(trigger_reason)
        logger.info(f"Forced improvement cycle: {trigger_reason}")
        return cycle
    
    def get_symbiosis_status(self) -> Dict[str, Any]:
        """Get status of the symbiosis protocol."""
        if not self.enable_symbiosis_protocol or not self.recursive_improvement:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'system_status': self.recursive_improvement.get_system_status(),
            'evolution_summary': self.recursive_improvement.get_evolution_summary()
        }
    
    def save_symbiosis_state(self, filepath: str) -> None:
        """Save the current symbiosis state."""
        if not self.enable_symbiosis_protocol or not self.recursive_improvement:
            logger.warning("Symbiosis protocol not enabled")
            return
        
        self.recursive_improvement.save_evolution_state(filepath)
        logger.info(f"Symbiosis state saved to {filepath}")
    
    def load_symbiosis_state(self, filepath: str) -> None:
        """Load symbiosis state from a file."""
        if not self.enable_symbiosis_protocol or not self.recursive_improvement:
            logger.warning("Symbiosis protocol not enabled")
            return
        
        self.recursive_improvement.load_evolution_state(filepath)
        logger.info(f"Symbiosis state loaded from {filepath}")
    
    def _update_symbiosis_protocol(self, 
                                  action: Tuple[int, Optional[Tuple[int, int]]],
                                  coordinate: Optional[Tuple[int, int]],
                                  context: Dict[str, Any],
                                  confidence: float,
                                  success: bool) -> None:
        """Update symbiosis protocol with decision data."""
        if not self.enable_symbiosis_protocol or not self.recursive_improvement:
            return
        
        # Prepare decision data
        decision_data = {
            'action': action,
            'coordinate': coordinate,
            'context': context
        }
        
        result = {
            'success': success,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        # Log the decision
        self.log_symbiosis_decision(
            decision_type='action_selection',
            decision_data=decision_data,
            result=result,
            success=success,
            confidence=confidence,
            energy_cost=context.get('energy_cost', 0.0),
            learning_gain=context.get('learning_gain', 0.0)
        )
        
        # Log performance snapshot
        performance_metrics = {
            'curiosity_level': self.current_state.curiosity_level,
            'boredom_level': self.current_state.boredom_level,
            'learning_acceleration': self.current_state.learning_acceleration,
            'system_health': self.current_state.system_health,
            'active_hypotheses': self.current_state.active_hypotheses,
            'semantic_concepts': self.current_state.semantic_concepts
        }
        
        system_state = {
            'cohesive_state': self.current_state.__dict__,
            'learning_metrics': self.learning_metrics,
            'decision_history_length': len(self.decision_history)
        }
        
        self.log_symbiosis_performance(performance_metrics, system_state)
    
    def _process_conscious_architecture(self, 
                                     frame: np.ndarray,
                                     context: Dict[str, Any],
                                     hypotheses: List[SimulationHypothesis],
                                     curiosity_response: Dict[str, Any]) -> Dict[str, Any]:
        """Process conscious architecture enhancements."""
        if not self.enable_conscious_architecture:
            return {'enabled': False}
        
        conscious_processing = {'enabled': True}
        
        # 1. Dual-Pathway Processing
        if self.dual_pathway_processor:
            # Update performance metrics
            performance_metrics = {
                'confidence': context.get('confidence', 0.5),
                'success_rate': context.get('success_rate', 0.5),
                'learning_progress': context.get('learning_progress', 0.0),
                'energy_efficiency': context.get('energy_efficiency', 0.5),
                'uncertainty': context.get('uncertainty', 0.5)
            }
            
            self.dual_pathway_processor.update_performance(performance_metrics)
            
            # Check for mode switching
            available_actions = context.get('available_actions', [1, 2, 3, 4, 5, 6, 7])
            mode_switch_decision = self.dual_pathway_processor.should_switch_mode(context, available_actions)
            
            if mode_switch_decision.should_switch:
                switch_result = self.dual_pathway_processor.switch_to_mode(
                    mode_switch_decision.target_mode,
                    mode_switch_decision.trigger,
                    context
                )
                conscious_processing['mode_switch'] = switch_result
                logger.info(f"Conscious mode switch: {switch_result['previous_mode']} -> {switch_result['current_mode']}")
            
            # Get mode-specific actions
            mode_actions = self.dual_pathway_processor.get_mode_specific_actions(available_actions, context)
            conscious_processing['mode_actions'] = mode_actions
            
            # Get consciousness metrics
            consciousness_metrics = self.dual_pathway_processor.get_consciousness_metrics()
            conscious_processing['consciousness_metrics'] = consciousness_metrics
            
            # Update system state
            self.current_state.current_cognitive_mode = consciousness_metrics['current_mode']
            self.current_state.consciousness_score = consciousness_metrics['consciousness_score']
        
        # 2. Enhanced Gut Feeling Processing
        if self.gut_feeling_engine:
            # Get gut feelings for current state
            current_state = {
                'frame_features': context.get('frame_features', {}),
                'spatial_features': context.get('spatial_features', {}),
                'temporal_features': context.get('temporal_features', {})
            }
            
            gut_feelings = self.gut_feeling_engine.get_gut_feelings(
                current_state, available_actions, context
            )
            
            conscious_processing['gut_feelings'] = [
                {
                    'action': gf.action,
                    'confidence': gf.confidence,
                    'type': gf.gut_feeling_type.value,
                    'reasoning': gf.reasoning,
                    'similarity_score': gf.similarity_score,
                    'success_rate': gf.success_rate
                }
                for gf in gut_feelings
            ]
            
            # Update gut feeling confidence in system state
            if gut_feelings:
                self.current_state.gut_feeling_confidence = max(gf.confidence for gf in gut_feelings)
            else:
                self.current_state.gut_feeling_confidence = 0.0
            
            # Get gut feeling metrics
            gut_metrics = self.gut_feeling_engine.get_gut_feeling_metrics()
            conscious_processing['gut_feeling_metrics'] = gut_metrics
        
        return conscious_processing
    
    def _select_cohesive_action(self, 
                               hypotheses: List[SimulationHypothesis],
                               context: Dict[str, Any],
                               curiosity_response: Dict[str, Any],
                               conscious_processing: Optional[Dict[str, Any]] = None) -> int:
        """Select action with conscious architecture integration."""
        if not conscious_processing or not conscious_processing.get('enabled', False):
            # Fallback to original action selection
            return self._original_select_cohesive_action(hypotheses, context, curiosity_response)
        
        # Integrate conscious architecture into action selection
        available_actions = context.get('available_actions', [1, 2, 3, 4, 5, 6, 7])
        
        # 1. Get gut feelings
        gut_feelings = conscious_processing.get('gut_feelings', [])
        
        # 2. Get mode-specific actions
        mode_actions = conscious_processing.get('mode_actions', [])
        
        # 3. Combine gut feelings with mode-specific actions
        action_scores = {}
        
        # Score based on gut feelings
        for gf in gut_feelings:
            action = gf['action']
            if action in available_actions:
                action_scores[action] = action_scores.get(action, 0.0) + gf['confidence'] * 0.4
        
        # Score based on mode-specific actions
        for ma in mode_actions:
            action = ma['action']
            if action in available_actions:
                action_scores[action] = action_scores.get(action, 0.0) + ma['priority'] * 0.3
        
        # Score based on hypotheses (original logic)
        for hypothesis in hypotheses:
            if hasattr(hypothesis, 'recommended_action') and hypothesis.recommended_action in available_actions:
                action_scores[hypothesis.recommended_action] = action_scores.get(hypothesis.recommended_action, 0.0) + 0.3
        
        # Select action with highest score
        if action_scores:
            selected_action = max(action_scores.items(), key=lambda x: x[1])[0]
            logger.debug(f"Conscious action selection: {selected_action} (score: {action_scores[selected_action]:.3f})")
            return selected_action
        
        # Fallback to original selection
        return self._original_select_cohesive_action(hypotheses, context, curiosity_response)
    
    def _original_select_cohesive_action(self, 
                                        hypotheses: List[SimulationHypothesis],
                                        context: Dict[str, Any],
                                        curiosity_response: Dict[str, Any]) -> int:
        """Original action selection logic (fallback)."""
        # This is the original action selection logic
        # For now, return a simple selection
        available_actions = context.get('available_actions', [1, 2, 3, 4, 5, 6, 7])
        return available_actions[0] if available_actions else 1
    
    def learn_from_conscious_outcome(self, 
                                   action: int,
                                   outcome: Dict[str, Any],
                                   context: Dict[str, Any]):
        """Learn from the outcome of a conscious architecture decision."""
        if not self.enable_conscious_architecture or not self.gut_feeling_engine:
            return
        
        # Find the gut feeling that led to this action
        recent_gut_feelings = getattr(self, '_recent_gut_feelings', [])
        for gf in recent_gut_feelings:
            if gf.action == action:
                self.gut_feeling_engine.learn_from_outcome(gf, outcome, context)
                break
    
    def get_conscious_architecture_status(self) -> Dict[str, Any]:
        """Get status of conscious architecture components."""
        if not self.enable_conscious_architecture:
            return {'enabled': False}
        
        status = {'enabled': True}
        
        if self.dual_pathway_processor:
            status['dual_pathway'] = self.dual_pathway_processor.get_consciousness_metrics()
        
        if self.gut_feeling_engine:
            status['gut_feeling'] = self.gut_feeling_engine.get_gut_feeling_metrics()
        
        return status

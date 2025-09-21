#!/usr/bin/env python3
"""
Enhanced Mutation Engine - Generates advanced mutations for system genomes.
"""

import time
import random
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from ..system_design.genome import SystemGenome, MutationType, MutationImpact
from .types import Mutation

@dataclass
class MutationContext:
    """Context for mutation generation."""
    performance_history: List[Dict[str, Any]]
    frame_analysis: Dict[str, Any]
    memory_state: Dict[str, Any]
    energy_state: Dict[str, Any]
    learning_progress: float
    stagnation_detected: bool
    recent_failures: int

class MutationEngine:
    """Enhanced mutation engine with advanced capabilities."""
    
    def __init__(self, base_genome: SystemGenome, logger: logging.Logger):
        self.base_genome = base_genome
        self.logger = logger
        self.mutation_templates = self._initialize_mutation_templates()
        self.mutation_history = []
        self.success_patterns = []
        self.failure_patterns = []
        self.adaptive_weights = self._initialize_adaptive_weights()
    
    def _initialize_adaptive_weights(self) -> Dict[str, float]:
        """Initialize adaptive weights for mutation strategies."""
        return {
            'parameter_adjustment': 0.3,
            'feature_toggle': 0.25,
            'mode_modification': 0.2,
            'architecture_enhancement': 0.15,
            'neural_architecture_search': 0.1
        }
    
    def _initialize_mutation_templates(self) -> List[Dict[str, Any]]:
        """Initialize templates for different types of mutations."""
        return [
            # Parameter adjustments
            {
                'type': MutationType.PARAMETER_ADJUSTMENT,
                'impact': MutationImpact.MINIMAL,
                'targets': ['max_actions_per_game', 'target_score', 'salience_threshold'],
                'strategies': ['increase', 'decrease', 'optimize']
            },
            
            # Feature toggles
            {
                'type': MutationType.FEATURE_TOGGLE,
                'impact': MutationImpact.MODERATE,
                'targets': ['enable_contrarian_strategy', 'enable_boredom_detection', 'enable_mid_game_sleep'],
                'strategies': ['enable', 'disable', 'conditional']
            },
            
            # Mode modifications
            {
                'type': MutationType.MODE_MODIFICATION,
                'impact': MutationImpact.MODERATE,
                'targets': ['salience_mode'],
                'strategies': ['switch_mode', 'hybrid_approach']
            },
            
            # Advanced system combinations
            {
                'type': MutationType.ARCHITECTURE_ENHANCEMENT,
                'impact': MutationImpact.SIGNIFICANT,
                'targets': ['cognitive_system_combinations'],
                'strategies': ['enable_synergies', 'optimize_coordination', 'reduce_conflicts']
            },
            
            # Neural Architecture Search (NAS) patterns
            {
                'type': MutationType.NEURAL_ARCHITECTURE_SEARCH,
                'impact': MutationImpact.SIGNIFICANT,
                'targets': ['hidden_dimensions', 'layer_depths', 'activation_functions', 'normalization_schemes'],
                'strategies': ['progressive_search', 'skip_connections', 'attention_heads', 'layer_scaling']
            },
            
            # Attention mechanism modifications
            {
                'type': MutationType.ATTENTION_MODIFICATION,
                'impact': MutationImpact.MODERATE,
                'targets': ['attention_span', 'focus_patterns', 'multi_head_attention', 'self_attention_depth'],
                'strategies': ['enhance_locality', 'global_attention', 'sparse_attention', 'dynamic_attention']
            },
            
            # Multi-objective optimization
            {
                'type': MutationType.MULTI_OBJECTIVE_OPTIMIZATION,
                'impact': MutationImpact.SIGNIFICANT,
                'targets': ['objective_weights', 'pareto_optimization', 'constraint_handling'],
                'strategies': ['balance_objectives', 'priority_weighting', 'adaptive_constraints', 'evolutionary_pareto']
            },
            
            # Learning rate scheduling
            {
                'type': MutationType.LEARNING_SCHEDULE_OPTIMIZATION,
                'impact': MutationImpact.MODERATE,
                'targets': ['learning_rate_schedule', 'warmup_steps', 'decay_strategy', 'adaptive_lr'],
                'strategies': ['cosine_annealing', 'exponential_decay', 'plateau_reduction', 'cyclic_learning']
            },
            
            # Memory and caching optimizations
            {
                'type': MutationType.MEMORY_OPTIMIZATION,
                'impact': MutationImpact.MINIMAL,
                'targets': ['cache_size', 'memory_patterns', 'garbage_collection', 'buffer_strategies'],
                'strategies': ['increase_cache', 'optimize_patterns', 'smart_gc', 'streaming_buffers']
            },
            
            # Ensemble and model fusion
            {
                'type': MutationType.ENSEMBLE_FUSION,
                'impact': MutationImpact.SIGNIFICANT,
                'targets': ['model_ensemble', 'voting_strategies', 'fusion_weights', 'diversity_metrics'],
                'strategies': ['weighted_voting', 'stacking_ensemble', 'dynamic_selection', 'adversarial_fusion']
            }
        ]
    
    def generate_mutation(self, request: Optional[Any] = None, 
                         context: Optional[MutationContext] = None) -> Mutation:
        """Generate a mutation based on request, context, or random exploration."""
        
        if request:
            return self._generate_targeted_mutation(request, context)
        elif context:
            return self._generate_context_aware_mutation(context)
        else:
            return self._generate_exploratory_mutation()
    
    def generate_exploratory_mutation(self) -> Mutation:
        """Public wrapper for exploratory mutation generation."""
        return self._generate_exploratory_mutation()
    
    def generate_context_aware_mutation(self, context: MutationContext) -> Mutation:
        """Generate mutation based on current system context."""
        return self._generate_context_aware_mutation(context)
    
    def _generate_exploratory_mutation(self) -> Mutation:
        """Generate a random exploratory mutation."""
        mutation_id = f"exploratory_{int(time.time())}"
        
        # Select random template
        template = random.choice(self.mutation_templates)
        
        # Generate changes based on template
        changes = {}
        rationale_parts = []
        
        if template['type'] == MutationType.PARAMETER_ADJUSTMENT:
            target = random.choice(template['targets'])
            strategy = random.choice(template['strategies'])
            
            if target == 'max_actions_per_game':
                if strategy == 'increase':
                    changes[target] = int(self.base_genome.max_actions_per_game * 1.2)
                elif strategy == 'decrease':
                    changes[target] = int(self.base_genome.max_actions_per_game * 0.8)
                else:  # optimize
                    changes[target] = int(self.base_genome.max_actions_per_game * random.uniform(0.9, 1.1))
                rationale_parts.append(f"Adjusted {target} using {strategy}")
                
        elif template['type'] == MutationType.FEATURE_TOGGLE:
            target = random.choice(template['targets'])
            strategy = random.choice(template['strategies'])
            
            if strategy == 'enable':
                changes[target] = True
            elif strategy == 'disable':
                changes[target] = False
            else:  # conditional
                changes[target] = random.choice([True, False])
            rationale_parts.append(f"Toggled {target} using {strategy}")
            
        elif template['type'] == MutationType.MODE_MODIFICATION:
            target = random.choice(template['targets'])
            if target == 'salience_mode':
                modes = ['lossless', 'decay_compression', 'hybrid']
                changes[target] = random.choice(modes)
                rationale_parts.append(f"Switched {target} to {changes[target]}")
        
        # Default fallback
        if not changes:
            changes = {'salience_threshold': max(0.1, self.base_genome.salience_threshold + random.uniform(-0.1, 0.1))}
            rationale_parts.append("Random parameter adjustment")
        
        rationale = f"Exploratory {template['type'].value}: " + " | ".join(rationale_parts)
        
        return Mutation(
            id=mutation_id,
            type=template['type'],
            impact=template['impact'],
            changes=changes,
            rationale=rationale,
            expected_improvement=random.uniform(0.05, 0.25),
            confidence=random.uniform(0.3, 0.8),
            test_duration_estimate=random.uniform(10.0, 30.0)
        )
    
    def _generate_context_aware_mutation(self, context: MutationContext) -> Mutation:
        """Generate mutation based on system context and performance patterns."""
        mutation_id = f"context_aware_{int(time.time())}"
        
        # Analyze context to determine mutation strategy
        mutation_strategy = self._analyze_context_for_mutation_strategy(context)
        
        # Generate changes based on strategy
        changes = self._generate_changes_for_strategy(mutation_strategy, context)
        
        # Calculate expected improvement and confidence
        expected_improvement, confidence = self._calculate_mutation_metrics(changes, context)
        
        rationale = f"Context-aware mutation: {mutation_strategy['description']}"
        
        return Mutation(
            id=mutation_id,
            type=mutation_strategy['type'],
            impact=mutation_strategy['impact'],
            changes=changes,
            rationale=rationale,
            expected_improvement=expected_improvement,
            confidence=confidence,
            test_duration_estimate=mutation_strategy['test_duration']
        )
    
    def _generate_targeted_mutation(self, request: Any, context: Optional[MutationContext] = None) -> Mutation:
        """Generate mutation to address specific issues with enhanced analysis."""
        mutation_id = f"targeted_{int(time.time())}"
        
        # Enhanced targeted mutation with context analysis
        issue_type = getattr(request, 'issue_type', 'general')
        changes = self._generate_targeted_changes(issue_type, context)
        
        rationale = f"Targeted improvement for {issue_type}"
        if context and context.stagnation_detected:
            rationale += " (stagnation detected)"
        
        return Mutation(
            id=mutation_id,
            type=MutationType.PARAMETER_ADJUSTMENT,
            impact=MutationImpact.MODERATE,
            changes=changes,
            rationale=rationale,
            expected_improvement=0.15,
            confidence=0.7,
            test_duration_estimate=15.0
        )
    
    def _analyze_context_for_mutation_strategy(self, context: MutationContext) -> Dict[str, Any]:
        """Analyze context to determine the best mutation strategy."""
        # Determine mutation type based on context
        if context.stagnation_detected and context.recent_failures > 3:
            return {
                'type': MutationType.ARCHITECTURE_ENHANCEMENT,
                'impact': MutationImpact.SIGNIFICANT,
                'description': 'Architectural enhancement for stagnation',
                'test_duration': 25.0
            }
        elif context.learning_progress < 0.1:
            return {
                'type': MutationType.PARAMETER_ADJUSTMENT,
                'impact': MutationImpact.MODERATE,
                'description': 'Parameter optimization for low learning progress',
                'test_duration': 15.0
            }
        elif context.energy_state.get('current_energy', 100) < 30:
            return {
                'type': MutationType.FEATURE_TOGGLE,
                'impact': MutationImpact.MINIMAL,
                'description': 'Energy optimization features',
                'test_duration': 10.0
            }
        else:
            # Default to parameter adjustment
            return {
                'type': MutationType.PARAMETER_ADJUSTMENT,
                'impact': MutationImpact.MINIMAL,
                'description': 'General parameter optimization',
                'test_duration': 12.0
            }
    
    def _generate_changes_for_strategy(self, strategy: Dict[str, Any], context: MutationContext) -> Dict[str, Any]:
        """Generate specific changes based on mutation strategy."""
        changes = {}
        
        if strategy['type'] == MutationType.PARAMETER_ADJUSTMENT:
            changes = self._generate_parameter_changes(context)
        elif strategy['type'] == MutationType.FEATURE_TOGGLE:
            changes = self._generate_feature_changes(context)
        elif strategy['type'] == MutationType.ARCHITECTURE_ENHANCEMENT:
            changes = self._generate_architecture_changes(context)
        elif strategy['type'] == MutationType.MODE_MODIFICATION:
            changes = self._generate_mode_changes(context)
        
        return changes
    
    def _generate_parameter_changes(self, context: MutationContext) -> Dict[str, Any]:
        """Generate parameter adjustment changes based on context."""
        changes = {}
        
        # Adjust salience threshold based on learning progress
        if context.learning_progress < 0.2:
            changes['salience_threshold'] = max(0.1, self.base_genome.salience_threshold - 0.1)
        
        # Adjust action limits based on recent failures
        if context.recent_failures > 2:
            changes['max_actions_per_game'] = min(1000, int(self.base_genome.max_actions_per_game * 1.2))
        
        # Adjust energy parameters based on energy state
        if context.energy_state.get('current_energy', 100) < 50:
            changes['energy_decay_rate'] = max(0.005, self.base_genome.energy_decay_rate * 0.8)
            changes['sleep_trigger_energy'] = min(80.0, self.base_genome.sleep_trigger_energy + 10.0)
        
        return changes
    
    def _generate_feature_changes(self, context: MutationContext) -> Dict[str, Any]:
        """Generate feature toggle changes based on context."""
        changes = {}
        
        # Enable exploration if stagnation detected
        if context.stagnation_detected:
            changes['enable_exploration_strategies'] = True
            changes['enable_action_experimentation'] = True
        
        # Enable contrarian strategy if recent failures
        if context.recent_failures > 3:
            changes['enable_contrarian_strategy'] = True
            changes['contrarian_threshold'] = 2
        
        return changes
    
    def _generate_architecture_changes(self, context: MutationContext) -> Dict[str, Any]:
        """Generate architectural enhancement changes based on context."""
        changes = {}
        
        # Enhance memory systems if memory state indicates issues
        if context.memory_state.get('fragmentation_ratio', 0) > 0.7:
            changes['memory_consolidation_strength'] = min(1.0, self.base_genome.memory_consolidation_strength + 0.2)
            changes['enable_memory_regularization'] = True
        
        # Enhance learning systems if learning progress is low
        if context.learning_progress < 0.1:
            changes['enable_meta_learning'] = True
            changes['enable_knowledge_transfer'] = True
        
        return changes
    
    def _generate_mode_changes(self, context: MutationContext) -> Dict[str, Any]:
        """Generate mode modification changes based on context."""
        changes = {}
        
        # Switch salience mode based on performance
        if context.learning_progress < 0.2:
            current_mode = self.base_genome.salience_mode
            if current_mode == 'decay_compression':
                changes['salience_mode'] = 'lossless'
            else:
                changes['salience_mode'] = 'decay_compression'
        
        return changes
    
    def _generate_targeted_changes(self, issue_type: str, context: Optional[MutationContext] = None) -> Dict[str, Any]:
        """Generate targeted changes for specific issue types."""
        changes = {}
        
        if issue_type == "low_efficiency":
            changes['max_actions_per_game'] = int(self.base_genome.max_actions_per_game * 1.2)
            changes['salience_threshold'] = max(0.1, self.base_genome.salience_threshold - 0.1)
        elif issue_type == "stagnation":
            changes['enable_contrarian_strategy'] = True
            changes['enable_exploration_strategies'] = True
            changes['contrarian_threshold'] = 2
        elif issue_type == "memory_issues":
            changes['memory_consolidation_strength'] = min(1.0, self.base_genome.memory_consolidation_strength + 0.2)
            changes['enable_memory_regularization'] = True
        else:
            # Default improvement
            changes['salience_threshold'] = max(0.1, self.base_genome.salience_threshold - 0.05)
        
        return changes
    
    def _calculate_mutation_metrics(self, changes: Dict[str, Any], context: MutationContext) -> Tuple[float, float]:
        """Calculate expected improvement and confidence for mutation."""
        # Base metrics
        expected_improvement = 0.1
        confidence = 0.6
        
        # Adjust based on context
        if context.stagnation_detected:
            expected_improvement += 0.1
            confidence += 0.1
        
        if context.learning_progress < 0.2:
            expected_improvement += 0.05
            confidence += 0.05
        
        # Adjust based on number of changes
        num_changes = len(changes)
        if num_changes > 3:
            expected_improvement += 0.05
            confidence -= 0.1  # More changes = less confidence
        
        return min(0.5, expected_improvement), min(0.9, confidence)
    
    def update_adaptive_weights(self, mutation: Mutation, success: bool, improvement: float):
        """Update adaptive weights based on mutation results."""
        mutation_type = mutation.type.value
        
        if success and improvement > 0.05:
            # Increase weight for successful mutation type
            self.adaptive_weights[mutation_type] = min(0.5, self.adaptive_weights[mutation_type] + 0.05)
            self.success_patterns.append({
                'type': mutation_type,
                'changes': mutation.changes,
                'improvement': improvement,
                'timestamp': time.time()
            })
        else:
            # Decrease weight for unsuccessful mutation type
            self.adaptive_weights[mutation_type] = max(0.05, self.adaptive_weights[mutation_type] - 0.02)
            self.failure_patterns.append({
                'type': mutation_type,
                'changes': mutation.changes,
                'improvement': improvement,
                'timestamp': time.time()
            })
        
        # Keep only recent patterns
        if len(self.success_patterns) > 100:
            self.success_patterns = self.success_patterns[-50:]
        if len(self.failure_patterns) > 100:
            self.failure_patterns = self.failure_patterns[-50:]

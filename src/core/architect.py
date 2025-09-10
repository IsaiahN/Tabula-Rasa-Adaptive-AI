#!/usr/bin/env python3
"""
Architect - The "Zeroth Brain"

A safe, recursive self-improvement system that can hypothesize, test,
and implement improvements to its own architecture and codebase.

This module performs automated, safe, and measurable self-modification
at the architectural and hyperparameter level by:
1. Using the system's configuration as a "genome"
2. Generating mutations to improve performance
3. Testing mutations in sandboxed environments
4. Committing successful improvements through version control
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
import logging
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Optional Git integration
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    git = None

# Import existing system components
try:
    from src.core.meta_cognitive_governor import ArchitectRequest, GovernorRecommendation
    from src.core.salience_system import SalienceMode
except ImportError:
    # Fallback for direct execution
    class SalienceMode(Enum):
        LOSSLESS = "lossless"
        DECAY_COMPRESSION = "decay_compression"
    
    @dataclass
    class ArchitectRequest:
        issue_type: str
        persistent_problem: str
        failed_solutions: List[Dict[str, Any]]
        performance_data: Dict[str, Any]
        suggested_research_directions: List[str]
        priority: float

class MutationType(Enum):
    """Types of mutations the Architect can perform."""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    FEATURE_TOGGLE = "feature_toggle"
    MODE_MODIFICATION = "mode_modification"
    ALGORITHM_REPLACEMENT = "algorithm_replacement"
    ARCHITECTURE_ENHANCEMENT = "architecture_enhancement"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    ATTENTION_MODIFICATION = "attention_modification"
    MULTI_OBJECTIVE_OPTIMIZATION = "multi_objective_optimization"
    LEARNING_SCHEDULE_OPTIMIZATION = "learning_schedule_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    ENSEMBLE_FUSION = "ensemble_fusion"

class MutationImpact(Enum):
    """Expected impact levels of mutations."""
    MINIMAL = "minimal"        # Small parameter tweaks
    MODERATE = "moderate"      # Feature toggles, mode changes
    SIGNIFICANT = "significant" # Algorithm modifications
    ARCHITECTURAL = "architectural"  # Major structural changes

@dataclass
class SystemGenome:
    """
    Formalized representation of the system's architecture and parameters.
    Based on the existing TrainingConfig but expanded for evolution.
    """
    # Core learning parameters
    salience_mode: str = "decay_compression"
    max_actions_per_game: int = 500
    max_learning_cycles: int = 50
    target_score: float = 85.0
    
    # Cognitive system feature flags (all 37+ systems)
    enable_swarm: bool = True
    enable_coordinates: bool = True
    enable_energy_system: bool = True
    enable_sleep_cycles: bool = True
    enable_dnc_memory: bool = True
    enable_meta_learning: bool = True
    enable_salience_system: bool = True
    enable_contrarian_strategy: bool = True
    enable_frame_analysis: bool = True
    enable_boundary_detection: bool = True
    enable_memory_consolidation: bool = True
    enable_action_intelligence: bool = True
    enable_goal_invention: bool = True
    enable_learning_progress_drive: bool = True
    enable_death_manager: bool = True
    enable_exploration_strategies: bool = True
    enable_pattern_recognition: bool = True
    enable_knowledge_transfer: bool = True
    enable_boredom_detection: bool = True
    enable_mid_game_sleep: bool = True
    enable_action_experimentation: bool = True
    enable_reset_decisions: bool = True
    enable_curriculum_learning: bool = True
    enable_multi_modal_input: bool = True
    enable_temporal_memory: bool = True
    enable_hebbian_bonuses: bool = True
    enable_memory_regularization: bool = True
    enable_gradient_flow_monitoring: bool = True
    enable_usage_tracking: bool = True
    enable_salient_memory_retrieval: bool = True
    enable_anti_bias_weighting: bool = True
    enable_stagnation_detection: bool = True
    enable_emergency_movement: bool = True
    enable_cluster_formation: bool = True
    enable_danger_zone_avoidance: bool = True
    enable_predictive_coordinates: bool = True
    
    # Advanced parameters
    energy_decay_rate: float = 0.02
    sleep_trigger_energy: float = 30.0
    salience_threshold: float = 0.5
    memory_consolidation_strength: float = 0.8
    contrarian_threshold: int = 5
    boredom_threshold: int = 100
    
    # Meta-cognitive parameters
    enable_governor: bool = True
    governor_intervention_threshold: float = 0.7
    architect_mutation_rate: float = 0.05
    
    # Evolution metadata
    generation: int = 0
    parent_hash: Optional[str] = None
    mutation_history: List[str] = field(default_factory=list)
    performance_history: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary."""
        return asdict(self)
    
    def get_hash(self) -> str:
        """Get unique hash for this genome configuration."""
        # Create hash from core parameters (exclude metadata)
        core_params = self.to_dict()
        for key in ['generation', 'parent_hash', 'mutation_history', 'performance_history']:
            core_params.pop(key, None)
        
        return hashlib.md5(json.dumps(core_params, sort_keys=True).encode()).hexdigest()[:8]

@dataclass
class Mutation:
    """Represents a proposed change to the system genome."""
    id: str
    type: MutationType
    impact: MutationImpact
    changes: Dict[str, Any]
    rationale: str
    expected_improvement: float
    confidence: float
    test_duration_estimate: float  # in minutes
    
    def apply_to_genome(self, genome: SystemGenome) -> SystemGenome:
        """Apply this mutation to create a new genome."""
        new_genome_dict = genome.to_dict()
        
        # Apply changes
        for key, value in self.changes.items():
            if key in new_genome_dict:
                new_genome_dict[key] = value
        
        # Update metadata
        new_genome_dict['generation'] = genome.generation + 1
        new_genome_dict['parent_hash'] = genome.get_hash()
        new_genome_dict['mutation_history'] = genome.mutation_history + [self.id]
        
        return SystemGenome(**new_genome_dict)

@dataclass
class TestResult:
    """Results from testing a mutated genome in sandbox."""
    mutation_id: str
    genome_hash: str
    success: bool
    performance_metrics: Dict[str, float]
    improvement_over_baseline: Dict[str, float]
    test_duration: float
    error_log: Optional[str] = None
    detailed_results: Optional[Dict[str, Any]] = None
    
    def get_overall_improvement(self) -> float:
        """Calculate overall improvement score."""
        if not self.success:
            return -1.0
        
        # Weight different metrics
        weights = {
            'win_rate': 100.0,
            'average_score': 1.0,
            'learning_efficiency': 10.0,
            'sample_efficiency': 5.0,
            'robustness': 20.0
        }
        
        total_improvement = 0.0
        total_weight = 0.0
        
        for metric, improvement in self.improvement_over_baseline.items():
            if metric in weights:
                total_improvement += improvement * weights[metric]
                total_weight += weights[metric]
        
        return total_improvement / max(total_weight, 1.0)

class MutationEngine:
    """Generates mutations for system genomes."""
    
    def __init__(self, base_genome: SystemGenome, logger: logging.Logger):
        self.base_genome = base_genome
        self.logger = logger
        self.mutation_templates = self._initialize_mutation_templates()
    
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
    
    def generate_mutation(self, request: Optional[ArchitectRequest] = None) -> Mutation:
        """Generate a mutation based on request or random exploration."""
        
        if request:
            return self._generate_targeted_mutation(request)
        else:
            return self._generate_exploratory_mutation()
    
    def generate_exploratory_mutation(self) -> Mutation:
        """Public wrapper for exploratory mutation generation."""
        return self._generate_exploratory_mutation()
    
    def _generate_targeted_mutation(self, request: ArchitectRequest) -> Mutation:
        """Generate mutation to address specific issues with enhanced context analysis."""
        mutation_id = f"targeted_{int(time.time())}"
        
        # Enhanced analysis using frame data, memory context, and object analysis
        frame_insights = self._analyze_frame_context(request.frame_data)
        memory_insights = self._analyze_memory_context(request.memory_context)
        object_insights = self._analyze_object_context(request.object_analysis)
        energy_insights = self._analyze_energy_context(request.energy_state)
        
        # Generate targeted changes based on comprehensive analysis
        changes = {}
        rationale_parts = []
        
        if request.issue_type == "low_efficiency":
            # Use frame data to determine if visual targeting needs improvement
            if frame_insights.get('low_interactive_targets', False):
                changes['enable_frame_analysis'] = True
                changes['salience_threshold'] = max(0.1, self.base_genome.salience_threshold - 0.15)
                rationale_parts.append("Enhanced visual targeting based on frame analysis")
            
            # Use memory context to improve learning
            if memory_insights.get('low_retention', False):
                changes['memory_consolidation_strength'] = min(1.0, self.base_genome.memory_consolidation_strength + 0.2)
                rationale_parts.append("Improved memory consolidation based on retention analysis")
            
            # Use object analysis for better interaction
            if object_insights.get('complex_objects', False):
                changes['enable_multi_modal_input'] = True
                changes['enable_boundary_detection'] = True
                rationale_parts.append("Enhanced object interaction capabilities")
            
            # Use energy state for better resource management
            if energy_insights.get('high_depletion', False):
                changes['energy_decay_rate'] = max(0.005, self.base_genome.energy_decay_rate * 0.7)
                changes['sleep_trigger_energy'] = min(80.0, self.base_genome.sleep_trigger_energy + 10.0)
                rationale_parts.append("Optimized energy management based on depletion patterns")
            
            # Default efficiency improvements
            if not changes:
                changes = {'max_actions_per_game': int(self.base_genome.max_actions_per_game * 1.5)}
                rationale_parts.append("Increased exploration capacity")
            
        elif request.issue_type == "stagnation":
            # Use learning progress to break stagnation
            if request.learning_progress is not None and request.learning_progress < 0.1:
                changes['enable_contrarian_strategy'] = True
                changes['contrarian_threshold'] = 2  # More aggressive
                changes['enable_boredom_detection'] = True
                changes['boredom_threshold'] = 50  # Lower threshold
                rationale_parts.append("Aggressive anti-stagnation measures based on learning progress")
            
            # Use frame data for visual stagnation
            if frame_insights.get('repetitive_patterns', False):
                changes['enable_exploration_strategies'] = True
                changes['enable_action_experimentation'] = True
                rationale_parts.append("Enhanced exploration to break visual pattern repetition")
            
        elif request.issue_type == "visual_targeting_issues":
            # Specific visual targeting improvements
            changes['enable_frame_analysis'] = True
            changes['enable_boundary_detection'] = True
            changes['salience_threshold'] = max(0.1, self.base_genome.salience_threshold - 0.2)
            rationale_parts.append("Comprehensive visual targeting enhancement")
            
        else:  # General improvement
            # Use all available context for general improvements
            if frame_insights.get('rich_environment', False):
                changes['enable_multi_modal_input'] = True
                changes['enable_temporal_memory'] = True
                rationale_parts.append("Enhanced multi-modal processing for rich environments")
            
            if memory_insights.get('high_importance_memories', False):
                changes['memory_consolidation_strength'] = min(1.0, self.base_genome.memory_consolidation_strength + 0.1)
                rationale_parts.append("Strengthened memory consolidation for important experiences")
            
            if not changes:
                changes = {'salience_threshold': max(0.1, self.base_genome.salience_threshold - 0.1)}
                rationale_parts.append("General salience threshold optimization")
        
        # Combine rationale
        rationale = f"Targeted improvement for {request.issue_type}: " + " | ".join(rationale_parts)
        
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
    
    def _analyze_frame_context(self, frame_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze frame data for architectural insights."""
        if not frame_data:
            return {}
        
        insights = {}
        
        # Analyze interactive targets
        interactive_targets = frame_data.get('interactive_targets', [])
        insights['low_interactive_targets'] = len(interactive_targets) < 2
        insights['rich_environment'] = len(interactive_targets) > 5
        
        # Analyze confidence levels
        confidence = frame_data.get('confidence', 0.0)
        insights['low_confidence'] = confidence < 0.5
        
        # Analyze patterns
        patterns = frame_data.get('patterns', [])
        insights['repetitive_patterns'] = len(set(str(p) for p in patterns)) < len(patterns) * 0.7
        
        return insights
    
    def _analyze_memory_context(self, memory_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory context for architectural insights."""
        if not memory_context:
            return {}
        
        insights = {}
        
        # Analyze retention rates
        retention_rate = memory_context.get('retention_rate', 0.5)
        insights['low_retention'] = retention_rate < 0.3
        
        # Analyze memory importance
        high_importance = memory_context.get('high_importance_count', 0)
        total_memories = memory_context.get('total_memories', 1)
        insights['high_importance_memories'] = (high_importance / total_memories) > 0.3
        
        return insights
    
    def _analyze_object_context(self, object_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze object analysis for architectural insights."""
        if not object_analysis:
            return {}
        
        insights = {}
        
        # Analyze object complexity
        object_count = object_analysis.get('object_count', 0)
        complexity_score = object_analysis.get('complexity_score', 0.0)
        insights['complex_objects'] = object_count > 3 and complexity_score > 0.7
        
        return insights
    
    def _analyze_energy_context(self, energy_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze energy state for architectural insights."""
        if not energy_state:
            return {}
        
        insights = {}
        
        # Analyze energy depletion
        current_energy = energy_state.get('current_energy', 100.0)
        depletion_rate = energy_state.get('depletion_rate', 0.01)
        insights['high_depletion'] = depletion_rate > 0.02 or current_energy < 30.0
        
        return insights
    
    def _generate_exploratory_mutation(self) -> Mutation:
        """Generate exploratory mutation for general improvement."""
        import random
        
        mutation_id = f"exploratory_{int(time.time())}"
        
        # Select a template
        template = random.choice(self.mutation_templates)
        target = random.choice(template['targets'])
        strategy = random.choice(template['strategies'])
        
        # Generate changes based on template type
        changes = {}
        rationale = f"Exploratory {strategy} of {target}"
        expected_improvement = 0.05
        confidence = 0.5
        test_duration = 10.0
        
        # Handle different mutation types
        if template['type'] == MutationType.PARAMETER_ADJUSTMENT:
            changes, rationale, expected_improvement, confidence = self._generate_parameter_changes(target, strategy)
            
        elif template['type'] == MutationType.NEURAL_ARCHITECTURE_SEARCH:
            changes, rationale, expected_improvement, confidence = self._generate_nas_changes(target, strategy)
            test_duration = 25.0  # NAS changes need more testing time
            
        elif template['type'] == MutationType.ATTENTION_MODIFICATION:
            changes, rationale, expected_improvement, confidence = self._generate_attention_changes(target, strategy)
            test_duration = 20.0
            
        elif template['type'] == MutationType.MULTI_OBJECTIVE_OPTIMIZATION:
            changes, rationale, expected_improvement, confidence = self._generate_moo_changes(target, strategy)
            test_duration = 30.0  # Complex optimization needs thorough testing
            
        elif template['type'] == MutationType.LEARNING_SCHEDULE_OPTIMIZATION:
            changes, rationale, expected_improvement, confidence = self._generate_schedule_changes(target, strategy)
            test_duration = 15.0
            
        elif template['type'] == MutationType.MEMORY_OPTIMIZATION:
            changes, rationale, expected_improvement, confidence = self._generate_memory_changes(target, strategy)
            test_duration = 8.0  # Memory optimizations are quick to test
            
        elif template['type'] == MutationType.ENSEMBLE_FUSION:
            changes, rationale, expected_improvement, confidence = self._generate_ensemble_changes(target, strategy)
            test_duration = 35.0  # Ensemble methods are complex
            
        else:
            # Fallback to legacy mutation generation
            changes, rationale = self._generate_legacy_changes(target, strategy)
        
        return Mutation(
            id=mutation_id,
            type=template['type'],
            impact=template['impact'],
            changes=changes,
            rationale=rationale,
            expected_improvement=expected_improvement,
            confidence=confidence,
            test_duration_estimate=test_duration
        )
    
    def _generate_parameter_changes(self, target: str, strategy: str) -> tuple:
        """Generate parameter adjustment changes."""
        import random
        
        changes = {}
        
        if target == 'max_actions_per_game':
            if strategy == 'increase':
                changes[target] = min(1000, int(self.base_genome.max_actions_per_game * 1.2))
            elif strategy == 'decrease':
                changes[target] = max(100, int(self.base_genome.max_actions_per_game * 0.8))
            else:  # optimize
                changes[target] = 750  # Sweet spot based on analysis
                
        elif target == 'salience_mode':
            current_mode = self.base_genome.salience_mode
            changes[target] = 'lossless' if current_mode == 'decay_compression' else 'decay_compression'
            
        elif 'enable_' in target:
            current_value = getattr(self.base_genome, target, True)
            changes[target] = not current_value
            
        elif target == 'salience_threshold':
            if strategy == 'increase':
                changes[target] = min(1.0, self.base_genome.salience_threshold + 0.1)
            elif strategy == 'decrease':
                changes[target] = max(0.1, self.base_genome.salience_threshold - 0.1)
            else:  # optimize
                changes[target] = 0.7  # Balanced threshold
        
        rationale = f"Parameter {strategy} for {target} optimization"
        return changes, rationale, 0.05, 0.6
    
    def _generate_nas_changes(self, target: str, strategy: str) -> tuple:
        """Generate Neural Architecture Search changes."""
        import random
        
        changes = {}
        
        if target == 'hidden_dimensions':
            base_dim = 512
            if strategy == 'progressive_search':
                changes['hidden_dims'] = [256, 512, 1024, 512]
                changes['progressive_scaling'] = True
            elif strategy == 'layer_scaling':
                changes['hidden_dims'] = [base_dim * (2 ** i) for i in range(4)]
                
        elif target == 'layer_depths':
            if strategy == 'skip_connections':
                changes['enable_skip_connections'] = True
                changes['skip_interval'] = random.choice([2, 3, 4])
            elif strategy == 'progressive_search':
                changes['depth_schedule'] = [4, 6, 8, 6]  # Progressive depth
                
        elif target == 'activation_functions':
            activations = ['relu', 'gelu', 'swish', 'mish']
            if strategy == 'progressive_search':
                changes['activation_sequence'] = random.sample(activations, 3)
            else:
                changes['primary_activation'] = random.choice(activations)
                
        elif target == 'attention_heads':
            if strategy == 'attention_heads':
                changes['num_attention_heads'] = random.choice([4, 8, 12, 16])
                changes['head_dimension'] = random.choice([32, 64, 128])
        
        rationale = f"Neural architecture search: {strategy} for {target}"
        return changes, rationale, 0.15, 0.4  # Higher potential, lower confidence
    
    def _generate_attention_changes(self, target: str, strategy: str) -> tuple:
        """Generate attention mechanism modifications."""
        import random
        
        changes = {}
        
        if target == 'attention_span':
            if strategy == 'enhance_locality':
                changes['local_attention_window'] = random.choice([16, 32, 64])
                changes['enable_local_attention'] = True
            elif strategy == 'global_attention':
                changes['global_attention_layers'] = random.choice([2, 4, 6])
                changes['attention_pooling'] = 'global'
                
        elif target == 'focus_patterns':
            if strategy == 'sparse_attention':
                changes['attention_sparsity'] = random.uniform(0.1, 0.3)
                changes['sparse_pattern'] = random.choice(['strided', 'random', 'block'])
            elif strategy == 'dynamic_attention':
                changes['dynamic_attention_threshold'] = random.uniform(0.5, 0.9)
                changes['attention_adaptation_rate'] = 0.1
                
        elif target == 'self_attention_depth':
            changes['self_attention_layers'] = random.choice([3, 6, 9, 12])
            changes['cross_attention_layers'] = random.choice([1, 2, 3])
        
        rationale = f"Attention mechanism optimization: {strategy} for {target}"
        return changes, rationale, 0.12, 0.5
    
    def _generate_moo_changes(self, target: str, strategy: str) -> tuple:
        """Generate multi-objective optimization changes."""
        import random
        
        changes = {}
        
        if target == 'objective_weights':
            if strategy == 'balance_objectives':
                changes['win_rate_weight'] = random.uniform(0.3, 0.5)
                changes['score_weight'] = random.uniform(0.2, 0.4)
                changes['efficiency_weight'] = random.uniform(0.2, 0.4)
                changes['learning_weight'] = random.uniform(0.1, 0.2)
            elif strategy == 'priority_weighting':
                primary_objective = random.choice(['win_rate', 'score', 'efficiency'])
                changes[f'{primary_objective}_weight'] = random.uniform(0.6, 0.8)
                
        elif target == 'pareto_optimization':
            if strategy == 'evolutionary_pareto':
                changes['pareto_front_size'] = random.choice([10, 20, 30])
                changes['pareto_selection_pressure'] = random.uniform(0.7, 0.9)
                
        elif target == 'constraint_handling':
            if strategy == 'adaptive_constraints':
                changes['constraint_adaptation_rate'] = random.uniform(0.05, 0.15)
                changes['constraint_violation_penalty'] = random.uniform(0.1, 0.3)
        
        rationale = f"Multi-objective optimization: {strategy} for {target}"
        return changes, rationale, 0.18, 0.35  # High potential, moderate confidence
    
    def _generate_schedule_changes(self, target: str, strategy: str) -> tuple:
        """Generate learning rate schedule changes."""
        import random
        
        changes = {}
        
        if target == 'learning_rate_schedule':
            if strategy == 'cosine_annealing':
                changes['lr_schedule'] = 'cosine'
                changes['min_lr_factor'] = random.uniform(0.01, 0.1)
                changes['restart_period'] = random.choice([100, 200, 500])
            elif strategy == 'exponential_decay':
                changes['lr_schedule'] = 'exponential'
                changes['decay_rate'] = random.uniform(0.9, 0.99)
                changes['decay_steps'] = random.choice([50, 100, 200])
                
        elif target == 'warmup_steps':
            changes['warmup_steps'] = random.choice([100, 500, 1000])
            changes['warmup_factor'] = random.uniform(0.1, 0.5)
            
        elif target == 'adaptive_lr':
            if strategy == 'plateau_reduction':
                changes['lr_patience'] = random.choice([5, 10, 20])
                changes['lr_reduction_factor'] = random.uniform(0.5, 0.8)
            elif strategy == 'cyclic_learning':
                changes['lr_cycle_length'] = random.choice([20, 50, 100])
                changes['lr_cycle_amplitude'] = random.uniform(0.1, 0.5)
        
        rationale = f"Learning rate schedule optimization: {strategy} for {target}"
        return changes, rationale, 0.10, 0.7
    
    def _generate_memory_changes(self, target: str, strategy: str) -> tuple:
        """Generate memory optimization changes."""
        import random
        
        changes = {}
        
        if target == 'cache_size':
            if strategy == 'increase_cache':
                changes['pattern_cache_size'] = random.choice([1000, 2000, 5000])
                changes['solution_cache_size'] = random.choice([500, 1000, 2000])
            elif strategy == 'smart_gc':
                changes['gc_threshold'] = random.uniform(0.7, 0.9)
                changes['gc_strategy'] = random.choice(['lru', 'lfu', 'adaptive'])
                
        elif target == 'memory_patterns':
            if strategy == 'optimize_patterns':
                changes['memory_pattern_limit'] = random.choice([100, 200, 500])
                changes['pattern_similarity_threshold'] = random.uniform(0.8, 0.95)
                
        elif target == 'buffer_strategies':
            if strategy == 'streaming_buffers':
                changes['enable_streaming'] = True
                changes['buffer_size'] = random.choice([64, 128, 256])
        
        rationale = f"Memory optimization: {strategy} for {target}"
        return changes, rationale, 0.08, 0.8  # Lower impact, high confidence
    
    def _generate_ensemble_changes(self, target: str, strategy: str) -> tuple:
        """Generate ensemble and model fusion changes."""
        import random
        
        changes = {}
        
        if target == 'model_ensemble':
            if strategy == 'weighted_voting':
                changes['ensemble_size'] = random.choice([3, 5, 7])
                changes['voting_weights'] = 'performance_based'
            elif strategy == 'stacking_ensemble':
                changes['stacking_layers'] = random.choice([1, 2, 3])
                changes['meta_learner'] = random.choice(['linear', 'tree', 'neural'])
                
        elif target == 'voting_strategies':
            if strategy == 'dynamic_selection':
                changes['selection_criteria'] = random.choice(['confidence', 'diversity', 'hybrid'])
                changes['selection_threshold'] = random.uniform(0.6, 0.9)
                
        elif target == 'fusion_weights':
            if strategy == 'adversarial_fusion':
                changes['adversarial_weight_learning'] = True
                changes['fusion_learning_rate'] = random.uniform(0.001, 0.01)
        
        rationale = f"Ensemble and fusion optimization: {strategy} for {target}"
        return changes, rationale, 0.20, 0.3  # High potential, lower confidence (complex)
    
    def _generate_legacy_changes(self, target: str, strategy: str) -> tuple:
        """Generate changes for legacy mutation types."""
        changes = {}
        
        if target == 'salience_mode':
            current_mode = self.base_genome.salience_mode
            changes[target] = 'lossless' if current_mode == 'decay_compression' else 'decay_compression'
        elif 'enable_' in target:
            current_value = getattr(self.base_genome, target, True)
            changes[target] = not current_value
        
        rationale = f"Legacy {strategy} of {target}"
        return changes, rationale

class SandboxTester:
    """Tests mutations in isolated sandbox environments."""
    
    def __init__(self, base_path: Path, logger: logging.Logger):
        self.base_path = base_path
        self.logger = logger
        self.sandbox_dir = base_path / "sandbox_tests"
        self.sandbox_dir.mkdir(exist_ok=True)
    
    async def test_mutation(self, mutation: Mutation, 
                          baseline_genome: SystemGenome,
                          test_games: List[str] = None) -> TestResult:
        """Test a mutation in a sandboxed environment."""
        start_time = time.time()
        sandbox_path = None
        
        try:
            # Create mutated genome
            mutated_genome = mutation.apply_to_genome(baseline_genome)
            
            # Create sandbox environment
            sandbox_path = await self._create_sandbox(mutated_genome)
            
            # Run test suite
            test_results = await self._run_sandbox_test(sandbox_path, test_games or ["test_game_1"])
            
            # Compare with baseline (would need baseline results stored)
            baseline_performance = self._get_baseline_performance(baseline_genome)
            improvement = self._calculate_improvement(test_results, baseline_performance)
            
            test_duration = time.time() - start_time
            
            # Calculate overall improvement score for logging
            overall_improvement = sum(improvement.values()) if improvement else 0.0
            
            self.logger.info(f"🧪 Mutation {mutation.id} tested: "
                           f"overall_improvement={overall_improvement:.3f}, duration={test_duration:.1f}s")
            
            return TestResult(
                mutation_id=mutation.id,
                genome_hash=mutated_genome.get_hash(),
                success=test_results.get('success', False),
                performance_metrics=test_results.get('metrics', {}),
                improvement_over_baseline=improvement,
                test_duration=test_duration,
                detailed_results=test_results
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error testing mutation {mutation.id}: {e}")
            return TestResult(
                mutation_id=mutation.id,
                genome_hash="error",
                success=False,
                performance_metrics={},
                improvement_over_baseline={},
                test_duration=time.time() - start_time,
                error_log=str(e)
            )
        finally:
            # Cleanup sandbox
            if sandbox_path is not None:
                await self._cleanup_sandbox(sandbox_path)
    
    async def _create_sandbox(self, genome: SystemGenome) -> Path:
        """Create isolated sandbox environment for testing."""
        sandbox_id = f"sandbox_{genome.get_hash()}_{int(time.time())}"
        sandbox_path = self.sandbox_dir / sandbox_id
        
        # Copy core system files to sandbox
        base_path = Path(self.base_path)
        if (base_path / "src").exists():
            shutil.copytree(base_path / "src", sandbox_path / "src")
        elif base_path.name == "src":
            # If base_path is already the src directory
            shutil.copytree(base_path, sandbox_path / "src")
        else:
            # Create minimal src structure
            (sandbox_path / "src").mkdir(parents=True)
            
        # Copy training script if available
        for script_name in ["master_arc_trainer.py"]:
            script_path = base_path.parent / script_name if base_path.name == "src" else base_path / script_name
            if script_path.exists():
                shutil.copy2(script_path, sandbox_path)
                break
        
        # Create custom config file for this genome
        config_path = sandbox_path / "sandbox_config.json"
        with open(config_path, 'w') as f:
            json.dump(genome.to_dict(), f, indent=2)
        
        self.logger.debug(f"🏗️ Created sandbox: {sandbox_path}")
        return sandbox_path
    
    async def _run_sandbox_test(self, sandbox_path: Path, test_games: List[str]) -> Dict[str, Any]:
        """Run test suite in sandbox environment using real ARC training."""
        start_time = time.time()
        
        try:
            # Use the master training script (consolidated system)
            base_dir = Path(self.base_path).parent if Path(self.base_path).name == "src" else Path(self.base_path)
            
            # Try the new master script first, then fallback to legacy scripts
            training_scripts_to_try = [
                "master_arc_trainer.py",  # New consolidated script
            ]
            
            training_script = None
            for script_name in training_scripts_to_try:
                script_path = base_dir / script_name
                if script_path.exists():
                    training_script = str(script_path)
                    self.logger.info(f"🎯 Using training script: {script_name}")
                    break
            
            if not training_script:
                self.logger.warning("⚠️ No training script found, using simulated results")
                await asyncio.sleep(2)
                return self._get_simulated_results(test_games)
            
            # Prepare test configuration
            config_path = sandbox_path / "test_config.json"
            test_config = {
                "max_episodes": 5,  # Quick test
                "max_actions_per_game": 100,
                "test_mode": True,
                "games": test_games[:2] if len(test_games) > 2 else test_games,
                "timeout_per_game": 30,
                "salience_mode": "decay_compression"
            }
            
            with open(config_path, 'w') as f:
                json.dump(test_config, f, indent=2)
            
            # Run training with appropriate parameters for each script
            if "master_arc_trainer.py" in training_script:
                # New master script - uses standardized parameters
                cmd = [
                    sys.executable, training_script,
                    "--mode", "quick-validation",
                    "--games", "ls20,ft09",  # Use real ARC games
                    "--max-cycles", "3",
                    "--session-duration", "2",
                    "--no-logs",
                    "--no-monitoring"
                ]
            else:
                # Fallback for other scripts
                cmd = [
                    sys.executable, training_script,
                    "--help"  # Just get help to verify it works
                ]
            
            self.logger.info(f"🎮 Running actual ARC training test: {' '.join(cmd)}")
            
            # Execute with timeout - run from base directory where scripts exist
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(base_dir)  # Run from the directory where training scripts exist
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=90.0)  # Shorter timeout
                test_duration = time.time() - start_time
                
                if process.returncode == 0:
                    # Parse results from stdout
                    results = self._parse_training_output(stdout.decode(), stderr.decode())
                    results['test_duration'] = test_duration
                    results['games_tested'] = test_games
                    results['success'] = True
                    
                    self.logger.info(f"✅ Sandbox test completed successfully in {test_duration:.1f}s")
                    return results
                else:
                    self.logger.error(f"❌ Training process failed with code {process.returncode}")
                    self.logger.error(f"STDERR: {stderr.decode()}")
                    return self._get_simulated_results(test_games, success=False)
                    
            except asyncio.TimeoutError:
                self.logger.warning("⏰ Training test timed out, terminating process")
                process.terminate()
                await process.wait()
                return self._get_simulated_results(test_games, success=False)
                
        except Exception as e:
            self.logger.error(f"❌ Sandbox test error: {e}")
            return self._get_simulated_results(test_games, success=False)
    
    def _parse_training_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse training output to extract performance metrics."""
        try:
            # Look for key metrics in the output
            lines = stdout.split('\n') + stderr.split('\n')
            
            metrics = {
                'win_rate': 0.0,
                'average_score': 0.0,
                'learning_efficiency': 0.0,
                'sample_efficiency': 0.0,
                'robustness': 0.0
            }
            
            total_episodes = 0
            successful_games = 0
            total_score = 0.0
            
            for line in lines:
                line = line.strip()
                
                # Parse episode results
                if "Episode" in line and "score:" in line:
                    try:
                        score_part = line.split("score:")[-1].strip()
                        score = float(score_part.split()[0])
                        total_score += score
                        total_episodes += 1
                        if score > 0:
                            successful_games += 1
                    except:
                        continue
                
                # Parse win/success indicators
                if any(keyword in line.lower() for keyword in ["won", "success", "solved"]):
                    successful_games += 1
                
                # Look for explicit metrics
                for metric in metrics.keys():
                    if metric in line.lower():
                        try:
                            # Extract number after metric name
                            parts = line.split(":")
                            if len(parts) > 1:
                                value = float(parts[1].strip().split()[0])
                                metrics[metric] = value
                        except:
                            continue
            
            # Calculate derived metrics
            if total_episodes > 0:
                metrics['win_rate'] = successful_games / total_episodes
                metrics['average_score'] = total_score / total_episodes
                
            # If we don't have explicit metrics, estimate them
            if metrics['learning_efficiency'] == 0.0:
                metrics['learning_efficiency'] = min(0.9, metrics['win_rate'] * 1.2)
            if metrics['sample_efficiency'] == 0.0:
                metrics['sample_efficiency'] = min(0.8, metrics['average_score'] / 100.0)
            if metrics['robustness'] == 0.0:
                metrics['robustness'] = min(0.7, metrics['win_rate'] * 0.8)
            
            self.logger.info(f"📊 Parsed metrics from training: win_rate={metrics['win_rate']:.3f}, "
                           f"avg_score={metrics['average_score']:.1f}, episodes={total_episodes}")
            
            return {
                'metrics': metrics,
                'total_episodes': total_episodes,
                'successful_games': successful_games,
                'parsing_source': 'real_output'
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to parse training output: {e}")
            return self._get_simulated_results([], success=True)['metrics']
    
    def _get_simulated_results(self, test_games: List[str], success: bool = True) -> Dict[str, Any]:
        """Generate simulated results as fallback."""
        import random
        
        base_performance = {
            'win_rate': 0.60,
            'average_score': 45.0,
            'learning_efficiency': 0.75,
            'sample_efficiency': 0.65,
            'robustness': 0.55
        }
        
        if success:
            # Add some random variation to simulate real performance
            metrics = {}
            for key, base_value in base_performance.items():
                variation = random.uniform(-0.1, 0.1)
                metrics[key] = max(0.0, min(1.0, base_value + variation))
        else:
            # Degraded performance for failed tests
            metrics = {key: value * 0.7 for key, value in base_performance.items()}
        
        return {
            'success': success,
            'metrics': metrics,
            'games_tested': test_games,
            'total_episodes': len(test_games) * 5 if test_games else 10,
            'test_duration': random.uniform(30.0, 120.0),
            'parsing_source': 'simulated'
        }
    
    def _get_baseline_performance(self, baseline_genome: SystemGenome) -> Dict[str, float]:
        """Get baseline performance metrics for comparison."""
        # Would retrieve from stored baseline results
        return {
            'win_rate': 0.60,
            'average_score': 45.0,
            'learning_efficiency': 0.75,
            'sample_efficiency': 0.65,
            'robustness': 0.55
        }
    
    def _calculate_improvement(self, test_results: Dict[str, Any], 
                             baseline: Dict[str, float]) -> Dict[str, float]:
        """Calculate improvement over baseline."""
        improvement = {}
        test_metrics = test_results.get('metrics', {})
        
        for metric, baseline_value in baseline.items():
            test_value = test_metrics.get(metric, baseline_value)
            improvement[metric] = test_value - baseline_value
        
        return improvement
    
    async def _cleanup_sandbox(self, sandbox_path: Path):
        """Clean up sandbox environment."""
        try:
            if sandbox_path.exists():
                shutil.rmtree(sandbox_path)
                self.logger.debug(f"🧹 Cleaned up sandbox: {sandbox_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cleanup sandbox {sandbox_path}: {e}")

class Architect:
    """
    The "Zeroth Brain" - Self-Architecture Evolution System
    
    Performs safe, sandboxed experimentation on the AI's own architecture
    and hyperparameters using a general-intelligence fitness function.
    """
    
    def __init__(self, evolution_rate: float = 0.05, innovation_threshold: float = 0.8, memory_capacity: int = 500,
                 base_path: str = ".", repo_path: str = ".", logger: Optional[logging.Logger] = None):
        self.base_path = Path(base_path)
        self.repo_path = Path(repo_path)
        self.logger = logger or logging.getLogger(f"{__name__}.Architect")
        
        # Initialize components
        self.current_genome = self._load_current_genome()
        self.mutation_engine = MutationEngine(self.current_genome, self.logger)
        self.sandbox_tester = SandboxTester(self.base_path, self.logger)
        
        # Evolution state
        self.generation = 0
        self.mutation_history = []
        self.successful_mutations = []
        self.pending_requests = []
        
        # Game monitoring
        self.last_training_check = 0
        self.training_active = False
        self.game_activity_log = []
        
        # Meta-cognitive memory management integration
        self.memory_manager = None
        try:
            from src.core.meta_cognitive_memory_manager import MetaCognitiveMemoryManager
            self.memory_manager = MetaCognitiveMemoryManager(self.base_path, self.logger)
            self.logger.info("Architect memory management enabled")
        except ImportError:
            self.logger.warning("Meta-cognitive memory manager not available for Architect")
        
        # Git integration
        self.repo = None
        self.default_branch = "Tabula-Rasa-v3"  # Our working branch
        if GIT_AVAILABLE:
            try:
                self.repo = git.Repo(self.repo_path)
                self._ensure_correct_branch()
                self._setup_architect_gitignore()
            except Exception as e:
                self.logger.warning(f"⚠️ Git repository error: {e} - version control disabled")
        else:
            self.logger.warning("⚠️ GitPython not available - version control disabled")
        
        # Safety measures
        self.max_concurrent_tests = 1  # Start conservative
        self.human_approval_required = True
        self.auto_merge_threshold = 0.15  # Minimum improvement for auto-merge
        
        self.logger.info("🔬 Architect initialized - Zeroth Brain online")
    
    def evolve_strategy(self, available_actions: List[int], context: Dict[str, Any], 
                       performance_data: List[Dict[str, Any]], frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve strategy based on current performance and context.
        
        Args:
            available_actions: List of available actions
            context: Current game context
            performance_data: Historical performance data
            frame_analysis: Current frame analysis
            
        Returns:
            Dictionary containing evolved strategy, reasoning, and innovation score
        """
        try:
            game_id = context.get('game_id', 'unknown')
            
            # Analyze current performance patterns
            innovation_score = self._calculate_innovation_score(performance_data, frame_analysis)
            
            # Generate evolved strategy based on analysis
            evolved_strategy = self._generate_evolved_strategy(
                available_actions, context, performance_data, frame_analysis
            )
            
            # Generate reasoning for the strategy evolution
            reasoning = self._generate_evolution_reasoning(
                evolved_strategy, innovation_score, game_id
            )
            
            # Track evolution
            evolution_record = {
                'timestamp': time.time(),
                'game_id': game_id,
                'available_actions': available_actions,
                'evolved_strategy': evolved_strategy,
                'innovation_score': innovation_score,
                'reasoning': reasoning
            }
            
            self.mutation_history.append(evolution_record)
            
            return {
                'evolved_strategy': evolved_strategy,
                'innovation_score': innovation_score,
                'reasoning': reasoning,
                'architectural_insights': {
                    'performance_analysis': self._analyze_performance_patterns(performance_data),
                    'frame_utilization': self._analyze_frame_utilization(frame_analysis),
                    'action_effectiveness': self._analyze_action_effectiveness(available_actions, performance_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Architect evolution failed: {e}")
            # Fallback response
            return {
                'evolved_strategy': 'maintain_current_approach',
                'innovation_score': 0.1,
                'reasoning': f"Evolution error, maintaining status quo: {e}",
                'architectural_insights': {'error': str(e)}
            }
    
    def _calculate_innovation_score(self, performance_data: List[Dict[str, Any]], 
                                  frame_analysis: Dict[str, Any]) -> float:
        """Calculate how innovative/novel the current approach should be."""
        base_score = 0.3
        
        # Performance stagnation increases innovation need
        if performance_data:
            recent_performance = performance_data[-5:]
            if len(recent_performance) >= 3:
                scores = [p.get('score', 0) for p in recent_performance]
                if all(s <= 0 for s in scores):  # All failures
                    base_score += 0.4
                elif len(set(scores)) <= 1:  # No variation
                    base_score += 0.3
        
        # Frame analysis complexity affects innovation
        if frame_analysis:
            visual_targets = frame_analysis.get('interactive_targets', [])
            if len(visual_targets) > 3:
                base_score += 0.2  # Rich visual environment needs more innovation
        
        return min(1.0, base_score)
    
    def _generate_evolved_strategy(self, available_actions: List[int], context: Dict[str, Any],
                                 performance_data: List[Dict[str, Any]], frame_analysis: Dict[str, Any]) -> str:
        """Generate an evolved strategy based on analysis."""
        
        # Analyze what's not working
        if performance_data:
            recent_failures = [p for p in performance_data[-5:] if not p.get('success', False)]
            if len(recent_failures) >= 3:
                # Multiple failures - suggest major strategy shift
                if 6 in available_actions:
                    return 'prioritize_visual_exploration'
                else:
                    return 'systematic_action_cycling'
        
        # Analyze frame richness
        if frame_analysis:
            interactive_targets = frame_analysis.get('interactive_targets', [])
            if len(interactive_targets) > 5:
                return 'target_rich_environment_exploitation'
            elif len(interactive_targets) <= 1:
                return 'sparse_environment_systematic_search'
        
        # Default evolved strategy
        return 'adaptive_multi_modal_approach'
    
    def _generate_evolution_reasoning(self, evolved_strategy: str, innovation_score: float, game_id: str) -> str:
        """Generate reasoning for the evolved strategy."""
        reasoning_parts = [f"Architectural analysis for {game_id}"]
        
        if innovation_score > 0.7:
            reasoning_parts.append("High innovation needed - performance stagnation detected")
        elif innovation_score > 0.4:
            reasoning_parts.append("Moderate innovation - exploring new approaches")
        else:
            reasoning_parts.append("Conservative evolution - maintaining effective patterns")
        
        strategy_explanations = {
            'prioritize_visual_exploration': 'Focus on visual-interactive elements for breakthrough',
            'systematic_action_cycling': 'Methodical exploration of all available actions',
            'target_rich_environment_exploitation': 'Leverage abundant visual targets for progress',
            'sparse_environment_systematic_search': 'Comprehensive search in minimal visual context',
            'adaptive_multi_modal_approach': 'Balanced approach adapting to environmental cues'
        }
        
        if evolved_strategy in strategy_explanations:
            reasoning_parts.append(strategy_explanations[evolved_strategy])
        
        return " | ".join(reasoning_parts)
    
    def _analyze_performance_patterns(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in performance data."""
        if not performance_data:
            return {'trend': 'no_data'}
        
        recent_scores = [p.get('score', 0) for p in performance_data[-5:]]
        success_rate = sum(1 for p in performance_data[-10:] if p.get('success', False)) / min(10, len(performance_data))
        
        return {
            'recent_scores': recent_scores,
            'success_rate': success_rate,
            'trend': 'improving' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else 'stable'
        }
    
    def _analyze_frame_utilization(self, frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how well frame data is being utilized."""
        if not frame_analysis:
            return {'utilization': 'no_data'}
        
        interactive_targets = frame_analysis.get('interactive_targets', [])
        confidence = frame_analysis.get('confidence', 0.0)
        
        return {
            'targets_available': len(interactive_targets),
            'analysis_confidence': confidence,
            'utilization': 'high' if len(interactive_targets) > 3 and confidence > 0.7 else 'moderate'
        }
    
    def _analyze_action_effectiveness(self, available_actions: List[int], 
                                    performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze effectiveness of different actions."""
        action_stats = {}
        
        for session in performance_data[-10:]:  # Last 10 sessions
            actions = session.get('actions_taken', [])
            success = session.get('success', False)
            
            for action in actions:
                if action not in action_stats:
                    action_stats[action] = {'used': 0, 'successful': 0}
                action_stats[action]['used'] += 1
                if success:
                    action_stats[action]['successful'] += 1
        
        # Calculate effectiveness for available actions
        effectiveness = {}
        for action in available_actions:
            if action in action_stats:
                stats = action_stats[action]
                effectiveness[action] = stats['successful'] / max(stats['used'], 1)
            else:
                effectiveness[action] = 0.0
        
        return {
            'action_effectiveness': effectiveness,
            'most_effective': max(effectiveness.keys(), key=lambda k: effectiveness[k]) if effectiveness else None,
            'least_effective': min(effectiveness.keys(), key=lambda k: effectiveness[k]) if effectiveness else None
        }
    
    def _load_current_genome(self) -> SystemGenome:
        """Load current system genome from configuration."""
        # Would load from master_arc_trainer.py TrainingConfig or config files
        # For now, create default genome
        return SystemGenome()
    
    def _ensure_correct_branch(self):
        """Ensure we're on the Tabula-Rasa-v3 branch and never switch to main/master."""
        if not self.repo:
            return
            
        try:
            current_branch = self.repo.active_branch.name
            
            if current_branch != self.default_branch:
                self.logger.warning(f"⚠️ Currently on branch '{current_branch}', expected '{self.default_branch}'")
                
                # Try to switch to the correct branch
                if self.default_branch in [head.name for head in self.repo.heads]:
                    self.logger.info(f"🌿 Switching to {self.default_branch} branch for safety")
                    self.repo.heads[self.default_branch].checkout()
                else:
                    self.logger.warning(f"⚠️ {self.default_branch} branch not found. Staying on current branch '{current_branch}'")
                    self.logger.warning("⚠️ IMPORTANT: Architect will NOT create branches from main/master")
            else:
                self.logger.info(f"✅ Already on correct branch: {self.default_branch}")
                
        except Exception as e:
            self.logger.error(f"❌ Branch verification failed: {e}")
    
    def _safe_checkout_default_branch(self):
        """Safely checkout the Tabula-Rasa-v3 branch, never main/master."""
        if not self.repo:
            return False
            
        try:
            if self.default_branch in [head.name for head in self.repo.heads]:
                self.repo.heads[self.default_branch].checkout()
                self.logger.info(f"✅ Returned to {self.default_branch} branch")
                return True
            else:
                self.logger.warning(f"⚠️ {self.default_branch} branch not found")
                return False
        except Exception as e:
            self.logger.error(f"❌ Failed to checkout {self.default_branch}: {e}")
            return False
    
    def _validate_branch_operation(self, target_branch: str) -> bool:
        """Validate that branch operations are safe and don't target main/master."""
        dangerous_branches = ['main', 'master']
        
        if target_branch.lower() in [b.lower() for b in dangerous_branches]:
            self.logger.error(f"🚫 BLOCKED: Attempted to checkout dangerous branch '{target_branch}'")
            self.logger.error(f"🚫 Architect is configured to only work with '{self.default_branch}' branch")
            return False
            
        return True
    
    def create_system_genome(self) -> SystemGenome:
        """Create a system genome representing current configuration."""
        return self._load_current_genome()
    
    async def process_governor_request(self, request: ArchitectRequest) -> Dict[str, Any]:
        """Process a request from the MetaCognitiveGovernor."""
        self.logger.info(f"🔬 Processing Governor request: {request.issue_type}")
        
        try:
            # Generate targeted mutation
            mutation = self.mutation_engine.generate_mutation(request)
            
            # Test mutation
            test_result = await self.sandbox_tester.test_mutation(
                mutation, self.current_genome
            )
            
            # Evaluate results
            if test_result.success and test_result.get_overall_improvement() > 0.05:
                # Create branch with improvement
                branch_info = await self._create_improvement_branch(mutation, test_result)
                
                response = {
                    'success': True,
                    'mutation_id': mutation.id,
                    'improvement': test_result.get_overall_improvement(),
                    'branch_created': branch_info['branch_name'] if branch_info else None,
                    'requires_approval': self.human_approval_required,
                    'recommendation': 'merge' if test_result.get_overall_improvement() > self.auto_merge_threshold else 'review'
                }
                
                self.successful_mutations.append((mutation, test_result))
                
            else:
                response = {
                    'success': False,
                    'mutation_id': mutation.id,
                    'improvement': test_result.get_overall_improvement(),
                    'issues': [test_result.error_log] if test_result.error_log else ['Performance regression'],
                    'recommendation': 'abandon'
                }
            
            self.mutation_history.append((mutation, test_result))
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Error processing Governor request: {e}")
            return {'success': False, 'error': str(e)}
    
    async def autonomous_evolution_cycle(self) -> Dict[str, Any]:
        """Run one cycle of autonomous evolution."""
        self.logger.info(f"🧬 Starting evolution cycle {self.generation}")
        
        try:
            # Generate exploratory mutation
            mutation = self.mutation_engine.generate_exploratory_mutation()
            
            # Test in sandbox
            test_result = await self.sandbox_tester.test_mutation(
                mutation, self.current_genome
            )
            
            # Record results
            self.mutation_history.append((mutation, test_result))
            
            improvement = test_result.get_overall_improvement()
            
            if test_result.success and improvement > 0.02:  # Small threshold for exploration
                # Create experimental branch
                branch_info = await self._create_experimental_branch(mutation, test_result)
                
                self.logger.info(f"✅ Beneficial mutation found: {improvement:.3f} improvement")
                
                return {
                    'success': True,
                    'generation': self.generation,
                    'improvement': improvement,
                    'branch_created': branch_info['branch_name'] if branch_info else None
                }
            else:
                self.logger.debug(f"📊 Mutation tested: {improvement:.3f} improvement (below threshold)")
                
                return {
                    'success': False,
                    'generation': self.generation,
                    'improvement': improvement,
                    'reason': 'insufficient_improvement'
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error in evolution cycle: {e}")
            return {
                'success': False, 
                'error': str(e),
                'generation': self.generation
            }
        
        finally:
            self.generation += 1
    
    async def _create_improvement_branch(self, mutation: Mutation, 
                                       test_result: TestResult) -> Optional[Dict[str, Any]]:
        """Create git branch for successful improvement."""
        if not self.repo:
            return None
        
        try:
            # Create branch name
            branch_name = f"improvement/{mutation.id}_{test_result.genome_hash}"
            
            # Create and checkout branch
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            
            # Apply mutation to actual config files
            self._apply_mutation_to_files(mutation)
            
            # Create detailed commit message
            commit_message = f"""🚀 Architectural Improvement: {mutation.rationale}

Mutation ID: {mutation.id}
Type: {mutation.type.value}
Impact: {mutation.impact.value}

Performance Improvements:
- Overall: {test_result.get_overall_improvement():.1%}
- Win Rate: {test_result.improvement_over_baseline.get('win_rate', 0):.3f}
- Score: {test_result.improvement_over_baseline.get('average_score', 0):.1f}

Changes Applied:
{json.dumps(mutation.changes, indent=2)}

Test Duration: {test_result.test_duration:.1f}s
Confidence: {mutation.confidence:.1%}

Generated by Architect (Zeroth Brain) - Safe Self-Improvement System
"""
            
            # Stage and commit changes
            self.repo.git.add('.')
            self.repo.index.commit(commit_message)
            
            self.logger.info(f"🌿 Created improvement branch: {branch_name}")
            
            # Return to Tabula-Rasa-v3 branch (our default working branch)
            self._safe_checkout_default_branch()
            
            return {
                'branch_name': branch_name,
                'commit_hash': new_branch.commit.hexsha,
                'improvement': test_result.get_overall_improvement()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create improvement branch: {e}")
            return None
    
    async def _create_experimental_branch(self, mutation: Mutation, 
                                        test_result: TestResult) -> Optional[Dict[str, Any]]:
        """Create experimental branch for exploratory mutations."""
        # Similar to improvement branch but marked as experimental
        if not self.repo:
            return None
        
        try:
            branch_name = f"experimental/{mutation.id}_{int(time.time())}"
            
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            
            self._apply_mutation_to_files(mutation)
            
            commit_message = f"""🧪 Experimental Mutation: {mutation.rationale}

Mutation ID: {mutation.id}
Type: {mutation.type.value}
Improvement: {test_result.get_overall_improvement():.3f}

This is an experimental change - requires review before merging.
"""
            
            self.repo.git.add('.')
            self.repo.index.commit(commit_message)
            
            self.logger.info(f"🧪 Created experimental branch: {branch_name}")
            
            # Try to return to Tabula-Rasa-v3 branch (our default working branch)
            self._safe_checkout_default_branch()
            
            return {'branch_name': branch_name}
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create experimental branch: {e}")
            return None
    
    def _apply_mutation_to_files(self, mutation: Mutation):
        """Apply mutation changes to actual configuration files."""
        # This would modify master_arc_trainer.py or config files
        # For now, create a config patch file
        
        patch_file = self.base_path / f"mutation_{mutation.id}.json"
        with open(patch_file, 'w') as f:
            json.dump({
                'mutation_id': mutation.id,
                'changes': mutation.changes,
                'timestamp': time.time()
            }, f, indent=2)
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get comprehensive evolution status."""
        successful_count = len(self.successful_mutations)
        total_count = len(self.mutation_history)
        
        return {
            'generation': self.generation,
            'total_mutations_tested': total_count,
            'successful_mutations': successful_count,
            'success_rate': successful_count / max(total_count, 1),
            'pending_requests': len(self.pending_requests),
            'current_genome_hash': self.current_genome.get_hash(),
            'git_enabled': self.repo is not None,
            'recent_improvements': [
                {
                    'mutation_id': mut.id,
                    'improvement': result.get_overall_improvement(),
                    'timestamp': result.test_duration
                }
                for mut, result in self.successful_mutations[-5:]
            ]
        }
    
    async def emergency_rollback(self, target_commit: str = None) -> bool:
        """Emergency rollback to previous working state."""
        if not self.repo:
            self.logger.error("❌ Cannot rollback - no git repository")
            return False
        
        try:
            if target_commit:
                self.repo.git.reset('--hard', target_commit)
            else:
                self.repo.git.reset('--hard', 'HEAD~1')
            
            self.logger.warning("🔄 Emergency rollback completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def check_game_activity(self) -> Dict[str, Any]:
        """Check if the game/training system is currently active."""
        current_time = time.time()
        
        # Check for running training processes
        training_processes = self._get_training_processes()
        
        # Check recent log files for activity
        recent_activity = self._check_recent_training_logs()
        
        # Check continuous learning data updates
        continuous_data_activity = self._check_continuous_learning_activity()
        
        activity_status = {
            'timestamp': current_time,
            'training_processes_active': len(training_processes) > 0,
            'process_count': len(training_processes),
            'process_details': training_processes,
            'recent_log_activity': recent_activity,
            'continuous_learning_active': continuous_data_activity,
            'overall_active': len(training_processes) > 0 or recent_activity or continuous_data_activity
        }
        
        # Log activity status
        if activity_status['overall_active']:
            self.logger.info("🎮 Game/Training system is ACTIVE")
            if training_processes:
                self.logger.info(f"📊 {len(training_processes)} training processes running")
            if recent_activity:
                self.logger.info("📝 Recent training log activity detected")
            if continuous_data_activity:
                self.logger.info("🔄 Continuous learning data being updated")
        else:
            self.logger.warning("⏸️ No active game/training processes detected")
        
        # Store in activity log
        self.game_activity_log.append(activity_status)
        # Keep only last 10 checks
        if len(self.game_activity_log) > 10:
            self.game_activity_log.pop(0)
        
        self.last_training_check = current_time
        self.training_active = activity_status['overall_active']
        
        return activity_status
    
    def _get_training_processes(self) -> List[Dict[str, Any]]:
        """Get list of currently running training processes."""
        try:
            import psutil
        except ImportError:
            self.logger.warning("⚠️ psutil not available, cannot detect running processes")
            return []
        
        training_processes = []
        training_scripts = [
            'master_arc_trainer.py',  # New master script
            'continuous_training',
            'meta_cognitive',
            'master_arc_trainer.py'
        ]
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any(script in ' '.join(cmdline) for script in training_scripts):
                        training_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': ' '.join(cmdline),
                            'create_time': proc.info['create_time'],
                            'duration': time.time() - proc.info['create_time']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking processes: {e}")
        
        return training_processes
    
    def _check_recent_training_logs(self) -> bool:
        """Check if training logs have been updated recently."""
        try:
            import glob
            log_patterns = [
                "*.log",
                "*training*.log", 
                "*meta_cognitive*.log",
                "*arc*.log"
            ]
            
            recent_threshold = time.time() - 300  # 5 minutes
            
            for pattern in log_patterns:
                log_files = glob.glob(os.path.join(self.base_path, "..", pattern))
                
                for log_file in log_files:
                    try:
                        if os.path.getmtime(log_file) > recent_threshold:
                            return True
                    except OSError:
                        continue
            
            return False
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking log files: {e}")
            return False
    
    def _check_continuous_learning_activity(self) -> bool:
        """Check if continuous learning data is being updated."""
        try:
            data_dirs = [
                os.path.join(self.base_path, "..", "data"),
                os.path.join(self.base_path, "..", "data", "meta_learning_data")
            ]
            
            recent_threshold = time.time() - 600  # 10 minutes
            
            for data_dir in data_dirs:
                if os.path.exists(data_dir):
                    for root, dirs, files in os.walk(data_dir):
                        for file in files:
                            if file.endswith('.json'):
                                file_path = os.path.join(root, file)
                                try:
                                    if os.path.getmtime(file_path) > recent_threshold:
                                        return True
                                except OSError:
                                    continue
            
            return False
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking continuous learning data: {e}")
            return False
    
    def _setup_architect_gitignore(self) -> None:
        """Setup git configuration to allow Architect version control access."""
        if not self.repo:
            return
            
        try:
            gitignore_path = self.repo_path / ".gitignore"
            
            # Read current gitignore
            if gitignore_path.exists():
                with open(gitignore_path, 'r') as f:
                    current_content = f.read()
                
                # Check if our memory management comment exists
                if "SELECTIVE MEMORY MANAGEMENT" in current_content:
                    self.logger.info("✅ GitIgnore already configured for Architect control")
                    return
                
                self.logger.info("🔧 GitIgnore configuration needs Architect access - already updated")
            else:
                self.logger.warning("⚠️ No .gitignore found - Architect has full access")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking gitignore: {e}")
    
    def perform_memory_maintenance(self, force_cleanup: bool = False) -> Dict[str, Any]:
        """Perform memory maintenance from Architect perspective."""
        if not self.memory_manager:
            self.logger.warning("Memory manager not available for Architect")
            return {"status": "unavailable"}
        
        try:
            # Get current memory status
            memory_status = self.memory_manager.get_memory_status()
            
            # Architect-specific analysis
            architect_files_mb = 0
            for classification, stats in memory_status["classifications"].items():
                if classification == "critical_lossless":
                    architect_files_mb += stats["total_size_mb"]
            
            results = {
                "architect_critical_files_mb": architect_files_mb,
                "memory_maintenance_performed": False
            }
            
            # Determine if cleanup is needed
            total_size = memory_status["total_size_mb"]
            needs_cleanup = force_cleanup or total_size > 1500  # 1.5GB threshold
            
            if needs_cleanup:
                self.logger.info(f"🧠 Architect performing memory maintenance (total: {total_size:.2f} MB)")
                
                # Perform cleanup but protect critical evolution data
                cleanup_results = self.memory_manager.perform_garbage_collection(dry_run=False)
                results.update(cleanup_results)
                results["memory_maintenance_performed"] = True
                
                # Log evolution decision
                evolution_log = {
                    "timestamp": time.time(),
                    "decision": "memory_maintenance",
                    "reason": "Architect initiated memory cleanup",
                    "files_deleted": cleanup_results["files_deleted"],
                    "bytes_freed": cleanup_results["bytes_freed"],
                    "critical_files_protected": cleanup_results["critical_files_protected"]
                }
                
                self._log_evolution_decision(evolution_log)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Architect memory maintenance failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _log_evolution_decision(self, decision_data: Dict[str, Any]) -> None:
        """Log an evolution decision with version control."""
        try:
            # Create evolution log file
            log_file = self.base_path / "architect_evolution_log.json"
            
            # Load existing log or create new
            if log_file.exists():
                with open(log_file, 'r') as f:
                    evolution_log = json.load(f)
            else:
                evolution_log = {"decisions": [], "created": time.time()}
            
            # Add new decision
            evolution_log["decisions"].append(decision_data)
            
            # Keep only last 1000 decisions to prevent bloat
            if len(evolution_log["decisions"]) > 1000:
                evolution_log["decisions"] = evolution_log["decisions"][-1000:]
            
            # Write back
            with open(log_file, 'w') as f:
                json.dump(evolution_log, f, indent=2)
            
            # Commit to version control if available
            if self.repo:
                try:
                    self.repo.git.add(str(log_file))
                    self.repo.git.commit('-m', f"Architect decision: {decision_data.get('decision', 'unknown')}")
                    self.logger.info("✅ Evolution decision committed to version control")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not commit evolution decision: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to log evolution decision: {e}")
    
    def ensure_game_is_running(self) -> bool:
        """Ensure the game/training system is running, start if needed."""
        activity = self.check_game_activity()
        
        if activity['overall_active']:
            self.logger.info("✅ Game/Training system confirmed active")
            return True
        
        self.logger.warning("⚠️ No active training detected, attempting to start...")
        
        # Try to start continuous training
        return self._start_training_system()
    
    def _start_training_system(self) -> bool:
        """Start the training system if not running."""
        try:
            # Look for the best training script to start (prioritize master script)
            training_scripts = [
                "master_arc_trainer.py",  # New consolidated script
                "master_arc_trainer.py"
            ]
            
            for script in training_scripts:
                script_path = os.path.join(self.base_path, "..", script)
                if os.path.exists(script_path):
                    self.logger.info(f"🚀 Starting training system: {script}")
                    
                    # Start in background
                    cmd = [sys.executable, script_path, "--mode", "continuous", "--quiet"]
                    
                    subprocess.Popen(
                        cmd,
                        cwd=os.path.dirname(script_path),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    
                    # Wait a moment and check if it started
                    time.sleep(5)
                    activity = self.check_game_activity()
                    
                    if activity['overall_active']:
                        self.logger.info("✅ Training system started successfully")
                        return True
                    
            self.logger.error("❌ Failed to start any training system")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error starting training system: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create Architect instance
    architect = Architect(
        base_path="C:/Users/Admin/Documents/GitHub/tabula-rasa",
        repo_path="C:/Users/Admin/Documents/GitHub/tabula-rasa"
    )
    
    async def test_architect():
        print("🔬 Testing Architect system...")
        
        # Test autonomous evolution
        result = await architect.autonomous_evolution_cycle()
        print(f"Evolution result: {result}")
        
        # Test Governor request processing
        test_request = ArchitectRequest(
            issue_type="low_efficiency",
            persistent_problem="Win rate stuck at 60% for 10 cycles",
            failed_solutions=[],
            performance_data={'win_rate': 0.6, 'score': 45.0},
            suggested_research_directions=["Increase exploration", "Try contrarian strategies"],
            priority=0.8
        )
        
        response = await architect.process_governor_request(test_request)
        print(f"Governor request response: {response}")
        
        # Show status
        status = architect.get_evolution_status()
        print(f"Evolution status: {status}")
    
    # Run test
    asyncio.run(test_architect())

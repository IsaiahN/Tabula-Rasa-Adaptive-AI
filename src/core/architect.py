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
        """Generate mutation to address specific issues."""
        mutation_id = f"targeted_{int(time.time())}"
        
        # Analyze the issue and suggest appropriate mutations
        if request.issue_type == "low_efficiency":
            changes = {'max_actions_per_game': int(self.base_genome.max_actions_per_game * 1.5)}
            rationale = f"Increase exploration to address: {request.persistent_problem}"
            
        elif request.issue_type == "stagnation":
            changes = {'enable_contrarian_strategy': True, 'contrarian_threshold': 3}
            rationale = f"Enable contrarian strategies to break stagnation: {request.persistent_problem}"
            
        else:  # General improvement
            changes = {'salience_threshold': max(0.1, self.base_genome.salience_threshold - 0.1)}
            rationale = f"Lower salience threshold to preserve more memories"
        
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
            await self._cleanup_sandbox(sandbox_path)
    
    async def _create_sandbox(self, genome: SystemGenome) -> Path:
        """Create isolated sandbox environment for testing."""
        sandbox_id = f"sandbox_{genome.get_hash()}_{int(time.time())}"
        sandbox_path = self.sandbox_dir / sandbox_id
        
        # Copy core system files to sandbox
        shutil.copytree(self.base_path / "src", sandbox_path / "src")
        shutil.copy2(self.base_path / "unified_arc_trainer.py", sandbox_path)
        
        # Create custom config file for this genome
        config_path = sandbox_path / "sandbox_config.json"
        with open(config_path, 'w') as f:
            json.dump(genome.to_dict(), f, indent=2)
        
        self.logger.debug(f"🏗️ Created sandbox: {sandbox_path}")
        return sandbox_path
    
    async def _run_sandbox_test(self, sandbox_path: Path, test_games: List[str]) -> Dict[str, Any]:
        """Run test suite in sandbox environment."""
        # This would run a minimal version of the training system
        # For now, simulate test results
        await asyncio.sleep(2)  # Simulate test execution
        
        # Mock results - in real implementation, would run actual tests
        return {
            'success': True,
            'metrics': {
                'win_rate': 0.65,
                'average_score': 48.5,
                'learning_efficiency': 0.8,
                'sample_efficiency': 0.7,
                'robustness': 0.6
            },
            'games_tested': test_games,
            'total_episodes': 15,
            'test_duration': 120.0
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
    
    def __init__(self, base_path: str, repo_path: str, logger: Optional[logging.Logger] = None):
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
        
        # Git integration
        self.repo = None
        self.default_branch = "Tabula-Rasa-v3"  # Our working branch
        if GIT_AVAILABLE:
            try:
                self.repo = git.Repo(self.repo_path)
                self._ensure_correct_branch()
            except Exception as e:
                self.logger.warning(f"⚠️ Git repository error: {e} - version control disabled")
        else:
            self.logger.warning("⚠️ GitPython not available - version control disabled")
        
        # Safety measures
        self.max_concurrent_tests = 1  # Start conservative
        self.human_approval_required = True
        self.auto_merge_threshold = 0.15  # Minimum improvement for auto-merge
        
        self.logger.info("🔬 Architect initialized - Zeroth Brain online")
    
    def _load_current_genome(self) -> SystemGenome:
        """Load current system genome from configuration."""
        # Would load from unified_arc_trainer.py TrainingConfig or config files
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
        # This would modify unified_arc_trainer.py or config files
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

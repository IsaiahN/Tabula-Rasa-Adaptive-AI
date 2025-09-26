#!/usr/bin/env python3
"""
System Genome - Formalized representation of the system's architecture and parameters.
"""

import hashlib
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum

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
    max_actions_per_game: int = 5000
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

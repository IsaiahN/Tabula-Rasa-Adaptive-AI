#!/usr/bin/env python3
"""
High Priority Enhancements Integration Module

Integrates all high-priority enhancements (Self-Prior Mechanism, Pattern Discovery Curiosity,
and Enhanced Architectural Systems) into a unified system that can be easily integrated
with existing Tabula Rasa components.

Key Features:
- Unified initialization and configuration
- Seamless integration with existing systems
- Comprehensive metrics and monitoring
- Easy-to-use factory functions
- Performance optimization and caching
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# Import the enhancement modules
from .self_prior_mechanism import (
    SelfPriorManager, create_self_prior_manager,
    SensoryExperience, IntrinsicGoal, GoalType
)
from .pattern_discovery_curiosity import (
    PatternDiscoveryCuriosity, create_pattern_discovery_curiosity,
    DiscoveredPattern, PatternType, CuriosityLevel
)
from .enhanced_architectural_systems import (
    EnhancedTreeBasedDirector, EnhancedTreeBasedArchitect, EnhancedImplicitMemoryManager,
    create_enhanced_tree_based_director, create_enhanced_tree_based_architect, 
    create_enhanced_implicit_memory_manager
)

logger = logging.getLogger(__name__)

class IntegrationStatus(Enum):
    """Status of integration components."""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class EnhancementConfig:
    """Configuration for high priority enhancements."""
    # Self-Prior Mechanism
    enable_self_prior: bool = True
    self_prior_visual_dim: int = 512
    self_prior_proprioceptive_dim: int = 64
    self_prior_tactile_dim: int = 32
    self_prior_auditory_dim: int = 128
    self_prior_latent_dim: int = 256
    self_prior_n_components: int = 10
    self_prior_max_experiences: int = 10000
    
    # Pattern Discovery Curiosity
    enable_pattern_discovery: bool = True
    pattern_compression_methods: List[str] = field(default_factory=lambda: ['gzip', 'pca', 'clustering'])
    pattern_learning_rate: float = 0.1
    pattern_curiosity_decay: float = 0.99
    pattern_max_patterns: int = 1000
    
    # Enhanced Architectural Systems
    enable_enhanced_architectural: bool = True
    enhancement_weight: float = 0.3
    motivation_weight: float = 0.2
    motivational_weight: float = 0.3
    
    # Integration settings
    enable_caching: bool = True
    cache_size: int = 1000
    performance_monitoring: bool = True
    metrics_update_interval: float = 1.0  # seconds

@dataclass
class IntegrationMetrics:
    """Comprehensive metrics for all enhancements."""
    self_prior_metrics: Dict[str, Any] = field(default_factory=dict)
    pattern_discovery_metrics: Dict[str, Any] = field(default_factory=dict)
    enhanced_director_metrics: Dict[str, Any] = field(default_factory=dict)
    enhanced_architect_metrics: Dict[str, Any] = field(default_factory=dict)
    enhanced_memory_metrics: Dict[str, Any] = field(default_factory=dict)
    integration_status: Dict[str, str] = field(default_factory=dict)
    performance_stats: Dict[str, float] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)

class HighPriorityEnhancementsIntegration:
    """
    Main integration class for all high priority enhancements.
    
    This class provides a unified interface for initializing, configuring,
    and using all the high priority enhancements together.
    """
    
    def __init__(self, 
                 config: Optional[EnhancementConfig] = None,
                 existing_components: Optional[Dict[str, Any]] = None):
        
        self.config = config or EnhancementConfig()
        self.existing_components = existing_components or {}
        
        # Initialize enhancement components
        self.self_prior_manager = None
        self.pattern_discovery_curiosity = None
        self.enhanced_director = None
        self.enhanced_architect = None
        self.enhanced_memory_manager = None
        
        # Integration status
        self.integration_status = {
            'self_prior': IntegrationStatus.NOT_INITIALIZED,
            'pattern_discovery': IntegrationStatus.NOT_INITIALIZED,
            'enhanced_director': IntegrationStatus.NOT_INITIALIZED,
            'enhanced_architect': IntegrationStatus.NOT_INITIALIZED,
            'enhanced_memory': IntegrationStatus.NOT_INITIALIZED
        }
        
        # Metrics and monitoring
        self.metrics = IntegrationMetrics()
        self.performance_cache = {}
        self.last_metrics_update = 0.0
        
        # Initialize components
        self._initialize_components()
        
        logger.info("HighPriorityEnhancementsIntegration initialized")
    
    def _initialize_components(self):
        """Initialize all enhancement components."""
        
        # Initialize Self-Prior Mechanism
        if self.config.enable_self_prior:
            try:
                self.integration_status['self_prior'] = IntegrationStatus.INITIALIZING
                self.self_prior_manager = create_self_prior_manager(
                    visual_dim=self.config.self_prior_visual_dim,
                    proprioceptive_dim=self.config.self_prior_proprioceptive_dim,
                    tactile_dim=self.config.self_prior_tactile_dim,
                    auditory_dim=self.config.self_prior_auditory_dim,
                    latent_dim=self.config.self_prior_latent_dim,
                    n_density_components=self.config.self_prior_n_components,
                    max_experiences=self.config.self_prior_max_experiences
                )
                self.integration_status['self_prior'] = IntegrationStatus.READY
                logger.info("Self-Prior Mechanism initialized successfully")
            except Exception as e:
                self.integration_status['self_prior'] = IntegrationStatus.ERROR
                logger.error(f"Self-Prior Mechanism initialization failed: {e}")
        
        # Initialize Pattern Discovery Curiosity
        if self.config.enable_pattern_discovery:
            try:
                self.integration_status['pattern_discovery'] = IntegrationStatus.INITIALIZING
                self.pattern_discovery_curiosity = create_pattern_discovery_curiosity(
                    compression_methods=self.config.pattern_compression_methods,
                    learning_rate=self.config.pattern_learning_rate,
                    curiosity_decay=self.config.pattern_curiosity_decay,
                    max_patterns=self.config.pattern_max_patterns
                )
                self.integration_status['pattern_discovery'] = IntegrationStatus.READY
                logger.debug("Pattern Discovery Curiosity initialized successfully")
            except Exception as e:
                self.integration_status['pattern_discovery'] = IntegrationStatus.ERROR
                logger.error(f"Pattern Discovery Curiosity initialization failed: {e}")
        
        # Initialize Enhanced Architectural Systems
        if self.config.enable_enhanced_architectural:
            self._initialize_enhanced_architectural_systems()
        
        # Cross-integrate components
        self._cross_integrate_components()
    
    def _initialize_enhanced_architectural_systems(self):
        """Initialize enhanced architectural systems."""
        
        # Get existing components
        original_director = self.existing_components.get('tree_based_director')
        original_architect = self.existing_components.get('tree_based_architect')
        original_memory_manager = self.existing_components.get('implicit_memory_manager')
        
        # Initialize Enhanced Tree-Based Director
        if original_director:
            try:
                self.integration_status['enhanced_director'] = IntegrationStatus.INITIALIZING
                self.enhanced_director = create_enhanced_tree_based_director(
                    original_director,
                    self_prior_manager=self.self_prior_manager,
                    pattern_discovery_curiosity=self.pattern_discovery_curiosity,
                    enhancement_weight=self.config.enhancement_weight
                )
                self.integration_status['enhanced_director'] = IntegrationStatus.READY
                logger.info("Enhanced Tree-Based Director initialized successfully")
            except Exception as e:
                self.integration_status['enhanced_director'] = IntegrationStatus.ERROR
                logger.error(f"Enhanced Tree-Based Director initialization failed: {e}")
        
        # Initialize Enhanced Tree-Based Architect
        if original_architect:
            try:
                self.integration_status['enhanced_architect'] = IntegrationStatus.INITIALIZING
                self.enhanced_architect = create_enhanced_tree_based_architect(
                    original_architect,
                    self_prior_manager=self.self_prior_manager,
                    pattern_discovery_curiosity=self.pattern_discovery_curiosity,
                    motivation_weight=self.config.motivation_weight
                )
                self.integration_status['enhanced_architect'] = IntegrationStatus.READY
                logger.info("Enhanced Tree-Based Architect initialized successfully")
            except Exception as e:
                self.integration_status['enhanced_architect'] = IntegrationStatus.ERROR
                logger.error(f"Enhanced Tree-Based Architect initialization failed: {e}")
        
        # Initialize Enhanced Implicit Memory Manager
        if original_memory_manager:
            try:
                self.integration_status['enhanced_memory'] = IntegrationStatus.INITIALIZING
                self.enhanced_memory_manager = create_enhanced_implicit_memory_manager(
                    original_memory_manager,
                    self_prior_manager=self.self_prior_manager,
                    pattern_discovery_curiosity=self.pattern_discovery_curiosity,
                    motivational_weight=self.config.motivational_weight
                )
                self.integration_status['enhanced_memory'] = IntegrationStatus.READY
                logger.info("Enhanced Implicit Memory Manager initialized successfully")
            except Exception as e:
                self.integration_status['enhanced_memory'] = IntegrationStatus.ERROR
                logger.error(f"Enhanced Implicit Memory Manager initialization failed: {e}")
    
    def _cross_integrate_components(self):
        """Cross-integrate components with each other."""
        
        # Integrate Self-Prior Manager with other components
        if self.self_prior_manager:
            predictive_core = self.existing_components.get('predictive_core')
            learning_progress_drive = self.existing_components.get('learning_progress_drive')
            tree_based_director = self.existing_components.get('tree_based_director')
            
            self.self_prior_manager.integrate_components(
                predictive_core=predictive_core,
                learning_progress_drive=learning_progress_drive,
                tree_based_director=tree_based_director
            )
        
        # Integrate Pattern Discovery Curiosity with other components
        if self.pattern_discovery_curiosity:
            enhanced_curiosity_system = self.existing_components.get('enhanced_curiosity_system')
            gut_feeling_engine = self.existing_components.get('enhanced_gut_feeling_engine')
            tree_based_architect = self.existing_components.get('tree_based_architect')
            
            self.pattern_discovery_curiosity.integrate_components(
                enhanced_curiosity_system=enhanced_curiosity_system,
                gut_feeling_engine=gut_feeling_engine,
                tree_based_architect=tree_based_architect
            )
    
    def process_sensory_experience(self, 
                                 visual_features: np.ndarray,
                                 proprioceptive_state: np.ndarray,
                                 tactile_feedback: Optional[np.ndarray] = None,
                                 auditory_input: Optional[np.ndarray] = None,
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a sensory experience through all relevant enhancements."""
        
        context = context or {}
        results = {}
        
        # Process through Self-Prior Mechanism
        if self.self_prior_manager and self.integration_status['self_prior'] == IntegrationStatus.READY:
            try:
                self_prior_result = self.self_prior_manager.process_sensory_experience(
                    visual_features, proprioceptive_state, tactile_feedback, auditory_input, context
                )
                results['self_prior'] = self_prior_result
            except Exception as e:
                logger.warning(f"Self-prior processing failed: {e}")
                results['self_prior'] = {'error': str(e)}
        
        # Process through Pattern Discovery Curiosity
        if self.pattern_discovery_curiosity and self.integration_status['pattern_discovery'] == IntegrationStatus.READY:
            try:
                pattern_result = self.pattern_discovery_curiosity.process_observation(
                    visual_features, context
                )
                results['pattern_discovery'] = pattern_result
            except Exception as e:
                logger.warning(f"Pattern discovery processing failed: {e}")
                results['pattern_discovery'] = {'error': str(e)}
        
        # Update metrics if enabled
        if self.config.performance_monitoring:
            self._update_metrics()
        
        return results
    
    def enhanced_goal_decomposition(self, 
                                  original_goal: Any,
                                  context: Dict[str, Any]) -> List[Any]:
        """Enhanced goal decomposition using all available enhancements."""
        
        if self.enhanced_director and self.integration_status['enhanced_director'] == IntegrationStatus.READY:
            try:
                enhanced_goals = self.enhanced_director.enhanced_goal_decomposition(original_goal, context)
                return enhanced_goals
            except Exception as e:
                logger.warning(f"Enhanced goal decomposition failed: {e}")
                return []
        
        # Fallback to original director if available
        original_director = self.existing_components.get('tree_based_director')
        if original_director:
            try:
                return original_director.decompose_goal(original_goal, context)
            except Exception as e:
                logger.warning(f"Original goal decomposition failed: {e}")
                return []
        
        return []
    
    def enhanced_architectural_evolution(self, 
                                       current_architecture: Any,
                                       performance_metrics: Dict[str, Any],
                                       context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Enhanced architectural evolution using all available enhancements."""
        
        if self.enhanced_architect and self.integration_status['enhanced_architect'] == IntegrationStatus.READY:
            try:
                return self.enhanced_architect.enhanced_architectural_evolution(
                    current_architecture, performance_metrics, context
                )
            except Exception as e:
                logger.warning(f"Enhanced architectural evolution failed: {e}")
        
        # Fallback to original architect if available
        original_architect = self.existing_components.get('tree_based_architect')
        if original_architect:
            try:
                return original_architect.evolve_architecture(current_architecture, performance_metrics, context)
            except Exception as e:
                logger.warning(f"Original architectural evolution failed: {e}")
        
        return current_architecture, {}
    
    def enhanced_memory_storage(self, 
                              memory_data: Any,
                              context: Dict[str, Any]) -> str:
        """Enhanced memory storage using all available enhancements."""
        
        if self.enhanced_memory_manager and self.integration_status['enhanced_memory'] == IntegrationStatus.READY:
            try:
                return self.enhanced_memory_manager.enhanced_memory_storage(memory_data, context)
            except Exception as e:
                logger.warning(f"Enhanced memory storage failed: {e}")
        
        # Fallback to original memory manager if available
        original_memory_manager = self.existing_components.get('implicit_memory_manager')
        if original_memory_manager:
            try:
                return original_memory_manager.store_memory(memory_data, context)
            except Exception as e:
                logger.warning(f"Original memory storage failed: {e}")
        
        return f"fallback_memory_{int(time.time())}"
    
    def get_intrinsic_rewards(self, current_state: Dict[str, Any]) -> Dict[str, float]:
        """Get intrinsic rewards from all enhancement systems."""
        
        rewards = {}
        
        # Get rewards from Self-Prior Mechanism
        if self.self_prior_manager and self.integration_status['self_prior'] == IntegrationStatus.READY:
            try:
                self_prior_rewards = self.self_prior_manager.get_intrinsic_rewards(current_state)
                rewards.update(self_prior_rewards)
            except Exception as e:
                logger.warning(f"Self-prior rewards failed: {e}")
        
        # Get rewards from Pattern Discovery Curiosity
        if self.pattern_discovery_curiosity and self.integration_status['pattern_discovery'] == IntegrationStatus.READY:
            try:
                if 'observation' in current_state:
                    pattern_result = self.pattern_discovery_curiosity.process_observation(
                        current_state['observation'], current_state
                    )
                    pattern_rewards = pattern_result.get('intrinsic_rewards', {})
                    rewards.update(pattern_rewards)
            except Exception as e:
                logger.warning(f"Pattern discovery rewards failed: {e}")
        
        return rewards
    
    def _update_metrics(self):
        """Update comprehensive metrics for all enhancements."""
        
        current_time = time.time()
        if current_time - self.last_metrics_update < self.config.metrics_update_interval:
            return
        
        try:
            # Update Self-Prior metrics
            if self.self_prior_manager and self.integration_status['self_prior'] == IntegrationStatus.READY:
                self.metrics.self_prior_metrics = self.self_prior_manager.get_self_prior_metrics()
            
            # Update Pattern Discovery metrics
            if self.pattern_discovery_curiosity and self.integration_status['pattern_discovery'] == IntegrationStatus.READY:
                self.metrics.pattern_discovery_metrics = self.pattern_discovery_curiosity.get_curiosity_metrics()
            
            # Update Enhanced Director metrics
            if self.enhanced_director and self.integration_status['enhanced_director'] == IntegrationStatus.READY:
                self.metrics.enhanced_director_metrics = self.enhanced_director.get_enhancement_metrics()
            
            # Update Enhanced Architect metrics
            if self.enhanced_architect and self.integration_status['enhanced_architect'] == IntegrationStatus.READY:
                self.metrics.enhanced_architect_metrics = self.enhanced_architect.get_enhancement_metrics()
            
            # Update Enhanced Memory metrics
            if self.enhanced_memory_manager and self.integration_status['enhanced_memory'] == IntegrationStatus.READY:
                self.metrics.enhanced_memory_metrics = self.enhanced_memory_manager.get_motivational_clusters()
            
            # Update integration status
            self.metrics.integration_status = {k: v.value for k, v in self.integration_status.items()}
            
            # Update performance stats
            self.metrics.performance_stats = {
                'total_components': len([c for c in self.integration_status.values() if c == IntegrationStatus.READY]),
                'error_components': len([c for c in self.integration_status.values() if c == IntegrationStatus.ERROR]),
                'last_update': current_time
            }
            
            self.metrics.last_update = current_time
            self.last_metrics_update = current_time
            
        except Exception as e:
            logger.warning(f"Metrics update failed: {e}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all enhancements."""
        
        if self.config.performance_monitoring:
            self._update_metrics()
        
        return {
            'self_prior_metrics': self.metrics.self_prior_metrics,
            'pattern_discovery_metrics': self.metrics.pattern_discovery_metrics,
            'enhanced_director_metrics': self.metrics.enhanced_director_metrics,
            'enhanced_architect_metrics': self.metrics.enhanced_architect_metrics,
            'enhanced_memory_metrics': self.metrics.enhanced_memory_metrics,
            'integration_status': self.metrics.integration_status,
            'performance_stats': self.metrics.performance_stats,
            'last_update': self.metrics.last_update,
            'config': {
                'enable_self_prior': self.config.enable_self_prior,
                'enable_pattern_discovery': self.config.enable_pattern_discovery,
                'enable_enhanced_architectural': self.config.enable_enhanced_architectural,
                'performance_monitoring': self.config.performance_monitoring
            }
        }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of the integration status."""
        
        ready_components = [k for k, v in self.integration_status.items() if v == IntegrationStatus.READY]
        error_components = [k for k, v in self.integration_status.items() if v == IntegrationStatus.ERROR]
        not_initialized_components = [k for k, v in self.integration_status.items() if v == IntegrationStatus.NOT_INITIALIZED]
        
        return {
            'total_components': len(self.integration_status),
            'ready_components': ready_components,
            'error_components': error_components,
            'not_initialized_components': not_initialized_components,
            'ready_count': len(ready_components),
            'error_count': len(error_components),
            'not_initialized_count': len(not_initialized_components),
            'overall_status': 'ready' if len(error_components) == 0 else 'partial' if len(ready_components) > 0 else 'error'
        }

# Factory function for easy integration
def create_high_priority_enhancements_integration(
    config: Optional[EnhancementConfig] = None,
    existing_components: Optional[Dict[str, Any]] = None
) -> HighPriorityEnhancementsIntegration:
    """Create a high priority enhancements integration system."""
    return HighPriorityEnhancementsIntegration(config, existing_components)

# Convenience function for quick setup
def setup_high_priority_enhancements(
    enable_self_prior: bool = True,
    enable_pattern_discovery: bool = True,
    enable_enhanced_architectural: bool = True,
    existing_components: Optional[Dict[str, Any]] = None
) -> HighPriorityEnhancementsIntegration:
    """Quick setup function for high priority enhancements."""
    
    config = EnhancementConfig(
        enable_self_prior=enable_self_prior,
        enable_pattern_discovery=enable_pattern_discovery,
        enable_enhanced_architectural=enable_enhanced_architectural
    )
    
    return create_high_priority_enhancements_integration(config, existing_components)

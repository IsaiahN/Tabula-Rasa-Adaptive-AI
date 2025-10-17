"""Thin orchestrator facade for TB-Seed extraction.

This module provides a small `Orchestrator` class that currently wraps
the existing `ContinuousLearningLoop`. It exists to allow incremental
refactors that move functionality out of the large monolith.
"""
from typing import Any, Dict, Optional
from .continuous_learning_loop import ContinuousLearningLoop


class Orchestrator:
    def __init__(self, core: Optional[ContinuousLearningLoop] = None, **kwargs):
        # Allow injection of an existing core for testing/migration; otherwise create one
        if core is not None:
            self._core = core
        else:
            self._core = ContinuousLearningLoop(**kwargs)

        # Allow the core to call back into this facade during migration
        try:
            setattr(self._core, 'orchestrator', self)
        except Exception:
            pass

    async def start_training(self, game_id: str, **kwargs) -> Dict[str, Any]:
        return await self._core.start_training_with_direct_control(game_id, **kwargs)

    async def get_available_games(self):
        return await self._core.get_available_games()

    def shutdown(self):
        # Provide a simple shutdown hook
        try:
            if hasattr(self._core, 'shutdown_handler'):
                self._core.shutdown_handler.request_shutdown()
        except Exception:
            pass

    def ensure_initialized(self):
        """Expose a synchronous ensure_initialized method mirroring the legacy API."""
        if hasattr(self._core, '_ensure_initialized'):
            return self._core._ensure_initialized()
        if hasattr(self._core, 'ensure_initialized'):
            return self._core.ensure_initialized()
        return None

    def initialize_components(self):
        """Initialize underlying modular components via the legacy core."""
        if hasattr(self._core, '_initialize_components'):
            return self._core._initialize_components()
        return None

    def initialize_losing_streak_systems(self):
        """Initialize the losing-streak detection systems via the core."""
        if hasattr(self._core, '_initialize_losing_streak_systems'):
            return self._core._initialize_losing_streak_systems()
        return None

    def initialize_real_time_learning_systems(self):
        """Initialize the real-time learning subsystems via the core."""
        if hasattr(self._core, '_initialize_real_time_learning_systems'):
            return self._core._initialize_real_time_learning_systems()
        return None

    def initialize_attention_communication_systems(self):
        """Initialize the enhanced attention + communication systems via the core."""
        # Implement initialization here to avoid circular delegation.
        try:
            # Respect idempotence
            if getattr(self._core, '_attention_communication_initialized', False):
                return None

            db_path = str(getattr(self._core, 'db_path', '.'))

            # Lazy import to avoid heavy dependencies at import time
            try:
                from src.core.central_attention_controller import CentralAttentionController
                from src.core.weighted_communication_system import WeightedCommunicationSystem
            except Exception:
                from core.central_attention_controller import CentralAttentionController
                from core.weighted_communication_system import WeightedCommunicationSystem

            # Instantiate and assign to core
            self._core.attention_controller = CentralAttentionController(db_path)
            self._core.communication_system = WeightedCommunicationSystem(db_path)

            # Set communication system on action selector if available
            if getattr(self._core, 'action_selector', None) and hasattr(self._core.action_selector, 'set_communication_system'):
                try:
                    self._core.action_selector.set_communication_system(self._core.communication_system)
                except Exception:
                    pass

            self._core._attention_communication_initialized = True
            return None
        except Exception:
            # Ensure flag is not left in inconsistent state
            self._core._attention_communication_initialized = False
            return None

    def initialize_fitness_evolution_system(self):
        """Initialize the context-dependent fitness evolution system via the core."""
        try:
            if getattr(self._core, '_fitness_evolution_initialized', False):
                return None

            db_path = str(getattr(self._core, 'db_path', '.'))

            try:
                from src.core.context_dependent_fitness_evolution import ContextDependentFitnessEvolution
            except Exception:
                from core.context_dependent_fitness_evolution import ContextDependentFitnessEvolution

            self._core.fitness_evolution_system = ContextDependentFitnessEvolution(db_path)

            # If attention coordination is available, link systems
            if getattr(self._core, '_attention_communication_initialized', False) and \
               getattr(self._core, 'attention_controller', None) and getattr(self._core, 'communication_system', None):
                try:
                    self._core.fitness_evolution_system.set_attention_coordination(
                        self._core.attention_controller, self._core.communication_system
                    )
                except Exception:
                    pass

            self._core._fitness_evolution_initialized = True
            return None
        except Exception:
            self._core._fitness_evolution_initialized = False
            return None

    def initialize_neat_architect_system(self):
        """Initialize the NEAT-based architect system via the core."""
        try:
            if getattr(self._core, '_neat_architect_initialized', False):
                return None

            db_path = str(getattr(self._core, 'db_path', '.'))

            try:
                from src.core.neat_based_architect import NEATBasedArchitect
            except Exception:
                from core.neat_based_architect import NEATBasedArchitect

            self._core.neat_architect_system = NEATBasedArchitect(db_path)

            # Link with attention coordination if available
            if getattr(self._core, '_attention_communication_initialized', False) and \
               getattr(self._core, 'attention_controller', None) and getattr(self._core, 'communication_system', None):
                try:
                    self._core.neat_architect_system.set_attention_coordination(
                        self._core.attention_controller, self._core.communication_system
                    )
                except Exception:
                    pass

            # Optionally add fitness observer if method exists
            if getattr(self._core, '_fitness_evolution_initialized', False) and getattr(self._core, 'fitness_evolution_system', None):
                try:
                    if hasattr(self._core.neat_architect_system, 'add_fitness_observer'):
                        self._core.neat_architect_system.add_fitness_observer(self._core.fitness_evolution_system)
                except Exception:
                    pass

            self._core._neat_architect_initialized = True
            return None
        except Exception:
            self._core._neat_architect_initialized = False
            return None

    def initialize_bayesian_inference_system(self):
        """Initialize the Bayesian inference engine via the core."""
        try:
            if getattr(self._core, '_bayesian_inference_initialized', False):
                return None

            db_path = str(getattr(self._core, 'db_path', '.'))

            try:
                from src.core.bayesian_inference_engine import BayesianInferenceEngine
            except Exception:
                from core.bayesian_inference_engine import BayesianInferenceEngine

            self._core.bayesian_inference_system = BayesianInferenceEngine(db_path)

            # Link attention coordination if available
            if getattr(self._core, '_attention_communication_initialized', False) and \
               getattr(self._core, 'attention_controller', None) and getattr(self._core, 'communication_system', None):
                try:
                    self._core.bayesian_inference_system.set_attention_coordination(
                        self._core.attention_controller, self._core.communication_system
                    )
                except Exception:
                    pass

            # Link with fitness evolution if available
            if getattr(self._core, '_fitness_evolution_initialized', False) and getattr(self._core, 'fitness_evolution_system', None):
                try:
                    if hasattr(self._core.bayesian_inference_system, 'add_fitness_data_source'):
                        self._core.bayesian_inference_system.add_fitness_data_source(self._core.fitness_evolution_system)
                except Exception:
                    pass

            self._core._bayesian_inference_initialized = True
            return None
        except Exception:
            self._core._bayesian_inference_initialized = False
            return None

    def initialize_graph_traversal_system(self):
        """Initialize the enhanced graph traversal system via the core."""
        try:
            if getattr(self._core, '_graph_traversal_initialized', False):
                return None

            db_path = str(getattr(self._core, 'db_path', '.'))

            try:
                from src.core.enhanced_graph_traversal import EnhancedGraphTraversal
            except Exception:
                from core.enhanced_graph_traversal import EnhancedGraphTraversal

            # Some implementations expect a DB connection; allow path or conn
            try:
                self._core.graph_traversal_system = EnhancedGraphTraversal(db_path)
            except Exception:
                # Fall back to passing a db connection object if core has one
                db_conn = getattr(self._core, 'db_connection', None)
                self._core.graph_traversal_system = EnhancedGraphTraversal(db_conn)

            # Set attention coordination if available
            if getattr(self._core, '_attention_communication_initialized', False) and \
               getattr(self._core, 'attention_controller', None) and getattr(self._core, 'communication_system', None):
                try:
                    self._core.graph_traversal_system.set_attention_coordination(
                        self._core.attention_controller, self._core.communication_system
                    )
                except Exception:
                    pass

            # Link with fitness evolution if available
            if getattr(self._core, '_fitness_evolution_initialized', False) and getattr(self._core, 'fitness_evolution_system', None):
                try:
                    if hasattr(self._core.graph_traversal_system, 'set_fitness_evolution_coordination'):
                        self._core.graph_traversal_system.set_fitness_evolution_coordination(self._core.fitness_evolution_system)
                except Exception:
                    pass

            self._core._graph_traversal_initialized = True
            return None
        except Exception:
            self._core._graph_traversal_initialized = False
            return None

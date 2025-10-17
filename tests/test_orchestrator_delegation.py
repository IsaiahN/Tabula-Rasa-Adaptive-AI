import types
import sys

from src.training.core.orchestrator import Orchestrator


class FakeCore:
    def __init__(self):
        self._attention_communication_initialized = False
        self._fitness_evolution_initialized = False
        self._neat_architect_initialized = False
        self._bayesian_inference_initialized = False
        self._graph_traversal_initialized = False
        self.db_path = ':memory:'


# Provide lightweight fake implementations for modules that Orchestrator may import.
class FakeAttention:
    def __init__(self, db_path):
        self.db_path = db_path


class FakeCommunication:
    def __init__(self, db_path):
        self.db_path = db_path


class FakeFitness:
    def __init__(self, db_path):
        self.db_path = db_path


class FakeNEAT:
    def __init__(self, db_path):
        self.db_path = db_path


class FakeBayesian:
    def __init__(self, db_path):
        self.db_path = db_path


class FakeGraphTraversal:
    def __init__(self, db_path):
        self.db_path = db_path


def _inject_fake(module_name, symbol_name, fake_cls):
    module = types.ModuleType(module_name)
    setattr(module, symbol_name, fake_cls)
    sys.modules[module_name] = module


def test_orchestrator_initializers_with_fake_core():
    # Inject fake modules to avoid heavy imports
    _inject_fake('src.core.central_attention_controller', 'CentralAttentionController', FakeAttention)
    _inject_fake('src.core.weighted_communication_system', 'WeightedCommunicationSystem', FakeCommunication)
    _inject_fake('src.core.context_dependent_fitness_evolution', 'ContextDependentFitnessEvolution', FakeFitness)
    _inject_fake('src.core.neat_based_architect', 'NEATBasedArchitect', FakeNEAT)
    _inject_fake('src.core.bayesian_inference_engine', 'BayesianInferenceEngine', FakeBayesian)
    _inject_fake('src.core.enhanced_graph_traversal', 'EnhancedGraphTraversal', FakeGraphTraversal)

    core = FakeCore()
    orch = Orchestrator(core=core)

    # Run initializers
    orch.initialize_attention_communication_systems()
    orch.initialize_fitness_evolution_system()
    orch.initialize_neat_architect_system()
    orch.initialize_bayesian_inference_system()
    orch.initialize_graph_traversal_system()

    # Check flags
    assert core._attention_communication_initialized is True
    assert core._fitness_evolution_initialized is True
    assert core._neat_architect_initialized is True
    assert core._bayesian_inference_initialized is True
    assert core._graph_traversal_initialized is True

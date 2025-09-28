"""
NEAT-based Architect System (Tier 2)

This system evolves the architecture itself using NEAT (NeuroEvolution of Augmenting Topologies)
with "use it or lose it" module pruning. It dynamically adds, removes, and modifies system
components based on their effectiveness and usage patterns.

Key Features:
- NEAT algorithm for evolving system topology
- Module usage tracking and effectiveness measurement
- "Use it or lose it" pruning of ineffective modules
- Species formation and maintenance for architectural diversity
- Integration with attention system for evolution priorities
- Genetic operations: crossover, mutation, speciation
- Innovation tracking for consistent evolution
"""

import asyncio
import json
import time
import uuid
import math
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from collections import defaultdict

# Import attention coordination components
try:
    from .central_attention_controller import SubsystemDemand
    from .weighted_communication_system import MessagePriority
    ATTENTION_COORDINATION_AVAILABLE = True
except ImportError:
    ATTENTION_COORDINATION_AVAILABLE = False
    SubsystemDemand = None
    MessagePriority = None

logger = logging.getLogger(__name__)

class ModuleType(Enum):
    """Types of system modules."""
    CORE = "core"
    ANALYSIS = "analysis"
    LEARNING = "learning"
    COORDINATION = "coordination"
    OPTIMIZATION = "optimization"

class ModuleCategory(Enum):
    """Categories for module importance."""
    ESSENTIAL = "essential"
    ENHANCEMENT = "enhancement"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"

class InnovationType(Enum):
    """Types of NEAT innovations."""
    ADD_NODE = "add_node"
    ADD_CONNECTION = "add_connection"
    MODIFY_WEIGHT = "modify_weight"
    REMOVE_CONNECTION = "remove_connection"
    MODIFY_MODULE = "modify_module"

@dataclass
class SystemModule:
    """Represents a system module that can evolve."""
    module_id: str
    module_name: str
    module_type: ModuleType
    module_category: ModuleCategory
    current_version: str = "1.0.0"
    module_definition: Dict[str, Any] = field(default_factory=dict)
    dependency_modules: List[str] = field(default_factory=list)
    dependent_modules: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    usage_frequency: int = 0
    effectiveness_score: float = 0.5
    last_used_timestamp: Optional[float] = None
    activation_threshold: float = 0.3
    pruning_candidate: bool = False
    is_active: bool = True

@dataclass
class NEATNode:
    """A node in the NEAT network representing a system component."""
    node_id: str
    node_type: str  # 'input', 'hidden', 'output'
    activation_function: str = "relu"
    bias: float = 0.0
    layer: int = 0
    module_reference: Optional[str] = None  # Reference to SystemModule
    innovation_number: Optional[int] = None

@dataclass
class NEATConnection:
    """A connection in the NEAT network representing data flow."""
    connection_id: str
    source_node: str
    target_node: str
    weight: float
    enabled: bool = True
    innovation_number: Optional[int] = None

@dataclass
class NEATGenome:
    """A NEAT genome representing a complete system architecture."""
    genome_id: str
    generation: int
    species_id: Optional[str] = None
    parent_genome_ids: List[str] = field(default_factory=list)
    fitness_score: float = 0.0
    adjusted_fitness: float = 0.0
    nodes: Dict[str, NEATNode] = field(default_factory=dict)
    connections: Dict[str, NEATConnection] = field(default_factory=dict)
    innovation_numbers: Set[int] = field(default_factory=set)
    complexity_score: float = 0.0
    specialization_score: float = 0.0
    robustness_score: float = 0.0
    age: int = 0
    last_improvement_generation: int = 0
    is_champion: bool = False
    is_active: bool = True

@dataclass
class NEATSpecies:
    """A NEAT species grouping similar genomes."""
    species_id: str
    generation: int
    representative_genome_id: str
    species_name: str
    genomes: List[str] = field(default_factory=list)
    average_fitness: float = 0.0
    max_fitness: float = 0.0
    fitness_stagnation_count: int = 0
    allowed_offspring: int = 0
    elitism_threshold: float = 0.8
    compatibility_threshold: float = 3.0
    species_traits: Dict[str, Any] = field(default_factory=dict)
    extinction_risk: float = 0.0
    is_extinct: bool = False
    extinction_generation: Optional[int] = None

@dataclass
class NEATInnovation:
    """Tracks structural innovations in NEAT evolution."""
    innovation_id: int
    innovation_type: InnovationType
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    weight_value: Optional[float] = None
    innovation_description: str = ""
    first_genome_id: str = ""
    generation_introduced: int = 0
    usage_frequency: int = 1
    average_fitness_impact: float = 0.0
    is_beneficial: Optional[bool] = None

class NEATBasedArchitect:
    """
    Main NEAT-based architect system for evolving system architecture.

    This system uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve
    the architecture of the AI system itself, with "use it or lose it" pruning
    of ineffective modules.
    """

    def __init__(self, db_manager):
        self.db = db_manager

        # Core system state
        self.current_generation = 0
        self.global_innovation_number = 0
        self.population_size = 20  # Smaller population for system architecture

        # NEAT algorithm components
        self.population: Dict[str, NEATGenome] = {}
        self.species: Dict[str, NEATSpecies] = {}
        self.innovations: Dict[int, NEATInnovation] = {}
        self.champion_genome: Optional[NEATGenome] = None

        # Module management
        self.system_modules: Dict[str, SystemModule] = {}
        self.module_usage_history: List[Dict[str, Any]] = []

        # Attention system integration
        self.attention_controller = None
        self.communication_system = None
        self.attention_integration_enabled = ATTENTION_COORDINATION_AVAILABLE

        # Evolution parameters
        self.config = {
            "population_size": 20,
            "species_threshold": 3.0,
            "survival_threshold": 0.2,
            "interspecies_mating_rate": 0.001,
            "mutation_rate": 0.8,
            "add_node_mutation_rate": 0.03,
            "add_connection_mutation_rate": 0.05,
            "weight_mutation_rate": 0.8,
            "weight_perturbation_rate": 0.9,
            "elitism_ratio": 0.1,
            "stagnation_threshold": 15,
            "module_pruning_threshold": 0.3,
            "usage_tracking_window": 100,
            "effectiveness_decay_rate": 0.95,
            "architectural_complexity_penalty": 0.01
        }

        # Initialize default system modules
        self._initialize_default_modules()

        logger.info("NEAT-based Architect System initialized")

    def set_attention_coordination(self, attention_controller, communication_system):
        """Set attention coordination systems for enhanced integration."""
        self.attention_controller = attention_controller
        self.communication_system = communication_system
        if attention_controller and communication_system:
            logger.info("NEAT architect enhanced with attention coordination")
        else:
            logger.warning("Attention coordination systems not fully available for NEAT architect")

    async def initialize_architecture(self, game_id: str, session_id: str) -> Dict[str, Any]:
        """Initialize the architectural evolution system for a new session."""
        try:
            # Create initial population if empty
            if not self.population:
                await self._create_initial_population()

            # Select current champion architecture or create one
            if not self.champion_genome:
                self.champion_genome = self._select_best_genome()

            # Initialize module tracking for this session
            await self._initialize_module_tracking(game_id, session_id)

            # Store initialization in database
            await self._store_architectural_state(game_id, session_id)

            logger.info(f"NEAT architect initialized for session {session_id}")

            return {
                "current_generation": self.current_generation,
                "population_size": len(self.population),
                "species_count": len(self.species),
                "champion_fitness": self.champion_genome.fitness_score if self.champion_genome else 0.0,
                "active_modules": len([m for m in self.system_modules.values() if m.is_active])
            }

        except Exception as e:
            logger.error(f"Error initializing NEAT architect: {e}")
            return {"error": str(e)}

    async def track_module_usage(self,
                               module_name: str,
                               game_id: str,
                               session_id: str,
                               usage_context: Dict[str, Any],
                               processing_time_ms: float,
                               effectiveness_score: float,
                               success: bool = True) -> None:
        """Track usage of a system module for 'use it or lose it' analysis."""
        try:
            # Find or create module
            module = self._get_or_create_module(module_name)

            # Update module usage statistics
            module.usage_frequency += 1
            module.last_used_timestamp = time.time()

            # Update effectiveness score with exponential moving average
            alpha = 0.1  # Learning rate
            module.effectiveness_score = (1 - alpha) * module.effectiveness_score + alpha * effectiveness_score

            # Store usage record
            usage_record = {
                "usage_id": f"usage_{module.module_id}_{int(time.time() * 1000)}",
                "module_id": module.module_id,
                "game_id": game_id,
                "session_id": session_id,
                "usage_timestamp": time.time(),
                "usage_context": usage_context,
                "processing_time_ms": processing_time_ms,
                "success": success,
                "effectiveness_score": effectiveness_score,
                "impact_on_performance": 0.0,  # Would be calculated based on system metrics
                "resource_consumption": {"cpu": 0.1, "memory": 50.0},  # Simplified
                "interactions_with_modules": []  # Would track actual interactions
            }

            await self._store_module_usage(usage_record)

            # Add to usage history for analysis
            self.module_usage_history.append(usage_record)

            # Keep history manageable
            if len(self.module_usage_history) > self.config["usage_tracking_window"]:
                self.module_usage_history = self.module_usage_history[-self.config["usage_tracking_window"]:]

            # Check if module should be considered for pruning
            await self._evaluate_module_for_pruning(module)

        except Exception as e:
            logger.error(f"Error tracking module usage for {module_name}: {e}")

    async def evolve_architecture(self, game_id: str, session_id: str) -> Dict[str, Any]:
        """Perform one generation of architectural evolution."""
        try:
            evolution_start_time = time.time()

            # Evaluate current population fitness
            await self._evaluate_population_fitness(game_id, session_id)

            # Perform speciation
            await self._perform_speciation()

            # Create next generation
            new_population = await self._create_next_generation()

            # Apply mutations
            await self._apply_mutations(new_population)

            # Update population
            self.population = new_population
            self.current_generation += 1

            # Prune ineffective modules
            pruning_results = await self._prune_ineffective_modules(game_id, session_id)

            # Update champion
            self.champion_genome = self._select_best_genome()

            # Store evolution results
            evolution_results = {
                "generation": self.current_generation,
                "population_size": len(self.population),
                "species_count": len(self.species),
                "champion_fitness": self.champion_genome.fitness_score if self.champion_genome else 0.0,
                "evolution_time": time.time() - evolution_start_time,
                "modules_pruned": len(pruning_results.get("pruned_modules", [])),
                "innovations_added": len([i for i in self.innovations.values()
                                        if i.generation_introduced == self.current_generation])
            }

            await self._store_evolution_results(game_id, session_id, evolution_results)

            logger.info(f"Architecture evolution completed for generation {self.current_generation}")

            return evolution_results

        except Exception as e:
            logger.error(f"Error in architectural evolution: {e}")
            return {"error": str(e)}

    async def request_architectural_attention(self,
                                            game_id: str,
                                            session_id: str,
                                            evolution_priorities: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Request attention allocation for architectural evolution."""
        if not (self.attention_integration_enabled and self.attention_controller and
                ATTENTION_COORDINATION_AVAILABLE):
            return None

        try:
            # Analyze current architectural bottlenecks
            bottlenecks = self._identify_architectural_bottlenecks()

            # Create subsystem demands based on evolution needs
            subsystem_demands = []

            for priority_area, priority_level in evolution_priorities.items():
                demand = SubsystemDemand(
                    subsystem_name=f"architecture_evolution_{priority_area}",
                    requested_priority=min(0.8, priority_level),
                    current_load=0.4,  # Moderate load for evolution
                    processing_complexity=0.8,  # High complexity
                    urgency_level=min(5, int(priority_level * 4) + 1),
                    justification=f"Architectural evolution needed for {priority_area}",
                    context_data={
                        "evolution_area": priority_area,
                        "priority_level": priority_level,
                        "bottlenecks": bottlenecks.get(priority_area, [])
                    }
                )
                subsystem_demands.append(demand)

            # Request attention allocation
            attention_allocation = await self.attention_controller.allocate_attention_resources(
                game_id, subsystem_demands, {"architectural_evolution_request": True}
            )

            # Store integration record
            await self._store_architectural_attention_integration(
                game_id, session_id, evolution_priorities, attention_allocation
            )

            return attention_allocation.__dict__ if attention_allocation else None

        except Exception as e:
            logger.error(f"Error requesting architectural attention: {e}")
            return None

    def _initialize_default_modules(self):
        """Initialize default system modules."""
        default_modules = [
            {
                "name": "real_time_learner",
                "type": ModuleType.LEARNING,
                "category": ModuleCategory.ESSENTIAL,
                "definition": {"interfaces": ["learning"], "capabilities": ["pattern_detection", "adaptation"]}
            },
            {
                "name": "pattern_detector",
                "type": ModuleType.ANALYSIS,
                "category": ModuleCategory.ENHANCEMENT,
                "definition": {"interfaces": ["analysis"], "capabilities": ["pattern_recognition"]}
            },
            {
                "name": "strategy_adjuster",
                "type": ModuleType.OPTIMIZATION,
                "category": ModuleCategory.ENHANCEMENT,
                "definition": {"interfaces": ["optimization"], "capabilities": ["strategy_modification"]}
            },
            {
                "name": "attention_controller",
                "type": ModuleType.COORDINATION,
                "category": ModuleCategory.ESSENTIAL,
                "definition": {"interfaces": ["coordination"], "capabilities": ["resource_allocation"]}
            },
            {
                "name": "fitness_evolution",
                "type": ModuleType.OPTIMIZATION,
                "category": ModuleCategory.ENHANCEMENT,
                "definition": {"interfaces": ["optimization"], "capabilities": ["fitness_adaptation"]}
            }
        ]

        for module_def in default_modules:
            module_id = f"module_{module_def['name']}_{int(time.time() * 1000)}"
            module = SystemModule(
                module_id=module_id,
                module_name=module_def["name"],
                module_type=module_def["type"],
                module_category=module_def["category"],
                module_definition=module_def["definition"]
            )
            self.system_modules[module_id] = module

    def _get_or_create_module(self, module_name: str) -> SystemModule:
        """Get existing module or create a new one."""
        # Check if module already exists
        for module in self.system_modules.values():
            if module.module_name == module_name:
                return module

        # Create new module
        module_id = f"module_{module_name}_{int(time.time() * 1000)}"
        module = SystemModule(
            module_id=module_id,
            module_name=module_name,
            module_type=ModuleType.EXPERIMENTAL,  # New modules start as experimental
            module_category=ModuleCategory.EXPERIMENTAL
        )
        self.system_modules[module_id] = module
        return module

    async def _create_initial_population(self):
        """Create the initial population of genomes."""
        for i in range(self.config["population_size"]):
            genome_id = f"genome_gen0_{i}_{int(time.time() * 1000)}"
            genome = self._create_minimal_genome(genome_id, 0)
            self.population[genome_id] = genome

        logger.info(f"Created initial population of {len(self.population)} genomes")

    def _create_minimal_genome(self, genome_id: str, generation: int) -> NEATGenome:
        """Create a minimal viable genome with basic architecture."""
        genome = NEATGenome(
            genome_id=genome_id,
            generation=generation
        )

        # Create input nodes (representing data sources)
        input_nodes = ["game_state", "score_change", "action_feedback"]
        for i, node_name in enumerate(input_nodes):
            node_id = f"input_{node_name}"
            node = NEATNode(
                node_id=node_id,
                node_type="input",
                layer=0,
                innovation_number=self._get_next_innovation_number()
            )
            genome.nodes[node_id] = node

        # Create output nodes (representing system actions)
        output_nodes = ["action_selection", "learning_trigger", "attention_allocation"]
        for i, node_name in enumerate(output_nodes):
            node_id = f"output_{node_name}"
            node = NEATNode(
                node_id=node_id,
                node_type="output",
                layer=2,
                innovation_number=self._get_next_innovation_number()
            )
            genome.nodes[node_id] = node

        # Create a minimal hidden layer with essential modules
        essential_modules = [m for m in self.system_modules.values()
                           if m.module_category == ModuleCategory.ESSENTIAL]
        for module in essential_modules:
            node_id = f"hidden_{module.module_name}"
            node = NEATNode(
                node_id=node_id,
                node_type="hidden",
                layer=1,
                module_reference=module.module_id,
                innovation_number=self._get_next_innovation_number()
            )
            genome.nodes[node_id] = node

        # Create basic connections (full connectivity for minimal genome)
        for input_node in [n for n in genome.nodes.values() if n.node_type == "input"]:
            for hidden_node in [n for n in genome.nodes.values() if n.node_type == "hidden"]:
                connection_id = f"conn_{input_node.node_id}_{hidden_node.node_id}"
                connection = NEATConnection(
                    connection_id=connection_id,
                    source_node=input_node.node_id,
                    target_node=hidden_node.node_id,
                    weight=random.uniform(-1.0, 1.0),
                    innovation_number=self._get_next_innovation_number()
                )
                genome.connections[connection_id] = connection

        for hidden_node in [n for n in genome.nodes.values() if n.node_type == "hidden"]:
            for output_node in [n for n in genome.nodes.values() if n.node_type == "output"]:
                connection_id = f"conn_{hidden_node.node_id}_{output_node.node_id}"
                connection = NEATConnection(
                    connection_id=connection_id,
                    source_node=hidden_node.node_id,
                    target_node=output_node.node_id,
                    weight=random.uniform(-1.0, 1.0),
                    innovation_number=self._get_next_innovation_number()
                )
                genome.connections[connection_id] = connection

        return genome

    async def _evaluate_population_fitness(self, game_id: str, session_id: str):
        """Evaluate fitness for all genomes in the population."""
        for genome in self.population.values():
            fitness = await self._evaluate_genome_fitness(genome, game_id, session_id)
            genome.fitness_score = fitness

        logger.debug(f"Evaluated fitness for {len(self.population)} genomes")

    async def _evaluate_genome_fitness(self, genome: NEATGenome, game_id: str, session_id: str) -> float:
        """Evaluate the fitness of a single genome."""
        try:
            fitness_components = {}

            # Module effectiveness component (40% of fitness)
            module_effectiveness = self._evaluate_module_effectiveness(genome)
            fitness_components["module_effectiveness"] = module_effectiveness * 0.4

            # Architectural efficiency component (30% of fitness)
            architectural_efficiency = self._evaluate_architectural_efficiency(genome)
            fitness_components["architectural_efficiency"] = architectural_efficiency * 0.3

            # System performance component (20% of fitness)
            system_performance = await self._evaluate_system_performance(genome, game_id, session_id)
            fitness_components["system_performance"] = system_performance * 0.2

            # Robustness component (10% of fitness)
            robustness = self._evaluate_robustness(genome)
            fitness_components["robustness"] = robustness * 0.1

            # Calculate total fitness
            total_fitness = sum(fitness_components.values())

            # Apply complexity penalty
            complexity_penalty = len(genome.nodes) * self.config["architectural_complexity_penalty"]
            total_fitness = max(0.0, total_fitness - complexity_penalty)

            return total_fitness

        except Exception as e:
            logger.error(f"Error evaluating genome fitness: {e}")
            return 0.0

    def _evaluate_module_effectiveness(self, genome: NEATGenome) -> float:
        """Evaluate how effective the modules in this genome are."""
        if not genome.nodes:
            return 0.0

        module_scores = []
        for node in genome.nodes.values():
            if node.module_reference and node.module_reference in self.system_modules:
                module = self.system_modules[node.module_reference]
                module_scores.append(module.effectiveness_score)

        return np.mean(module_scores) if module_scores else 0.5

    def _evaluate_architectural_efficiency(self, genome: NEATGenome) -> float:
        """Evaluate the efficiency of the genome's architecture."""
        if not genome.nodes or not genome.connections:
            return 0.0

        # Connection efficiency (well-connected but not over-connected)
        node_count = len(genome.nodes)
        connection_count = len([c for c in genome.connections.values() if c.enabled])

        # Optimal connectivity is roughly O(n log n)
        optimal_connections = max(1, node_count * math.log(node_count + 1))
        connection_efficiency = 1.0 - abs(connection_count - optimal_connections) / optimal_connections

        # Layer organization efficiency
        layers = set(node.layer for node in genome.nodes.values())
        layer_efficiency = min(1.0, len(layers) / max(1, node_count))  # Prefer layered organization

        # Module diversity (having different types of modules)
        module_types = set()
        for node in genome.nodes.values():
            if node.module_reference and node.module_reference in self.system_modules:
                module = self.system_modules[node.module_reference]
                module_types.add(module.module_type)

        diversity_score = len(module_types) / len(ModuleType)

        return (connection_efficiency * 0.5 + layer_efficiency * 0.3 + diversity_score * 0.2)

    async def _evaluate_system_performance(self, genome: NEATGenome, game_id: str, session_id: str) -> float:
        """Evaluate system performance metrics for this genome."""
        # This would integrate with actual system performance metrics
        # For now, return a simulated score based on module usage

        total_performance = 0.0
        active_modules = 0

        for node in genome.nodes.values():
            if node.module_reference and node.module_reference in self.system_modules:
                module = self.system_modules[node.module_reference]
                if module.is_active and module.usage_frequency > 0:
                    # Performance based on usage frequency and effectiveness
                    performance = (module.effectiveness_score *
                                 min(1.0, module.usage_frequency / 10.0))
                    total_performance += performance
                    active_modules += 1

        return total_performance / max(1, active_modules)

    def _evaluate_robustness(self, genome: NEATGenome) -> float:
        """Evaluate the robustness of the genome architecture."""
        if not genome.connections:
            return 0.0

        # Measure connectivity robustness
        enabled_connections = [c for c in genome.connections.values() if c.enabled]
        total_connections = len(genome.connections)

        connectivity_ratio = len(enabled_connections) / max(1, total_connections)

        # Measure redundancy (multiple paths between nodes)
        redundancy_score = self._calculate_path_redundancy(genome)

        # Measure module backup availability
        module_backup_score = self._calculate_module_backup_score(genome)

        return (connectivity_ratio * 0.4 + redundancy_score * 0.4 + module_backup_score * 0.2)

    def _calculate_path_redundancy(self, genome: NEATGenome) -> float:
        """Calculate path redundancy in the genome network."""
        # Simplified redundancy calculation
        input_nodes = [n for n in genome.nodes.values() if n.node_type == "input"]
        output_nodes = [n for n in genome.nodes.values() if n.node_type == "output"]

        if not input_nodes or not output_nodes:
            return 0.0

        # Count alternative paths (simplified)
        path_count = 0
        for input_node in input_nodes:
            for output_node in output_nodes:
                # Count connections that could form paths
                intermediate_connections = sum(1 for c in genome.connections.values()
                                            if c.enabled and c.source_node != input_node.node_id
                                            and c.target_node != output_node.node_id)
                if intermediate_connections > 0:
                    path_count += 1

        max_possible_paths = len(input_nodes) * len(output_nodes)
        return path_count / max(1, max_possible_paths)

    def _calculate_module_backup_score(self, genome: NEATGenome) -> float:
        """Calculate how well modules have backups/alternatives."""
        module_types_present = defaultdict(int)

        for node in genome.nodes.values():
            if node.module_reference and node.module_reference in self.system_modules:
                module = self.system_modules[node.module_reference]
                module_types_present[module.module_type] += 1

        # Score based on having multiple modules of each type
        backup_scores = []
        for module_type, count in module_types_present.items():
            backup_score = min(1.0, count / 2.0)  # Optimal is 2 modules per type
            backup_scores.append(backup_score)

        return np.mean(backup_scores) if backup_scores else 0.0

    async def _perform_speciation(self):
        """Organize genomes into species based on architectural similarity."""
        # Clear existing species assignments
        for species in self.species.values():
            species.genomes.clear()

        new_species = {}
        unassigned_genomes = list(self.population.keys())

        # Assign genomes to species
        for genome_id in unassigned_genomes:
            genome = self.population[genome_id]
            assigned = False

            # Try to assign to existing species
            for species_id, species in new_species.items():
                representative = self.population[species.representative_genome_id]
                compatibility = self._calculate_compatibility(genome, representative)

                if compatibility < species.compatibility_threshold:
                    species.genomes.append(genome_id)
                    genome.species_id = species_id
                    assigned = True
                    break

            # Create new species if not assigned
            if not assigned:
                species_id = f"species_{self.current_generation}_{len(new_species)}"
                species = NEATSpecies(
                    species_id=species_id,
                    generation=self.current_generation,
                    representative_genome_id=genome_id,
                    species_name=f"Species_{len(new_species) + 1}",
                    genomes=[genome_id]
                )
                new_species[species_id] = species
                genome.species_id = species_id

        # Update species fitness statistics
        for species in new_species.values():
            if species.genomes:
                fitnesses = [self.population[gid].fitness_score for gid in species.genomes]
                species.average_fitness = np.mean(fitnesses)
                species.max_fitness = np.max(fitnesses)
                species.genome_count = len(species.genomes)

        self.species = new_species
        logger.debug(f"Speciation created {len(self.species)} species")

    def _calculate_compatibility(self, genome1: NEATGenome, genome2: NEATGenome) -> float:
        """Calculate compatibility distance between two genomes."""
        # Get innovation numbers for both genomes
        innovations1 = set(genome1.innovation_numbers)
        innovations2 = set(genome2.innovation_numbers)

        # Count disjoint and excess innovations
        all_innovations = innovations1.union(innovations2)
        common_innovations = innovations1.intersection(innovations2)

        disjoint_excess = len(all_innovations) - len(common_innovations)

        # Calculate weight differences for common connections
        weight_differences = []
        for connection1 in genome1.connections.values():
            for connection2 in genome2.connections.values():
                if (connection1.innovation_number and connection2.innovation_number and
                    connection1.innovation_number == connection2.innovation_number):
                    weight_differences.append(abs(connection1.weight - connection2.weight))

        avg_weight_diff = np.mean(weight_differences) if weight_differences else 0.0

        # Calculate genome sizes
        n = max(len(genome1.connections), len(genome2.connections), 1)

        # Compatibility formula (coefficients from original NEAT paper)
        c1, c2, c3 = 1.0, 1.0, 0.4
        compatibility = (c1 * disjoint_excess / n) + (c3 * avg_weight_diff)

        return compatibility

    async def _create_next_generation(self) -> Dict[str, NEATGenome]:
        """Create the next generation through selection and reproduction."""
        new_population = {}

        # Calculate offspring allocation for each species
        self._calculate_offspring_allocation()

        # Reproduce each species
        for species in self.species.values():
            if species.is_extinct or not species.genomes:
                continue

            offspring = await self._reproduce_species(species)
            new_population.update(offspring)

        # Fill population to target size if needed
        while len(new_population) < self.config["population_size"]:
            # Create a new random genome
            genome_id = f"genome_gen{self.current_generation + 1}_{len(new_population)}"
            genome = self._create_minimal_genome(genome_id, self.current_generation + 1)
            new_population[genome_id] = genome

        return new_population

    def _calculate_offspring_allocation(self):
        """Calculate how many offspring each species should produce."""
        total_adjusted_fitness = 0.0

        # Calculate adjusted fitness for each species
        for species in self.species.values():
            if not species.is_extinct and species.genomes:
                # Fitness sharing within species
                adjusted_fitness = species.average_fitness / len(species.genomes)
                total_adjusted_fitness += adjusted_fitness

        # Allocate offspring based on adjusted fitness
        for species in self.species.values():
            if species.is_extinct or not species.genomes:
                species.allowed_offspring = 0
            else:
                adjusted_fitness = species.average_fitness / len(species.genomes)
                species.allowed_offspring = max(1, int(
                    (adjusted_fitness / max(0.001, total_adjusted_fitness)) * self.config["population_size"]
                ))

    async def _reproduce_species(self, species: NEATSpecies) -> Dict[str, NEATGenome]:
        """Reproduce a species to create offspring."""
        offspring = {}

        if species.allowed_offspring <= 0:
            return offspring

        # Sort genomes by fitness
        species_genomes = [(gid, self.population[gid]) for gid in species.genomes]
        species_genomes.sort(key=lambda x: x[1].fitness_score, reverse=True)

        # Preserve elite
        elite_count = max(1, int(len(species_genomes) * self.config["elitism_ratio"]))

        for i in range(min(elite_count, species.allowed_offspring)):
            if i < len(species_genomes):
                original_genome = species_genomes[i][1]
                # Create a copy of the elite genome
                elite_genome_id = f"genome_gen{self.current_generation + 1}_elite_{i}"
                elite_genome = self._copy_genome(original_genome, elite_genome_id, self.current_generation + 1)
                offspring[elite_genome_id] = elite_genome

        # Create remaining offspring through crossover and mutation
        remaining_offspring = species.allowed_offspring - len(offspring)

        for i in range(remaining_offspring):
            if len(species_genomes) >= 2 and random.random() < self.config["interspecies_mating_rate"]:
                # Crossover within species
                parent1 = random.choice(species_genomes[:len(species_genomes)//2])[1]  # Top half
                parent2 = random.choice(species_genomes)[1]
                child_id = f"genome_gen{self.current_generation + 1}_cross_{i}"
                child = self._crossover(parent1, parent2, child_id, self.current_generation + 1)
            else:
                # Mutation of existing genome
                parent = random.choice(species_genomes[:len(species_genomes)//2])[1]  # Top half
                child_id = f"genome_gen{self.current_generation + 1}_mut_{i}"
                child = self._copy_genome(parent, child_id, self.current_generation + 1)

            offspring[child_id] = child

        return offspring

    def _copy_genome(self, original: NEATGenome, new_id: str, generation: int) -> NEATGenome:
        """Create a deep copy of a genome with new ID."""
        new_genome = NEATGenome(
            genome_id=new_id,
            generation=generation,
            species_id=original.species_id,
            parent_genome_ids=[original.genome_id]
        )

        # Copy nodes
        for node in original.nodes.values():
            new_node = NEATNode(
                node_id=node.node_id,
                node_type=node.node_type,
                activation_function=node.activation_function,
                bias=node.bias,
                layer=node.layer,
                module_reference=node.module_reference,
                innovation_number=node.innovation_number
            )
            new_genome.nodes[node.node_id] = new_node

        # Copy connections
        for connection in original.connections.values():
            new_connection = NEATConnection(
                connection_id=connection.connection_id,
                source_node=connection.source_node,
                target_node=connection.target_node,
                weight=connection.weight,
                enabled=connection.enabled,
                innovation_number=connection.innovation_number
            )
            new_genome.connections[connection.connection_id] = new_connection

        new_genome.innovation_numbers = original.innovation_numbers.copy()

        return new_genome

    def _crossover(self, parent1: NEATGenome, parent2: NEATGenome, child_id: str, generation: int) -> NEATGenome:
        """Perform crossover between two parent genomes."""
        child = NEATGenome(
            genome_id=child_id,
            generation=generation,
            parent_genome_ids=[parent1.genome_id, parent2.genome_id]
        )

        # Determine which parent is more fit
        more_fit_parent = parent1 if parent1.fitness_score >= parent2.fitness_score else parent2
        less_fit_parent = parent2 if more_fit_parent == parent1 else parent1

        # Copy nodes from more fit parent
        child.nodes = {nid: self._copy_node(node) for nid, node in more_fit_parent.nodes.items()}

        # Crossover connections
        for connection in more_fit_parent.connections.values():
            if connection.innovation_number in [c.innovation_number for c in less_fit_parent.connections.values()]:
                # Matching connection - randomly choose from either parent
                parent_connection = connection
                if random.random() < 0.5:
                    for other_conn in less_fit_parent.connections.values():
                        if other_conn.innovation_number == connection.innovation_number:
                            parent_connection = other_conn
                            break

                child_connection = NEATConnection(
                    connection_id=parent_connection.connection_id,
                    source_node=parent_connection.source_node,
                    target_node=parent_connection.target_node,
                    weight=parent_connection.weight,
                    enabled=parent_connection.enabled,
                    innovation_number=parent_connection.innovation_number
                )
                child.connections[parent_connection.connection_id] = child_connection
            else:
                # Disjoint/excess connection - inherit from more fit parent
                child_connection = NEATConnection(
                    connection_id=connection.connection_id,
                    source_node=connection.source_node,
                    target_node=connection.target_node,
                    weight=connection.weight,
                    enabled=connection.enabled,
                    innovation_number=connection.innovation_number
                )
                child.connections[connection.connection_id] = child_connection

        # Update innovation numbers
        child.innovation_numbers = set(c.innovation_number for c in child.connections.values()
                                     if c.innovation_number is not None)

        return child

    def _copy_node(self, original: NEATNode) -> NEATNode:
        """Create a copy of a node."""
        return NEATNode(
            node_id=original.node_id,
            node_type=original.node_type,
            activation_function=original.activation_function,
            bias=original.bias,
            layer=original.layer,
            module_reference=original.module_reference,
            innovation_number=original.innovation_number
        )

    async def _apply_mutations(self, population: Dict[str, NEATGenome]):
        """Apply mutations to the population."""
        for genome in population.values():
            if random.random() < self.config["mutation_rate"]:
                await self._mutate_genome(genome)

    async def _mutate_genome(self, genome: NEATGenome):
        """Apply various mutations to a genome."""
        mutations_applied = []

        # Weight mutation
        if random.random() < self.config["weight_mutation_rate"]:
            self._mutate_weights(genome)
            mutations_applied.append("weight_mutation")

        # Add node mutation
        if random.random() < self.config["add_node_mutation_rate"]:
            await self._mutate_add_node(genome)
            mutations_applied.append("add_node")

        # Add connection mutation
        if random.random() < self.config["add_connection_mutation_rate"]:
            await self._mutate_add_connection(genome)
            mutations_applied.append("add_connection")

        # Module mutation (specific to this system)
        if random.random() < 0.1:  # 10% chance
            await self._mutate_module_assignment(genome)
            mutations_applied.append("module_mutation")

        if mutations_applied:
            logger.debug(f"Applied mutations to {genome.genome_id}: {mutations_applied}")

    def _mutate_weights(self, genome: NEATGenome):
        """Mutate connection weights."""
        for connection in genome.connections.values():
            if random.random() < self.config["weight_perturbation_rate"]:
                # Perturb existing weight
                connection.weight += random.uniform(-0.1, 0.1)
                connection.weight = max(-5.0, min(5.0, connection.weight))  # Clamp weights
            else:
                # Assign new random weight
                connection.weight = random.uniform(-1.0, 1.0)

    async def _mutate_add_node(self, genome: NEATGenome):
        """Add a new node by splitting an existing connection."""
        if not genome.connections:
            return

        # Choose a random enabled connection to split
        enabled_connections = [c for c in genome.connections.values() if c.enabled]
        if not enabled_connections:
            return

        connection_to_split = random.choice(enabled_connections)

        # Create new node
        node_id = f"hidden_mut_{len(genome.nodes)}"
        new_node = NEATNode(
            node_id=node_id,
            node_type="hidden",
            layer=1,  # Place in middle layer
            innovation_number=self._get_next_innovation_number()
        )

        # Optionally assign a module to the new node
        available_modules = [m for m in self.system_modules.values()
                           if m.is_active and m.module_category != ModuleCategory.DEPRECATED]
        if available_modules:
            selected_module = random.choice(available_modules)
            new_node.module_reference = selected_module.module_id

        genome.nodes[node_id] = new_node

        # Disable the original connection
        connection_to_split.enabled = False

        # Create two new connections
        conn1_id = f"conn_{connection_to_split.source_node}_{node_id}"
        conn1 = NEATConnection(
            connection_id=conn1_id,
            source_node=connection_to_split.source_node,
            target_node=node_id,
            weight=1.0,  # Keep signal strength
            innovation_number=self._get_next_innovation_number()
        )

        conn2_id = f"conn_{node_id}_{connection_to_split.target_node}"
        conn2 = NEATConnection(
            connection_id=conn2_id,
            source_node=node_id,
            target_node=connection_to_split.target_node,
            weight=connection_to_split.weight,  # Preserve original weight
            innovation_number=self._get_next_innovation_number()
        )

        genome.connections[conn1_id] = conn1
        genome.connections[conn2_id] = conn2

        # Update innovation numbers
        genome.innovation_numbers.update([new_node.innovation_number, conn1.innovation_number, conn2.innovation_number])

        # Record innovation
        await self._record_innovation(InnovationType.ADD_NODE, node_id, None, None, genome.genome_id)

    async def _mutate_add_connection(self, genome: NEATGenome):
        """Add a new connection between existing nodes."""
        if len(genome.nodes) < 2:
            return

        # Find potential connections that don't already exist
        existing_connections = set((c.source_node, c.target_node) for c in genome.connections.values())

        # Try to create a new connection
        attempts = 0
        while attempts < 10:  # Limit attempts to avoid infinite loop
            source_node = random.choice(list(genome.nodes.keys()))
            target_node = random.choice(list(genome.nodes.keys()))

            if (source_node != target_node and
                (source_node, target_node) not in existing_connections and
                self._is_valid_connection(genome, source_node, target_node)):

                # Create new connection
                conn_id = f"conn_{source_node}_{target_node}"
                new_connection = NEATConnection(
                    connection_id=conn_id,
                    source_node=source_node,
                    target_node=target_node,
                    weight=random.uniform(-1.0, 1.0),
                    innovation_number=self._get_next_innovation_number()
                )

                genome.connections[conn_id] = new_connection
                genome.innovation_numbers.add(new_connection.innovation_number)

                # Record innovation
                await self._record_innovation(InnovationType.ADD_CONNECTION, source_node, target_node,
                                            new_connection.weight, genome.genome_id)
                break

            attempts += 1

    def _is_valid_connection(self, genome: NEATGenome, source_id: str, target_id: str) -> bool:
        """Check if a connection between two nodes is valid."""
        source_node = genome.nodes.get(source_id)
        target_node = genome.nodes.get(target_id)

        if not source_node or not target_node:
            return False

        # Prevent connections that would create cycles (feedforward only for now)
        if source_node.layer >= target_node.layer:
            return False

        # Output nodes cannot be sources
        if source_node.node_type == "output":
            return False

        # Input nodes cannot be targets
        if target_node.node_type == "input":
            return False

        return True

    async def _mutate_module_assignment(self, genome: NEATGenome):
        """Mutate module assignments for hidden nodes."""
        hidden_nodes = [n for n in genome.nodes.values() if n.node_type == "hidden"]
        if not hidden_nodes:
            return

        node_to_mutate = random.choice(hidden_nodes)

        # Change module assignment
        available_modules = [m for m in self.system_modules.values()
                           if m.is_active and m.module_category != ModuleCategory.DEPRECATED]

        if available_modules:
            new_module = random.choice(available_modules)
            old_module_id = node_to_mutate.module_reference
            node_to_mutate.module_reference = new_module.module_id

            # Record innovation
            await self._record_innovation(InnovationType.MODIFY_MODULE, None, None, None, genome.genome_id)

    async def _prune_ineffective_modules(self, game_id: str, session_id: str) -> Dict[str, Any]:
        """Prune modules that fall below effectiveness thresholds."""
        pruning_results = {
            "modules_evaluated": 0,
            "pruned_modules": [],
            "preserved_modules": [],
            "modified_modules": []
        }

        try:
            for module in list(self.system_modules.values()):
                pruning_results["modules_evaluated"] += 1

                # Skip essential modules
                if module.module_category == ModuleCategory.ESSENTIAL:
                    pruning_results["preserved_modules"].append(module.module_id)
                    continue

                # Evaluate for pruning
                should_prune = await self._should_prune_module(module)

                if should_prune:
                    # Mark as pruning candidate
                    module.pruning_candidate = True

                    # Create pruning decision record
                    pruning_decision = {
                        "pruning_id": f"prune_{module.module_id}_{int(time.time() * 1000)}",
                        "module_id": module.module_id,
                        "decision_timestamp": time.time(),
                        "decision_type": "prune",
                        "decision_reason": {
                            "effectiveness_score": module.effectiveness_score,
                            "usage_frequency": module.usage_frequency,
                            "last_used": module.last_used_timestamp,
                            "threshold": self.config["module_pruning_threshold"]
                        },
                        "usage_statistics": {
                            "total_usage": module.usage_frequency,
                            "effectiveness": module.effectiveness_score,
                            "category": module.module_category.value
                        }
                    }

                    await self._store_pruning_decision(pruning_decision)

                    # Actually perform pruning for experimental modules
                    if module.module_category == ModuleCategory.EXPERIMENTAL:
                        await self._execute_module_pruning(module, game_id, session_id)
                        pruning_results["pruned_modules"].append(module.module_id)
                    else:
                        # Just mark as candidate for review
                        pruning_results["modified_modules"].append(module.module_id)
                else:
                    pruning_results["preserved_modules"].append(module.module_id)

            logger.info(f"Module pruning completed: {len(pruning_results['pruned_modules'])} pruned, "
                       f"{len(pruning_results['preserved_modules'])} preserved")

            return pruning_results

        except Exception as e:
            logger.error(f"Error in module pruning: {e}")
            return pruning_results

    async def _should_prune_module(self, module: SystemModule) -> bool:
        """Determine if a module should be pruned based on usage and effectiveness."""
        # Don't prune essential modules
        if module.module_category == ModuleCategory.ESSENTIAL:
            return False

        # Prune if effectiveness is below threshold
        if module.effectiveness_score < self.config["module_pruning_threshold"]:
            return True

        # Prune if not used recently (for experimental modules)
        if module.module_category == ModuleCategory.EXPERIMENTAL:
            if module.usage_frequency == 0:
                return True

            if (module.last_used_timestamp and
                time.time() - module.last_used_timestamp > 3600):  # 1 hour threshold
                return True

        return False

    async def _execute_module_pruning(self, module: SystemModule, game_id: str, session_id: str):
        """Execute the actual pruning of a module."""
        try:
            # Remove module from active modules
            module.is_active = False
            module.module_category = ModuleCategory.DEPRECATED

            # Remove module references from all genomes
            for genome in self.population.values():
                nodes_to_update = []
                for node in genome.nodes.values():
                    if node.module_reference == module.module_id:
                        nodes_to_update.append(node)

                # Update nodes to remove module reference
                for node in nodes_to_update:
                    node.module_reference = None

            logger.info(f"Module {module.module_name} pruned successfully")

        except Exception as e:
            logger.error(f"Error executing module pruning for {module.module_name}: {e}")

    def _select_best_genome(self) -> Optional[NEATGenome]:
        """Select the best genome from the current population."""
        if not self.population:
            return None

        best_genome = max(self.population.values(), key=lambda g: g.fitness_score)
        best_genome.is_champion = True
        return best_genome

    def _get_next_innovation_number(self) -> int:
        """Get the next global innovation number."""
        self.global_innovation_number += 1
        return self.global_innovation_number

    async def _record_innovation(self, innovation_type: InnovationType, source_node: Optional[str],
                                target_node: Optional[str], weight: Optional[float], genome_id: str):
        """Record a new innovation in the innovation database."""
        innovation = NEATInnovation(
            innovation_id=self._get_next_innovation_number(),
            innovation_type=innovation_type,
            source_node_id=source_node,
            target_node_id=target_node,
            weight_value=weight,
            innovation_description=f"{innovation_type.value} in {genome_id}",
            first_genome_id=genome_id,
            generation_introduced=self.current_generation
        )

        self.innovations[innovation.innovation_id] = innovation

        # Store in database
        await self._store_innovation(innovation)

    def _identify_architectural_bottlenecks(self) -> Dict[str, List[str]]:
        """Identify current architectural bottlenecks."""
        bottlenecks = defaultdict(list)

        # Analyze module effectiveness
        low_effectiveness_modules = [
            m for m in self.system_modules.values()
            if m.is_active and m.effectiveness_score < 0.5
        ]

        if low_effectiveness_modules:
            bottlenecks["module_effectiveness"] = [m.module_name for m in low_effectiveness_modules]

        # Analyze architectural complexity
        if self.champion_genome:
            node_count = len(self.champion_genome.nodes)
            connection_count = len(self.champion_genome.connections)

            if connection_count > node_count * 2:
                bottlenecks["over_connectivity"] = ["excessive_connections"]

            if node_count > 20:
                bottlenecks["architectural_bloat"] = ["too_many_nodes"]

        # Analyze species diversity
        if len(self.species) < 3:
            bottlenecks["diversity"] = ["insufficient_species_diversity"]

        return dict(bottlenecks)

    async def _initialize_module_tracking(self, game_id: str, session_id: str):
        """Initialize module tracking for a new session."""
        # Reset usage frequencies for the session
        for module in self.system_modules.values():
            # Don't reset completely, but decay from previous sessions
            module.usage_frequency = int(module.usage_frequency * self.config["effectiveness_decay_rate"])

        logger.debug(f"Module tracking initialized for session {session_id}")

    async def _store_architectural_state(self, game_id: str, session_id: str):
        """Store current architectural state in database."""
        try:
            # Store current modules
            for module in self.system_modules.values():
                await self.db.execute_query(
                    """INSERT OR REPLACE INTO system_modules
                       (module_id, module_name, module_type, module_category, current_version,
                        module_definition, dependency_modules, dependent_modules, resource_requirements,
                        performance_metrics, usage_frequency, effectiveness_score, last_used_timestamp,
                        activation_threshold, pruning_candidate, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (module.module_id, module.module_name, module.module_type.value, module.module_category.value,
                     module.current_version, json.dumps(module.module_definition),
                     json.dumps(module.dependency_modules), json.dumps(module.dependent_modules),
                     json.dumps(module.resource_requirements), json.dumps(module.performance_metrics),
                     module.usage_frequency, module.effectiveness_score, module.last_used_timestamp,
                     module.activation_threshold, module.pruning_candidate, module.is_active)
                )

            # Store current population state
            population_record = {
                "population_id": f"pop_{session_id}_{self.current_generation}",
                "generation": self.current_generation,
                "population_size": len(self.population),
                "species_count": len(self.species),
                "champion_genome_id": self.champion_genome.genome_id if self.champion_genome else None,
                "average_fitness": np.mean([g.fitness_score for g in self.population.values()]) if self.population else 0.0,
                "max_fitness": max([g.fitness_score for g in self.population.values()]) if self.population else 0.0,
                "generation_timestamp": time.time()
            }

            await self.db.execute_query(
                """INSERT INTO neat_populations
                   (population_id, generation, population_size, species_count, champion_genome_id,
                    average_fitness, max_fitness, generation_timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (population_record["population_id"], population_record["generation"],
                 population_record["population_size"], population_record["species_count"],
                 population_record["champion_genome_id"], population_record["average_fitness"],
                 population_record["max_fitness"], population_record["generation_timestamp"])
            )

        except Exception as e:
            logger.error(f"Error storing architectural state: {e}")

    async def _store_module_usage(self, usage_record: Dict[str, Any]):
        """Store module usage record in database."""
        try:
            await self.db.execute_query(
                """INSERT INTO module_usage_tracking
                   (usage_id, module_id, game_id, session_id, usage_timestamp, usage_context,
                    processing_time_ms, success, effectiveness_score, impact_on_performance,
                    resource_consumption, interactions_with_modules)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (usage_record["usage_id"], usage_record["module_id"], usage_record["game_id"],
                 usage_record["session_id"], usage_record["usage_timestamp"],
                 json.dumps(usage_record["usage_context"]), usage_record["processing_time_ms"],
                 usage_record["success"], usage_record["effectiveness_score"],
                 usage_record["impact_on_performance"], json.dumps(usage_record["resource_consumption"]),
                 json.dumps(usage_record["interactions_with_modules"]))
            )
        except Exception as e:
            logger.error(f"Error storing module usage: {e}")

    async def _store_pruning_decision(self, pruning_decision: Dict[str, Any]):
        """Store module pruning decision in database."""
        try:
            await self.db.execute_query(
                """INSERT INTO module_pruning_decisions
                   (pruning_id, module_id, decision_timestamp, decision_type, decision_reason,
                    usage_statistics, pruning_scheduled, pruning_executed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (pruning_decision["pruning_id"], pruning_decision["module_id"],
                 pruning_decision["decision_timestamp"], pruning_decision["decision_type"],
                 json.dumps(pruning_decision["decision_reason"]), json.dumps(pruning_decision["usage_statistics"]),
                 False, False)
            )
        except Exception as e:
            logger.error(f"Error storing pruning decision: {e}")

    async def _store_innovation(self, innovation: NEATInnovation):
        """Store innovation record in database."""
        try:
            await self.db.execute_query(
                """INSERT INTO neat_innovations
                   (innovation_id, innovation_type, source_node_id, target_node_id, weight_value,
                    innovation_description, first_genome_id, generation_introduced, usage_frequency,
                    average_fitness_impact)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (innovation.innovation_id, innovation.innovation_type.value, innovation.source_node_id,
                 innovation.target_node_id, innovation.weight_value, innovation.innovation_description,
                 innovation.first_genome_id, innovation.generation_introduced, innovation.usage_frequency,
                 innovation.average_fitness_impact)
            )
        except Exception as e:
            logger.error(f"Error storing innovation: {e}")

    async def _store_evolution_results(self, game_id: str, session_id: str, results: Dict[str, Any]):
        """Store evolution results in database."""
        try:
            experiment_record = {
                "experiment_id": f"evolution_{session_id}_{results['generation']}",
                "experiment_name": f"Generation {results['generation']} Evolution",
                "experiment_type": "full_evolution",
                "experiment_timestamp": time.time(),
                "experiment_duration_seconds": results["evolution_time"],
                "experimental_performance": results,
                "experiment_success": results["champion_fitness"] > 0.0
            }

            await self.db.execute_query(
                """INSERT INTO architectural_evolution_experiments
                   (experiment_id, experiment_name, experiment_type, experiment_timestamp,
                    experiment_duration_seconds, experimental_performance, experiment_success)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (experiment_record["experiment_id"], experiment_record["experiment_name"],
                 experiment_record["experiment_type"], experiment_record["experiment_timestamp"],
                 experiment_record["experiment_duration_seconds"], json.dumps(experiment_record["experimental_performance"]),
                 experiment_record["experiment_success"])
            )
        except Exception as e:
            logger.error(f"Error storing evolution results: {e}")

    async def _store_architectural_attention_integration(self, game_id: str, session_id: str,
                                                       evolution_priorities: Dict[str, float],
                                                       attention_allocation: Optional[Any]):
        """Store architectural attention integration record in database."""
        try:
            integration_id = f"arch_attn_{game_id}_{int(time.time() * 1000)}"

            allocation_received = {}
            if attention_allocation:
                allocation_received = {
                    "allocation_id": getattr(attention_allocation, 'allocation_id', ''),
                    "allocations": getattr(attention_allocation, 'allocations', {}),
                    "reasoning": getattr(attention_allocation, 'allocation_reasoning', '')
                }

            await self.db.execute_query(
                """INSERT INTO architectural_attention_integration
                   (integration_id, game_id, session_id, integration_timestamp,
                    current_architecture_id, architectural_priorities, attention_allocation_request,
                    attention_allocation_received, evolution_targets, integration_effectiveness)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (integration_id, game_id, session_id, time.time(),
                 self.champion_genome.genome_id if self.champion_genome else None,
                 json.dumps(evolution_priorities), json.dumps({"requested": True}),
                 json.dumps(allocation_received), json.dumps(list(evolution_priorities.keys())),
                 0.5)  # Default effectiveness
            )
        except Exception as e:
            logger.error(f"Error storing architectural attention integration: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics for the NEAT architect system."""
        return {
            "current_generation": self.current_generation,
            "population_size": len(self.population),
            "species_count": len(self.species),
            "champion_fitness": self.champion_genome.fitness_score if self.champion_genome else 0.0,
            "total_innovations": len(self.innovations),
            "active_modules": len([m for m in self.system_modules.values() if m.is_active]),
            "pruning_candidates": len([m for m in self.system_modules.values() if m.pruning_candidate]),
            "average_module_effectiveness": np.mean([m.effectiveness_score for m in self.system_modules.values()]) if self.system_modules else 0.0,
            "total_module_usage": sum(m.usage_frequency for m in self.system_modules.values()),
            "attention_integration_enabled": self.attention_integration_enabled,
            "config": self.config.copy()
        }
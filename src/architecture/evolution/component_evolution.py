"""
Component Evolution

Advanced component evolution system for evolving individual components
within the modular architecture.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

from ...training.interfaces import ComponentInterface
from ...training.caching import CacheManager, CacheConfig
from ...training.monitoring import PerformanceMonitor


class ComponentType(Enum):
    """Types of components."""
    CORE = "core"
    INTERFACE = "interface"
    UTILITY = "utility"
    ADAPTER = "adapter"
    FACTORY = "factory"
    STRATEGY = "strategy"
    OBSERVER = "observer"
    DECORATOR = "decorator"


@dataclass
class ComponentMutation:
    """Component mutation data structure."""
    mutation_type: str
    component_name: str
    changes: Dict[str, Any]
    timestamp: datetime
    success: bool
    fitness_delta: float


@dataclass
class ComponentFitness:
    """Component fitness data structure."""
    component_name: str
    fitness_score: float
    metrics: Dict[str, float]
    timestamp: datetime
    evaluation_time: float


class ComponentEvolution(ComponentInterface):
    """
    Advanced component evolution system for evolving individual components
    within the modular architecture.
    """
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """Initialize the component evolution system."""
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Evolution state
        self.component_mutations: List[ComponentMutation] = []
        self.component_fitness: List[ComponentFitness] = []
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.mutation_times: List[float] = []
        self.fitness_evaluation_times: List[float] = []
        
        # Evolution components
        self.mutation_operators: List[Callable] = []
        self.fitness_functions: List[Callable] = []
        self.component_templates: Dict[str, Dict[str, Any]] = {}
        
        # Component registry
        self.component_registry: Dict[str, Dict[str, Any]] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the component evolution system."""
        try:
            self.cache.initialize()
            self.performance_monitor.start_monitoring()
            
            # Initialize evolution components
            self._initialize_mutation_operators()
            self._initialize_fitness_functions()
            self._initialize_component_templates()
            
            # Load component registry
            self._load_component_registry()
            
            self._initialized = True
            self.logger.info("Component evolution system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize component evolution: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'ComponentEvolution',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'component_mutations_count': len(self.component_mutations),
                'component_fitness_count': len(self.component_fitness),
                'evolution_history_count': len(self.evolution_history),
                'mutation_operators_count': len(self.mutation_operators),
                'fitness_functions_count': len(self.fitness_functions),
                'component_templates_count': len(self.component_templates),
                'component_registry_count': len(self.component_registry),
                'average_mutation_time': np.mean(self.mutation_times) if self.mutation_times else 0.0
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.performance_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Component evolution system cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def evolve_component(self, component_name: str, 
                        target_improvements: Optional[List[str]] = None) -> ComponentMutation:
        """Evolve a specific component."""
        try:
            start_time = datetime.now()
            
            # Get current component
            if component_name not in self.component_registry:
                raise ValueError(f"Component {component_name} not found in registry")
            
            current_component = self.component_registry[component_name]
            
            # Select mutation operator
            mutation_operator = np.random.choice(self.mutation_operators)
            
            # Apply mutation
            mutated_component = mutation_operator(current_component.copy())
            
            # Evaluate fitness
            current_fitness = self._evaluate_component_fitness(current_component)
            mutated_fitness = self._evaluate_component_fitness(mutated_component)
            
            # Calculate fitness delta
            fitness_delta = mutated_fitness - current_fitness
            
            # Determine success
            success = fitness_delta > 0 or np.random.random() < 0.1  # 10% chance of accepting worse fitness
            
            # Create mutation record
            mutation = ComponentMutation(
                mutation_type=mutation_operator.__name__,
                component_name=component_name,
                changes=self._calculate_changes(current_component, mutated_component),
                timestamp=datetime.now(),
                success=success,
                fitness_delta=fitness_delta
            )
            
            # Store mutation
            self.component_mutations.append(mutation)
            
            # Update component if successful
            if success:
                self.component_registry[component_name] = mutated_component
                self._save_component_registry()
            
            # Update performance metrics
            mutation_time = (datetime.now() - start_time).total_seconds()
            self.mutation_times.append(mutation_time)
            
            # Cache mutation
            cache_key = f"mutation_{component_name}_{datetime.now().timestamp()}"
            self.cache.set(cache_key, mutation, ttl=3600)
            
            self.logger.info(f"Component {component_name} evolved in {mutation_time:.3f}s (fitness delta: {fitness_delta:.3f})")
            
            return mutation
            
        except Exception as e:
            self.logger.error(f"Error evolving component {component_name}: {e}")
            raise
    
    def evaluate_component_fitness(self, component_name: str) -> ComponentFitness:
        """Evaluate fitness of a specific component."""
        try:
            start_time = datetime.now()
            
            # Get component
            if component_name not in self.component_registry:
                raise ValueError(f"Component {component_name} not found in registry")
            
            component = self.component_registry[component_name]
            
            # Evaluate fitness
            fitness_score, metrics = self._evaluate_component_fitness_detailed(component)
            
            # Create fitness record
            fitness = ComponentFitness(
                component_name=component_name,
                fitness_score=fitness_score,
                metrics=metrics,
                timestamp=datetime.now(),
                evaluation_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Store fitness
            self.component_fitness.append(fitness)
            
            # Update performance metrics
            evaluation_time = (datetime.now() - start_time).total_seconds()
            self.fitness_evaluation_times.append(evaluation_time)
            
            # Cache fitness
            cache_key = f"fitness_{component_name}_{datetime.now().timestamp()}"
            self.cache.set(cache_key, fitness, ttl=3600)
            
            self.logger.debug(f"Component {component_name} fitness evaluated: {fitness_score:.3f}")
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error evaluating component fitness for {component_name}: {e}")
            raise
    
    def get_component_fitness_history(self, component_name: str) -> List[ComponentFitness]:
        """Get fitness history for a specific component."""
        try:
            return [f for f in self.component_fitness if f.component_name == component_name]
        except Exception as e:
            self.logger.error(f"Error getting fitness history for {component_name}: {e}")
            return []
    
    def get_component_mutations(self, component_name: str) -> List[ComponentMutation]:
        """Get mutations for a specific component."""
        try:
            return [m for m in self.component_mutations if m.component_name == component_name]
        except Exception as e:
            self.logger.error(f"Error getting mutations for {component_name}: {e}")
            return []
    
    def get_component_statistics(self, component_name: str) -> Dict[str, Any]:
        """Get statistics for a specific component."""
        try:
            # Get fitness history
            fitness_history = self.get_component_fitness_history(component_name)
            
            # Get mutation history
            mutation_history = self.get_component_mutations(component_name)
            
            # Calculate statistics
            if fitness_history:
                fitness_scores = [f.fitness_score for f in fitness_history]
                fitness_trend = self._calculate_fitness_trend(fitness_scores)
            else:
                fitness_scores = []
                fitness_trend = 'no_data'
            
            if mutation_history:
                successful_mutations = [m for m in mutation_history if m.success]
                success_rate = len(successful_mutations) / len(mutation_history)
                average_fitness_delta = np.mean([m.fitness_delta for m in mutation_history])
            else:
                success_rate = 0.0
                average_fitness_delta = 0.0
            
            return {
                'component_name': component_name,
                'fitness_scores_count': len(fitness_scores),
                'mutation_count': len(mutation_history),
                'successful_mutations': len([m for m in mutation_history if m.success]),
                'success_rate': success_rate,
                'average_fitness_delta': average_fitness_delta,
                'fitness_trend': fitness_trend,
                'current_fitness': fitness_scores[-1] if fitness_scores else 0.0,
                'max_fitness': np.max(fitness_scores) if fitness_scores else 0.0,
                'min_fitness': np.min(fitness_scores) if fitness_scores else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics for {component_name}: {e}")
            return {'error': str(e)}
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get evolution system statistics."""
        try:
            return {
                'total_mutations': len(self.component_mutations),
                'total_fitness_evaluations': len(self.component_fitness),
                'evolution_history_count': len(self.evolution_history),
                'mutation_operators_count': len(self.mutation_operators),
                'fitness_functions_count': len(self.fitness_functions),
                'component_templates_count': len(self.component_templates),
                'component_registry_count': len(self.component_registry),
                'average_mutation_time': np.mean(self.mutation_times) if self.mutation_times else 0.0,
                'average_fitness_evaluation_time': np.mean(self.fitness_evaluation_times) if self.fitness_evaluation_times else 0.0
            }
        except Exception as e:
            self.logger.error(f"Error getting evolution statistics: {e}")
            return {'error': str(e)}
    
    def _initialize_mutation_operators(self) -> None:
        """Initialize mutation operators."""
        try:
            # Add method mutation
            self.mutation_operators.append(self._mutate_methods)
            
            # Add property mutation
            self.mutation_operators.append(self._mutate_properties)
            
            # Add dependency mutation
            self.mutation_operators.append(self._mutate_dependencies)
            
            # Add configuration mutation
            self.mutation_operators.append(self._mutate_configuration)
            
            # Add interface mutation
            self.mutation_operators.append(self._mutate_interfaces)
            
            self.logger.info(f"Initialized {len(self.mutation_operators)} mutation operators")
            
        except Exception as e:
            self.logger.error(f"Error initializing mutation operators: {e}")
    
    def _initialize_fitness_functions(self) -> None:
        """Initialize fitness functions."""
        try:
            # Add complexity fitness
            self.fitness_functions.append(self._evaluate_complexity_fitness)
            
            # Add cohesion fitness
            self.fitness_functions.append(self._evaluate_cohesion_fitness)
            
            # Add coupling fitness
            self.fitness_functions.append(self._evaluate_coupling_fitness)
            
            # Add reusability fitness
            self.fitness_functions.append(self._evaluate_reusability_fitness)
            
            # Add testability fitness
            self.fitness_functions.append(self._evaluate_testability_fitness)
            
            self.logger.info(f"Initialized {len(self.fitness_functions)} fitness functions")
            
        except Exception as e:
            self.logger.error(f"Error initializing fitness functions: {e}")
    
    def _initialize_component_templates(self) -> None:
        """Initialize component templates."""
        try:
            # Add core component template
            self.component_templates['core'] = {
                'type': 'core',
                'methods': ['initialize', 'get_state', 'cleanup', 'is_healthy'],
                'properties': ['name', 'status', 'last_updated'],
                'dependencies': [],
                'interfaces': ['ComponentInterface'],
                'config': {}
            }
            
            # Add interface component template
            self.component_templates['interface'] = {
                'type': 'interface',
                'methods': ['method1', 'method2'],
                'properties': ['property1', 'property2'],
                'dependencies': [],
                'interfaces': [],
                'config': {}
            }
            
            # Add utility component template
            self.component_templates['utility'] = {
                'type': 'utility',
                'methods': ['utility_method1', 'utility_method2'],
                'properties': ['utility_property1'],
                'dependencies': [],
                'interfaces': [],
                'config': {}
            }
            
            # Add adapter component template
            self.component_templates['adapter'] = {
                'type': 'adapter',
                'methods': ['adapt', 'convert'],
                'properties': ['source', 'target'],
                'dependencies': ['source_component', 'target_component'],
                'interfaces': ['AdapterInterface'],
                'config': {}
            }
            
            self.logger.info(f"Initialized {len(self.component_templates)} component templates")
            
        except Exception as e:
            self.logger.error(f"Error initializing component templates: {e}")
    
    def _load_component_registry(self) -> None:
        """Load component registry from cache or create default."""
        try:
            # Try to load from cache
            cache_key = "component_registry"
            cached_registry = self.cache.get(cache_key)
            
            if cached_registry:
                self.component_registry = cached_registry
            else:
                # Create default registry
                self._create_default_registry()
                self._save_component_registry()
            
            self.logger.info("Component registry loaded")
            
        except Exception as e:
            self.logger.error(f"Error loading component registry: {e}")
            self._create_default_registry()
    
    def _save_component_registry(self) -> None:
        """Save component registry to cache."""
        try:
            cache_key = "component_registry"
            self.cache.set(cache_key, self.component_registry, ttl=3600)
        except Exception as e:
            self.logger.error(f"Error saving component registry: {e}")
    
    def _create_default_registry(self) -> None:
        """Create default component registry."""
        try:
            # Add some default components
            self.component_registry = {
                'training_core': {
                    'type': 'core',
                    'methods': ['train', 'evaluate', 'predict'],
                    'properties': ['model', 'config'],
                    'dependencies': ['memory', 'governor'],
                    'interfaces': ['TrainingInterface'],
                    'config': {'version': '1.0.0'}
                },
                'memory_manager': {
                    'type': 'core',
                    'methods': ['store', 'retrieve', 'delete'],
                    'properties': ['capacity', 'usage'],
                    'dependencies': [],
                    'interfaces': ['MemoryInterface'],
                    'config': {'max_capacity': 1000}
                },
                'governor': {
                    'type': 'core',
                    'methods': ['allocate', 'deallocate', 'monitor'],
                    'properties': ['resources', 'limits'],
                    'dependencies': ['memory_manager'],
                    'interfaces': ['GovernorInterface'],
                    'config': {'max_resources': 100}
                }
            }
            
            # Update dependencies
            self.component_dependencies = {
                'training_core': ['memory_manager', 'governor'],
                'governor': ['memory_manager']
            }
            
        except Exception as e:
            self.logger.error(f"Error creating default registry: {e}")
    
    def _evaluate_component_fitness(self, component: Dict[str, Any]) -> float:
        """Evaluate component fitness."""
        try:
            # Calculate fitness using all fitness functions
            fitness_scores = []
            for fitness_func in self.fitness_functions:
                try:
                    score = fitness_func(component)
                    fitness_scores.append(score)
                except Exception as e:
                    self.logger.warning(f"Error in fitness function: {e}")
                    continue
            
            # Calculate weighted average
            if fitness_scores:
                fitness = np.mean(fitness_scores)
            else:
                fitness = 0.0
            
            return max(0.0, min(1.0, fitness))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Error evaluating component fitness: {e}")
            return 0.0
    
    def _evaluate_component_fitness_detailed(self, component: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Evaluate component fitness with detailed metrics."""
        try:
            metrics = {}
            fitness_scores = []
            
            for fitness_func in self.fitness_functions:
                try:
                    score = fitness_func(component)
                    fitness_scores.append(score)
                    metrics[fitness_func.__name__] = score
                except Exception as e:
                    self.logger.warning(f"Error in fitness function: {e}")
                    continue
            
            # Calculate overall fitness
            if fitness_scores:
                overall_fitness = np.mean(fitness_scores)
            else:
                overall_fitness = 0.0
            
            return max(0.0, min(1.0, overall_fitness)), metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating detailed component fitness: {e}")
            return 0.0, {}
    
    def _calculate_changes(self, old_component: Dict[str, Any], 
                          new_component: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate changes between old and new component."""
        try:
            changes = {}
            
            # Compare methods
            old_methods = set(old_component.get('methods', []))
            new_methods = set(new_component.get('methods', []))
            
            added_methods = new_methods - old_methods
            removed_methods = old_methods - new_methods
            
            if added_methods:
                changes['added_methods'] = list(added_methods)
            if removed_methods:
                changes['removed_methods'] = list(removed_methods)
            
            # Compare properties
            old_properties = set(old_component.get('properties', []))
            new_properties = set(new_component.get('properties', []))
            
            added_properties = new_properties - old_properties
            removed_properties = old_properties - new_properties
            
            if added_properties:
                changes['added_properties'] = list(added_properties)
            if removed_properties:
                changes['removed_properties'] = list(removed_properties)
            
            # Compare dependencies
            old_dependencies = set(old_component.get('dependencies', []))
            new_dependencies = set(new_component.get('dependencies', []))
            
            added_dependencies = new_dependencies - old_dependencies
            removed_dependencies = old_dependencies - new_dependencies
            
            if added_dependencies:
                changes['added_dependencies'] = list(added_dependencies)
            if removed_dependencies:
                changes['removed_dependencies'] = list(removed_dependencies)
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error calculating changes: {e}")
            return {}
    
    def _calculate_fitness_trend(self, fitness_scores: List[float]) -> str:
        """Calculate fitness trend direction."""
        try:
            if len(fitness_scores) < 2:
                return 'insufficient_data'
            
            # Calculate trend using linear regression
            x = np.arange(len(fitness_scores))
            y = np.array(fitness_scores)
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            self.logger.error(f"Error calculating fitness trend: {e}")
            return 'unknown'
    
    # Mutation operators
    def _mutate_methods(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate component methods."""
        try:
            methods = component.get('methods', [])
            
            if methods and np.random.random() < 0.5:
                # Remove a method
                method_to_remove = np.random.choice(methods)
                methods.remove(method_to_remove)
            else:
                # Add a method
                new_method = f'method_{len(methods)}'
                methods.append(new_method)
            
            component['methods'] = methods
            return component
            
        except Exception as e:
            self.logger.error(f"Error mutating methods: {e}")
            return component
    
    def _mutate_properties(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate component properties."""
        try:
            properties = component.get('properties', [])
            
            if properties and np.random.random() < 0.5:
                # Remove a property
                property_to_remove = np.random.choice(properties)
                properties.remove(property_to_remove)
            else:
                # Add a property
                new_property = f'property_{len(properties)}'
                properties.append(new_property)
            
            component['properties'] = properties
            return component
            
        except Exception as e:
            self.logger.error(f"Error mutating properties: {e}")
            return component
    
    def _mutate_dependencies(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate component dependencies."""
        try:
            dependencies = component.get('dependencies', [])
            
            if dependencies and np.random.random() < 0.5:
                # Remove a dependency
                dependency_to_remove = np.random.choice(dependencies)
                dependencies.remove(dependency_to_remove)
            else:
                # Add a dependency
                available_components = [c for c in self.component_registry.keys() 
                                      if c != component.get('name', '')]
                if available_components:
                    new_dependency = np.random.choice(available_components)
                    if new_dependency not in dependencies:
                        dependencies.append(new_dependency)
            
            component['dependencies'] = dependencies
            return component
            
        except Exception as e:
            self.logger.error(f"Error mutating dependencies: {e}")
            return component
    
    def _mutate_configuration(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate component configuration."""
        try:
            config = component.get('config', {})
            
            # Add or modify a configuration parameter
            config_key = f'param_{len(config)}'
            config_value = np.random.randint(1, 100)
            config[config_key] = config_value
            
            component['config'] = config
            return component
            
        except Exception as e:
            self.logger.error(f"Error mutating configuration: {e}")
            return component
    
    def _mutate_interfaces(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate component interfaces."""
        try:
            interfaces = component.get('interfaces', [])
            
            if interfaces and np.random.random() < 0.5:
                # Remove an interface
                interface_to_remove = np.random.choice(interfaces)
                interfaces.remove(interface_to_remove)
            else:
                # Add an interface
                new_interface = f'Interface_{len(interfaces)}'
                interfaces.append(new_interface)
            
            component['interfaces'] = interfaces
            return component
            
        except Exception as e:
            self.logger.error(f"Error mutating interfaces: {e}")
            return component
    
    # Fitness functions
    def _evaluate_complexity_fitness(self, component: Dict[str, Any]) -> float:
        """Evaluate complexity fitness."""
        try:
            methods = component.get('methods', [])
            properties = component.get('properties', [])
            dependencies = component.get('dependencies', [])
            
            # Calculate complexity
            complexity = len(methods) + len(properties) + len(dependencies)
            
            # Lower complexity is better
            complexity_fitness = 1.0 / (1.0 + complexity * 0.1)
            
            return max(0.0, min(1.0, complexity_fitness))
            
        except Exception as e:
            self.logger.error(f"Error evaluating complexity fitness: {e}")
            return 0.0
    
    def _evaluate_cohesion_fitness(self, component: Dict[str, Any]) -> float:
        """Evaluate cohesion fitness."""
        try:
            methods = component.get('methods', [])
            properties = component.get('properties', [])
            
            # Cohesion is better with more methods and properties
            cohesion_score = (len(methods) + len(properties)) / 20.0  # Normalize
            
            return max(0.0, min(1.0, cohesion_score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating cohesion fitness: {e}")
            return 0.0
    
    def _evaluate_coupling_fitness(self, component: Dict[str, Any]) -> float:
        """Evaluate coupling fitness."""
        try:
            dependencies = component.get('dependencies', [])
            
            # Lower coupling is better
            coupling_score = 1.0 / (1.0 + len(dependencies) * 0.2)
            
            return max(0.0, min(1.0, coupling_score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating coupling fitness: {e}")
            return 0.0
    
    def _evaluate_reusability_fitness(self, component: Dict[str, Any]) -> float:
        """Evaluate reusability fitness."""
        try:
            methods = component.get('methods', [])
            interfaces = component.get('interfaces', [])
            
            # Reusability is better with more methods and interfaces
            reusability_score = (len(methods) + len(interfaces)) / 15.0  # Normalize
            
            return max(0.0, min(1.0, reusability_score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating reusability fitness: {e}")
            return 0.0
    
    def _evaluate_testability_fitness(self, component: Dict[str, Any]) -> float:
        """Evaluate testability fitness."""
        try:
            methods = component.get('methods', [])
            dependencies = component.get('dependencies', [])
            
            # Testability is better with more methods and fewer dependencies
            testability_score = len(methods) / (1.0 + len(dependencies) * 0.5)
            
            return max(0.0, min(1.0, testability_score))
            
        except Exception as e:
            self.logger.error(f"Error evaluating testability fitness: {e}")
            return 0.0

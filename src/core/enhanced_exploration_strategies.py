"""
Enhanced Exploration Strategies with Intelligent Search Algorithms

This module implements advanced exploration strategies that combine multiple
intelligent search algorithms for optimal exploration in complex environments.
"""

import logging
import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import torch
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExplorationType(Enum):
    """Types of exploration strategies."""
    RANDOM = "random"
    CURIOSITY_DRIVEN = "curiosity_driven"
    GOAL_ORIENTED = "goal_oriented"
    MEMORY_BASED = "memory_based"
    UCB = "ucb"  # Upper Confidence Bound
    THOMPSON_SAMPLING = "thompson_sampling"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    TREE_SEARCH = "tree_search"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class SearchAlgorithm(Enum):
    """Search algorithms for exploration."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    A_STAR = "a_star"
    DIJKSTRA = "dijkstra"
    BEAM_SEARCH = "beam_search"
    BEST_FIRST = "best_first"
    HILL_CLIMBING = "hill_climbing"
    GENETIC_SEARCH = "genetic_search"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    MONTE_CARLO = "monte_carlo"
    UCT = "uct"  # Upper Confidence Trees


@dataclass
class ExplorationState:
    """Represents the current state for exploration."""
    position: Tuple[float, float, float]
    energy_level: float
    learning_progress: float
    visited_positions: Set[Tuple[float, float, float]]
    success_history: List[bool]
    action_history: List[int]
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExplorationResult:
    """Result of an exploration action."""
    action: int
    position: Tuple[float, float, float]
    reward: float
    confidence: float
    exploration_value: float
    strategy_used: ExplorationType
    search_algorithm: SearchAlgorithm
    metadata: Dict[str, Any]


@dataclass
class SearchNode:
    """Node in a search tree."""
    state: ExplorationState
    parent: Optional['SearchNode'] = None
    children: List['SearchNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    ucb_value: float = 0.0
    depth: int = 0
    path: List[int] = field(default_factory=list)


class ExplorationStrategy(ABC):
    """Abstract base class for exploration strategies."""
    
    @abstractmethod
    def explore(self, state: ExplorationState, available_actions: List[int]) -> ExplorationResult:
        """Perform exploration and return the best action."""
        pass
    
    @abstractmethod
    def update(self, result: ExplorationResult, success: bool):
        """Update the strategy based on the result."""
        pass
    
    @abstractmethod
    def get_exploration_value(self, state: ExplorationState, action: int) -> float:
        """Get the exploration value for a state-action pair."""
        pass


class RandomExploration(ExplorationStrategy):
    """Random exploration strategy."""
    
    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        self.name = "Random"
    
    def explore(self, state: ExplorationState, available_actions: List[int]) -> ExplorationResult:
        """Randomly select an action."""
        action = random.choice(available_actions)
        return ExplorationResult(
            action=action,
            position=state.position,
            reward=0.0,
            confidence=0.5,
            exploration_value=1.0,
            strategy_used=ExplorationType.RANDOM,
            search_algorithm=SearchAlgorithm.MONTE_CARLO,
            metadata={'random': True}
        )
    
    def update(self, result: ExplorationResult, success: bool):
        """No update needed for random strategy."""
        pass
    
    def get_exploration_value(self, state: ExplorationState, action: int) -> float:
        """Return constant exploration value."""
        return 1.0


class CuriosityDrivenExploration(ExplorationStrategy):
    """Curiosity-driven exploration based on prediction error."""
    
    def __init__(self, curiosity_weight: float = 0.3, prediction_error_threshold: float = 0.1):
        self.curiosity_weight = curiosity_weight
        self.prediction_error_threshold = prediction_error_threshold
        self.prediction_errors = defaultdict(list)
        self.name = "Curiosity"
    
    def explore(self, state: ExplorationState, available_actions: List[int]) -> ExplorationResult:
        """Select action with highest curiosity value."""
        curiosity_values = []
        for action in available_actions:
            curiosity_value = self._calculate_curiosity_value(state, action)
            curiosity_values.append((action, curiosity_value))
        
        # Select action with highest curiosity
        best_action, best_curiosity = max(curiosity_values, key=lambda x: x[1])
        
        return ExplorationResult(
            action=best_action,
            position=state.position,
            reward=0.0,
            confidence=min(best_curiosity, 1.0),
            exploration_value=best_curiosity,
            strategy_used=ExplorationType.CURIOSITY_DRIVEN,
            search_algorithm=SearchAlgorithm.BEST_FIRST,
            metadata={'curiosity_value': best_curiosity}
        )
    
    def update(self, result: ExplorationResult, success: bool):
        """Update prediction errors based on result."""
        action_key = (result.position, result.action)
        prediction_error = abs(result.reward - self._predict_reward(result.position, result.action))
        self.prediction_errors[action_key].append(prediction_error)
        
        # Keep only recent errors
        if len(self.prediction_errors[action_key]) > 100:
            self.prediction_errors[action_key] = self.prediction_errors[action_key][-50:]
    
    def get_exploration_value(self, state: ExplorationState, action: int) -> float:
        """Get curiosity-based exploration value."""
        return self._calculate_curiosity_value(state, action)
    
    def _calculate_curiosity_value(self, state: ExplorationState, action: int) -> float:
        """Calculate curiosity value for a state-action pair."""
        action_key = (state.position, action)
        
        if action_key not in self.prediction_errors:
            return 1.0  # High curiosity for unexplored actions
        
        recent_errors = self.prediction_errors[action_key][-10:]  # Last 10 errors
        avg_error = np.mean(recent_errors) if recent_errors else 0.0
        
        # Higher prediction error = higher curiosity
        curiosity = min(avg_error / self.prediction_error_threshold, 1.0)
        return curiosity * self.curiosity_weight
    
    def _predict_reward(self, position: Tuple[float, float, float], action: int) -> float:
        """Predict reward for a state-action pair (simplified)."""
        # This would be replaced with actual prediction model
        return random.uniform(0.0, 1.0)


class UCBExploration(ExplorationStrategy):
    """Upper Confidence Bound exploration strategy."""
    
    def __init__(self, exploration_constant: float = 1.414):  # sqrt(2)
        self.exploration_constant = exploration_constant
        self.action_counts = defaultdict(int)
        self.action_rewards = defaultdict(list)
        self.total_visits = 0
        self.name = "UCB"
    
    def explore(self, state: ExplorationState, available_actions: List[int]) -> ExplorationResult:
        """Select action using UCB formula."""
        ucb_values = []
        
        for action in available_actions:
            ucb_value = self._calculate_ucb_value(action)
            ucb_values.append((action, ucb_value))
        
        # Select action with highest UCB value
        best_action, best_ucb = max(ucb_values, key=lambda x: x[1])
        
        return ExplorationResult(
            action=best_action,
            position=state.position,
            reward=0.0,
            confidence=min(best_ucb, 1.0),
            exploration_value=best_ucb,
            strategy_used=ExplorationType.UCB,
            search_algorithm=SearchAlgorithm.BEST_FIRST,
            metadata={'ucb_value': best_ucb}
        )
    
    def update(self, result: ExplorationResult, success: bool):
        """Update action statistics."""
        self.action_counts[result.action] += 1
        self.action_rewards[result.action].append(1.0 if success else 0.0)
        self.total_visits += 1
        
        # Keep only recent rewards
        if len(self.action_rewards[result.action]) > 100:
            self.action_rewards[result.action] = self.action_rewards[result.action][-50:]
    
    def get_exploration_value(self, state: ExplorationState, action: int) -> float:
        """Get UCB-based exploration value."""
        return self._calculate_ucb_value(action)
    
    def _calculate_ucb_value(self, action: int) -> float:
        """Calculate UCB value for an action."""
        if self.action_counts[action] == 0:
            return float('inf')  # Unvisited actions get highest priority
        
        # Average reward
        rewards = self.action_rewards[action]
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        # Exploration bonus
        exploration_bonus = self.exploration_constant * math.sqrt(
            math.log(self.total_visits) / self.action_counts[action]
        )
        
        return avg_reward + exploration_bonus


class TreeSearchExploration(ExplorationStrategy):
    """Tree search-based exploration strategy."""
    
    def __init__(self, max_depth: int = 5, max_iterations: int = 100, 
                 exploration_constant: float = 1.414):
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.search_trees = defaultdict(lambda: SearchNode(None))
        self.name = "TreeSearch"
    
    def explore(self, state: ExplorationState, available_actions: List[int]) -> ExplorationResult:
        """Perform tree search to find best action."""
        # Create root node
        root = SearchNode(state)
        
        # Perform tree search
        for _ in range(self.max_iterations):
            # Selection
            node = self._select_node(root)
            
            # Expansion
            if node.depth < self.max_depth and not node.children:
                self._expand_node(node, available_actions)
            
            # Simulation
            if node.children:
                child = random.choice(node.children)
                reward = self._simulate(child)
                
                # Backpropagation
                self._backpropagate(child, reward)
            else:
                reward = self._simulate(node)
                self._backpropagate(node, reward)
        
        # Select best action
        best_action = self._select_best_action(root)
        
        return ExplorationResult(
            action=best_action,
            position=state.position,
            reward=0.0,
            confidence=0.8,
            exploration_value=1.0,
            strategy_used=ExplorationType.TREE_SEARCH,
            search_algorithm=SearchAlgorithm.UCT,
            metadata={'tree_depth': root.depth, 'iterations': self.max_iterations}
        )
    
    def update(self, result: ExplorationResult, success: bool):
        """Update tree based on result."""
        # This would update the tree with actual results
        pass
    
    def get_exploration_value(self, state: ExplorationState, action: int) -> float:
        """Get tree search-based exploration value."""
        # This would calculate based on tree search results
        return 0.5
    
    def _select_node(self, node: SearchNode) -> SearchNode:
        """Select node using UCB."""
        while node.children:
            if any(child.visits == 0 for child in node.children):
                # Select unvisited child
                return random.choice([child for child in node.children if child.visits == 0])
            
            # Select child with highest UCB value
            ucb_values = [self._calculate_ucb(child) for child in node.children]
            best_child_idx = np.argmax(ucb_values)
            node = node.children[best_child_idx]
        
        return node
    
    def _expand_node(self, node: SearchNode, available_actions: List[int]):
        """Expand a node by adding children."""
        for action in available_actions:
            # Create new state (simplified)
            new_state = ExplorationState(
                position=node.state.position,
                energy_level=node.state.energy_level,
                learning_progress=node.state.learning_progress,
                visited_positions=node.state.visited_positions.copy(),
                success_history=node.state.success_history.copy(),
                action_history=node.state.action_history + [action],
                context=node.state.context.copy()
            )
            
            child = SearchNode(
                state=new_state,
                parent=node,
                depth=node.depth + 1,
                path=node.path + [action]
            )
            node.children.append(child)
    
    def _simulate(self, node: SearchNode) -> float:
        """Simulate from a node to get reward estimate."""
        # Simplified simulation - would use actual environment model
        return random.uniform(0.0, 1.0)
    
    def _backpropagate(self, node: SearchNode, reward: float):
        """Backpropagate reward up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
    
    def _calculate_ucb(self, node: SearchNode) -> float:
        """Calculate UCB value for a node."""
        if node.visits == 0:
            return float('inf')
        
        avg_reward = node.total_reward / node.visits
        exploration_bonus = self.exploration_constant * math.sqrt(
            math.log(node.parent.visits) / node.visits
        ) if node.parent else 0.0
        
        return avg_reward + exploration_bonus
    
    def _select_best_action(self, root: SearchNode) -> int:
        """Select the best action from root."""
        if not root.children:
            return 0
        
        # Select child with highest average reward
        best_child = max(root.children, key=lambda child: child.total_reward / max(child.visits, 1))
        return best_child.path[0] if best_child.path else 0


class GeneticAlgorithmExploration(ExplorationStrategy):
    """Genetic algorithm-based exploration strategy."""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1, 
                 crossover_rate: float = 0.8, generations: int = 10):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.population = []
        self.fitness_history = []
        self.name = "Genetic"
    
    def explore(self, state: ExplorationState, available_actions: List[int]) -> ExplorationResult:
        """Use genetic algorithm to find best action sequence."""
        # Initialize population
        self._initialize_population(available_actions)
        
        # Evolve population
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual, state) 
                            for individual in self.population]
            
            # Selection
            parents = self._select_parents(fitness_scores)
            
            # Crossover
            offspring = self._crossover(parents)
            
            # Mutation
            offspring = self._mutate(offspring, available_actions)
            
            # Replace population
            self.population = offspring
            
            # Track best fitness
            best_fitness = max(fitness_scores)
            self.fitness_history.append(best_fitness)
        
        # Select best individual
        best_individual = max(self.population, key=lambda ind: self._evaluate_fitness(ind, state))
        best_action = best_individual[0] if best_individual else random.choice(available_actions)
        
        return ExplorationResult(
            action=best_action,
            position=state.position,
            reward=0.0,
            confidence=0.7,
            exploration_value=1.0,
            strategy_used=ExplorationType.GENETIC_ALGORITHM,
            search_algorithm=SearchAlgorithm.GENETIC_SEARCH,
            metadata={'generations': self.generations, 'population_size': self.population_size}
        )
    
    def update(self, result: ExplorationResult, success: bool):
        """Update genetic algorithm based on result."""
        # This would update the fitness function based on actual results
        pass
    
    def get_exploration_value(self, state: ExplorationState, action: int) -> float:
        """Get genetic algorithm-based exploration value."""
        return 0.6
    
    def _initialize_population(self, available_actions: List[int]):
        """Initialize population with random action sequences."""
        self.population = []
        for _ in range(self.population_size):
            # Create random action sequence
            sequence_length = random.randint(1, 5)
            individual = [random.choice(available_actions) for _ in range(sequence_length)]
            self.population.append(individual)
    
    def _evaluate_fitness(self, individual: List[int], state: ExplorationState) -> float:
        """Evaluate fitness of an individual."""
        # Simplified fitness function - would use actual environment model
        return random.uniform(0.0, 1.0)
    
    def _select_parents(self, fitness_scores: List[float]) -> List[List[int]]:
        """Select parents using tournament selection."""
        parents = []
        for _ in range(self.population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
        return parents
    
    def _crossover(self, parents: List[List[int]]) -> List[List[int]]:
        """Perform crossover between parents."""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i + 1]
                
                if random.random() < self.crossover_rate and parent1 and parent2:
                    # Single-point crossover
                    min_len = min(len(parent1), len(parent2))
                    if min_len > 1:
                        crossover_point = random.randint(1, min_len - 1)
                        child1 = parent1[:crossover_point] + parent2[crossover_point:]
                        child2 = parent2[:crossover_point] + parent1[crossover_point:]
                        offspring.extend([child1, child2])
                    else:
                        offspring.extend([parent1, parent2])
                else:
                    offspring.extend([parent1, parent2])
            else:
                offspring.append(parents[i])
        
        return offspring
    
    def _mutate(self, offspring: List[List[int]], available_actions: List[int]) -> List[List[int]]:
        """Mutate offspring."""
        for individual in offspring:
            if random.random() < self.mutation_rate:
                # Random mutation
                if individual:
                    mutation_point = random.randint(0, len(individual) - 1)
                    individual[mutation_point] = random.choice(available_actions)
        
        return offspring


class EnhancedExplorationSystem:
    """Enhanced exploration system that combines multiple strategies."""
    
    def __init__(self, strategies: List[ExplorationStrategy] = None, 
                 adaptive_weights: bool = True):
        self.strategies = strategies or self._create_default_strategies()
        self.adaptive_weights = adaptive_weights
        self.strategy_weights = {strategy.name: 1.0 for strategy in self.strategies}
        self.strategy_performance = {strategy.name: [] for strategy in self.strategies}
        self.exploration_history = deque(maxlen=1000)
        
        logger.info(f"Enhanced Exploration System initialized with {len(self.strategies)} strategies")
    
    def _create_default_strategies(self) -> List[ExplorationStrategy]:
        """Create default exploration strategies."""
        return [
            RandomExploration(exploration_rate=0.1),
            CuriosityDrivenExploration(curiosity_weight=0.3),
            UCBExploration(exploration_constant=1.414),
            TreeSearchExploration(max_depth=3, max_iterations=50),
            GeneticAlgorithmExploration(population_size=10, generations=5)
        ]
    
    def explore(self, state: ExplorationState, available_actions: List[int]) -> ExplorationResult:
        """Perform exploration using the best strategy."""
        if not available_actions:
            raise ValueError("No available actions for exploration")
        
        # Get results from all strategies
        strategy_results = []
        for strategy in self.strategies:
            try:
                result = strategy.explore(state, available_actions)
                strategy_results.append((strategy.name, result))
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")
                continue
        
        if not strategy_results:
            # Fallback to random
            result = ExplorationResult(
                action=random.choice(available_actions),
                position=state.position,
                reward=0.0,
                confidence=0.5,
                exploration_value=1.0,
                strategy_used=ExplorationType.RANDOM,
                search_algorithm=SearchAlgorithm.MONTE_CARLO,
                metadata={'fallback': True}
            )
        else:
            # Select best result based on weighted combination
            result = self._select_best_result(strategy_results)
        
        # Record exploration
        self.exploration_history.append(result)
        
        return result
    
    def update(self, result: ExplorationResult, success: bool):
        """Update all strategies based on the result."""
        # Update individual strategies
        for strategy in self.strategies:
            try:
                strategy.update(result, success)
            except Exception as e:
                logger.warning(f"Failed to update strategy {strategy.name}: {e}")
        
        # Update strategy performance
        strategy_name = result.strategy_used.value
        # Map strategy type to strategy name
        strategy_name_mapping = {
            'random': 'Random',
            'curiosity_driven': 'Curiosity',
            'ucb': 'UCB',
            'tree_search': 'TreeSearch',
            'genetic_algorithm': 'Genetic'
        }
        mapped_name = strategy_name_mapping.get(strategy_name, strategy_name)
        
        if mapped_name in self.strategy_performance:
            self.strategy_performance[mapped_name].append(1.0 if success else 0.0)
            
            # Keep only recent performance
            if len(self.strategy_performance[mapped_name]) > 100:
                self.strategy_performance[mapped_name] = self.strategy_performance[mapped_name][-50:]
        
        # Update adaptive weights
        if self.adaptive_weights:
            self._update_adaptive_weights()
    
    def _select_best_result(self, strategy_results: List[Tuple[str, ExplorationResult]]) -> ExplorationResult:
        """Select the best result from multiple strategies."""
        if len(strategy_results) == 1:
            return strategy_results[0][1]
        
        # Calculate weighted scores
        weighted_scores = []
        for strategy_name, result in strategy_results:
            weight = self.strategy_weights.get(strategy_name, 1.0)
            score = result.confidence * result.exploration_value * weight
            weighted_scores.append((score, result))
        
        # Select result with highest weighted score
        best_score, best_result = max(weighted_scores, key=lambda x: x[0])
        return best_result
    
    def _update_adaptive_weights(self):
        """Update strategy weights based on performance."""
        for strategy_name, performance in self.strategy_performance.items():
            if performance:
                avg_performance = np.mean(performance)
                # Update weight based on performance (exponential moving average)
                current_weight = self.strategy_weights.get(strategy_name, 1.0)
                new_weight = 0.9 * current_weight + 0.1 * avg_performance
                self.strategy_weights[strategy_name] = max(0.1, min(2.0, new_weight))
    
    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        stats = {
            'total_explorations': len(self.exploration_history),
            'strategy_weights': self.strategy_weights.copy(),
            'strategy_performance': {},
            'recent_explorations': list(self.exploration_history)[-10:]
        }
        
        # Calculate performance statistics
        for strategy_name, performance in self.strategy_performance.items():
            if performance:
                stats['strategy_performance'][strategy_name] = {
                    'avg_performance': np.mean(performance),
                    'total_attempts': len(performance),
                    'recent_performance': performance[-10:] if len(performance) >= 10 else performance
                }
        
        return stats
    
    def add_strategy(self, strategy: ExplorationStrategy):
        """Add a new exploration strategy."""
        self.strategies.append(strategy)
        self.strategy_weights[strategy.name] = 1.0
        self.strategy_performance[strategy.name] = []
        logger.info(f"Added exploration strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str):
        """Remove an exploration strategy."""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
        self.strategy_weights.pop(strategy_name, None)
        self.strategy_performance.pop(strategy_name, None)
        logger.info(f"Removed exploration strategy: {strategy_name}")


def create_enhanced_exploration_system(strategies: List[ExplorationStrategy] = None,
                                     adaptive_weights: bool = True) -> EnhancedExplorationSystem:
    """Create an enhanced exploration system."""
    return EnhancedExplorationSystem(strategies, adaptive_weights)

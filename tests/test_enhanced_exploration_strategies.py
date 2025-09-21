"""
Test suite for Enhanced Exploration Strategies

This module tests the comprehensive exploration strategies with
intelligent search algorithms and adaptive learning.
"""

import unittest
import time
from typing import Dict, List, Any

from src.core.enhanced_exploration_strategies import (
    ExplorationType, SearchAlgorithm, ExplorationState, ExplorationResult,
    RandomExploration, CuriosityDrivenExploration, UCBExploration,
    TreeSearchExploration, GeneticAlgorithmExploration, EnhancedExplorationSystem,
    create_enhanced_exploration_system
)


class TestExplorationStrategies(unittest.TestCase):
    """Test cases for individual exploration strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_state = ExplorationState(
            position=(10.0, 20.0, 5.0),
            energy_level=80.0,
            learning_progress=0.6,
            visited_positions={(10.0, 20.0, 5.0), (15.0, 25.0, 3.0)},
            success_history=[True, False, True, True],
            action_history=[1, 2, 3, 4],
            context={'test': 'value'}
        )
        self.available_actions = [1, 2, 3, 4, 5]
    
    def test_random_exploration(self):
        """Test random exploration strategy."""
        strategy = RandomExploration(exploration_rate=0.1)
        
        # Test exploration
        result = strategy.explore(self.test_state, self.available_actions)
        
        self.assertIsInstance(result, ExplorationResult)
        self.assertIn(result.action, self.available_actions)
        self.assertEqual(result.strategy_used, ExplorationType.RANDOM)
        self.assertEqual(result.search_algorithm, SearchAlgorithm.MONTE_CARLO)
        self.assertEqual(result.exploration_value, 1.0)
        self.assertTrue(result.metadata['random'])
        
        # Test update
        strategy.update(result, True)
        
        # Test exploration value
        exploration_value = strategy.get_exploration_value(self.test_state, 1)
        self.assertEqual(exploration_value, 1.0)
    
    def test_curiosity_driven_exploration(self):
        """Test curiosity-driven exploration strategy."""
        strategy = CuriosityDrivenExploration(curiosity_weight=0.3)
        
        # Test exploration
        result = strategy.explore(self.test_state, self.available_actions)
        
        self.assertIsInstance(result, ExplorationResult)
        self.assertIn(result.action, self.available_actions)
        self.assertEqual(result.strategy_used, ExplorationType.CURIOSITY_DRIVEN)
        self.assertEqual(result.search_algorithm, SearchAlgorithm.BEST_FIRST)
        self.assertIn('curiosity_value', result.metadata)
        
        # Test update
        strategy.update(result, True)
        
        # Test exploration value
        exploration_value = strategy.get_exploration_value(self.test_state, 1)
        self.assertGreaterEqual(exploration_value, 0.0)
        self.assertLessEqual(exploration_value, 1.0)
    
    def test_ucb_exploration(self):
        """Test UCB exploration strategy."""
        strategy = UCBExploration(exploration_constant=1.414)
        
        # Test exploration
        result = strategy.explore(self.test_state, self.available_actions)
        
        self.assertIsInstance(result, ExplorationResult)
        self.assertIn(result.action, self.available_actions)
        self.assertEqual(result.strategy_used, ExplorationType.UCB)
        self.assertEqual(result.search_algorithm, SearchAlgorithm.BEST_FIRST)
        self.assertIn('ucb_value', result.metadata)
        
        # Test update
        strategy.update(result, True)
        
        # Test exploration value
        exploration_value = strategy.get_exploration_value(self.test_state, 1)
        self.assertGreaterEqual(exploration_value, 0.0)
    
    def test_tree_search_exploration(self):
        """Test tree search exploration strategy."""
        strategy = TreeSearchExploration(max_depth=3, max_iterations=10)
        
        # Test exploration
        result = strategy.explore(self.test_state, self.available_actions)
        
        self.assertIsInstance(result, ExplorationResult)
        self.assertIn(result.action, self.available_actions)
        self.assertEqual(result.strategy_used, ExplorationType.TREE_SEARCH)
        self.assertEqual(result.search_algorithm, SearchAlgorithm.UCT)
        self.assertIn('tree_depth', result.metadata)
        self.assertIn('iterations', result.metadata)
        
        # Test update
        strategy.update(result, True)
        
        # Test exploration value
        exploration_value = strategy.get_exploration_value(self.test_state, 1)
        self.assertEqual(exploration_value, 0.5)
    
    def test_genetic_algorithm_exploration(self):
        """Test genetic algorithm exploration strategy."""
        strategy = GeneticAlgorithmExploration(population_size=5, generations=3)
        
        # Test exploration
        result = strategy.explore(self.test_state, self.available_actions)
        
        self.assertIsInstance(result, ExplorationResult)
        self.assertIn(result.action, self.available_actions)
        self.assertEqual(result.strategy_used, ExplorationType.GENETIC_ALGORITHM)
        self.assertEqual(result.search_algorithm, SearchAlgorithm.GENETIC_SEARCH)
        self.assertIn('generations', result.metadata)
        self.assertIn('population_size', result.metadata)
        
        # Test update
        strategy.update(result, True)
        
        # Test exploration value
        exploration_value = strategy.get_exploration_value(self.test_state, 1)
        self.assertEqual(exploration_value, 0.6)


class TestEnhancedExplorationSystem(unittest.TestCase):
    """Test cases for the enhanced exploration system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_state = ExplorationState(
            position=(10.0, 20.0, 5.0),
            energy_level=80.0,
            learning_progress=0.6,
            visited_positions={(10.0, 20.0, 5.0)},
            success_history=[True, False, True],
            action_history=[1, 2, 3],
            context={'test': 'value'}
        )
        self.available_actions = [1, 2, 3, 4, 5]
    
    def test_initialization(self):
        """Test system initialization."""
        system = create_enhanced_exploration_system()
        
        self.assertIsInstance(system, EnhancedExplorationSystem)
        self.assertGreater(len(system.strategies), 0)
        self.assertTrue(system.adaptive_weights)
        self.assertEqual(len(system.strategy_weights), len(system.strategies))
        self.assertEqual(len(system.strategy_performance), len(system.strategies))
    
    def test_custom_strategies(self):
        """Test system with custom strategies."""
        strategies = [
            RandomExploration(exploration_rate=0.2),
            UCBExploration(exploration_constant=2.0)
        ]
        
        system = create_enhanced_exploration_system(strategies=strategies, adaptive_weights=False)
        
        self.assertEqual(len(system.strategies), 2)
        self.assertFalse(system.adaptive_weights)
        self.assertEqual(system.strategy_weights['Random'], 1.0)
        self.assertEqual(system.strategy_weights['UCB'], 1.0)
    
    def test_exploration(self):
        """Test exploration functionality."""
        system = create_enhanced_exploration_system()
        
        # Test exploration
        result = system.explore(self.test_state, self.available_actions)
        
        self.assertIsInstance(result, ExplorationResult)
        self.assertIn(result.action, self.available_actions)
        self.assertGreater(len(system.exploration_history), 0)
        self.assertEqual(system.exploration_history[-1], result)
    
    def test_exploration_with_empty_actions(self):
        """Test exploration with no available actions."""
        system = create_enhanced_exploration_system()
        
        with self.assertRaises(ValueError):
            system.explore(self.test_state, [])
    
    def test_update(self):
        """Test system update functionality."""
        system = create_enhanced_exploration_system()
        
        # Perform exploration
        result = system.explore(self.test_state, self.available_actions)
        
        # Update with success
        system.update(result, True)
        
        # Check that performance was recorded
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
        
        self.assertIn(mapped_name, system.strategy_performance)
        self.assertEqual(len(system.strategy_performance[mapped_name]), 1)
        self.assertEqual(system.strategy_performance[mapped_name][0], 1.0)
        
        # Update with failure
        system.update(result, False)
        self.assertEqual(len(system.strategy_performance[mapped_name]), 2)
        self.assertEqual(system.strategy_performance[mapped_name][1], 0.0)
    
    def test_adaptive_weights(self):
        """Test adaptive weight updates."""
        system = create_enhanced_exploration_system(adaptive_weights=True)
        
        # Perform multiple explorations and updates
        for _ in range(10):
            result = system.explore(self.test_state, self.available_actions)
            system.update(result, True)  # Always success
        
        # Check that weights have been updated
        for strategy_name in system.strategy_weights:
            self.assertGreater(system.strategy_weights[strategy_name], 0.0)
    
    def test_exploration_statistics(self):
        """Test exploration statistics."""
        system = create_enhanced_exploration_system()
        
        # Perform some explorations
        for _ in range(5):
            result = system.explore(self.test_state, self.available_actions)
            system.update(result, True)
        
        # Get statistics
        stats = system.get_exploration_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_explorations'], 5)
        self.assertIn('strategy_weights', stats)
        self.assertIn('strategy_performance', stats)
        self.assertIn('recent_explorations', stats)
        self.assertEqual(len(stats['recent_explorations']), 5)
    
    def test_add_strategy(self):
        """Test adding a new strategy."""
        system = create_enhanced_exploration_system()
        initial_count = len(system.strategies)
        
        # Add new strategy
        new_strategy = RandomExploration(exploration_rate=0.3)
        system.add_strategy(new_strategy)
        
        self.assertEqual(len(system.strategies), initial_count + 1)
        self.assertIn('Random', system.strategy_weights)
        self.assertIn('Random', system.strategy_performance)
    
    def test_remove_strategy(self):
        """Test removing a strategy."""
        system = create_enhanced_exploration_system()
        initial_count = len(system.strategies)
        
        # Remove a strategy
        strategy_name = system.strategies[0].name
        system.remove_strategy(strategy_name)
        
        self.assertEqual(len(system.strategies), initial_count - 1)
        self.assertNotIn(strategy_name, system.strategy_weights)
        self.assertNotIn(strategy_name, system.strategy_performance)
    
    def test_strategy_failure_handling(self):
        """Test handling of strategy failures."""
        # Create a system with a failing strategy
        class FailingStrategy(RandomExploration):
            def explore(self, state, available_actions):
                raise Exception("Strategy failed")
        
        strategies = [FailingStrategy(), RandomExploration()]
        system = create_enhanced_exploration_system(strategies=strategies)
        
        # Exploration should still work with fallback
        result = system.explore(self.test_state, self.available_actions)
        
        self.assertIsInstance(result, ExplorationResult)
        self.assertIn(result.action, self.available_actions)
    
    def test_fallback_exploration(self):
        """Test fallback when all strategies fail."""
        # Create a system with all failing strategies
        class FailingStrategy(RandomExploration):
            def explore(self, state, available_actions):
                raise Exception("Strategy failed")
        
        strategies = [FailingStrategy(), FailingStrategy()]
        system = create_enhanced_exploration_system(strategies=strategies)
        
        # Exploration should use fallback
        result = system.explore(self.test_state, self.available_actions)
        
        self.assertIsInstance(result, ExplorationResult)
        self.assertIn(result.action, self.available_actions)
        self.assertTrue(result.metadata.get('fallback', False))


class TestExplorationState(unittest.TestCase):
    """Test cases for ExplorationState dataclass."""
    
    def test_exploration_state_creation(self):
        """Test creating an exploration state."""
        state = ExplorationState(
            position=(10.0, 20.0, 5.0),
            energy_level=80.0,
            learning_progress=0.6,
            visited_positions={(10.0, 20.0, 5.0)},
            success_history=[True, False, True],
            action_history=[1, 2, 3],
            context={'test': 'value'}
        )
        
        self.assertEqual(state.position, (10.0, 20.0, 5.0))
        self.assertEqual(state.energy_level, 80.0)
        self.assertEqual(state.learning_progress, 0.6)
        self.assertEqual(state.visited_positions, {(10.0, 20.0, 5.0)})
        self.assertEqual(state.success_history, [True, False, True])
        self.assertEqual(state.action_history, [1, 2, 3])
        self.assertEqual(state.context, {'test': 'value'})
        self.assertIsInstance(state.timestamp, float)


class TestExplorationResult(unittest.TestCase):
    """Test cases for ExplorationResult dataclass."""
    
    def test_exploration_result_creation(self):
        """Test creating an exploration result."""
        result = ExplorationResult(
            action=1,
            position=(10.0, 20.0, 5.0),
            reward=0.8,
            confidence=0.9,
            exploration_value=0.7,
            strategy_used=ExplorationType.RANDOM,
            search_algorithm=SearchAlgorithm.MONTE_CARLO,
            metadata={'test': 'value'}
        )
        
        self.assertEqual(result.action, 1)
        self.assertEqual(result.position, (10.0, 20.0, 5.0))
        self.assertEqual(result.reward, 0.8)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.exploration_value, 0.7)
        self.assertEqual(result.strategy_used, ExplorationType.RANDOM)
        self.assertEqual(result.search_algorithm, SearchAlgorithm.MONTE_CARLO)
        self.assertEqual(result.metadata, {'test': 'value'})


class TestEnums(unittest.TestCase):
    """Test cases for enums."""
    
    def test_exploration_type_enum(self):
        """Test ExplorationType enum values."""
        self.assertEqual(ExplorationType.RANDOM.value, "random")
        self.assertEqual(ExplorationType.CURIOSITY_DRIVEN.value, "curiosity_driven")
        self.assertEqual(ExplorationType.GOAL_ORIENTED.value, "goal_oriented")
        self.assertEqual(ExplorationType.MEMORY_BASED.value, "memory_based")
        self.assertEqual(ExplorationType.UCB.value, "ucb")
        self.assertEqual(ExplorationType.THOMPSON_SAMPLING.value, "thompson_sampling")
        self.assertEqual(ExplorationType.GENETIC_ALGORITHM.value, "genetic_algorithm")
        self.assertEqual(ExplorationType.SIMULATED_ANNEALING.value, "simulated_annealing")
        self.assertEqual(ExplorationType.PARTICLE_SWARM.value, "particle_swarm")
        self.assertEqual(ExplorationType.BAYESIAN_OPTIMIZATION.value, "bayesian_optimization")
        self.assertEqual(ExplorationType.MULTI_ARMED_BANDIT.value, "multi_armed_bandit")
        self.assertEqual(ExplorationType.TREE_SEARCH.value, "tree_search")
        self.assertEqual(ExplorationType.REINFORCEMENT_LEARNING.value, "reinforcement_learning")
    
    def test_search_algorithm_enum(self):
        """Test SearchAlgorithm enum values."""
        self.assertEqual(SearchAlgorithm.BREADTH_FIRST.value, "breadth_first")
        self.assertEqual(SearchAlgorithm.DEPTH_FIRST.value, "depth_first")
        self.assertEqual(SearchAlgorithm.A_STAR.value, "a_star")
        self.assertEqual(SearchAlgorithm.DIJKSTRA.value, "dijkstra")
        self.assertEqual(SearchAlgorithm.BEAM_SEARCH.value, "beam_search")
        self.assertEqual(SearchAlgorithm.BEST_FIRST.value, "best_first")
        self.assertEqual(SearchAlgorithm.HILL_CLIMBING.value, "hill_climbing")
        self.assertEqual(SearchAlgorithm.GENETIC_SEARCH.value, "genetic_search")
        self.assertEqual(SearchAlgorithm.SIMULATED_ANNEALING.value, "simulated_annealing")
        self.assertEqual(SearchAlgorithm.PARTICLE_SWARM.value, "particle_swarm")
        self.assertEqual(SearchAlgorithm.MONTE_CARLO.value, "monte_carlo")
        self.assertEqual(SearchAlgorithm.UCT.value, "uct")


if __name__ == '__main__':
    unittest.main()

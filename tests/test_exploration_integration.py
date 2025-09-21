"""
Test suite for Exploration Integration

This module tests the integration between enhanced exploration strategies
and the existing ARC-AGI-3 system components.
"""

import unittest
from typing import Dict, List, Any, Tuple

from src.core.exploration_integration import ExplorationIntegration, create_exploration_integration


class TestExplorationIntegration(unittest.TestCase):
    """Test cases for Exploration Integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integration = create_exploration_integration(enable_database_storage=False)
        self.test_position = (10.0, 20.0, 5.0)
        self.test_available_actions = [1, 2, 3, 4, 5]
        self.test_visited_positions = [(10.0, 20.0, 5.0), (15.0, 25.0, 3.0)]
        self.test_success_history = [True, False, True, True]
        self.test_action_history = [1, 2, 3, 4]
        self.test_context = {'test': 'value'}
    
    def test_initialization(self):
        """Test integration initialization."""
        self.assertIsInstance(self.integration, ExplorationIntegration)
        self.assertIsNotNone(self.integration.enhanced_system)
        self.assertIsNotNone(self.integration.legacy_strategy)
        self.assertFalse(self.integration.enable_database_storage)
        self.assertEqual(self.integration.exploration_stats['total_explorations'], 0)
    
    def test_explore(self):
        """Test exploration functionality."""
        result = self.integration.explore(
            position=self.test_position,
            energy_level=80.0,
            learning_progress=0.6,
            visited_positions=self.test_visited_positions,
            success_history=self.test_success_history,
            action_history=self.test_action_history,
            available_actions=self.test_available_actions,
            context=self.test_context
        )
        
        self.assertIsNotNone(result)
        self.assertIn(result.action, self.test_available_actions)
        self.assertEqual(result.position, self.test_position)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertGreaterEqual(result.exploration_value, 0.0)
        # exploration_value can be infinity for UCB, so we check for reasonable range or infinity
        self.assertTrue(result.exploration_value <= 1.0 or result.exploration_value == float('inf'))
        
        # Check statistics were updated
        self.assertEqual(self.integration.exploration_stats['total_explorations'], 1)
        self.assertGreater(self.integration.exploration_stats['avg_confidence'], 0.0)
    
    def test_explore_with_empty_actions(self):
        """Test exploration with empty available actions."""
        with self.assertRaises(ValueError):
            self.integration.explore(
                position=self.test_position,
                energy_level=80.0,
                learning_progress=0.6,
                visited_positions=self.test_visited_positions,
                success_history=self.test_success_history,
                action_history=self.test_action_history,
                available_actions=[],
                context=self.test_context
            )
    
    def test_update_exploration(self):
        """Test exploration update functionality."""
        # Perform exploration
        result = self.integration.explore(
            position=self.test_position,
            energy_level=80.0,
            learning_progress=0.6,
            visited_positions=self.test_visited_positions,
            success_history=self.test_success_history,
            action_history=self.test_action_history,
            available_actions=self.test_available_actions,
            context=self.test_context
        )
        
        # Update with success
        self.integration.update_exploration(result, True)
        
        # Check statistics
        self.assertEqual(self.integration.exploration_stats['successful_explorations'], 1)
        self.assertEqual(self.integration.exploration_stats['failed_explorations'], 0)
        self.assertIn(result.strategy_used.value, self.integration.exploration_stats['strategy_usage'])
        
        # Update with failure
        self.integration.update_exploration(result, False)
        
        # Check statistics
        self.assertEqual(self.integration.exploration_stats['successful_explorations'], 1)
        self.assertEqual(self.integration.exploration_stats['failed_explorations'], 1)
    
    def test_get_exploration_bonus(self):
        """Test exploration bonus calculation."""
        bonus = self.integration.get_exploration_bonus(
            position=self.test_position,
            learning_progress=0.6
        )
        
        self.assertIsInstance(bonus, float)
        self.assertGreaterEqual(bonus, 0.0)
    
    def test_should_explore(self):
        """Test exploration decision."""
        should_explore = self.integration.should_explore(
            learning_progress=0.6,
            energy_level=80.0
        )
        
        self.assertIsInstance(should_explore, bool)
    
    def test_get_exploration_statistics(self):
        """Test exploration statistics retrieval."""
        # Perform some explorations
        for _ in range(3):
            result = self.integration.explore(
                position=self.test_position,
                energy_level=80.0,
                learning_progress=0.6,
                visited_positions=self.test_visited_positions,
                success_history=self.test_success_history,
                action_history=self.test_action_history,
                available_actions=self.test_available_actions,
                context=self.test_context
            )
            self.integration.update_exploration(result, True)
        
        # Get statistics
        stats = self.integration.get_exploration_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_explorations'], 3)
        self.assertEqual(stats['successful_explorations'], 3)
        self.assertEqual(stats['failed_explorations'], 0)
        self.assertIn('enhanced_system_stats', stats)
        self.assertIn('legacy_strategy_stats', stats)
        self.assertIn('strategy_usage', stats)
        self.assertIn('avg_confidence', stats)
        self.assertIn('avg_exploration_value', stats)
    
    def test_add_custom_strategy(self):
        """Test adding custom strategy."""
        from src.core.enhanced_exploration_strategies import RandomExploration
        
        custom_strategy = RandomExploration(exploration_rate=0.2)
        self.integration.add_custom_strategy(custom_strategy)
        
        # Check that strategy was added
        strategy_names = [s.name for s in self.integration.enhanced_system.strategies]
        self.assertIn('Random', strategy_names)
    
    def test_remove_strategy(self):
        """Test removing strategy."""
        # Remove a strategy
        self.integration.remove_strategy('Random')
        
        # Check that strategy was removed
        strategy_names = [s.name for s in self.integration.enhanced_system.strategies]
        self.assertNotIn('Random', strategy_names)
    
    def test_fallback_exploration(self):
        """Test fallback exploration when enhanced system fails."""
        # This test would require mocking the enhanced system to fail
        # For now, we'll test that the method exists and handles errors gracefully
        result = self.integration._fallback_exploration(
            position=self.test_position,
            available_actions=self.test_available_actions
        )
        
        self.assertIsNotNone(result)
        self.assertIn(result.action, self.test_available_actions)
        self.assertEqual(result.position, self.test_position)
    
    def test_statistics_update(self):
        """Test statistics update functionality."""
        # Create a mock result
        from src.core.enhanced_exploration_strategies import ExplorationResult, ExplorationType, SearchAlgorithm
        
        result = ExplorationResult(
            action=1,
            position=self.test_position,
            reward=0.8,
            confidence=0.9,
            exploration_value=0.7,
            strategy_used=ExplorationType.RANDOM,
            search_algorithm=SearchAlgorithm.MONTE_CARLO,
            metadata={'test': 'value'}
        )
        
        # Update statistics
        self.integration._update_statistics(result)
        
        # Check that statistics were updated
        self.assertEqual(self.integration.exploration_stats['total_explorations'], 1)
        self.assertEqual(self.integration.exploration_stats['avg_confidence'], 0.9)
        self.assertEqual(self.integration.exploration_stats['avg_exploration_value'], 0.7)
    
    def test_factory_function(self):
        """Test the factory function."""
        integration = create_exploration_integration(enable_database_storage=False)
        
        self.assertIsInstance(integration, ExplorationIntegration)
        self.assertFalse(integration.enable_database_storage)
        
        integration_with_db = create_exploration_integration(enable_database_storage=True)
        self.assertTrue(integration_with_db.enable_database_storage)


if __name__ == '__main__':
    unittest.main()

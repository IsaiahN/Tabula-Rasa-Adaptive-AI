#!/usr/bin/env python3
"""
Comprehensive tests for Action Sequence Optimization system.

Tests the integration of Tree Evaluation Engine, OpenCV target detection,
and ActionSequenceOptimizer for ARC-AGI action optimization.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple

# Import the components to test
try:
    from src.core.action_sequence_optimizer import ActionSequenceOptimizer, OptimizationConfig, ActionSequenceResult
    from src.core.tree_evaluation_simulation import TreeEvaluationSimulationEngine, TreeEvaluationConfig
    from src.arc_integration.opencv_feature_extractor import OpenCVFeatureExtractor, ActionableTarget
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    print(f"WARNING: Action sequence optimization not available for testing: {e}")

@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Action sequence optimization not available")
class TestActionSequenceOptimizer:
    """Test the ActionSequenceOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock tree engine
        self.mock_tree_engine = Mock()
        self.mock_tree_engine.evaluate_action_sequence_tree.return_value = {
            'optimal_sequence': [6, 1, 2],
            'sequence_value': 0.8,
            'wasted_moves_avoided': 2,
            'target_reached': True,
            'reasoning': 'Test sequence found'
        }
        
        # Mock OpenCV extractor
        self.mock_opencv_extractor = Mock()
        self.mock_opencv_extractor.identify_actionable_targets.return_value = [
            ActionableTarget(
                id=1,
                coordinates=(10, 15),
                object_type="button",
                action_type="click",
                priority=0.9,
                confidence=0.8,
                description="Test button",
                bounding_box=(8, 12, 4, 6),
                color=2,
                area=24
            )
        ]
        
        # Create optimizer
        self.config = OptimizationConfig(
            max_sequence_length=10,
            max_evaluation_time=2.0,
            confidence_threshold=0.7
        )
        self.optimizer = ActionSequenceOptimizer(
            tree_engine=self.mock_tree_engine,
            opencv_extractor=self.mock_opencv_extractor,
            config=self.config
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.tree_engine == self.mock_tree_engine
        assert self.optimizer.opencv_extractor == self.mock_opencv_extractor
        assert self.optimizer.config == self.config
        assert self.optimizer.optimization_stats['total_optimizations'] == 0
    
    def test_optimize_action_sequence_with_targets(self):
        """Test action sequence optimization with target detection."""
        # Test data
        current_state = {'position_x': 5, 'position_y': 5, 'score': 0}
        available_actions = [1, 2, 3, 4, 6]
        grid = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        game_id = "test_game"
        
        # Run optimization
        result = self.optimizer.optimize_action_sequence(
            current_state=current_state,
            available_actions=available_actions,
            grid=grid,
            game_id=game_id
        )
        
        # Verify result
        assert isinstance(result, ActionSequenceResult)
        assert len(result.optimal_sequence) > 0
        assert result.target_coordinates == (10, 15)
        assert result.sequence_value == 0.8
        assert result.wasted_moves_avoided == 2
        assert result.targets_reached == True
        assert result.confidence == 0.8
        
        # Verify tree engine was called
        self.mock_tree_engine.evaluate_action_sequence_tree.assert_called_once()
        
        # Verify OpenCV extractor was called
        self.mock_opencv_extractor.identify_actionable_targets.assert_called_once_with(grid, game_id)
    
    def test_optimize_action_sequence_without_targets(self):
        """Test action sequence optimization without target detection."""
        # Test data
        current_state = {'position_x': 5, 'position_y': 5, 'score': 0}
        available_actions = [1, 2, 3, 4]
        game_id = "test_game"
        
        # Run optimization without grid (no targets will be found)
        result = self.optimizer.optimize_action_sequence(
            current_state=current_state,
            available_actions=available_actions,
            grid=None,
            game_id=game_id
        )
        
        # Verify result
        assert isinstance(result, ActionSequenceResult)
        assert len(result.optimal_sequence) > 0  # Should have fallback sequence
        assert result.target_coordinates is None
        
        # Verify OpenCV extractor was not called
        self.mock_opencv_extractor.identify_actionable_targets.assert_not_called()
    
    def test_optimize_action_sequence_fallback(self):
        """Test fallback behavior when tree evaluation fails."""
        # Mock tree engine to raise exception
        self.mock_tree_engine.evaluate_action_sequence_tree.side_effect = Exception("Tree evaluation failed")
        
        # Test data
        current_state = {'position_x': 5, 'position_y': 5, 'score': 0}
        available_actions = [1, 2, 3, 4]
        grid = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        game_id = "test_game"
        
        # Run optimization
        result = self.optimizer.optimize_action_sequence(
            current_state=current_state,
            available_actions=available_actions,
            grid=grid,
            game_id=game_id
        )
        
        # Verify fallback result
        assert isinstance(result, ActionSequenceResult)
        assert len(result.optimal_sequence) > 0
        assert "Fallback sequence" in result.reasoning
    
    def test_optimize_for_action6(self):
        """Test ACTION6 optimization convenience method."""
        # Test data
        current_state = {'position_x': 5, 'position_y': 5, 'score': 0}
        available_actions = [1, 2, 3, 4, 6]
        grid = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        game_id = "test_game"
        
        # Run ACTION6 optimization
        best_action, target_coordinates = self.optimizer.optimize_for_action6(
            current_state=current_state,
            available_actions=available_actions,
            grid=grid,
            game_id=game_id
        )
        
        # Verify result
        assert best_action in available_actions
        assert target_coordinates == (10, 15)
    
    def test_sequence_validation(self):
        """Test sequence validation and optimization."""
        # Test data
        current_state = {'position_x': 5, 'position_y': 5, 'score': 0}
        available_actions = [1, 2, 3, 4]
        
        # Test with invalid actions
        invalid_sequence = [1, 2, 5, 3]  # Action 5 not in available_actions
        
        # Mock tree engine to return invalid sequence
        self.mock_tree_engine.evaluate_action_sequence_tree.return_value = {
            'optimal_sequence': invalid_sequence,
            'sequence_value': 0.6,
            'wasted_moves_avoided': 0,
            'target_reached': False,
            'reasoning': 'Invalid sequence test'
        }
        
        # Run optimization
        result = self.optimizer.optimize_action_sequence(
            current_state=current_state,
            available_actions=available_actions,
            grid=None,
            game_id="test_game"
        )
        
        # Verify invalid actions were removed
        assert all(action in available_actions for action in result.optimal_sequence)
    
    def test_redundant_pair_removal(self):
        """Test removal of redundant action pairs."""
        # Test data
        current_state = {'position_x': 5, 'position_y': 5, 'score': 0}
        available_actions = [1, 2, 3, 4]
        
        # Mock tree engine to return sequence with redundant pairs
        redundant_sequence = [1, 2, 3, 4, 1, 2]  # 1,2 and 3,4 are redundant pairs
        
        self.mock_tree_engine.evaluate_action_sequence_tree.return_value = {
            'optimal_sequence': redundant_sequence,
            'sequence_value': 0.6,
            'wasted_moves_avoided': 0,
            'target_reached': False,
            'reasoning': 'Redundant sequence test'
        }
        
        # Run optimization
        result = self.optimizer.optimize_action_sequence(
            current_state=current_state,
            available_actions=available_actions,
            grid=None,
            game_id="test_game"
        )
        
        # Verify redundant pairs were removed
        assert len(result.optimal_sequence) < len(redundant_sequence)
    
    def test_statistics_tracking(self):
        """Test optimization statistics tracking."""
        # Test data
        current_state = {'position_x': 5, 'position_y': 5, 'score': 0}
        available_actions = [1, 2, 3, 4]
        
        # Run multiple optimizations
        for i in range(3):
            result = self.optimizer.optimize_action_sequence(
                current_state=current_state,
                available_actions=available_actions,
                grid=None,
                game_id=f"test_game_{i}"
            )
        
        # Verify statistics
        stats = self.optimizer.get_optimization_stats()
        assert stats['total_optimizations'] == 3
        assert stats['successful_optimizations'] == 3
        assert stats['average_sequence_length'] > 0
        assert stats['average_confidence'] > 0
    
    def test_statistics_reset(self):
        """Test statistics reset functionality."""
        # Run one optimization
        result = self.optimizer.optimize_action_sequence(
            current_state={'position_x': 5, 'position_y': 5, 'score': 0},
            available_actions=[1, 2, 3, 4],
            grid=None,
            game_id="test_game"
        )
        
        # Verify statistics were updated
        stats = self.optimizer.get_optimization_stats()
        assert stats['total_optimizations'] == 1
        
        # Reset statistics
        self.optimizer.reset_statistics()
        
        # Verify statistics were reset
        stats = self.optimizer.get_optimization_stats()
        assert stats['total_optimizations'] == 0
        assert stats['successful_optimizations'] == 0

@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Action sequence optimization not available")
class TestTreeEvaluationIntegration:
    """Test integration with Tree Evaluation Engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tree_config = TreeEvaluationConfig(
            max_depth=5,
            branching_factor=3,
            state_representation_bits=64,
            memory_limit_mb=10.0,
            timeout_seconds=5.0,
            confidence_threshold=0.6,
            pruning_threshold=0.1
        )
        self.tree_engine = TreeEvaluationSimulationEngine(self.tree_config)
        
        self.optimizer = ActionSequenceOptimizer(
            tree_engine=self.tree_engine,
            opencv_extractor=None,
            config=OptimizationConfig()
        )
    
    def test_tree_evaluation_integration(self):
        """Test integration with real Tree Evaluation Engine."""
        # Test data
        current_state = {
            'position_x': 0,
            'position_y': 0,
            'score': 0,
            'active': True
        }
        available_actions = [1, 2, 3, 4, 6]
        target_goals = [
            {
                'type': 'coordinate',
                'coordinates': (5, 5),
                'priority': 1.0,
                'action_type': 'move_to'
            }
        ]
        
        # Run tree evaluation
        result = self.tree_engine.evaluate_action_sequence_tree(
            current_state=current_state,
            target_goals=target_goals,
            available_actions=available_actions,
            max_sequence_length=10
        )
        
        # Verify result
        assert 'optimal_sequence' in result
        assert 'sequence_value' in result
        assert 'wasted_moves_avoided' in result
        assert isinstance(result['optimal_sequence'], list)
        assert isinstance(result['sequence_value'], (int, float))
    
    def test_sequence_value_calculation(self):
        """Test sequence value calculation."""
        # Test data
        current_state = {'position_x': 0, 'position_y': 0, 'score': 0}
        target_goals = [{'type': 'coordinate', 'coordinates': (5, 5), 'priority': 1.0}]
        sequence = [1, 1, 1, 1, 1]  # Move up 5 times
        
        # Test sequence value calculation
        value = self.tree_engine._evaluate_sequence_value(
            state=current_state,
            sequence=sequence,
            target_goals=target_goals
        )
        
        assert isinstance(value, (int, float))
        assert value >= 0.0
    
    def test_wasted_moves_detection(self):
        """Test wasted moves detection."""
        # Test sequences with different levels of waste
        sequences = [
            [1, 2, 1, 2],  # Oscillation pattern
            [1, 1, 1, 1],  # Repetitive actions
            [1, 2, 3, 4],  # No waste
            [1, 2, 1, 2, 1, 2]  # Multiple oscillations
        ]
        
        for sequence in sequences:
            wasted_penalty = self.tree_engine._calculate_wasted_moves_penalty(sequence)
            assert isinstance(wasted_penalty, (int, float))
            assert wasted_penalty >= 0.0

@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Action sequence optimization not available")
class TestOpenCVIntegration:
    """Test integration with OpenCV target detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.opencv_extractor = OpenCVFeatureExtractor()
        self.optimizer = ActionSequenceOptimizer(
            tree_engine=None,
            opencv_extractor=self.opencv_extractor,
            config=OptimizationConfig()
        )
    
    def test_actionable_target_detection(self):
        """Test actionable target detection."""
        # Test grid with button-like pattern
        grid = [
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0]
        ]
        
        # Detect actionable targets
        targets = self.opencv_extractor.identify_actionable_targets(grid, "test_game")
        
        # Verify targets were detected
        assert isinstance(targets, list)
        if targets:  # If targets were found
            for target in targets:
                assert hasattr(target, 'coordinates')
                assert hasattr(target, 'priority')
                assert hasattr(target, 'action_type')
                assert target.priority > 0.0
    
    def test_action6_target_coordinates(self):
        """Test ACTION6 target coordinate extraction."""
        # Test grid with multiple potential targets
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 3, 0, 4, 0],
            [0, 0, 0, 0, 0]
        ]
        
        # Get ACTION6 target coordinates
        coordinates = self.opencv_extractor.get_actionable_targets_for_action6(grid, "test_game")
        
        # Verify coordinates
        assert isinstance(coordinates, list)
        for coord in coordinates:
            assert isinstance(coord, tuple)
            assert len(coord) == 2
            assert isinstance(coord[0], int)
            assert isinstance(coord[1], int)
    
    def test_target_classification(self):
        """Test target classification logic."""
        # Test different object types
        test_objects = [
            {
                'type': 'square',
                'area': 25,
                'color': 2,
                'centroid': (5, 5),
                'bounding_box': (3, 3, 5, 5),
                'id': 1
            },
            {
                'type': 'rectangle',
                'area': 100,
                'color': 3,
                'centroid': (10, 10),
                'bounding_box': (8, 8, 10, 10),
                'id': 2
            },
            {
                'type': 'blob',
                'area': 5,
                'color': 1,
                'centroid': (2, 2),
                'bounding_box': (1, 1, 3, 3),
                'id': 3
            }
        ]
        
        grid = [[0] * 20 for _ in range(20)]
        
        for obj in test_objects:
            target = self.opencv_extractor._classify_as_actionable_target(obj, grid)
            if target:  # If target was classified as actionable
                assert hasattr(target, 'priority')
                assert hasattr(target, 'action_type')
                assert target.priority > 0.0

@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Action sequence optimization not available")
class TestIntegration:
    """Test full integration of all components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create real instances
        self.tree_config = TreeEvaluationConfig(
            max_depth=8,
            branching_factor=4,
            state_representation_bits=64,
            memory_limit_mb=20.0,
            timeout_seconds=3.0,
            confidence_threshold=0.6,
            pruning_threshold=0.1
        )
        self.tree_engine = TreeEvaluationSimulationEngine(self.tree_config)
        
        self.opencv_extractor = OpenCVFeatureExtractor()
        
        self.opt_config = OptimizationConfig(
            max_sequence_length=15,
            max_evaluation_time=3.0,
            confidence_threshold=0.6,
            wasted_move_penalty=0.1,
            strategic_action_bonus=0.2,
            target_priority_weight=0.4
        )
        
        self.optimizer = ActionSequenceOptimizer(
            tree_engine=self.tree_engine,
            opencv_extractor=self.opencv_extractor,
            config=self.opt_config
        )
    
    def test_full_integration(self):
        """Test full integration of all components."""
        # Test data
        current_state = {
            'position_x': 0,
            'position_y': 0,
            'score': 0,
            'active': True
        }
        available_actions = [1, 2, 3, 4, 6]
        
        # Test grid with actionable targets
        grid = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        
        game_id = "integration_test"
        
        # Run full optimization
        result = self.optimizer.optimize_action_sequence(
            current_state=current_state,
            available_actions=available_actions,
            grid=grid,
            game_id=game_id
        )
        
        # Verify result
        assert isinstance(result, ActionSequenceResult)
        assert len(result.optimal_sequence) > 0
        assert all(action in available_actions for action in result.optimal_sequence)
        assert result.confidence >= 0.0
        assert result.evaluation_time > 0.0
        
        # Verify statistics were updated
        stats = self.optimizer.get_optimization_stats()
        assert stats['total_optimizations'] == 1
        assert stats['successful_optimizations'] == 1
    
    def test_performance_benchmark(self):
        """Test performance characteristics."""
        # Test data
        current_state = {'position_x': 0, 'position_y': 0, 'score': 0}
        available_actions = [1, 2, 3, 4, 6]
        grid = [[0] * 10 for _ in range(10)]
        
        # Run multiple optimizations to test performance
        start_time = time.time()
        
        for i in range(5):
            result = self.optimizer.optimize_action_sequence(
                current_state=current_state,
                available_actions=available_actions,
                grid=grid,
                game_id=f"perf_test_{i}"
            )
        
        total_time = time.time() - start_time
        
        # Verify performance is reasonable (should complete in reasonable time)
        assert total_time < 30.0  # Should complete in under 30 seconds
        
        # Verify all optimizations succeeded
        stats = self.optimizer.get_optimization_stats()
        assert stats['successful_optimizations'] == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

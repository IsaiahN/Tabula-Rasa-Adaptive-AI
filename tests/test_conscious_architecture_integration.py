#!/usr/bin/env python3
"""
Integration Test for Conscious Architecture in Main Training System

Tests the integration of conscious architecture enhancements with the main
Tabula Rasa training system, including the continuous learning loop.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Mock the required modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
from src.core.cohesive_integration_system import CohesiveIntegrationSystem


class TestConsciousArchitectureIntegration:
    """Test conscious architecture integration with main training system."""
    
    def test_continuous_learning_loop_initialization(self):
        """Test that continuous learning loop initializes with conscious architecture."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock paths
            arc_agents_path = temp_dir
            tabula_rasa_path = temp_dir
            
            # Initialize continuous learning loop
            loop = ContinuousLearningLoop(
                arc_agents_path=arc_agents_path,
                tabula_rasa_path=tabula_rasa_path,
                api_key="test_key"
            )
            
            # Ensure initialization
            loop._ensure_initialized()
            
            # Check that cohesive system is initialized
            assert hasattr(loop, 'cohesive_system')
            assert loop.cohesive_system is not None
            assert hasattr(loop.cohesive_system, 'dual_pathway_processor')
            assert hasattr(loop.cohesive_system, 'gut_feeling_engine')
    
    def test_conscious_action_selection_integration(self):
        """Test that conscious architecture influences action selection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock paths
            arc_agents_path = temp_dir
            tabula_rasa_path = temp_dir
            
            # Initialize continuous learning loop
            loop = ContinuousLearningLoop(
                arc_agents_path=arc_agents_path,
                tabula_rasa_path=tabula_rasa_path,
                api_key="test_key"
            )
            
            # Ensure initialization
            loop._ensure_initialized()
            
            # Mock context for action selection
            context = {
                'game_id': 'test_game',
                'available_actions': [1, 2, 3, 4, 5, 6, 7],
                'confidence': 0.6,
                'success_rate': 0.5,
                'frame': np.zeros((10, 10, 3), dtype=np.uint8),
                'frame_features': {'position': [5, 5]},
                'spatial_features': {'similarity': 0.7},
                'current_frame': np.zeros((10, 10, 3), dtype=np.uint8),
                'current_frame': np.zeros((10, 10, 3), dtype=np.uint8)
            }
            
            # Test action selection with conscious architecture
            selected_action = loop._select_intelligent_action_with_relevance(
                available_actions=[1, 2, 3, 4, 5, 6, 7],
                context=context
            )
            
            # Verify action is valid
            assert selected_action in [1, 2, 3, 4, 5, 6, 7]
    
    def test_conscious_learning_integration(self):
        """Test that conscious architecture learns from action outcomes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock paths
            arc_agents_path = temp_dir
            tabula_rasa_path = temp_dir
            
            # Initialize continuous learning loop
            loop = ContinuousLearningLoop(
                arc_agents_path=arc_agents_path,
                tabula_rasa_path=tabula_rasa_path,
                api_key="test_key"
            )
            
            # Ensure initialization
            loop._ensure_initialized()
            
            # Mock governor for learning
            loop.governor = Mock()
            loop.governor.learning_manager = Mock()
            loop.governor.learning_manager.learn_pattern = Mock(return_value="test_pattern_id")
            
            # Test learning from action outcome
            game_id = "test_game"
            action_number = 6
            x, y = 5, 5
            response_data = {
                'state': 'WIN',
                'score': 10,
                'grid_width': 10,
                'grid_height': 10,
                'available_actions': [1, 2, 3, 4, 5, 6, 7]
            }
            before_state = {'score': 5}
            after_state = {'score': 10}
            
            # This should not raise an exception
            loop._learn_from_action_outcome(
                game_id, action_number, x, y, response_data, before_state, after_state
            )
            
            # Verify learning was called
            assert loop.governor.learning_manager.learn_pattern.called
    
    def test_conscious_architecture_status_monitoring(self):
        """Test that conscious architecture status can be monitored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock paths
            arc_agents_path = temp_dir
            tabula_rasa_path = temp_dir
            
            # Initialize continuous learning loop
            loop = ContinuousLearningLoop(
                arc_agents_path=arc_agents_path,
                tabula_rasa_path=tabula_rasa_path,
                api_key="test_key"
            )
            
            # Ensure initialization
            loop._ensure_initialized()
            
            # Test status monitoring
            status = loop.cohesive_system.get_conscious_architecture_status()
            
            assert 'enabled' in status
            assert status['enabled'] == True
            assert 'dual_pathway' in status
            assert 'gut_feeling' in status
    
    def test_conscious_architecture_performance(self):
        """Test that conscious architecture doesn't significantly impact performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock paths
            arc_agents_path = temp_dir
            tabula_rasa_path = temp_dir
            
            # Initialize continuous learning loop
            loop = ContinuousLearningLoop(
                arc_agents_path=arc_agents_path,
                tabula_rasa_path=tabula_rasa_path,
                api_key="test_key"
            )
            
            # Ensure initialization
            loop._ensure_initialized()
            
            # Mock context
            context = {
                'game_id': 'test_game',
                'available_actions': [1, 2, 3, 4, 5, 6, 7],
                'confidence': 0.6,
                'success_rate': 0.5,
                'frame': np.zeros((10, 10, 3), dtype=np.uint8),
                'frame_features': {'position': [5, 5]},
                'spatial_features': {'similarity': 0.7},
                'current_frame': np.zeros((10, 10, 3), dtype=np.uint8)
            }
            
            # Measure processing time
            import time
            start_time = time.time()
            
            # Process multiple action selections
            for _ in range(10):
                selected_action = loop._select_intelligent_action_with_relevance(
                    available_actions=[1, 2, 3, 4, 5, 6, 7],
                    context=context
                )
                assert selected_action in [1, 2, 3, 4, 5, 6, 7]
            
            processing_time = time.time() - start_time
            
            # Should complete within reasonable time (less than 5 seconds for 10 iterations)
            assert processing_time < 5.0
    
    def test_conscious_architecture_error_handling(self):
        """Test that conscious architecture handles errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock paths
            arc_agents_path = temp_dir
            tabula_rasa_path = temp_dir
            
            # Initialize continuous learning loop
            loop = ContinuousLearningLoop(
                arc_agents_path=arc_agents_path,
                tabula_rasa_path=tabula_rasa_path,
                api_key="test_key"
            )
            
            # Ensure initialization
            loop._ensure_initialized()
            
            # Mock cohesive system to raise an error
            loop.cohesive_system.process_environment_update = Mock(side_effect=Exception("Test error"))
            
            # Mock context
            context = {
                'game_id': 'test_game',
                'available_actions': [1, 2, 3, 4, 5, 6, 7],
                'confidence': 0.6,
                'success_rate': 0.5,
                'frame': np.zeros((10, 10, 3), dtype=np.uint8),
                'frame_features': {'position': [5, 5]},
                'spatial_features': {'similarity': 0.7},
                'current_frame': np.zeros((10, 10, 3), dtype=np.uint8),
                'frame_analysis': {}
            }
            
            # Should not raise an exception, should fall back to other methods
            selected_action = loop._select_intelligent_action_with_relevance(
                available_actions=[1, 2, 3, 4, 5, 6, 7],
                context=context
            )
            
            # Should still return a valid action
            assert selected_action in [1, 2, 3, 4, 5, 6, 7]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

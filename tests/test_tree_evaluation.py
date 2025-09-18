#!/usr/bin/env python3
"""
Tests for Tree Evaluation Simulation Engine

Tests the space-efficient tree evaluation system and its integration
with Tabula Rasa's simulation capabilities.
"""

import pytest
import time
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.tree_evaluation_simulation import (
    TreeEvaluationSimulationEngine, 
    TreeEvaluationConfig, 
    TreeNode, 
    NodeType
)
from src.core.tree_evaluation_integration import (
    TreeEvaluationEnhancedSimulationAgent,
    create_tree_evaluation_enhanced_agent
)


class TestTreeEvaluationSimulation:
    """Test the core tree evaluation simulation engine."""
    
    def test_tree_evaluation_engine_initialization(self):
        """Test that the tree evaluation engine initializes correctly."""
        config = TreeEvaluationConfig(
            max_depth=5,
            branching_factor=3,
            memory_limit_mb=50.0
        )
        
        engine = TreeEvaluationSimulationEngine(config)
        
        assert engine.config.max_depth == 5
        assert engine.config.branching_factor == 3
        assert engine.config.memory_limit_mb == 50.0
        assert len(engine.active_nodes) == 0
        assert engine.evaluation_stats['total_evaluations'] == 0
    
    def test_node_creation_and_compression(self):
        """Test node creation and state compression."""
        config = TreeEvaluationConfig(state_representation_bits=1024)  # Much larger limit for testing
        engine = TreeEvaluationSimulationEngine(config)
        
        # Test state compression
        test_state = {
            'position': 10,
            'rotation': 45,
            'action_count': 5,
            'score': 100
        }
        
        compressed = engine._compress_state(test_state)
        decompressed = engine._decompress_state(compressed)
        
        assert isinstance(compressed, bytes)
        assert len(compressed) <= engine.config.state_representation_bits // 8
        
        # For large enough limits, we should be able to recover the original state
        if engine.config.state_representation_bits >= 1024:
            assert decompressed == test_state
        else:
            # For smaller limits, we might get a hash, which is expected
            assert isinstance(decompressed, dict)
    
    def test_tree_evaluation_basic(self):
        """Test basic tree evaluation functionality."""
        config = TreeEvaluationConfig(
            max_depth=3,
            branching_factor=2,
            memory_limit_mb=10.0
        )
        
        engine = TreeEvaluationSimulationEngine(config)
        
        # Test state
        current_state = {
            'position': 0,
            'rotation': 0,
            'action_count': 0,
            'score': 0
        }
        
        available_actions = [1, 2, 3, 4]
        
        # Run evaluation
        result = engine.evaluate_simulation_tree(
            current_state=current_state,
            available_actions=available_actions
        )
        
        # Verify result structure
        assert 'recommended_action' in result
        assert 'confidence' in result
        assert 'value' in result
        assert 'reasoning' in result
        assert 'evaluation_depth' in result
        assert 'memory_usage_mb' in result
        assert 'memory_savings_bytes' in result
        
        # Verify values are reasonable
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['evaluation_depth'] <= config.max_depth
        assert result['memory_usage_mb'] <= config.memory_limit_mb
        assert result['memory_savings_bytes'] >= 0
    
    def test_memory_efficiency(self):
        """Test that tree evaluation is memory efficient."""
        config = TreeEvaluationConfig(
            max_depth=8,
            branching_factor=4,
            memory_limit_mb=20.0
        )
        
        engine = TreeEvaluationSimulationEngine(config)
        
        current_state = {'position': 0, 'score': 0}
        available_actions = list(range(1, 9))  # 8 actions
        
        result = engine.evaluate_simulation_tree(
            current_state=current_state,
            available_actions=available_actions
        )
        
        # Verify memory usage is within limits
        assert result['memory_usage_mb'] <= config.memory_limit_mb
        
        # Verify memory savings are significant
        theoretical_max_nodes = config.branching_factor ** config.max_depth  # Correct calculation: branching_factor^max_depth
        actual_nodes = result.get('nodes_evaluated', 0)
        
        # Should use significantly fewer nodes than theoretical maximum
        assert actual_nodes < theoretical_max_nodes
        assert result['memory_savings_bytes'] > 0
    
    def test_deep_simulation(self):
        """Test that tree evaluation can handle deep simulations."""
        config = TreeEvaluationConfig(
            max_depth=15,
            branching_factor=3,
            memory_limit_mb=50.0
        )
        
        engine = TreeEvaluationSimulationEngine(config)
        
        current_state = {'position': 0, 'score': 0}
        available_actions = [1, 2, 3]
        
        result = engine.evaluate_simulation_tree(
            current_state=current_state,
            available_actions=available_actions
        )
        
        print(f"Debug: evaluation_depth={result['evaluation_depth']}, max_depth={config.max_depth}")
        print(f"Debug: nodes_evaluated={result['nodes_evaluated']}, path_length={result['path_length']}")
        print(f"Debug: best_value={result['value']}, confidence={result['confidence']}")
        
        # Should be able to simulate deep trees
        assert result['evaluation_depth'] >= 5  # At least some depth
        assert result['evaluation_depth'] <= config.max_depth
        
        # Memory should still be reasonable
        assert result['memory_usage_mb'] <= config.memory_limit_mb
    
    def test_caching_effectiveness(self):
        """Test that caching improves performance."""
        config = TreeEvaluationConfig(max_depth=5, branching_factor=2)
        engine = TreeEvaluationSimulationEngine(config)
        
        current_state = {'position': 0, 'score': 0}
        available_actions = [1, 2]
        
        # Single evaluation with caching
        result = engine.evaluate_simulation_tree(
            current_state=current_state,
            available_actions=available_actions
        )
        
        # Cache hit rate should be > 0 (caching works within evaluation)
        assert result['cache_hit_rate'] > 0
        
        # Should have evaluated multiple nodes
        assert result['nodes_evaluated'] > 1
    
    def test_cleanup(self):
        """Test that cleanup works correctly."""
        config = TreeEvaluationConfig()
        engine = TreeEvaluationSimulationEngine(config)
        
        # Run some evaluations
        current_state = {'position': 0, 'score': 0}
        available_actions = [1, 2, 3]
        
        engine.evaluate_simulation_tree(
            current_state=current_state,
            available_actions=available_actions
        )
        
        # Verify state before cleanup
        assert len(engine.active_nodes) > 0
        assert engine.evaluation_stats['total_evaluations'] > 0
        
        # Cleanup
        engine.cleanup()
        
        # Verify state after cleanup
        assert len(engine.active_nodes) == 0
        assert len(engine.node_hashes) == 0
        assert len(engine.evaluation_cache) == 0
        assert engine.evaluation_stats['total_evaluations'] == 0


class TestTreeEvaluationIntegration:
    """Test the integration of tree evaluation with existing systems."""
    
    def test_enhanced_agent_creation(self):
        """Test creation of tree evaluation enhanced agent."""
        # Mock predictive core
        class MockPredictiveCore:
            def __init__(self):
                self.name = "MockPredictiveCore"
        
        predictive_core = MockPredictiveCore()
        
        # Create enhanced agent
        agent = create_tree_evaluation_enhanced_agent(
            predictive_core=predictive_core,
            max_depth=8,
            branching_factor=4,
            memory_limit_mb=100.0
        )
        
        assert isinstance(agent, TreeEvaluationEnhancedSimulationAgent)
        assert agent.tree_engine is not None
        assert agent.tree_engine.config.max_depth == 8
        assert agent.tree_engine.config.branching_factor == 4
        assert agent.tree_engine.config.memory_limit_mb == 100.0
    
    def test_action_plan_generation(self):
        """Test action plan generation with tree evaluation."""
        # Mock predictive core
        class MockPredictiveCore:
            def __init__(self):
                self.name = "MockPredictiveCore"
        
        predictive_core = MockPredictiveCore()
        
        # Create enhanced agent
        agent = create_tree_evaluation_enhanced_agent(
            predictive_core=predictive_core,
            max_depth=5,
            branching_factor=3,
            memory_limit_mb=50.0
        )
        
        # Test action plan generation
        current_state = {
            'position': 0,
            'rotation': 0,
            'score': 0
        }
        available_actions = [1, 2, 3, 4]
        frame_analysis = {'objects': [], 'features': []}
        
        action, coordinates, reasoning = agent.generate_action_plan(
            current_state=current_state,
            available_actions=available_actions,
            frame_analysis=frame_analysis
        )
        
        # Verify results
        assert action is not None
        assert action in available_actions
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        
        # Verify tree evaluation was used
        assert "Tree depth:" in reasoning or "Tree evaluation" in reasoning
    
    def test_enhanced_statistics(self):
        """Test that enhanced statistics are collected."""
        # Mock predictive core
        class MockPredictiveCore:
            def __init__(self):
                self.name = "MockPredictiveCore"
        
        predictive_core = MockPredictiveCore()
        
        # Create enhanced agent
        agent = create_tree_evaluation_enhanced_agent(
            predictive_core=predictive_core,
            max_depth=5,
            branching_factor=3,
            memory_limit_mb=50.0
        )
        
        # Run some evaluations
        current_state = {'position': 0, 'score': 0}
        available_actions = [1, 2, 3]
        
        for _ in range(3):
            agent.generate_action_plan(
                current_state=current_state,
                available_actions=available_actions
            )
        
        # Get enhanced statistics
        stats = agent.get_enhanced_simulation_stats()
        
        # Verify tree evaluation statistics are present
        assert 'tree_evaluation' in stats
        assert 'tree_evaluations' in stats['tree_evaluation']
        assert 'memory_savings_total_mb' in stats['tree_evaluation']
        assert 'deepest_simulation' in stats['tree_evaluation']
        
        # Verify values are reasonable
        assert stats['tree_evaluation']['tree_evaluations'] >= 3
        assert stats['tree_evaluation']['memory_savings_total_mb'] >= 0
        assert stats['tree_evaluation']['deepest_simulation'] >= 0
    
    def test_fallback_behavior(self):
        """Test fallback behavior when tree evaluation fails."""
        # Mock predictive core
        class MockPredictiveCore:
            def __init__(self):
                self.name = "MockPredictiveCore"
        
        predictive_core = MockPredictiveCore()
        
        # Create enhanced agent with very restrictive config to force fallback
        agent = create_tree_evaluation_enhanced_agent(
            predictive_core=predictive_core,
            max_depth=1,  # Very shallow
            branching_factor=1,  # Very limited
            memory_limit_mb=1.0  # Very low memory
        )
        
        # Test with valid but simple state
        current_state = {'position': 0, 'score': 0}
        available_actions = [1, 2]
        
        action, coordinates, reasoning = agent.generate_action_plan(
            current_state=current_state,
            available_actions=available_actions
        )
        
        # Should handle gracefully
        assert action is not None
        assert action in available_actions
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0


class TestTreeEvaluationPerformance:
    """Test performance characteristics of tree evaluation."""
    
    def test_memory_scaling(self):
        """Test that memory usage scales well with depth."""
        depths = [5, 10, 15]
        memory_usage = []
        
        for depth in depths:
            config = TreeEvaluationConfig(
                max_depth=depth,
                branching_factor=3,
                memory_limit_mb=100.0
            )
            
            engine = TreeEvaluationSimulationEngine(config)
            
            current_state = {'position': 0, 'score': 0}
            available_actions = [1, 2, 3]
            
            result = engine.evaluate_simulation_tree(
                current_state=current_state,
                available_actions=available_actions
            )
            
            memory_usage.append(result['memory_usage_mb'])
            engine.cleanup()
        
        # Memory usage should not grow exponentially with depth
        # (due to pruning and implicit generation)
        # Allow some growth but not exponential
        assert memory_usage[1] < memory_usage[0] * 3  # Allow some growth
        assert memory_usage[2] < memory_usage[1] * 3  # Allow some growth
    
    def test_evaluation_speed(self):
        """Test that evaluation is reasonably fast."""
        config = TreeEvaluationConfig(
            max_depth=8,
            branching_factor=4,
            memory_limit_mb=50.0
        )
        
        engine = TreeEvaluationSimulationEngine(config)
        
        current_state = {'position': 0, 'score': 0}
        available_actions = [1, 2, 3, 4, 5]
        
        start_time = time.time()
        result = engine.evaluate_simulation_tree(
            current_state=current_state,
            available_actions=available_actions
        )
        end_time = time.time()
        
        evaluation_time = end_time - start_time
        
        # Should complete within reasonable time (5 seconds)
        assert evaluation_time < 5.0
        
        # Should have achieved some depth
        assert result['evaluation_depth'] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

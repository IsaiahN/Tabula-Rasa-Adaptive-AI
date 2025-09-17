#!/usr/bin/env python3
"""
Tests for Space-Time Aware Governor

Tests the space-time aware governor's dynamic parameter optimization
and resource-aware decision making capabilities.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.space_time_governor import (
    SpaceTimeAwareGovernor,
    TreeParameterOptimizer,
    SpaceTimeParameters,
    ResourceProfile,
    ResourceLevel,
    ProblemComplexity,
    create_space_time_aware_governor
)


class TestSpaceTimeParameters:
    """Test space-time parameters functionality."""
    
    def test_parameters_initialization(self):
        """Test parameter initialization."""
        params = SpaceTimeParameters(
            branching_factor=5,
            state_bits=128,
            max_depth=10,
            memory_limit_mb=100.0
        )
        
        assert params.branching_factor == 5
        assert params.state_bits == 128
        assert params.max_depth == 10
        assert params.memory_limit_mb == 100.0
        assert params.timeout_seconds == 30.0  # Default value
    
    def test_parameters_to_dict(self):
        """Test parameter serialization."""
        params = SpaceTimeParameters(
            branching_factor=3,
            state_bits=64,
            max_depth=8
        )
        
        params_dict = params.to_dict()
        
        assert params_dict['branching_factor'] == 3
        assert params_dict['state_bits'] == 64
        assert params_dict['max_depth'] == 8
        assert params_dict['memory_limit_mb'] == 100.0
        assert params_dict['timeout_seconds'] == 30.0


class TestResourceProfile:
    """Test resource profile functionality."""
    
    def test_resource_profile_initialization(self):
        """Test resource profile initialization."""
        profile = ResourceProfile(
            available_memory_mb=200.0,
            cpu_utilization=0.5,
            active_processes=5,
            problem_complexity=ProblemComplexity.MODERATE,
            time_constraint=30.0
        )
        
        assert profile.available_memory_mb == 200.0
        assert profile.cpu_utilization == 0.5
        assert profile.active_processes == 5
        assert profile.problem_complexity == ProblemComplexity.MODERATE
        assert profile.time_constraint == 30.0
    
    def test_resource_level_detection(self):
        """Test resource level detection."""
        # High resources
        high_profile = ResourceProfile(
            available_memory_mb=300.0,
            cpu_utilization=0.3,
            active_processes=3,
            problem_complexity=ProblemComplexity.SIMPLE,
            time_constraint=30.0
        )
        assert high_profile.get_resource_level() == ResourceLevel.HIGH
        
        # Medium resources
        medium_profile = ResourceProfile(
            available_memory_mb=150.0,
            cpu_utilization=0.4,
            active_processes=5,
            problem_complexity=ProblemComplexity.MODERATE,
            time_constraint=30.0
        )
        assert medium_profile.get_resource_level() == ResourceLevel.MEDIUM
        
        # Low resources
        low_profile = ResourceProfile(
            available_memory_mb=80.0,
            cpu_utilization=0.6,
            active_processes=8,
            problem_complexity=ProblemComplexity.COMPLEX,
            time_constraint=30.0
        )
        assert low_profile.get_resource_level() == ResourceLevel.LOW
        
        # Critical resources
        critical_profile = ResourceProfile(
            available_memory_mb=30.0,
            cpu_utilization=0.95,
            active_processes=15,
            problem_complexity=ProblemComplexity.EXTREME,
            time_constraint=30.0
        )
        assert critical_profile.get_resource_level() == ResourceLevel.CRITICAL


class TestTreeParameterOptimizer:
    """Test tree parameter optimizer functionality."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = TreeParameterOptimizer()
        
        assert len(optimizer.optimization_history) == 0
        assert len(optimizer.parameter_effectiveness) == 0
        assert len(optimizer.parameter_templates) > 0
    
    def test_find_optimal_parameters(self):
        """Test finding optimal parameters."""
        optimizer = TreeParameterOptimizer()
        
        # Test with high resources and simple problem
        resource_profile = ResourceProfile(
            available_memory_mb=300.0,
            cpu_utilization=0.3,
            active_processes=3,
            problem_complexity=ProblemComplexity.SIMPLE,
            time_constraint=30.0
        )
        
        performance_history = [
            {'confidence': 0.7, 'evaluation_depth': 5, 'memory_usage_mb': 50.0},
            {'confidence': 0.8, 'evaluation_depth': 6, 'memory_usage_mb': 60.0}
        ]
        
        optimal_params = optimizer.find_optimal_parameters(
            resource_profile, performance_history, "test_problem"
        )
        
        assert isinstance(optimal_params, SpaceTimeParameters)
        assert optimal_params.branching_factor > 0
        assert optimal_params.max_depth > 0
        assert optimal_params.memory_limit_mb <= resource_profile.available_memory_mb
        assert optimal_params.timeout_seconds <= resource_profile.time_constraint
    
    def test_parameter_optimization_with_constraints(self):
        """Test parameter optimization with resource constraints."""
        optimizer = TreeParameterOptimizer()
        
        # Test with low resources
        resource_profile = ResourceProfile(
            available_memory_mb=50.0,
            cpu_utilization=0.8,
            active_processes=10,
            problem_complexity=ProblemComplexity.COMPLEX,
            time_constraint=10.0
        )
        
        optimal_params = optimizer.find_optimal_parameters(
            resource_profile, [], "test_problem"
        )
        
        # Should respect resource constraints
        assert optimal_params.memory_limit_mb <= 50.0
        assert optimal_params.timeout_seconds <= 10.0
        assert optimal_params.branching_factor <= 3  # Limited by CPU
        assert optimal_params.max_depth <= 6  # Limited by processes
    
    def test_learned_optimizations(self):
        """Test learned optimizations from performance history."""
        optimizer = TreeParameterOptimizer()
        
        # Simulate learning from performance history
        performance_history = [
            {'confidence': 0.3, 'evaluation_depth': 3, 'memory_usage_mb': 20.0},
            {'confidence': 0.4, 'evaluation_depth': 4, 'memory_usage_mb': 25.0},
            {'confidence': 0.2, 'evaluation_depth': 2, 'memory_usage_mb': 15.0}
        ]
        
        resource_profile = ResourceProfile(
            available_memory_mb=200.0,
            cpu_utilization=0.4,
            active_processes=5,
            problem_complexity=ProblemComplexity.MODERATE,
            time_constraint=30.0
        )
        
        optimal_params = optimizer.find_optimal_parameters(
            resource_profile, performance_history, "test_problem"
        )
        
        # Should try to improve from low confidence
        assert optimal_params.max_depth >= 5  # Should try deeper simulation
    
    def test_effectiveness_tracking(self):
        """Test effectiveness tracking for problem types."""
        optimizer = TreeParameterOptimizer()
        
        # Update effectiveness for a problem type
        params = SpaceTimeParameters(branching_factor=5, max_depth=8, memory_limit_mb=100.0)
        performance_result = {'confidence': 0.8, 'evaluation_depth': 8}
        
        optimizer.update_effectiveness("test_problem", params, performance_result)
        
        assert "test_problem" in optimizer.parameter_effectiveness
        effectiveness = optimizer.parameter_effectiveness["test_problem"]
        assert effectiveness['sample_count'] == 1
        assert effectiveness['avg_performance'] == 0.8
        assert effectiveness['high_branching_effective'] == True
        assert effectiveness['high_depth_effective'] == True
    
    def test_optimization_stats(self):
        """Test optimization statistics."""
        optimizer = TreeParameterOptimizer()
        
        # Add some optimization history
        for i in range(5):
            resource_profile = ResourceProfile(
                available_memory_mb=100.0 + i * 10,
                cpu_utilization=0.3 + i * 0.1,
                active_processes=3 + i,
                problem_complexity=ProblemComplexity.SIMPLE,
                time_constraint=30.0
            )
            
            optimizer.find_optimal_parameters(resource_profile, [], f"problem_{i}")
        
        stats = optimizer.get_optimization_stats()
        
        assert stats['total_optimizations'] == 5
        assert stats['recent_optimizations'] == 5
        assert 'average_parameters' in stats
        assert 'resource_distribution' in stats


class TestSpaceTimeAwareGovernor:
    """Test space-time aware governor functionality."""
    
    def test_governor_initialization(self):
        """Test governor initialization."""
        governor = SpaceTimeAwareGovernor()
        
        assert governor.parameter_optimizer is not None
        assert governor.resource_monitor is not None
        assert governor.current_parameters is not None
        assert governor.optimization_stats['total_optimizations'] == 0
    
    def test_governor_with_base_governor(self):
        """Test governor with base governor."""
        # Mock base governor
        class MockBaseGovernor:
            def make_decision(self, available_actions, context, performance_history, current_energy):
                return {
                    'recommended_action': available_actions[0] if available_actions else 1,
                    'confidence': 0.7,
                    'reasoning': 'Mock decision'
                }
        
        base_governor = MockBaseGovernor()
        governor = SpaceTimeAwareGovernor(base_governor)
        
        assert governor.base_governor == base_governor
    
    def test_decision_making_with_space_time_awareness(self):
        """Test decision making with space-time awareness."""
        governor = SpaceTimeAwareGovernor()
        
        available_actions = [1, 2, 3, 4]
        context = {
            'game_id': 'test_game',
            'frame_analysis': {'object_count': 5},
            'available_actions': available_actions
        }
        performance_history = [
            {'confidence': 0.6, 'evaluation_depth': 4, 'memory_usage_mb': 30.0}
        ]
        current_energy = 80.0
        
        decision = governor.make_decision_with_space_time_awareness(
            available_actions, context, performance_history, current_energy
        )
        
        # Verify decision structure
        assert 'recommended_action' in decision
        assert 'confidence' in decision
        assert 'reasoning' in decision
        assert 'space_time_parameters' in decision
        assert 'resource_profile' in decision
        assert 'optimization_reasoning' in decision
        
        # Verify space-time parameters
        params = decision['space_time_parameters']
        assert 'branching_factor' in params
        assert 'state_bits' in params
        assert 'max_depth' in params
        assert 'memory_limit_mb' in params
        
        # Verify resource profile
        resource_profile = decision['resource_profile']
        assert 'memory_mb' in resource_profile
        assert 'cpu_utilization' in resource_profile
        assert 'resource_level' in resource_profile
        assert 'problem_complexity' in resource_profile
    
    def test_problem_complexity_assessment(self):
        """Test problem complexity assessment."""
        governor = SpaceTimeAwareGovernor()
        
        # Simple problem
        simple_context = {
            'frame_analysis': {'object_count': 2},
            'available_actions': [1, 2, 3]
        }
        simple_performance = [{'confidence': 0.8}]
        
        complexity = governor._assess_problem_complexity(simple_context, simple_performance)
        assert complexity == ProblemComplexity.SIMPLE
        
        # Complex problem
        complex_context = {
            'frame_analysis': {'object_count': 15},
            'available_actions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        complex_performance = [{'confidence': 0.2}, {'confidence': 0.3}]
        
        complexity = governor._assess_problem_complexity(complex_context, complex_performance)
        assert complexity in [ProblemComplexity.COMPLEX, ProblemComplexity.EXTREME]
    
    def test_fallback_decision(self):
        """Test fallback decision making."""
        governor = SpaceTimeAwareGovernor()
        
        available_actions = [1, 2, 3]
        context = {'game_id': 'test_game'}
        
        decision = governor._make_fallback_decision(available_actions, context)
        
        assert decision['recommended_action'] == 1
        assert decision['confidence'] == 0.5
        assert 'space_time_parameters' in decision
    
    def test_performance_feedback_update(self):
        """Test performance feedback update."""
        governor = SpaceTimeAwareGovernor()
        
        problem_type = "test_problem"
        performance_result = {
            'confidence': 0.8,
            'evaluation_depth': 6,
            'memory_usage_mb': 50.0
        }
        
        # This should not raise an exception
        governor.update_performance_feedback(problem_type, performance_result)
        
        # Verify effectiveness was updated
        assert problem_type in governor.parameter_optimizer.parameter_effectiveness
    
    def test_space_time_stats(self):
        """Test space-time statistics."""
        governor = SpaceTimeAwareGovernor()
        
        # Make some decisions to generate stats
        for i in range(3):
            context = {'game_id': f'test_game_{i}'}
            governor.make_decision_with_space_time_awareness([1, 2, 3], context, [], 80.0)
        
        stats = governor.get_space_time_stats()
        
        assert 'total_optimizations' in stats
        assert 'successful_optimizations' in stats
        assert 'parameter_adjustments' in stats
        assert 'parameter_optimizer' in stats
        assert 'current_parameters' in stats
        
        assert stats['total_optimizations'] >= 3
        assert stats['successful_optimizations'] >= 3


class TestIntegration:
    """Test integration functionality."""
    
    def test_create_space_time_aware_governor(self):
        """Test creating space-time aware governor."""
        governor = create_space_time_aware_governor()
        
        assert isinstance(governor, SpaceTimeAwareGovernor)
        assert governor.base_governor is None
    
    def test_create_with_base_governor(self):
        """Test creating with base governor."""
        class MockBaseGovernor:
            def make_decision(self, available_actions, context, performance_history, current_energy):
                return {'recommended_action': 1, 'confidence': 0.7, 'reasoning': 'Mock'}
        
        base_governor = MockBaseGovernor()
        governor = create_space_time_aware_governor(base_governor)
        
        assert isinstance(governor, SpaceTimeAwareGovernor)
        assert governor.base_governor == base_governor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

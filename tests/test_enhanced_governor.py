#!/usr/bin/env python3
"""
Tests for Enhanced Space-Time Aware Governor

Tests the consolidated governor functionality that combines space-time awareness
with all legacy governor capabilities.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.enhanced_space_time_governor import (
    EnhancedSpaceTimeGovernor,
    create_enhanced_space_time_governor,
    CognitiveCost,
    CognitiveBenefit,
    GovernorRecommendationType,
    GovernorRecommendation,
    ArchitectRequest
)


class TestEnhancedSpaceTimeGovernor:
    """Test the enhanced space-time aware governor functionality."""
    
    def test_governor_initialization(self):
        """Test governor initialization."""
        governor = create_enhanced_space_time_governor()
        
        assert governor is not None
        assert hasattr(governor, 'space_time_governor')
        assert hasattr(governor, 'action_limits')
        assert hasattr(governor, 'decision_history')
        assert hasattr(governor, 'performance_history')
    
    def test_decision_making(self):
        """Test decision making functionality."""
        governor = create_enhanced_space_time_governor()
        
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
        
        decision = governor.make_decision(
            available_actions=available_actions,
            context=context,
            performance_history=performance_history,
            current_energy=current_energy
        )
        
        # Verify decision structure
        assert 'recommended_action' in decision
        assert 'confidence' in decision
        assert 'reasoning' in decision
        assert 'space_time_parameters' in decision
        assert 'resource_profile' in decision
        assert 'cognitive_cost' in decision
        assert 'efficiency_ratio' in decision
        
        # Verify space-time parameters
        params = decision['space_time_parameters']
        assert 'branching_factor' in params
        assert 'state_bits' in params
        assert 'max_depth' in params
        assert 'memory_limit_mb' in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Test lifecycle analyzer integration with CoreGameplay.
"""
import sys
sys.dont_write_bytecode = True

import pytest
import asyncio
import os
import sys
import json
from unittest.mock import Mock, patch, AsyncMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gameplay.enhanced_gameplay import CoreGameplay, GameSessionManager
from src.analysis.game_lifecycle_analyzer import GameLifecycleAnalyzer, FailureMode, StrategyType

class TestLifecycleAnalyzerIntegration:
    """Test lifecycle analyzer integration."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock session manager."""
        session_manager = Mock(spec=GameSessionManager)
        session_manager.enhanced_gameplay = None
        return session_manager

    def test_lifecycle_analyzer_initialization(self, mock_session_manager):
        """Test that lifecycle analyzer is initialized properly."""
        gameplay = CoreGameplay(mock_session_manager)

        # Check that lifecycle analyzer was initialized
        assert hasattr(gameplay, '_lifecycle_analyzer')
        assert isinstance(gameplay._lifecycle_analyzer, GameLifecycleAnalyzer)

    def test_lifecycle_analyzer_setter(self, mock_session_manager):
        """Test setting the lifecycle analyzer."""
        gameplay = CoreGameplay(mock_session_manager)
        mock_analyzer = Mock(spec=GameLifecycleAnalyzer)

        # Set mock analyzer
        gameplay.set_lifecycle_analyzer(mock_analyzer)

        # Check it was set
        assert gameplay._lifecycle_analyzer is mock_analyzer

    def test_lifecycle_analyzer_setter_none(self, mock_session_manager):
        """Test setting lifecycle analyzer to None creates new one."""
        gameplay = CoreGameplay(mock_session_manager)
        old_analyzer = gameplay._lifecycle_analyzer

        # Set to None - should create new analyzer
        gameplay.set_lifecycle_analyzer(None)
        
        # Check new analyzer was created
        assert gameplay._lifecycle_analyzer is not None
        assert gameplay._lifecycle_analyzer is not old_analyzer
        assert isinstance(gameplay._lifecycle_analyzer, GameLifecycleAnalyzer)

    @pytest.mark.asyncio
    async def test_get_lifecycle_aware_action_recommendation(self, mock_session_manager):
        """Test getting lifecycle-aware action recommendation."""
        gameplay = CoreGameplay(mock_session_manager)

        # Test getting recommendation
        result = await gameplay.get_lifecycle_aware_action_recommendation(
            frame=[[0 for _ in range(64)] for _ in range(64)],
            available_actions=[1, 2, 3, 4, 5, 6],
            game_context={
                'game_type': 'test_game',
                'current_action_count': 10,
                'recent_score_change': 0.0
            },
            action_count=10,
            recent_actions=[1, 2, 3, 4, 5]
        )

        # Verify result has required fields
        assert isinstance(result, dict)
        assert 'action' in result
        assert 'reason' in result
        assert 'confidence' in result
        assert 'lifecycle_analysis' in result

        # Verify lifecycle analysis has required fields
        lifecycle = result['lifecycle_analysis']
        assert 'failure_risk' in lifecycle
        assert 'strategy_switch_triggered' in lifecycle
        assert 'actions_avoided' in lifecycle
        assert 'oscillation_detected' in lifecycle
        assert 'safe_actions' in lifecycle

if __name__ == '__main__':
    pytest.main([__file__])
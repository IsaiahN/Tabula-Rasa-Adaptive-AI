"""
Test Win Pattern Learning Integration

Verifies that when a game is won, the system:
1. Captures the action sequence that led to the win
2. Calls StrategyDiscoverySystem to learn from the win
3. Can apply learned patterns to future games
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.training.core.continuous_learning_loop import ContinuousLearningLoop
from src.core.strategy_discovery_system import WinningStrategy


class TestWinPatternLearning:

    @pytest.fixture
    def learning_loop(self):
        """Create a ContinuousLearningLoop instance for testing."""
        loop = ContinuousLearningLoop("test_api_key")
        return loop

    @pytest.fixture
    def mock_action_selector(self):
        """Create a mock action selector with strategy discovery system."""
        selector = Mock()
        strategy_system = AsyncMock()
        selector.strategy_discovery_system = strategy_system
        return selector, strategy_system

    @pytest.fixture
    def mock_api_manager(self):
        """Create a mock API manager for testing."""
        api_manager = AsyncMock()

        # Mock game state responses
        reset_response = Mock()
        reset_response.state = 'NOT_FINISHED'
        reset_response.guid = 'test_guid_123'
        api_manager.reset_game.return_value = reset_response

        # Mock game state progression
        game_states = [
            Mock(state='NOT_FINISHED', score=0.0, available_actions=[1, 2, 3], frame=[]),
            Mock(state='NOT_FINISHED', score=25.0, available_actions=[1, 2, 3], frame=[]),
            Mock(state='NOT_FINISHED', score=50.0, available_actions=[1, 2, 3], frame=[]),
            Mock(state='WIN', score=100.0, available_actions=[], frame=[])
        ]
        api_manager.get_game_state.side_effect = game_states

        # Mock action results
        action_results = [
            Mock(state='NOT_FINISHED', score=25.0, available_actions=[1, 2, 3], frame=[]),
            Mock(state='NOT_FINISHED', score=50.0, available_actions=[1, 2, 3], frame=[]),
            Mock(state='WIN', score=100.0, available_actions=[], frame=[])
        ]
        api_manager.take_action.side_effect = action_results

        api_manager.create_scorecard.return_value = 'test_scorecard'
        api_manager.close_scorecard.return_value = True
        api_manager.close.return_value = True
        api_manager.get_rate_limit_status.return_value = {'current_usage': 10, 'max_requests': 550}
        api_manager.rate_limiter.get_usage_warning.return_value = None
        api_manager.rate_limiter.should_pause.return_value = (False, 0)

        return api_manager

    @pytest.mark.asyncio
    async def test_win_triggers_strategy_learning(self, learning_loop, mock_action_selector, mock_api_manager):
        """Test that winning a game triggers strategy discovery."""
        selector, strategy_system = mock_action_selector
        learning_loop.action_selector = selector
        learning_loop.api_manager = mock_api_manager

        # Mock action selection to return predictable actions
        selector.select_action.side_effect = [
            {'id': 1, 'reason': 'test_action_1'},
            {'id': 2, 'reason': 'test_action_2'},
            {'id': 3, 'reason': 'test_action_3'}
        ]

        # Mock strategy discovery to return a winning strategy
        mock_strategy = WinningStrategy(
            strategy_id='test_strategy_123',
            game_type='test_game',
            game_id='test_game_456',
            action_sequence=[1, 2, 3],
            score_progression=[0.0, 25.0, 50.0, 100.0],
            total_score_increase=100.0,
            efficiency=33.33,
            discovery_timestamp=1234567890
        )
        strategy_system.discover_winning_strategy.return_value = mock_strategy

        # Mock database integration
        with patch('src.database.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_get_integration.return_value = mock_integration

            # Run the training session
            result = await learning_loop.start_training_with_direct_control('test_game_456')

        # Verify the game was won
        assert result['win'] is True
        assert result['score'] == 100.0
        assert result['actions_taken'] == 3

        # Verify strategy discovery was called with correct parameters
        strategy_system.discover_winning_strategy.assert_called_once()
        call_args = strategy_system.discover_winning_strategy.call_args

        assert call_args[1]['game_id'] == 'test_game_456'
        assert call_args[1]['action_sequence'] == [1, 2, 3]  # Actions that led to win
        assert call_args[1]['score_progression'] == [0.0, 25.0, 50.0, 100.0]  # Score progression

    @pytest.mark.asyncio
    async def test_strategy_replication_at_game_start(self, learning_loop, mock_action_selector, mock_api_manager):
        """Test that learned strategies are checked at game start."""
        selector, strategy_system = mock_action_selector
        learning_loop.action_selector = selector
        learning_loop.api_manager = mock_api_manager

        # Mock strategy replication checks
        strategy_system.should_attempt_strategy_replication.return_value = True

        mock_strategy = WinningStrategy(
            strategy_id='existing_strategy_789',
            game_type='test_game',
            game_id='previous_game',
            action_sequence=[2, 1, 3],
            score_progression=[0.0, 30.0, 60.0, 95.0],
            total_score_increase=95.0,
            efficiency=31.67,
            discovery_timestamp=1234567800
        )
        strategy_system.get_best_strategy_for_game.return_value = mock_strategy

        # Mock action selection
        selector.select_action.side_effect = [
            {'id': 1}, {'id': 2}, {'id': 3}
        ]

        # Mock database integration
        with patch('src.database.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_get_integration.return_value = mock_integration

            # Run the training session
            result = await learning_loop.start_training_with_direct_control('test_game_new')

        # Verify strategy replication was checked
        strategy_system.should_attempt_strategy_replication.assert_called_once_with('test_game_new')
        strategy_system.get_best_strategy_for_game.assert_called_once_with('test_game_new')

    @pytest.mark.asyncio
    async def test_no_strategy_learning_without_action_selector(self, learning_loop, mock_api_manager):
        """Test that system handles missing action selector gracefully."""
        learning_loop.action_selector = None  # No action selector
        learning_loop.api_manager = mock_api_manager

        # Mock database integration
        with patch('src.database.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_get_integration.return_value = mock_integration

            # Run the training session - should complete without errors
            result = await learning_loop.start_training_with_direct_control('test_game_basic')

        # Should still complete the game even without strategy learning
        assert result['win'] is True
        assert result['training_completed'] is True

    @pytest.mark.asyncio
    async def test_strategy_learning_handles_errors_gracefully(self, learning_loop, mock_action_selector, mock_api_manager):
        """Test that strategy learning errors don't break the training loop."""
        selector, strategy_system = mock_action_selector
        learning_loop.action_selector = selector
        learning_loop.api_manager = mock_api_manager

        # Mock action selection
        selector.select_action.side_effect = [
            {'id': 1}, {'id': 2}, {'id': 3}
        ]

        # Mock strategy discovery to raise an exception
        strategy_system.discover_winning_strategy.side_effect = Exception("Strategy discovery failed")

        # Mock database integration
        with patch('src.database.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_get_integration.return_value = mock_integration

            # Run the training session - should complete despite error
            result = await learning_loop.start_training_with_direct_control('test_game_error')

        # Game should still complete successfully
        assert result['win'] is True
        assert result['training_completed'] is True

        # Strategy discovery should have been attempted
        strategy_system.discover_winning_strategy.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_sequence_tracking(self, learning_loop, mock_action_selector, mock_api_manager):
        """Test that action sequences are correctly tracked during gameplay."""
        selector, strategy_system = mock_action_selector
        learning_loop.action_selector = selector
        learning_loop.api_manager = mock_api_manager

        # Mock action selection with different action formats
        selector.select_action.side_effect = [
            {'id': 5, 'reason': 'strategic_move'},  # Dict format
            7,  # Integer format
            {'id': 2, 'x': 10, 'y': 20}  # Dict with coordinates
        ]

        mock_strategy = WinningStrategy(
            strategy_id='sequence_test',
            game_type='test_game',
            game_id='test_sequence',
            action_sequence=[5, 7, 2],
            score_progression=[0.0, 25.0, 50.0, 100.0],
            total_score_increase=100.0,
            efficiency=33.33,
            discovery_timestamp=1234567890
        )
        strategy_system.discover_winning_strategy.return_value = mock_strategy

        # Mock database integration
        with patch('src.database.system_integration.get_system_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_get_integration.return_value = mock_integration

            # Run the training session
            result = await learning_loop.start_training_with_direct_control('test_sequence')

        # Verify action sequence was tracked correctly
        call_args = strategy_system.discover_winning_strategy.call_args
        assert call_args[1]['action_sequence'] == [5, 7, 2]
        assert len(call_args[1]['score_progression']) == 4  # Initial + 3 actions
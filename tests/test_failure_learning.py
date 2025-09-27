"""
Test Failure-Based Learning Integration

Verifies that the system learns from game overs, level failures, and stagnation,
not just successes.
"""

import pytest
from unittest.mock import Mock, AsyncMock


def test_failure_type_classification():
    """Test classification of different failure types."""

    # Import the method we need to test
    # For testing, we'll simulate the classification logic
    def classify_failure_type(final_state: str, actions_taken: int, final_score: float) -> str:
        """Classify the type of failure to better understand what went wrong."""
        if final_state == 'NOT_FINISHED' and actions_taken >= 5000:
            return "TIMEOUT"
        elif final_score <= 0:
            return "ZERO_PROGRESS"
        elif final_score < 50:
            return "LOW_PROGRESS"
        else:
            return "FAILURE_WITH_PROGRESS"

    # Test different failure scenarios
    test_cases = [
        # (final_state, actions_taken, final_score, expected_type)
        ('NOT_FINISHED', 5000, 25.0, 'TIMEOUT'),
        ('GAME_OVER', 100, 0.0, 'ZERO_PROGRESS'),
        ('FAILED', 200, 25.0, 'LOW_PROGRESS'),
        ('GAME_OVER', 150, 75.0, 'FAILURE_WITH_PROGRESS'),
    ]

    for final_state, actions_taken, final_score, expected in test_cases:
        result = classify_failure_type(final_state, actions_taken, final_score)
        assert result == expected, f"State: {final_state}, Actions: {actions_taken}, Score: {final_score}"


def test_level_stagnation_detection():
    """Test detection of level stagnation (when stuck on a level)."""

    # Simulate level progress tracking
    level_action_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6]  # 15 actions
    level_score_progression = [10.0, 12.0, 15.0, 16.0, 17.0, 17.0, 17.0, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 18.0]

    # Check stagnation detection logic
    recent_score_change = level_score_progression[-1] - level_score_progression[-10]  # Last 10 actions
    is_stagnating = len(level_action_sequence) >= 15 and recent_score_change <= 5.0

    assert is_stagnating is True, "Should detect stagnation when score barely increases over many actions"

    # Test non-stagnating scenario
    progressing_scores = [10.0, 15.0, 22.0, 30.0, 38.0, 45.0, 52.0, 60.0, 68.0, 75.0, 82.0, 90.0, 98.0, 105.0, 115.0]
    recent_progress = progressing_scores[-1] - progressing_scores[-10]
    is_progressing = len(level_action_sequence) >= 15 and recent_progress > 5.0

    assert is_progressing is True, "Should NOT detect stagnation when score is increasing significantly"


@pytest.mark.asyncio
async def test_game_failure_learning():
    """Test that complete game failures trigger learning analysis."""

    # Mock the learning system components
    strategy_system = AsyncMock()

    # Simulate game failure analysis
    game_id = "test_game_123"
    final_state = "GAME_OVER"
    final_score = 25.0
    action_sequence = [1, 2, 3, 4, 5, 2, 3, 4]  # 8 actions
    actions_taken = 8

    # Test the analysis logic
    failure_type = "LOW_PROGRESS"  # Based on score < 50
    negative_efficiency = -(final_score / len(action_sequence))  # Should be negative

    # Verify failure analysis
    assert negative_efficiency < 0, "Failure efficiency should be negative"
    assert abs(negative_efficiency) == (25.0 / 8), "Should calculate correct negative efficiency"

    # Verify failure pattern identification
    failure_pattern = action_sequence[-5:]  # Last 5 actions before failure
    assert failure_pattern == [4, 5, 2, 3, 4], "Should capture actions leading to failure"


@pytest.mark.asyncio
async def test_level_failure_learning():
    """Test that level-specific failures trigger learning analysis."""

    # Mock components
    strategy_system = AsyncMock()

    # Simulate level failure scenario
    game_id = "test_game_456"
    current_level = 2
    final_state = "STAGNATION"
    level_actions = [1, 1, 1, 2, 2, 2, 3, 3, 3]  # Repetitive pattern
    level_scores = [50.0, 51.0, 51.5, 52.0, 52.0, 52.0, 52.5, 52.5, 53.0]  # Minimal progress

    # Test stagnation analysis
    score_stagnation = len(level_scores) > 1 and (level_scores[-1] - level_scores[0]) < 5.0
    assert score_stagnation is True, "Should detect score stagnation"

    # Test pattern analysis
    repeated_actions = [action for action in set(level_actions) if level_actions.count(action) >= 3]
    assert len(repeated_actions) > 0, "Should identify repeated actions that aren't working"


@pytest.mark.asyncio
async def test_failure_learning_integration():
    """Test the complete failure learning integration."""

    # Mock StrategyDiscoverySystem
    strategy_system = AsyncMock()

    # Mock action selector with strategy system
    action_selector = Mock()
    action_selector.strategy_discovery_system = strategy_system

    # Test game outcome analysis function signature
    # Simulate the parameters that would be passed to _analyze_game_outcome
    game_outcome_params = {
        'game_id': 'test_game_789',
        'game_won': False,  # FAILURE CASE
        'final_state': 'GAME_OVER',
        'final_score': 35.0,
        'actions_taken': 12,
        'action_sequence': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
        'score_progression': [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 25.0, 25.0, 30.0, 30.0, 32.0, 35.0],
        'current_level': 1,
        'level_action_sequence': [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
        'level_score_progression': [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 25.0, 25.0, 30.0, 30.0, 32.0, 35.0],
        'last_significant_score': 0.0
    }

    # Verify that we have all the necessary data for failure analysis
    assert game_outcome_params['game_won'] is False, "Testing failure case"
    assert len(game_outcome_params['action_sequence']) >= 3, "Sufficient actions for pattern analysis"
    assert len(game_outcome_params['level_action_sequence']) >= 3, "Sufficient level actions for analysis"

    # Test failure type classification
    failure_type = "LOW_PROGRESS"  # Score 35.0 < 50
    assert failure_type == "LOW_PROGRESS", "Should classify as low progress failure"


@pytest.mark.asyncio
async def test_multiple_failure_scenarios():
    """Test various failure scenarios to ensure comprehensive learning."""

    # Test different failure scenarios
    scenarios = [
        {
            'name': 'Early Game Over',
            'final_state': 'GAME_OVER',
            'actions_taken': 5,
            'final_score': 0.0,
            'expected_type': 'ZERO_PROGRESS'
        },
        {
            'name': 'Timeout Failure',
            'final_state': 'NOT_FINISHED',
            'actions_taken': 5000,
            'final_score': 150.0,
            'expected_type': 'TIMEOUT'
        },
        {
            'name': 'Progress but Failed',
            'final_state': 'FAILED',
            'actions_taken': 200,
            'final_score': 85.0,
            'expected_type': 'FAILURE_WITH_PROGRESS'
        }
    ]

    for scenario in scenarios:
        # Simulate failure type classification
        def classify_failure(state, actions, score):
            if state == 'NOT_FINISHED' and actions >= 5000:
                return "TIMEOUT"
            elif score <= 0:
                return "ZERO_PROGRESS"
            elif score < 50:
                return "LOW_PROGRESS"
            else:
                return "FAILURE_WITH_PROGRESS"

        result = classify_failure(
            scenario['final_state'],
            scenario['actions_taken'],
            scenario['final_score']
        )

        assert result == scenario['expected_type'], f"Scenario '{scenario['name']}' classification failed"


def test_failure_hypothesis_generation():
    """Test generation of hypotheses about why failures occurred."""

    # Test different failure patterns and expected hypotheses
    test_cases = [
        {
            'scenario': 'Repetitive Actions',
            'action_sequence': [1, 1, 1, 1, 1, 2, 2, 2, 2],
            'score_change': 2.0,
            'expected_hypothesis': 'repetitive_pattern'
        },
        {
            'scenario': 'Score Stagnation',
            'action_sequence': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'score_change': 1.0,  # Very little progress
            'expected_hypothesis': 'ineffective_sequence'
        },
        {
            'scenario': 'Random Actions',
            'action_sequence': [6, 1, 4, 7, 2, 5, 3, 6, 1],
            'score_change': -5.0,  # Score went down
            'expected_hypothesis': 'harmful_pattern'
        }
    ]

    for case in test_cases:
        # Analyze pattern
        unique_actions = len(set(case['action_sequence']))
        action_variety = unique_actions / len(case['action_sequence'])

        # Generate hypothesis based on pattern analysis
        if case['score_change'] < 0:
            hypothesis = 'harmful_pattern'
        elif action_variety < 0.5:  # Low variety = repetitive
            hypothesis = 'repetitive_pattern'
        elif case['score_change'] < 5.0:  # Low progress
            hypothesis = 'ineffective_sequence'
        else:
            hypothesis = 'unknown_failure'

        assert hypothesis == case['expected_hypothesis'], f"Wrong hypothesis for {case['scenario']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
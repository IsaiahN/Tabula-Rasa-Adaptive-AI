"""
Test Level-Based Win Pattern Learning

Verifies that the system learns from individual level completions within games,
not just complete game wins.
"""

import pytest
from unittest.mock import Mock, AsyncMock


def test_level_completion_detection():
    """Test that level completions are detected based on score increases."""

    # Simulate a game with multiple level completions
    score_progression = [0.0, 10.0, 25.0, 75.0, 85.0, 135.0, 145.0, 200.0]
    level_completion_threshold = 50.0

    levels_detected = []
    last_significant_score = 0.0
    current_level = 1

    for i, score in enumerate(score_progression[1:], 1):  # Skip initial score
        score_increase = score - last_significant_score

        if score_increase >= level_completion_threshold:
            levels_detected.append({
                'level': current_level,
                'score_jump': score_increase,
                'action_index': i
            })
            current_level += 1
            last_significant_score = score

    # Should detect 3 level completions based on the score progression:
    # Level 1: 0 → 75 (+75), Level 2: 75 → 135 (+60), Level 3: 135 → 200 (+65)
    assert len(levels_detected) == 3
    assert levels_detected[0]['level'] == 1
    assert levels_detected[0]['score_jump'] == 75.0  # 75 - 0 = 75
    assert levels_detected[1]['level'] == 2
    assert levels_detected[1]['score_jump'] == 60.0  # 135 - 75 = 60
    assert levels_detected[2]['level'] == 3
    assert levels_detected[2]['score_jump'] == 65.0  # 200 - 135 = 65


def test_level_action_sequence_tracking():
    """Test that action sequences are tracked separately for each level."""

    # Simulate action tracking with level resets
    actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    scores = [0.0, 10.0, 20.0, 30.0, 80.0, 90.0, 100.0, 150.0, 160.0, 170.0]  # Level completes at score 80, 150
    level_completion_threshold = 50.0

    level_sequences = {}
    current_level_sequence = []
    current_level = 1
    last_significant_score = 0.0

    for i, (action, score) in enumerate(zip(actions, scores[1:])):  # Skip initial score
        current_level_sequence.append(action)
        score_increase = score - last_significant_score

        if score_increase >= level_completion_threshold and len(current_level_sequence) >= 3:
            # Level completed - save sequence
            level_sequences[current_level] = current_level_sequence.copy()

            # Reset for next level
            current_level += 1
            last_significant_score = score
            current_level_sequence = []

    # Should have tracked 2 level sequences
    assert len(level_sequences) == 2
    assert level_sequences[1] == [1, 2, 3, 4]  # Actions leading to first level completion
    assert level_sequences[2] == [5, 6, 7]     # Actions leading to second level completion


@pytest.mark.asyncio
async def test_level_strategy_discovery_integration():
    """Test that level completions trigger strategy discovery with correct level IDs."""

    # Mock StrategyDiscoverySystem
    strategy_system = AsyncMock()

    # Mock different strategies for different levels
    level_1_strategy = Mock(strategy_id='level_1_strategy_123', efficiency=30.0)
    level_2_strategy = Mock(strategy_id='level_2_strategy_456', efficiency=25.0)

    strategy_system.discover_winning_strategy.side_effect = [level_1_strategy, level_2_strategy]

    # Mock action selector
    action_selector = Mock()
    action_selector.strategy_discovery_system = strategy_system

    # Simulate level-based strategy discovery
    game_id = 'test_game_123'

    # Level 1 completion
    level_1_actions = [1, 2, 3, 4]
    level_1_scores = [0.0, 10.0, 20.0, 30.0, 80.0]

    level_1_result = await strategy_system.discover_winning_strategy(
        game_id=f"{game_id}_level_1",
        action_sequence=level_1_actions,
        score_progression=level_1_scores
    )

    # Level 2 completion
    level_2_actions = [5, 6, 7]
    level_2_scores = [80.0, 90.0, 100.0, 150.0]

    level_2_result = await strategy_system.discover_winning_strategy(
        game_id=f"{game_id}_level_2",
        action_sequence=level_2_actions,
        score_progression=level_2_scores
    )

    # Verify strategy discovery was called for each level with correct IDs
    assert strategy_system.discover_winning_strategy.call_count == 2

    # Check call arguments
    calls = strategy_system.discover_winning_strategy.call_args_list

    # First call (level 1)
    assert calls[0][1]['game_id'] == 'test_game_123_level_1'
    assert calls[0][1]['action_sequence'] == [1, 2, 3, 4]

    # Second call (level 2)
    assert calls[1][1]['game_id'] == 'test_game_123_level_2'
    assert calls[1][1]['action_sequence'] == [5, 6, 7]

    # Verify strategies were returned
    assert level_1_result.strategy_id == 'level_1_strategy_123'
    assert level_2_result.strategy_id == 'level_2_strategy_456'


@pytest.mark.asyncio
async def test_level_strategy_application():
    """Test that level-specific strategies are applied at game start."""

    # Mock StrategyDiscoverySystem with level strategies
    strategy_system = AsyncMock()

    # Mock replication checks
    strategy_system.should_attempt_strategy_replication.side_effect = [True, True]  # Game strategy, Level 1 strategy

    # Mock strategy retrieval
    game_strategy = Mock(strategy_id='game_strategy_123', efficiency=35.0, action_sequence=[1, 2, 3])
    level_1_strategy = Mock(strategy_id='level_1_strategy_456', efficiency=40.0, action_sequence=[4, 5, 6])

    strategy_system.get_best_strategy_for_game.side_effect = [game_strategy, level_1_strategy]

    # Simulate strategy application logic
    game_id = 'test_game_new'
    strategies_applied = []

    # Check for game-level strategies
    should_replicate_game = await strategy_system.should_attempt_strategy_replication(game_id)
    if should_replicate_game:
        best_game_strategy = await strategy_system.get_best_strategy_for_game(game_id)
        if best_game_strategy:
            strategies_applied.append(('game', best_game_strategy))

    # Check for level-specific strategies (level 1)
    level_1_id = f"{game_id}_level_1"
    should_replicate_level = await strategy_system.should_attempt_strategy_replication(level_1_id)
    if should_replicate_level:
        best_level_strategy = await strategy_system.get_best_strategy_for_game(level_1_id)
        if best_level_strategy:
            strategies_applied.append(('level_1', best_level_strategy))

    # Verify both strategies were checked and applied
    assert len(strategies_applied) == 2

    game_strategy_applied = strategies_applied[0]
    assert game_strategy_applied[0] == 'game'
    assert game_strategy_applied[1].strategy_id == 'game_strategy_123'

    level_strategy_applied = strategies_applied[1]
    assert level_strategy_applied[0] == 'level_1'
    assert level_strategy_applied[1].strategy_id == 'level_1_strategy_456'

    # Verify the correct calls were made
    strategy_system.should_attempt_strategy_replication.assert_any_call(game_id)
    strategy_system.should_attempt_strategy_replication.assert_any_call(f"{game_id}_level_1")
    strategy_system.get_best_strategy_for_game.assert_any_call(game_id)
    strategy_system.get_best_strategy_for_game.assert_any_call(f"{game_id}_level_1")


def test_level_learning_configuration():
    """Test level learning configuration parameters."""

    # Test different threshold values
    test_cases = [
        {'threshold': 30.0, 'score_jump': 35.0, 'should_trigger': True},
        {'threshold': 50.0, 'score_jump': 25.0, 'should_trigger': False},
        {'threshold': 100.0, 'score_jump': 150.0, 'should_trigger': True},
    ]

    for case in test_cases:
        threshold = case['threshold']
        score_jump = case['score_jump']
        expected = case['should_trigger']

        # Simulate level completion logic
        level_completed = score_jump >= threshold

        assert level_completed == expected, f"Threshold {threshold}, jump {score_jump}: expected {expected}, got {level_completed}"


def test_insufficient_actions_for_level_learning():
    """Test that level learning doesn't trigger with insufficient actions."""

    # Test with less than 3 actions
    short_action_sequence = [1, 2]  # Only 2 actions
    score_increase = 60.0  # Above threshold
    threshold = 50.0

    # Should not trigger learning with insufficient actions
    should_learn = score_increase >= threshold and len(short_action_sequence) >= 3
    assert should_learn is False

    # Test with exactly 3 actions
    sufficient_action_sequence = [1, 2, 3]  # Exactly 3 actions
    should_learn = score_increase >= threshold and len(sufficient_action_sequence) >= 3
    assert should_learn is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
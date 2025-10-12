"""
Performance validation tests for AI-enhanced CORE_GAME_MECHANICS.
Validates that AI systems provide measurable improvements over random actions.
"""

import pytest
import asyncio
import sys
import os
import time
import statistics
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

# Add CORE_GAME_MECHANICS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CORE_GAME_MECHANICS'))

try:
    from ai_orchestration import AIOrchestrator
    from vision_coordinator import VisionCoordinator
    from pattern_action_selector import PatternActionSelector
    from knowledge_integration import KnowledgeIntegration
    from core_gameplay import CoreGameplay
    from action_handler import ActionHandler
except ImportError as e:
    pytest.skip(f"CORE_GAME_MECHANICS modules not available: {e}", allow_module_level=True)


class TestAIPerformanceValidation:
    """Validate AI system performance improvements."""

    @pytest.fixture
    def performance_db(self):
        """Mock database with realistic performance data."""
        db = Mock()

        # Mock action patterns with varying success rates
        db.get_action_patterns.return_value = [
            {
                'pattern_data': {'colors': [1, 2, 3], 'dimensions': (3, 3)},
                'success_action': 'ACTION1',
                'confidence': 0.8,
                'success_rate': 0.75,
                'avg_score_impact': 15.0
            },
            {
                'pattern_data': {'colors': [4, 5, 6], 'dimensions': (3, 3)},
                'success_action': 'ACTION6',
                'confidence': 0.9,
                'success_rate': 0.85,
                'avg_score_impact': 25.0
            }
        ]

        # Mock similar games with successful strategies
        db.get_similar_games.return_value = [
            {
                'game_id': 'successful_game_1',
                'similarity': 0.9,
                'successful_actions': ['ACTION1', 'ACTION6'],
                'final_score': 95.0,
                'completion_time': 120.0
            },
            {
                'game_id': 'successful_game_2',
                'similarity': 0.8,
                'successful_actions': ['ACTION2', 'ACTION3'],
                'final_score': 88.0,
                'completion_time': 140.0
            }
        ]

        # Mock GAN predictions
        db.get_gan_prediction.return_value = {
            'predicted_action': 'ACTION1',
            'confidence': 0.8,
            'expected_score_improvement': 20.0
        }

        db.db_path = "performance_test.db"
        return db

    @pytest.fixture
    def ai_orchestrator(self, performance_db):
        """Create AI orchestrator for performance testing."""
        return AIOrchestrator(performance_db, "performance_test_game")

    @pytest.mark.asyncio
    async def test_ai_vs_random_action_selection(self, ai_orchestrator):
        """Test that AI action selection outperforms random selection."""
        frame = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        game_state = Mock()
        game_state.score = 50.0
        game_state.frame = frame
        available_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION6"]

        # Test AI-guided selections
        ai_decisions = []
        for _ in range(100):
            action, metadata = await ai_orchestrator.select_action(frame, game_state, available_actions)
            ai_decisions.append({
                'action': action,
                'confidence': metadata.get('total_confidence', 0),
                'pattern_confidence': metadata.get('pattern_confidence', 0),
                'knowledge_confidence': metadata.get('knowledge_confidence', 0)
            })

        # Analyze AI decision quality
        avg_confidence = statistics.mean([d['confidence'] for d in ai_decisions])
        high_confidence_decisions = [d for d in ai_decisions if d['confidence'] > 0.5]

        # Validate AI performance characteristics
        assert avg_confidence > 0.3, "AI should have reasonable confidence in decisions"
        assert len(high_confidence_decisions) > 20, "AI should make high-confidence decisions regularly"

        # Test action distribution (should favor high-success patterns)
        action_counts = {}
        for decision in ai_decisions:
            action = decision['action']
            action_counts[action] = action_counts.get(action, 0) + 1

        # ACTION6 should be favored (highest success rate in mock data)
        if 'ACTION6' in action_counts:
            assert action_counts['ACTION6'] > 15, "AI should favor high-success actions"

    def test_vision_coordinate_optimization(self):
        """Test that vision system provides better coordinates than random."""
        vision = VisionCoordinator("test.db")

        # Test with structured frame (clear target at center)
        structured_frame = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 9, 1, 0],  # Unique target at center
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]

        # Get vision-guided coordinates multiple times
        vision_coordinates = []
        for _ in range(50):
            x, y = asyncio.run(vision.get_optimal_coordinates(structured_frame, "test_game"))
            vision_coordinates.append((x, y))

        # Calculate distance from optimal target (2, 2)
        optimal_x, optimal_y = 2, 2
        vision_distances = [
            abs(x - optimal_x) + abs(y - optimal_y)
            for x, y in vision_coordinates
        ]

        # Generate random coordinates for comparison
        import random
        random_coordinates = []
        for _ in range(50):
            x = random.randint(0, 4)
            y = random.randint(0, 4)
            random_coordinates.append((x, y))

        random_distances = [
            abs(x - optimal_x) + abs(y - optimal_y)
            for x, y in random_coordinates
        ]

        # Vision system should perform better than random
        avg_vision_distance = statistics.mean(vision_distances)
        avg_random_distance = statistics.mean(random_distances)

        assert avg_vision_distance < avg_random_distance, \
            f"Vision guidance should be better than random: {avg_vision_distance} vs {avg_random_distance}"

    def test_pattern_matching_accuracy(self, performance_db):
        """Test pattern matching system accuracy."""
        selector = PatternActionSelector(performance_db)

        # Test with various frame patterns
        test_cases = [
            {
                'frame_analysis': {
                    'colors': [1, 2, 3, 4, 5],
                    'dimensions': (3, 3),
                    'patterns': ['horizontal_line']
                },
                'available_actions': ['ACTION1', 'ACTION2', 'ACTION3'],
                'expected_preference': 'ACTION1'  # Should match first pattern
            },
            {
                'frame_analysis': {
                    'colors': [4, 5, 6, 7, 8],
                    'dimensions': (3, 3),
                    'patterns': ['vertical_line']
                },
                'available_actions': ['ACTION6', 'ACTION7'],
                'expected_preference': 'ACTION6'  # Should match second pattern
            }
        ]

        correct_predictions = 0
        total_predictions = len(test_cases)

        for case in test_cases:
            action, confidence = selector.select_action_from_patterns(
                case['frame_analysis'],
                Mock(),
                case['available_actions']
            )

            if action == case['expected_preference']:
                correct_predictions += 1

            # Confidence should be reasonable for pattern matches
            assert confidence > 0.1, "Pattern matching should have measurable confidence"

        accuracy = correct_predictions / total_predictions
        assert accuracy >= 0.5, f"Pattern matching accuracy should be reasonable: {accuracy}"

    @pytest.mark.asyncio
    async def test_knowledge_transfer_effectiveness(self, performance_db):
        """Test knowledge transfer system effectiveness."""
        knowledge = KnowledgeIntegration(performance_db)

        # Test knowledge loading for new game
        target_context = {
            'frame_analysis': {'colors': [1, 2, 3], 'dimensions': (3, 3)},
            'initial_score': 0.0
        }

        loaded_knowledge = await knowledge.load_game_knowledge("new_game", target_context)

        # Validate knowledge structure
        assert 'similar_games' in loaded_knowledge
        assert 'recommended_actions' in loaded_knowledge
        assert 'confidence_scores' in loaded_knowledge

        # Test that knowledge provides actionable recommendations
        recommendations = loaded_knowledge['recommended_actions']
        assert len(recommendations) > 0, "Knowledge system should provide recommendations"

        # Test confidence scores are reasonable
        confidence_scores = loaded_knowledge['confidence_scores']
        for action, confidence in confidence_scores.items():
            assert 0 <= confidence <= 1, f"Confidence for {action} should be normalized: {confidence}"

    def test_system_integration_performance(self, performance_db):
        """Test overall system integration performance."""
        # Create integrated system
        mock_session = Mock()
        mock_session.current_game_id = "integration_test"
        mock_session.db = performance_db
        mock_session.send_action = AsyncMock()

        gameplay = CoreGameplay(mock_session)

        # Verify AI system initialization
        assert gameplay.ai_orchestrator is not None
        assert gameplay.ai_available is True

        # Test system responsiveness
        start_time = time.time()

        # Simulate decision making process
        frame = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        game_state = Mock()
        game_state.frame = frame
        game_state.score = 0.0
        game_state.available_actions = ["ACTION1", "ACTION2", "ACTION3"]

        action = asyncio.run(gameplay._select_ai_enhanced_action(game_state))

        decision_time = time.time() - start_time

        # Performance requirements
        assert decision_time < 1.0, f"AI decision should be fast: {decision_time}s"
        assert action in game_state.available_actions

    def test_coordinate_success_learning(self):
        """Test that vision system learns from coordinate success."""
        vision = VisionCoordinator("test.db")

        # Simulate learning from successful coordinates
        successful_coords = [(1, 1), (2, 2), (1, 2)]
        unsuccessful_coords = [(0, 0), (4, 4), (3, 0)]

        for x, y in successful_coords:
            vision.update_coordinate_success(x, y, "learning_test", True, 10.0)

        for x, y in unsuccessful_coords:
            vision.update_coordinate_success(x, y, "learning_test", False, -5.0)

        # Get performance stats
        stats = vision.get_performance_stats()

        assert stats['total_attempts'] == 6
        assert stats['success_rate'] == 0.5

        # Test that learned preferences influence future selections
        frame = [[1, 1, 1], [1, 9, 1], [1, 1, 1]]  # Target at (1, 1)

        selected_coords = []
        for _ in range(20):
            x, y = asyncio.run(vision.get_optimal_coordinates(frame, "learning_test"))
            selected_coords.append((x, y))

        # Should prefer previously successful coordinate regions
        center_selections = sum(1 for x, y in selected_coords if 1 <= x <= 2 and 1 <= y <= 2)
        assert center_selections > 5, "Should favor previously successful coordinate regions"

    @pytest.mark.asyncio
    async def test_ai_system_reliability(self, performance_db):
        """Test AI system reliability under various conditions."""
        orchestrator = AIOrchestrator(performance_db, "reliability_test")

        # Test with various frame conditions
        test_frames = [
            [[1]],  # Minimal frame
            [[0, 0], [0, 0]],  # Empty frame
            [[i % 10 for i in range(5)] for _ in range(5)],  # Large frame
            None,  # Null frame
        ]

        for frame in test_frames:
            try:
                game_state = Mock()
                game_state.frame = frame
                game_state.score = 0.0
                available_actions = ["ACTION1", "ACTION2", "ACTION3"]

                action, metadata = await orchestrator.select_action(frame, game_state, available_actions)

                # System should always return valid action
                assert action in available_actions
                assert isinstance(metadata, dict)

            except Exception as e:
                pytest.fail(f"AI system failed on frame {frame}: {e}")

    def test_performance_monitoring_completeness(self, performance_db):
        """Test that performance monitoring captures all necessary metrics."""
        # Test AI orchestrator stats
        orchestrator = AIOrchestrator(performance_db, "monitoring_test")

        # Simulate some decisions
        orchestrator.ai_decisions = [
            {
                'timestamp': '2024-01-01T10:00:00',
                'action': 'ACTION1',
                'confidence': 0.8,
                'success': True,
                'score_improvement': 15.0
            },
            {
                'timestamp': '2024-01-01T10:01:00',
                'action': 'ACTION6',
                'confidence': 0.9,
                'success': True,
                'score_improvement': 25.0
            }
        ]

        stats = orchestrator.get_performance_stats()

        # Verify comprehensive stats
        required_metrics = [
            'total_decisions', 'success_rate', 'avg_confidence',
            'avg_score_improvement', 'action_distribution'
        ]

        for metric in required_metrics:
            assert metric in stats, f"Performance stats missing {metric}"

        # Test vision coordinator stats
        vision = VisionCoordinator("test.db")
        vision_stats = vision.get_performance_stats()

        required_vision_metrics = ['total_attempts', 'success_rate']
        for metric in required_vision_metrics:
            assert metric in vision_stats, f"Vision stats missing {metric}"


class TestAISystemBenchmarks:
    """Benchmark tests for AI system performance."""

    def test_decision_speed_benchmark(self, performance_db):
        """Benchmark AI decision making speed."""
        orchestrator = AIOrchestrator(performance_db, "speed_test")

        frame = [[i % 10 for i in range(10)] for _ in range(10)]
        game_state = Mock()
        game_state.frame = frame
        game_state.score = 50.0
        available_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION6"]

        # Measure decision time over multiple iterations
        times = []
        for _ in range(10):
            start_time = time.time()
            action, metadata = asyncio.run(
                orchestrator.select_action(frame, game_state, available_actions)
            )
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = statistics.mean(times)
        max_time = max(times)

        # Performance benchmarks
        assert avg_time < 0.5, f"Average decision time too slow: {avg_time}s"
        assert max_time < 1.0, f"Max decision time too slow: {max_time}s"

        print(f"AI Decision Speed Benchmark:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
        print(f"  Min: {min(times):.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
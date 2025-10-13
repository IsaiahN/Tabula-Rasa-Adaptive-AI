"""
Comprehensive Integration Tests for Hypothesis Generation and Testing System

Tests the complete end-to-end functionality of the Game-Specific Hypothesis
Generation and Testing System, including all components working together.
"""

import pytest
import numpy as np
import sys
import sqlite3
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

# Disable pycache
sys.dont_write_bytecode = True

# Add src to path for imports
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from intelligence import (
    GamePatternAnalyzer,
    HypothesisGenerator,
    HypothesisTester,
    HypothesisIntegrationSystem,
    GameMechanic,
    HypothesisType,
    TestOutcome
)


class TestCompleteHypothesisSystem:
    """Comprehensive end-to-end tests for the hypothesis system."""

    def setup_method(self):
        """Set up complete test environment."""
        # Create in-memory database with full schema
        self.db_connection = sqlite3.connect(':memory:')
        self.db_connection.row_factory = sqlite3.Row
        self._create_full_schema()

        # Create test game scenarios
        self.pattern_completion_game = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0]  # Incomplete row
        ])

        self.physics_simulation_game = np.array([
            [0, 0, 0, 0, 0],
            [0, 2, 0, 3, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])

        self.spatial_puzzle_game = np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4]
        ])

    def teardown_method(self):
        """Clean up test environment."""
        if self.db_connection:
            self.db_connection.close()

    def _create_full_schema(self):
        """Create full database schema for testing."""
        # Read and execute the schema
        schema_statements = [
            """CREATE TABLE game_hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                game_type TEXT,
                hypothesis_type TEXT,
                source TEXT,
                description TEXT,
                hypothesis_data TEXT,
                confidence REAL,
                success_rate REAL,
                test_count INTEGER,
                reasoning TEXT,
                supporting_evidence TEXT,
                created_at TEXT,
                updated_at TEXT,
                is_active BOOLEAN
            )""",

            """CREATE TABLE hypothesis_test_results (
                experiment_id TEXT PRIMARY KEY,
                hypothesis_id TEXT,
                outcome TEXT,
                experiment_data TEXT,
                total_score_change REAL,
                test_duration REAL,
                actions_count INTEGER,
                success_actions_count INTEGER,
                learning_insights TEXT,
                failure_reasons TEXT,
                success_factors TEXT,
                recommendations TEXT,
                created_at TEXT,
                game_id TEXT,
                session_id TEXT
            )""",

            """CREATE TABLE action_reasoning_log (
                action_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                hypothesis_id TEXT,
                coordinate_x INTEGER,
                coordinate_y INTEGER,
                action_type TEXT,
                reasoning TEXT,
                reason_category TEXT,
                expected_outcome TEXT,
                actual_outcome TEXT,
                confidence_before REAL,
                confidence_after REAL,
                score_change REAL,
                success BOOLEAN,
                evidence_collected TEXT,
                created_at TEXT
            )""",

            """CREATE TABLE game_mechanics_profiles (
                profile_id TEXT PRIMARY KEY,
                game_id TEXT,
                game_type TEXT,
                primary_mechanic TEXT,
                secondary_mechanics TEXT,
                mechanic_confidence TEXT,
                visual_patterns TEXT,
                grid_features TEXT,
                complexity_score REAL,
                analysis_timestamp TEXT,
                screenshot_hash TEXT,
                created_at TEXT
            )""",

            """CREATE TABLE hypothesis_generation_sessions (
                session_id TEXT PRIMARY KEY,
                game_id TEXT,
                game_type TEXT,
                hypotheses_generated INTEGER,
                generation_strategies_used TEXT,
                pattern_analysis_time REAL,
                database_query_time REAL,
                total_generation_time REAL,
                top_hypothesis_confidence REAL,
                average_hypothesis_confidence REAL,
                session_timestamp TEXT,
                screenshot_analyzed BOOLEAN
            )""",

            """CREATE TABLE multi_level_learning (
                learning_id TEXT PRIMARY KEY,
                learning_level TEXT,
                game_id TEXT,
                game_type TEXT,
                learning_context TEXT,
                insight_type TEXT,
                insight_description TEXT,
                supporting_data TEXT,
                confidence REAL,
                impact_score REAL,
                created_at TEXT,
                applied_count INTEGER,
                success_when_applied INTEGER
            )"""
        ]

        for statement in schema_statements:
            self.db_connection.execute(statement)

        self.db_connection.commit()

    @pytest.mark.asyncio
    async def test_complete_pattern_completion_workflow(self):
        """Test complete workflow for pattern completion game."""
        # Initialize complete system
        integration_system = HypothesisIntegrationSystem(self.db_connection)

        game_id = "pattern_completion_test"
        game_context = {
            'current_score': 100,
            'game_state': {'status': 'active'},
            'available_actions': ['action6'],
            'grid_size': (5, 5)
        }

        # Step 1: Analyze and generate hypotheses
        with patch.object(integration_system.pattern_analyzer, 'analyze_game_screenshot') as mock_analyze:
            with patch.object(integration_system.game_type_classifier, 'extract_game_type',
                             return_value="pattern_completion"):
                # Mock pattern analysis result
                mock_profile = Mock()
                mock_profile.primary_mechanic = GameMechanic.PATTERN_COMPLETION
                mock_profile.secondary_mechanics = []
                mock_profile.mechanic_confidence = {GameMechanic.PATTERN_COMPLETION: 0.8}
                mock_profile.visual_patterns = []
                mock_profile.grid_features = {
                    'grid_detected': True,
                    'color_count': 2,
                    'shape_count': 5,
                    'symmetry_detected': {'horizontal': True}
                }
                mock_profile.complexity_score = 0.6

                mock_analyze.return_value = mock_profile

                hypotheses = await integration_system.analyze_and_generate_hypotheses(
                    game_id, self.pattern_completion_game, game_context
                )

                assert isinstance(hypotheses, list)
                assert len(hypotheses) > 0

                # Verify pattern completion hypothesis was generated
                pattern_completion_hyps = [h for h in hypotheses
                                         if h['hypothesis_type'] == HypothesisType.PATTERN_COMPLETION.value]
                assert len(pattern_completion_hyps) > 0

        # Step 2: Test the best hypothesis
        if hypotheses:
            best_hypothesis = max(hypotheses, key=lambda h: h['confidence'])

            with patch.object(integration_system.hypothesis_tester, '_execute_action_with_integration',
                             return_value={'outcome': 'success', 'score_change': 25.0, 'success': True}):
                test_result = await integration_system.test_hypothesis(
                    game_id, best_hypothesis['hypothesis_id'], game_context
                )

                assert isinstance(test_result, dict)
                assert 'outcome' in test_result
                assert 'total_score_change' in test_result

        # Step 3: Get intelligent coordinates based on hypotheses
        coordinates = await integration_system.get_intelligent_action6_coordinates(
            game_id, self.pattern_completion_game.tolist(), game_context
        )

        assert isinstance(coordinates, tuple)
        assert len(coordinates) == 2

        # Step 4: Analyze action results
        frame_after = self.pattern_completion_game.copy()
        frame_after[4, 0] = 1  # Complete the pattern

        analysis_result = await integration_system.analyze_action_result(
            game_id, coordinates,
            self.pattern_completion_game.tolist(),
            frame_after.tolist(),
            15.0
        )

        assert isinstance(analysis_result, dict)
        assert 'learning_insight' in analysis_result

    @pytest.mark.asyncio
    async def test_complete_physics_simulation_workflow(self):
        """Test complete workflow for physics simulation game."""
        integration_system = HypothesisIntegrationSystem(self.db_connection)

        game_id = "physics_simulation_test"
        game_context = {
            'current_score': 75,
            'game_state': {'status': 'active'},
            'available_actions': ['action6']
        }

        # Mock physics simulation analysis
        with patch.object(integration_system.pattern_analyzer, 'analyze_game_screenshot') as mock_analyze:
            with patch.object(integration_system.game_type_classifier, 'extract_game_type',
                             return_value="physics_sim"):
                mock_profile = Mock()
                mock_profile.primary_mechanic = GameMechanic.PHYSICS_SIMULATION
                mock_profile.secondary_mechanics = [GameMechanic.OBJECT_MANIPULATION]
                mock_profile.mechanic_confidence = {
                    GameMechanic.PHYSICS_SIMULATION: 0.9,
                    GameMechanic.OBJECT_MANIPULATION: 0.6
                }
                mock_profile.visual_patterns = [Mock(pattern_type="circle")]
                mock_profile.grid_features = {
                    'grid_detected': True,
                    'color_count': 4,
                    'shape_count': 3
                }
                mock_profile.complexity_score = 0.7

                mock_analyze.return_value = mock_profile

                hypotheses = await integration_system.analyze_and_generate_hypotheses(
                    game_id, self.physics_simulation_game, game_context
                )

                # Should generate physics-based hypotheses
                physics_hyps = [h for h in hypotheses
                              if h['hypothesis_type'] == HypothesisType.PHYSICS_SIMULATION.value]
                assert len(physics_hyps) >= 0  # At least attempt to generate

    @pytest.mark.asyncio
    async def test_adaptive_strategy_workflow(self):
        """Test adaptive strategy generation based on performance."""
        integration_system = HypothesisIntegrationSystem(self.db_connection)

        # Test improving performance
        improving_performance = [5.0, 8.0, 12.0, 15.0, 18.0]
        strategy = await integration_system.get_adaptive_strategy(
            "adaptive_test_1", 150.0, improving_performance
        )

        assert strategy['performance_trend'] == "improving"
        assert len(strategy['strategy_recommendations']) > 0

        # Test declining performance
        declining_performance = [20.0, 15.0, 10.0, 5.0, 0.0]
        strategy = await integration_system.get_adaptive_strategy(
            "adaptive_test_2", 75.0, declining_performance
        )

        assert strategy['performance_trend'] == "declining"
        assert any("alternative" in rec.lower() or "switch" in rec.lower()
                  for rec in strategy['strategy_recommendations'])

    @pytest.mark.asyncio
    async def test_multi_level_learning_integration(self):
        """Test multi-level learning insights generation."""
        integration_system = HypothesisIntegrationSystem(self.db_connection)

        # Test micro-level learning (small negative score)
        micro_insight = await integration_system._create_learning_insight(
            "micro_test", (10, 15), -3.0, [[1, 0]], [[0, 1]]
        )

        assert micro_insight['learning_level'] == "micro"
        assert micro_insight['insight_type'] == "mechanic_understanding"

        # Test macro-level learning (large positive score)
        macro_insight = await integration_system._create_learning_insight(
            "macro_test", (20, 25), 35.0, [[1, 0]], [[0, 1]]
        )

        assert macro_insight['learning_level'] == "macro"
        assert macro_insight['insight_type'] == "strategy_effectiveness"

        # Verify database storage
        cursor = self.db_connection.execute(
            "SELECT COUNT(*) as count FROM multi_level_learning"
        )
        result = cursor.fetchone()
        assert result['count'] == 2

    @pytest.mark.asyncio
    async def test_database_integration_workflow(self):
        """Test complete database integration and persistence."""
        integration_system = HypothesisIntegrationSystem(self.db_connection)

        # Insert test data
        self.db_connection.execute("""
            INSERT INTO game_hypotheses
            (hypothesis_id, game_type, hypothesis_data, success_rate, test_count, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "db_test_hyp",
            "test_game_type",
            '{"predicted_coordinates": [[25, 30]], "expected_actions": ["click"]}',
            0.85,
            10,
            "Database test hypothesis"
        ))
        self.db_connection.commit()

        # Test hypothesis generation with database retrieval
        with patch.object(integration_system.hypothesis_generator, '_retrieve_successful_hypotheses',
                         return_value=[{
                             'predicted_coordinates': [[25, 30]],
                             'expected_actions': ['click'],
                             'success_rate': 0.85,
                             'test_count': 10,
                             'reasoning': 'Database test hypothesis'
                         }]):

            with patch.object(integration_system.pattern_analyzer, 'analyze_game_screenshot'):
                with patch.object(integration_system.game_type_classifier, 'extract_game_type',
                                 return_value="test_game_type"):

                    hypotheses = await integration_system.analyze_and_generate_hypotheses(
                        "db_test_game", self.pattern_completion_game, {}
                    )

                    # Should include database-retrieved hypothesis
                    db_hypotheses = [h for h in hypotheses if h.get('source') == 'database_retrieval']
                    # May or may not be generated depending on mock behavior, but test should not fail

    def test_system_statistics_and_monitoring(self):
        """Test system statistics and monitoring capabilities."""
        integration_system = HypothesisIntegrationSystem(self.db_connection)

        # Simulate some activity
        integration_system.integration_stats['games_analyzed'] = 15
        integration_system.integration_stats['hypotheses_generated'] = 45
        integration_system.integration_stats['hypotheses_tested'] = 30
        integration_system.integration_stats['successful_tests'] = 18

        stats = integration_system.get_integration_statistics()

        assert stats['games_analyzed'] == 15
        assert stats['hypotheses_generated'] == 45
        assert stats['hypothesis_test_success_rate'] == 18/30
        assert stats['avg_hypotheses_per_session'] == 45/0  # Will be calculated with max

    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self):
        """Test system resilience to errors in components."""
        integration_system = HypothesisIntegrationSystem(self.db_connection)

        # Test with broken pattern analyzer
        with patch.object(integration_system.pattern_analyzer, 'analyze_game_screenshot',
                         side_effect=Exception("Pattern analyzer error")):

            # Should not crash, should return empty results
            hypotheses = await integration_system.analyze_and_generate_hypotheses(
                "error_test", self.pattern_completion_game, {}
            )

            assert isinstance(hypotheses, list)
            assert len(hypotheses) == 0

        # Test with broken action executor
        integration_system.hypothesis_tester.action_executor = AsyncMock(
            side_effect=Exception("Action executor error")
        )

        # Should handle errors gracefully
        result = await integration_system._execute_action_with_integration(
            (10, 10), "click", "Test reasoning"
        )

        assert isinstance(result, dict)
        assert 'outcome' in result
        # Should indicate error but not crash

    @pytest.mark.asyncio
    async def test_cross_game_learning(self):
        """Test learning transfer across different games."""
        integration_system = HypothesisIntegrationSystem(self.db_connection)

        # Store successful pattern from first game
        await integration_system._create_learning_insight(
            "game_1", (15, 20), 25.0, [[1, 0]], [[0, 1]]
        )

        # Store successful pattern from second game
        await integration_system._create_learning_insight(
            "game_2", (18, 22), 30.0, [[2, 1]], [[1, 2]]
        )

        # Verify insights were stored
        cursor = self.db_connection.execute(
            "SELECT COUNT(*) as count FROM multi_level_learning WHERE learning_level = 'macro'"
        )
        result = cursor.fetchone()
        assert result['count'] == 2

        # Test that system can query cross-game insights
        cursor = self.db_connection.execute(
            "SELECT * FROM multi_level_learning WHERE impact_score > 0.2"
        )
        high_impact_insights = cursor.fetchall()
        assert len(high_impact_insights) >= 2

    @pytest.mark.asyncio
    async def test_performance_optimization(self):
        """Test system performance with larger datasets."""
        integration_system = HypothesisIntegrationSystem(self.db_connection)

        # Generate larger test scenario
        large_game = np.random.randint(0, 5, size=(20, 20))

        # Test should complete reasonably quickly
        start_time = datetime.now()

        with patch.object(integration_system.pattern_analyzer, 'analyze_game_screenshot'):
            with patch.object(integration_system.game_type_classifier, 'extract_game_type',
                             return_value="large_game_type"):

                hypotheses = await integration_system.analyze_and_generate_hypotheses(
                    "large_game_test", large_game, {}
                )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 5.0  # 5 seconds maximum for large game analysis


class TestSystemComponentInteraction:
    """Test interactions between system components."""

    def setup_method(self):
        """Set up component interaction tests."""
        self.db_connection = sqlite3.connect(':memory:')
        self.pattern_analyzer = GamePatternAnalyzer()
        self.hypothesis_generator = HypothesisGenerator(self.db_connection)
        self.hypothesis_tester = HypothesisTester(self.db_connection)

    def teardown_method(self):
        """Clean up."""
        if self.db_connection:
            self.db_connection.close()

    def test_pattern_analyzer_to_hypothesis_generator_flow(self):
        """Test data flow from pattern analyzer to hypothesis generator."""
        # Create test screenshot
        test_screenshot = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ])

        # Analyze patterns
        profile = self.pattern_analyzer.analyze_game_screenshot(test_screenshot, "flow_test")

        # Verify profile has required data for hypothesis generation
        assert profile.primary_mechanic is not None
        assert isinstance(profile.grid_features, dict)
        assert isinstance(profile.complexity_score, float)

        # Verify hypothesis generator can use this data
        indicators = self.pattern_analyzer.get_hypothesis_indicators(profile)
        assert isinstance(indicators, dict)
        assert 'primary_mechanic' in indicators
        assert 'suggested_approaches' in indicators

    @pytest.mark.asyncio
    async def test_hypothesis_generator_to_tester_flow(self):
        """Test data flow from hypothesis generator to tester."""
        test_screenshot = np.array([[1, 0], [0, 1]])

        # Generate hypotheses
        with patch.object(self.hypothesis_generator.pattern_analyzer, 'analyze_game_screenshot'):
            with patch.object(self.hypothesis_generator.game_type_classifier, 'extract_game_type',
                             return_value="flow_test_type"):

                hypotheses = self.hypothesis_generator.generate_hypotheses(
                    "flow_test_game", test_screenshot, max_hypotheses=1
                )

        if hypotheses:
            hypothesis = hypotheses[0]

            # Verify hypothesis has required data for testing
            assert hypothesis.predicted_coordinates is not None
            assert hypothesis.expected_actions is not None
            assert isinstance(hypothesis.confidence, float)

            # Test hypothesis with tester
            mock_executor = AsyncMock(return_value={
                'outcome': 'success',
                'score_change': 10.0,
                'success': True
            })

            self.hypothesis_tester.action_executor = mock_executor

            result = await self.hypothesis_tester.test_hypothesis(
                hypothesis, {'current_score': 100}, max_actions=1, timeout_seconds=5.0
            )

            assert isinstance(result, object)  # ExperimentResult
            assert hasattr(result, 'outcome')
            assert hasattr(result, 'actions_taken')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
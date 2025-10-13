"""
Tests for Hypothesis Integration System

Tests the integration between the hypothesis generation and testing system
with existing Action6Coordinator and Enhanced Gameplay systems.
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

from intelligence.hypothesis_integration import (
    HypothesisIntegrationSystem,
    get_hypothesis_integration_system,
    create_hypothesis_integration_system
)

from intelligence.hypothesis_generator import (
    Hypothesis,
    HypothesisType,
    HypothesisSource
)

from intelligence.game_pattern_analyzer import GameMechanic


class TestHypothesisIntegrationSystem:
    """Test suite for HypothesisIntegrationSystem."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create in-memory database for testing
        self.db_connection = sqlite3.connect(':memory:')
        self.db_connection.row_factory = sqlite3.Row

        # Create test tables
        self._create_test_tables()

        # Mock existing systems
        self.mock_action6_coordinator = Mock()
        self.mock_enhanced_gameplay = Mock()

        # Configure async methods
        self.mock_action6_coordinator.get_optimal_action6_coordinates = AsyncMock(return_value=(15, 20))
        self.mock_action6_coordinator.analyze_action6_effectiveness = AsyncMock()
        self.mock_enhanced_gameplay.analyze_action6_result = AsyncMock()

        self.integration_system = HypothesisIntegrationSystem(
            self.db_connection,
            self.mock_action6_coordinator,
            self.mock_enhanced_gameplay
        )

        # Test data
        self.test_screenshot = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1]
        ])

        self.test_game_context = {
            'current_score': 150,
            'game_state': {'status': 'active'},
            'available_actions': ['action6'],
            'grid_size': (10, 10)
        }

    def teardown_method(self):
        """Clean up after tests."""
        if self.db_connection:
            self.db_connection.close()

    def _create_test_tables(self):
        """Create test database tables."""
        # Create tables needed for hypothesis integration
        self.db_connection.execute("""
            CREATE TABLE hypothesis_generation_sessions (
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
            )
        """)

        self.db_connection.execute("""
            CREATE TABLE multi_level_learning (
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
                created_at TEXT
            )
        """)

        self.db_connection.commit()

    def test_integration_system_initialization(self):
        """Test that integration system initializes correctly."""
        assert self.integration_system is not None
        assert self.integration_system.db_connection is not None
        assert self.integration_system.action6_coordinator is not None
        assert self.integration_system.enhanced_gameplay is not None
        assert hasattr(self.integration_system, 'pattern_analyzer')
        assert hasattr(self.integration_system, 'hypothesis_generator')
        assert hasattr(self.integration_system, 'hypothesis_tester')

    @pytest.mark.asyncio
    @patch('intelligence.game_pattern_analyzer.get_game_pattern_analyzer')
    @patch('intelligence.hypothesis_generator.get_hypothesis_generator')
    async def test_analyze_and_generate_hypotheses(self, mock_gen, mock_analyzer):
        """Test hypothesis analysis and generation."""
        # Mock the hypothesis generator to return test hypotheses
        test_hypotheses = [
            Hypothesis(
                hypothesis_id="integration_test_1",
                hypothesis_type=HypothesisType.PATTERN_COMPLETION,
                source=HypothesisSource.PATTERN_ANALYSIS,
                description="Test integration hypothesis",
                predicted_coordinates=[(10, 10), (15, 15)],
                expected_actions=["click", "drag"],
                confidence=0.8,
                reasoning="Integration test reasoning",
                supporting_evidence={},
                game_mechanics=[GameMechanic.PATTERN_COMPLETION],
                created_at=datetime.now()
            )
        ]

        mock_gen.return_value.generate_hypotheses.return_value = test_hypotheses

        results = await self.integration_system.analyze_and_generate_hypotheses(
            "test_game_1", self.test_screenshot, self.test_game_context
        )

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]['hypothesis_id'] == "integration_test_1"
        assert results[0]['hypothesis_type'] == HypothesisType.PATTERN_COMPLETION.value
        assert len(results[0]['predicted_coordinates']) == 2

        # Verify statistics updated
        assert self.integration_system.integration_stats['games_analyzed'] == 1
        assert self.integration_system.integration_stats['hypotheses_generated'] == 1

    @pytest.mark.asyncio
    @patch('intelligence.hypothesis_tester.get_hypothesis_tester')
    async def test_test_hypothesis(self, mock_tester_getter):
        """Test hypothesis testing through integration system."""
        # Create test hypothesis in session
        test_hypothesis = Hypothesis(
            hypothesis_id="test_hypothesis_integration",
            hypothesis_type=HypothesisType.OBJECT_MANIPULATION,
            source=HypothesisSource.DATABASE_RETRIEVAL,
            description="Test hypothesis for integration testing",
            predicted_coordinates=[(20, 25)],
            expected_actions=["click_object"],
            confidence=0.75,
            reasoning="Integration testing",
            supporting_evidence={},
            game_mechanics=[GameMechanic.OBJECT_MANIPULATION],
            created_at=datetime.now()
        )

        # Add hypothesis to session
        session = self.integration_system._get_or_create_session("test_game_integration")
        session['current_hypotheses'] = [test_hypothesis]

        # Mock the tester
        mock_tester = Mock()
        mock_experiment_result = Mock()
        mock_experiment_result.experiment_id = "exp_123"
        mock_experiment_result.outcome = Mock()
        mock_experiment_result.outcome.value = "success"
        mock_experiment_result.total_score_change = 25.0
        mock_experiment_result.test_duration = 8.5
        mock_experiment_result.actions_taken = [Mock()]
        mock_experiment_result.learning_insights = ["Test insight"]
        mock_experiment_result.recommendations = ["Test recommendation"]

        mock_tester.test_hypothesis = AsyncMock(return_value=mock_experiment_result)
        mock_tester_getter.return_value = mock_tester

        result = await self.integration_system.test_hypothesis(
            "test_game_integration", "test_hypothesis_integration", self.test_game_context
        )

        assert isinstance(result, dict)
        assert 'experiment_id' in result
        assert 'outcome' in result
        assert result['hypothesis_id'] == "test_hypothesis_integration"

        # Verify statistics updated
        assert self.integration_system.integration_stats['hypotheses_tested'] == 1

    @pytest.mark.asyncio
    async def test_get_intelligent_action6_coordinates(self):
        """Test intelligent Action6 coordinate selection."""
        coordinates = await self.integration_system.get_intelligent_action6_coordinates(
            "test_game_coords", [[1, 0], [0, 1]], self.test_game_context
        )

        assert isinstance(coordinates, tuple)
        assert len(coordinates) == 2
        assert isinstance(coordinates[0], int)
        assert isinstance(coordinates[1], int)

        # Should have called Action6Coordinator as fallback
        self.mock_action6_coordinator.get_optimal_action6_coordinates.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_intelligent_action6_coordinates_with_hypotheses(self):
        """Test coordinate selection when hypotheses are available."""
        # Create session with hypothesis
        test_hypothesis = Hypothesis(
            hypothesis_id="coord_test_hypothesis",
            hypothesis_type=HypothesisType.NAVIGATION_PATH,
            source=HypothesisSource.PATTERN_ANALYSIS,
            description="Coordinate test hypothesis",
            predicted_coordinates=[(30, 35), (40, 45)],
            expected_actions=["click"],
            confidence=0.9,
            reasoning="High confidence coordinate prediction",
            supporting_evidence={},
            game_mechanics=[GameMechanic.NAVIGATION_PATHFINDING],
            created_at=datetime.now()
        )

        session = self.integration_system._get_or_create_session("test_game_with_hyp")
        session['current_hypotheses'] = [test_hypothesis]

        coordinates = await self.integration_system.get_intelligent_action6_coordinates(
            "test_game_with_hyp", [[1, 0], [0, 1]], self.test_game_context
        )

        # Should return coordinate from hypothesis
        assert coordinates == (30, 35)

    @pytest.mark.asyncio
    async def test_analyze_action_result(self):
        """Test action result analysis."""
        frame_before = [[1, 0], [0, 1]]
        frame_after = [[0, 1], [1, 0]]

        result = await self.integration_system.analyze_action_result(
            "test_game_analysis", (25, 30), frame_before, frame_after, 12.0
        )

        assert isinstance(result, dict)
        assert 'analysis_completed' in result
        assert result['score_change'] == 12.0
        assert 'learning_insight' in result

        # Verify integration calls were made
        self.mock_action6_coordinator.analyze_action6_effectiveness.assert_called_once()
        self.mock_enhanced_gameplay.analyze_action6_result.assert_called_once()

        # Verify statistics updated
        assert self.integration_system.integration_stats['action6_integrations'] == 1
        assert self.integration_system.integration_stats['enhanced_gameplay_feedbacks'] == 1

    @pytest.mark.asyncio
    async def test_get_adaptive_strategy(self):
        """Test adaptive strategy generation."""
        recent_performance = [5.0, 10.0, -2.0, 15.0, 8.0]

        strategy = await self.integration_system.get_adaptive_strategy(
            "test_game_strategy", 125.0, recent_performance
        )

        assert isinstance(strategy, dict)
        assert 'game_type' in strategy
        assert 'performance_trend' in strategy
        assert 'strategy_recommendations' in strategy
        assert 'adaptive_confidence' in strategy

        assert isinstance(strategy['strategy_recommendations'], list)
        assert len(strategy['strategy_recommendations']) > 0

    def test_get_integration_statistics(self):
        """Test integration statistics retrieval."""
        # Add some test data
        self.integration_system.integration_stats['games_analyzed'] = 5
        self.integration_system.integration_stats['hypotheses_generated'] = 15
        self.integration_system.integration_stats['successful_tests'] = 8
        self.integration_system.integration_stats['hypotheses_tested'] = 12

        stats = self.integration_system.get_integration_statistics()

        assert isinstance(stats, dict)
        assert 'games_analyzed' in stats
        assert 'hypotheses_generated' in stats
        assert 'hypothesis_test_success_rate' in stats
        assert 'total_sessions' in stats

        assert stats['games_analyzed'] == 5
        assert stats['hypotheses_generated'] == 15
        assert stats['hypothesis_test_success_rate'] == 8/12

    @pytest.mark.asyncio
    async def test_execute_action_with_integration(self):
        """Test integrated action execution."""
        result = await self.integration_system._execute_action_with_integration(
            (35, 40), "click", "Integration test reasoning"
        )

        assert isinstance(result, dict)
        assert 'outcome' in result
        assert 'score_change' in result
        assert 'success' in result
        assert 'integration_used' in result
        assert result['integration_used'] == True

    def test_get_or_create_session(self):
        """Test session management."""
        # First call should create new session
        session1 = self.integration_system._get_or_create_session("new_game")
        assert isinstance(session1, dict)
        assert session1['game_id'] == "new_game"
        assert 'created_at' in session1
        assert 'current_hypotheses' in session1

        # Second call should return same session
        session2 = self.integration_system._get_or_create_session("new_game")
        assert session1 is session2

    @pytest.mark.asyncio
    async def test_store_hypothesis_generation_session(self):
        """Test storing hypothesis generation session in database."""
        test_hypotheses = [
            Mock(confidence=0.8, source=Mock(value="pattern_analysis")),
            Mock(confidence=0.6, source=Mock(value="database_retrieval"))
        ]

        await self.integration_system._store_hypothesis_generation_session(
            "test_game_store", test_hypotheses, self.test_game_context
        )

        # Verify data was stored
        cursor = self.db_connection.execute(
            "SELECT * FROM hypothesis_generation_sessions WHERE game_id = ?",
            ("test_game_store",)
        )
        result = cursor.fetchone()
        assert result is not None
        assert result['hypotheses_generated'] == 2
        assert result['screenshot_analyzed'] == 1

    @pytest.mark.asyncio
    async def test_create_learning_insight(self):
        """Test multi-level learning insight creation."""
        insight = await self.integration_system._create_learning_insight(
            "test_game_insight", (45, 50), 20.0, [[1, 0]], [[0, 1]]
        )

        assert isinstance(insight, dict)
        assert 'learning_level' in insight
        assert 'insight_type' in insight
        assert 'insight_description' in insight
        assert 'confidence' in insight
        assert 'impact_score' in insight

        # High score change should create macro-level insight
        assert insight['learning_level'] == "macro"

        # Verify stored in database
        cursor = self.db_connection.execute(
            "SELECT * FROM multi_level_learning WHERE game_id = ?",
            ("test_game_insight",)
        )
        result = cursor.fetchone()
        assert result is not None

    def test_get_center_coordinates(self):
        """Test center coordinate calculation."""
        frame = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        coordinates = self.integration_system._get_center_coordinates(frame)

        assert isinstance(coordinates, tuple)
        assert coordinates == (1, 1)  # Center of 3x3 grid

        # Test empty frame
        empty_coords = self.integration_system._get_center_coordinates([])
        assert empty_coords == (25, 25)


class TestHypothesisIntegrationSystemFactory:
    """Test factory functions."""

    def test_create_hypothesis_integration_system(self):
        """Test factory function."""
        system = create_hypothesis_integration_system()
        assert isinstance(system, HypothesisIntegrationSystem)

    def test_create_hypothesis_integration_system_with_components(self):
        """Test factory function with components."""
        mock_action6 = Mock()
        mock_enhanced = Mock()
        db_connection = sqlite3.connect(':memory:')

        system = create_hypothesis_integration_system(
            db_connection, mock_action6, mock_enhanced
        )

        assert isinstance(system, HypothesisIntegrationSystem)
        assert system.db_connection is db_connection
        assert system.action6_coordinator is mock_action6
        assert system.enhanced_gameplay is mock_enhanced

    def test_get_hypothesis_integration_system_singleton(self):
        """Test singleton pattern."""
        system1 = get_hypothesis_integration_system()
        system2 = get_hypothesis_integration_system()
        assert system1 is system2


class TestHypothesisIntegrationSystemErrorHandling:
    """Test error handling in integration system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.integration_system = HypothesisIntegrationSystem()

    @pytest.mark.asyncio
    async def test_analyze_and_generate_hypotheses_with_error(self):
        """Test hypothesis generation with error."""
        # Mock pattern analyzer to raise an exception
        with patch.object(self.integration_system.pattern_analyzer, 'analyze_game_screenshot',
                         side_effect=Exception("Test error")):
            results = await self.integration_system.analyze_and_generate_hypotheses(
                "error_game", np.array([[1, 0]]), {}
            )

            assert isinstance(results, list)
            assert len(results) == 0  # Should return empty list on error

    @pytest.mark.asyncio
    async def test_test_hypothesis_not_found(self):
        """Test testing non-existent hypothesis."""
        result = await self.integration_system.test_hypothesis(
            "test_game", "non_existent_hypothesis", {}
        )

        assert isinstance(result, dict)
        assert 'error' in result
        assert result['error'] == 'Hypothesis not found'

    @pytest.mark.asyncio
    async def test_analyze_action_result_with_error(self):
        """Test action result analysis with error."""
        # Configure mocks to raise exceptions
        if self.integration_system.action6_coordinator:
            self.integration_system.action6_coordinator.analyze_action6_effectiveness = AsyncMock(
                side_effect=Exception("Analysis error")
            )

        result = await self.integration_system.analyze_action_result(
            "error_game", (10, 10), [[1]], [[0]], 5.0
        )

        # Should still return result even with errors in sub-components
        assert isinstance(result, dict)
        assert 'analysis_completed' in result


if __name__ == "__main__":
    pytest.main([__file__])
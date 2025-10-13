"""
Tests for Hypothesis Generator

Tests the automatic hypothesis generation capabilities including database integration,
pattern-based generation, and multi-source hypothesis creation.
"""

import pytest
import numpy as np
import sys
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Disable pycache
sys.dont_write_bytecode = True

# Add src to path for imports
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from intelligence.hypothesis_generator import (
    HypothesisGenerator,
    Hypothesis,
    HypothesisType,
    HypothesisSource,
    HypothesisContext,
    get_hypothesis_generator,
    create_hypothesis_generator
)

from intelligence.game_pattern_analyzer import GameMechanic, GameMechanicsProfile, VisualPattern


class TestHypothesisGenerator:
    """Test suite for HypothesisGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create in-memory database for testing
        self.db_connection = sqlite3.connect(':memory:')
        self.db_connection.row_factory = sqlite3.Row

        # Create test tables
        self._create_test_tables()

        self.generator = HypothesisGenerator(self.db_connection)

        # Create test data
        self.test_screenshot = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1]
        ])

        self.test_mechanics_profile = GameMechanicsProfile(
            primary_mechanic=GameMechanic.PATTERN_COMPLETION,
            secondary_mechanics=[GameMechanic.COLOR_MATCHING],
            mechanic_confidence={
                GameMechanic.PATTERN_COMPLETION: 0.8,
                GameMechanic.COLOR_MATCHING: 0.6
            },
            visual_patterns=[
                VisualPattern(
                    pattern_id="test_pattern_1",
                    pattern_type="rectangle",
                    confidence=0.7,
                    location=(5, 5),
                    size=(10, 10),
                    features={},
                    timestamp=datetime.now()
                )
            ],
            grid_features={
                'grid_detected': True,
                'color_count': 3,
                'shape_count': 5,
                'symmetry_detected': {'horizontal': True, 'vertical': False}
            },
            complexity_score=0.6,
            timestamp=datetime.now()
        )

    def teardown_method(self):
        """Clean up after tests."""
        if self.db_connection:
            self.db_connection.close()

    def _create_test_tables(self):
        """Create test database tables."""
        self.db_connection.execute("""
            CREATE TABLE game_hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                game_type TEXT,
                hypothesis_data TEXT,
                success_rate REAL,
                test_count INTEGER,
                reasoning TEXT
            )
        """)

        self.db_connection.execute("""
            CREATE TABLE learned_patterns (
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                frequency INTEGER,
                success_rate REAL,
                game_context TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Insert test data
        self.db_connection.execute("""
            INSERT INTO game_hypotheses
            (hypothesis_id, game_type, hypothesis_data, success_rate, test_count, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "test_hypothesis_1",
            "test_type",
            '{"predicted_coordinates": [[10, 10], [15, 15]], "expected_actions": ["click", "drag"]}',
            0.8,
            5,
            "Test hypothesis for pattern completion"
        ))

        self.db_connection.commit()

    def test_generator_initialization(self):
        """Test that generator initializes correctly."""
        assert self.generator is not None
        assert self.generator.db_connection is not None
        assert hasattr(self.generator, 'pattern_analyzer')
        assert hasattr(self.generator, 'game_type_classifier')
        assert hasattr(self.generator, 'hypothesis_cache')

    @patch('intelligence.game_pattern_analyzer.get_game_pattern_analyzer')
    @patch('learning.game_type_classifier.get_game_type_classifier')
    def test_generate_hypotheses_basic(self, mock_classifier, mock_analyzer):
        """Test basic hypothesis generation."""
        # Mock dependencies
        mock_analyzer.return_value.analyze_game_screenshot.return_value = self.test_mechanics_profile
        mock_classifier.return_value.extract_game_type.return_value = "test_type"
        mock_classifier.return_value.get_similar_game_types.return_value = []

        hypotheses = self.generator.generate_hypotheses(
            "test_game_1", self.test_screenshot, max_hypotheses=3
        )

        assert isinstance(hypotheses, list)
        assert len(hypotheses) <= 3
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, Hypothesis)
            assert hypothesis.hypothesis_id is not None
            assert hypothesis.confidence >= 0 and hypothesis.confidence <= 1

    def test_generate_pattern_based_hypotheses(self):
        """Test pattern-based hypothesis generation."""
        context = HypothesisContext(
            game_id="test_game",
            game_type="test_type",
            mechanics_profile=self.test_mechanics_profile,
            screenshot_features=self.test_mechanics_profile.grid_features,
            historical_successes=[],
            similar_games_data=[]
        )

        hypotheses = self.generator._generate_pattern_based_hypotheses(context)

        assert isinstance(hypotheses, list)
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, Hypothesis)
            assert hypothesis.source == HypothesisSource.PATTERN_ANALYSIS
            assert len(hypothesis.predicted_coordinates) > 0

    def test_predict_coordinates_pattern_completion(self):
        """Test coordinate prediction for pattern completion."""
        context = HypothesisContext(
            game_id="test_game",
            game_type="test_type",
            mechanics_profile=self.test_mechanics_profile,
            screenshot_features=self.test_mechanics_profile.grid_features,
            historical_successes=[],
            similar_games_data=[]
        )

        coordinates = self.generator._predict_coordinates_for_mechanic(
            context, GameMechanic.PATTERN_COMPLETION, "complete_missing_elements"
        )

        assert isinstance(coordinates, list)
        for coord in coordinates:
            assert isinstance(coord, tuple)
            assert len(coord) == 2
            assert isinstance(coord[0], int)
            assert isinstance(coord[1], int)

    def test_predict_actions_for_strategy(self):
        """Test action prediction for different strategies."""
        actions = self.generator._predict_actions_for_strategy(
            GameMechanic.OBJECT_MANIPULATION, "move_discrete_objects"
        )

        assert isinstance(actions, list)
        assert len(actions) > 0
        for action in actions:
            assert isinstance(action, str)

    def test_generate_reasoning(self):
        """Test reasoning generation."""
        context = HypothesisContext(
            game_id="test_game",
            game_type="test_type",
            mechanics_profile=self.test_mechanics_profile,
            screenshot_features=self.test_mechanics_profile.grid_features,
            historical_successes=[],
            similar_games_data=[]
        )

        reasoning = self.generator._generate_reasoning(
            context, GameMechanic.PATTERN_COMPLETION, "complete_missing_elements", [(10, 10), (15, 15)]
        )

        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "pattern" in reasoning.lower()

    def test_database_hypothesis_retrieval(self):
        """Test retrieval of hypotheses from database."""
        hypotheses = self.generator._retrieve_successful_hypotheses("test_type")

        assert isinstance(hypotheses, list)
        if len(hypotheses) > 0:
            for hypothesis in hypotheses:
                assert isinstance(hypothesis, dict)
                assert 'success_rate' in hypothesis
                assert 'test_count' in hypothesis

    def test_hypothesis_caching(self):
        """Test hypothesis caching mechanism."""
        game_id = "cache_test_game"

        # Mock dependencies
        with patch.object(self.generator.pattern_analyzer, 'analyze_game_screenshot',
                         return_value=self.test_mechanics_profile):
            with patch.object(self.generator.game_type_classifier, 'extract_game_type',
                             return_value="test_type"):
                # First generation
                hypotheses1 = self.generator.generate_hypotheses(game_id, self.test_screenshot)

                # Second generation should use cache
                hypotheses2 = self.generator.generate_hypotheses(game_id, self.test_screenshot)

                assert game_id in self.generator.hypothesis_cache
                assert len(hypotheses1) == len(hypotheses2)

    def test_save_hypothesis_result(self):
        """Test saving hypothesis test results."""
        hypothesis = Hypothesis(
            hypothesis_id="test_save_hypothesis",
            hypothesis_type=HypothesisType.PATTERN_COMPLETION,
            source=HypothesisSource.PATTERN_ANALYSIS,
            description="Test hypothesis",
            predicted_coordinates=[(10, 10)],
            expected_actions=["click"],
            confidence=0.7,
            reasoning="Test reasoning",
            supporting_evidence={},
            game_mechanics=[GameMechanic.PATTERN_COMPLETION],
            created_at=datetime.now()
        )

        # Save successful result
        self.generator.save_hypothesis_result(hypothesis, True, 25.0, {"test_data": "value"})

        assert hypothesis.tested == True
        assert hypothesis.test_count == 1
        assert hypothesis.success_rate == 1.0

        # Save failed result
        self.generator.save_hypothesis_result(hypothesis, False, -10.0)

        assert hypothesis.test_count == 2
        assert hypothesis.success_rate == 0.5

    def test_collect_supporting_evidence(self):
        """Test evidence collection."""
        context = HypothesisContext(
            game_id="test_game",
            game_type="test_type",
            mechanics_profile=self.test_mechanics_profile,
            screenshot_features=self.test_mechanics_profile.grid_features,
            historical_successes=[],
            similar_games_data=[]
        )

        evidence = self.generator._collect_supporting_evidence(
            context, GameMechanic.PATTERN_COMPLETION, "complete_missing_elements"
        )

        assert isinstance(evidence, dict)
        assert 'visual_patterns' in evidence
        assert 'mechanic_confidence' in evidence
        assert 'complexity_score' in evidence
        assert 'strategy_applied' in evidence

    def test_score_and_rank_hypotheses(self):
        """Test hypothesis scoring and ranking."""
        hypotheses = [
            Hypothesis(
                hypothesis_id=f"test_hypothesis_{i}",
                hypothesis_type=HypothesisType.PATTERN_COMPLETION,
                source=HypothesisSource.PATTERN_ANALYSIS,
                description=f"Test hypothesis {i}",
                predicted_coordinates=[(i*5, i*5)],
                expected_actions=["click"],
                confidence=0.5 + i*0.1,
                reasoning=f"Test reasoning {i}",
                supporting_evidence={},
                game_mechanics=[GameMechanic.PATTERN_COMPLETION],
                created_at=datetime.now()
            )
            for i in range(3)
        ]

        context = HypothesisContext(
            game_id="test_game",
            game_type="test_type",
            mechanics_profile=self.test_mechanics_profile,
            screenshot_features=self.test_mechanics_profile.grid_features,
            historical_successes=[],
            similar_games_data=[]
        )

        ranked_hypotheses = self.generator._score_and_rank_hypotheses(hypotheses, context)

        assert isinstance(ranked_hypotheses, list)
        assert len(ranked_hypotheses) == len(hypotheses)

        # Check that they're ranked by confidence (descending)
        for i in range(len(ranked_hypotheses) - 1):
            assert ranked_hypotheses[i].confidence >= ranked_hypotheses[i+1].confidence


class TestHypothesis:
    """Test Hypothesis dataclass."""

    def test_hypothesis_creation(self):
        """Test hypothesis creation."""
        hypothesis = Hypothesis(
            hypothesis_id="test_hypothesis_creation",
            hypothesis_type=HypothesisType.OBJECT_MANIPULATION,
            source=HypothesisSource.DATABASE_RETRIEVAL,
            description="Test hypothesis for object manipulation",
            predicted_coordinates=[(20, 30), (40, 50)],
            expected_actions=["click_object", "drag_to_position"],
            confidence=0.85,
            reasoning="Objects detected that can be manipulated based on pattern analysis",
            supporting_evidence={"object_count": 3, "boundary_clarity": "clear"},
            game_mechanics=[GameMechanic.OBJECT_MANIPULATION, GameMechanic.SPATIAL_PUZZLE],
            created_at=datetime.now()
        )

        assert hypothesis.hypothesis_id == "test_hypothesis_creation"
        assert hypothesis.hypothesis_type == HypothesisType.OBJECT_MANIPULATION
        assert hypothesis.source == HypothesisSource.DATABASE_RETRIEVAL
        assert len(hypothesis.predicted_coordinates) == 2
        assert len(hypothesis.expected_actions) == 2
        assert hypothesis.confidence == 0.85
        assert hypothesis.tested == False
        assert hypothesis.success_rate == 0.0


class TestHypothesisTypes:
    """Test hypothesis type enums."""

    def test_hypothesis_type_values(self):
        """Test hypothesis type enum values."""
        for hypothesis_type in HypothesisType:
            assert isinstance(hypothesis_type.value, str)
            assert len(hypothesis_type.value) > 0

    def test_hypothesis_source_values(self):
        """Test hypothesis source enum values."""
        for source in HypothesisSource:
            assert isinstance(source.value, str)
            assert len(source.value) > 0


class TestHypothesisGeneratorFactory:
    """Test factory functions."""

    def test_create_hypothesis_generator(self):
        """Test factory function."""
        generator = create_hypothesis_generator()
        assert isinstance(generator, HypothesisGenerator)

    def test_create_hypothesis_generator_with_db(self):
        """Test factory function with database."""
        db_connection = sqlite3.connect(':memory:')
        generator = create_hypothesis_generator(db_connection)
        assert isinstance(generator, HypothesisGenerator)
        assert generator.db_connection is db_connection

    def test_get_hypothesis_generator_singleton(self):
        """Test singleton pattern."""
        generator1 = get_hypothesis_generator()
        generator2 = get_hypothesis_generator()
        assert generator1 is generator2


class TestHypothesisContext:
    """Test HypothesisContext dataclass."""

    def test_hypothesis_context_creation(self):
        """Test hypothesis context creation."""
        context = HypothesisContext(
            game_id="context_test_game",
            game_type="context_test_type",
            mechanics_profile=Mock(),
            screenshot_features={"test": "features"},
            historical_successes=[{"test": "success"}],
            similar_games_data=[{"test": "data"}]
        )

        assert context.game_id == "context_test_game"
        assert context.game_type == "context_test_type"
        assert context.screenshot_features == {"test": "features"}
        assert len(context.historical_successes) == 1
        assert len(context.similar_games_data) == 1


if __name__ == "__main__":
    pytest.main([__file__])
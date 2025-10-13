"""
Tests for Hypothesis Tester

Tests the experimental framework for systematic hypothesis testing with
detailed action reasoning and evidence collection.
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

from intelligence.hypothesis_tester import (
    HypothesisTester,
    TestOutcome,
    ActionReason,
    ActionRecord,
    ExperimentResult,
    get_hypothesis_tester,
    create_hypothesis_tester
)

from intelligence.hypothesis_generator import (
    Hypothesis,
    HypothesisType,
    HypothesisSource
)

from intelligence.game_pattern_analyzer import GameMechanic


class TestHypothesisTester:
    """Test suite for HypothesisTester."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create in-memory database for testing
        self.db_connection = sqlite3.connect(':memory:')
        self.db_connection.row_factory = sqlite3.Row

        # Create test tables
        self._create_test_tables()

        # Mock action executor
        self.mock_action_executor = AsyncMock()
        self.mock_action_executor.return_value = {
            'outcome': 'successful_click_execution',
            'score_change': 15.0,
            'success': True
        }

        self.tester = HypothesisTester(self.db_connection, self.mock_action_executor)

        # Create test hypothesis
        self.test_hypothesis = Hypothesis(
            hypothesis_id="test_hypothesis_1",
            hypothesis_type=HypothesisType.PATTERN_COMPLETION,
            source=HypothesisSource.PATTERN_ANALYSIS,
            description="Test hypothesis for pattern completion",
            predicted_coordinates=[(10, 10), (15, 15), (20, 20)],
            expected_actions=["click", "drag", "release"],
            confidence=0.8,
            reasoning="Pattern analysis suggests completing missing elements",
            supporting_evidence={"pattern_strength": 0.7},
            game_mechanics=[GameMechanic.PATTERN_COMPLETION],
            created_at=datetime.now()
        )

        self.test_game_context = {
            'current_score': 100,
            'game_state': {'status': 'active'},
            'available_actions': ['action6'],
            'screenshot_data': {}
        }

    def teardown_method(self):
        """Clean up after tests."""
        if self.db_connection:
            self.db_connection.close()

    def _create_test_tables(self):
        """Create test database tables."""
        self.db_connection.execute("""
            CREATE TABLE hypothesis_test_results (
                experiment_id TEXT PRIMARY KEY,
                hypothesis_id TEXT,
                outcome TEXT,
                experiment_data TEXT,
                total_score_change REAL,
                test_duration REAL,
                created_at TEXT
            )
        """)

        self.db_connection.execute("""
            CREATE TABLE action_reasoning_log (
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
                created_at TEXT
            )
        """)

        self.db_connection.commit()

    def test_tester_initialization(self):
        """Test that tester initializes correctly."""
        assert self.tester is not None
        assert self.tester.db_connection is not None
        assert self.tester.action_executor is not None
        assert hasattr(self.tester, 'test_cache')
        assert hasattr(self.tester, 'reasoning_templates')

    @pytest.mark.asyncio
    async def test_test_hypothesis_basic(self):
        """Test basic hypothesis testing."""
        result = await self.tester.test_hypothesis(
            self.test_hypothesis, self.test_game_context, max_actions=2, timeout_seconds=10.0
        )

        assert isinstance(result, ExperimentResult)
        assert result.hypothesis.hypothesis_id == self.test_hypothesis.hypothesis_id
        assert result.outcome in [TestOutcome.SUCCESS, TestOutcome.FAILURE,
                                TestOutcome.PARTIAL_SUCCESS, TestOutcome.INCONCLUSIVE]
        assert len(result.actions_taken) <= 2
        assert result.test_duration > 0

    @pytest.mark.asyncio
    async def test_test_hypothesis_with_successful_actions(self):
        """Test hypothesis testing with successful actions."""
        # Configure mock to return successful results
        self.mock_action_executor.return_value = {
            'outcome': 'successful_click_execution',
            'score_change': 25.0,
            'success': True
        }

        result = await self.tester.test_hypothesis(
            self.test_hypothesis, self.test_game_context, max_actions=3
        )

        assert result.outcome == TestOutcome.SUCCESS
        assert result.total_score_change > 0
        assert all(action.success for action in result.actions_taken)

    @pytest.mark.asyncio
    async def test_test_hypothesis_with_failed_actions(self):
        """Test hypothesis testing with failed actions."""
        # Configure mock to return failed results
        self.mock_action_executor.return_value = {
            'outcome': 'failed_click_execution',
            'score_change': -10.0,
            'success': False
        }

        result = await self.tester.test_hypothesis(
            self.test_hypothesis, self.test_game_context, max_actions=3
        )

        assert result.outcome == TestOutcome.FAILURE
        assert result.total_score_change < 0
        assert all(not action.success for action in result.actions_taken)

    def test_generate_action_reasoning(self):
        """Test action reasoning generation."""
        reasoning = self.tester._generate_action_reasoning(
            self.test_hypothesis, (10, 10), "click", 0, {"test": "evidence"}
        )

        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "pattern_completion" in reasoning.lower() or "click" in reasoning.lower()

    @pytest.mark.asyncio
    async def test_execute_reasoned_action(self):
        """Test reasoned action execution."""
        action_record = await self.tester._execute_reasoned_action(
            (10, 10), "click", "Test reasoning", self.test_hypothesis, self.test_game_context
        )

        assert isinstance(action_record, ActionRecord)
        assert action_record.coordinates == (10, 10)
        assert action_record.action_type == "click"
        assert action_record.reasoning == "Test reasoning"
        assert action_record.success == True

    def test_determine_experiment_outcome(self):
        """Test experiment outcome determination."""
        # Test successful outcome
        successful_actions = [
            Mock(success=True, score_change=10.0),
            Mock(success=True, score_change=15.0),
            Mock(success=False, score_change=-2.0)
        ]
        outcome = self.tester._determine_experiment_outcome(successful_actions, 23.0)
        assert outcome == TestOutcome.PARTIAL_SUCCESS

        # Test failed outcome
        failed_actions = [
            Mock(success=False, score_change=-10.0),
            Mock(success=False, score_change=-15.0)
        ]
        outcome = self.tester._determine_experiment_outcome(failed_actions, -25.0)
        assert outcome == TestOutcome.FAILURE

        # Test inconclusive outcome
        inconclusive_actions = []
        outcome = self.tester._determine_experiment_outcome(inconclusive_actions, 0.0)
        assert outcome == TestOutcome.INCONCLUSIVE

    def test_generate_learning_insights(self):
        """Test learning insights generation."""
        actions = [
            Mock(success=True, confidence_before=0.8, confidence_after=0.9),
            Mock(success=False, confidence_before=0.7, confidence_after=0.6)
        ]

        insights = self.tester._generate_learning_insights(
            self.test_hypothesis, actions, {"test": "evidence"}, TestOutcome.PARTIAL_SUCCESS
        )

        assert isinstance(insights, list)
        assert len(insights) > 0
        for insight in insights:
            assert isinstance(insight, str)

    def test_analyze_failure_reasons(self):
        """Test failure reason analysis."""
        failed_actions = [
            Mock(success=False, actual_outcome="failed_click_execution", score_change=-5.0),
            Mock(success=False, actual_outcome="failed_drag_execution", score_change=-8.0)
        ]

        reasons = self.tester._analyze_failure_reasons(failed_actions, TestOutcome.FAILURE)

        assert isinstance(reasons, list)
        for reason in reasons:
            assert isinstance(reason, str)

    def test_analyze_success_factors(self):
        """Test success factor analysis."""
        successful_actions = [
            Mock(success=True, action_type="click", score_change=15.0),
            Mock(success=True, action_type="drag", score_change=20.0)
        ]

        factors = self.tester._analyze_success_factors(successful_actions, TestOutcome.SUCCESS)

        assert isinstance(factors, list)
        for factor in factors:
            assert isinstance(factor, str)

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        actions = [
            Mock(success=True, score_change=10.0),
            Mock(success=False, score_change=-5.0)
        ]

        recommendations = self.tester._generate_recommendations(
            self.test_hypothesis, actions, {"test": "evidence"}, TestOutcome.PARTIAL_SUCCESS
        )

        assert isinstance(recommendations, list)
        for recommendation in recommendations:
            assert isinstance(recommendation, str)

    def test_collect_initial_evidence(self):
        """Test initial evidence collection."""
        evidence = self.tester._collect_initial_evidence(self.test_game_context)

        assert isinstance(evidence, dict)
        assert 'initial_score' in evidence
        assert 'game_state' in evidence
        assert 'collection_timestamp' in evidence

    def test_update_confidence(self):
        """Test confidence updating."""
        action_record = Mock(success=True, score_change=15.0)
        new_confidence = self.tester._update_confidence(0.7, action_record, self.test_hypothesis)

        assert isinstance(new_confidence, float)
        assert 0.0 <= new_confidence <= 1.0
        assert new_confidence >= 0.7  # Should increase for successful action

        # Test with failed action
        action_record.success = False
        action_record.score_change = -10.0
        new_confidence = self.tester._update_confidence(0.7, action_record, self.test_hypothesis)
        assert new_confidence <= 0.7  # Should decrease for failed action

    def test_simulate_action_execution(self):
        """Test action execution simulation."""
        outcome, score_change, success = self.tester._simulate_action_execution(
            (10, 10), "click", self.test_hypothesis, self.test_game_context
        )

        assert isinstance(outcome, str)
        assert isinstance(score_change, float)
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_save_experiment_result(self):
        """Test saving experiment results to database."""
        # Create a test experiment result
        actions = [
            ActionRecord(
                action_id="test_action_1",
                action_type="click",
                coordinates=(10, 10),
                timestamp=datetime.now(),
                reasoning="Test reasoning",
                reason_category=ActionReason.HYPOTHESIS_PREDICTION,
                hypothesis_context="test_context",
                expected_outcome="expected",
                actual_outcome="actual",
                confidence_before=0.8,
                confidence_after=0.9,
                score_change=15.0,
                success=True,
                evidence_collected={}
            )
        ]

        experiment_result = ExperimentResult(
            experiment_id="test_experiment_1",
            hypothesis=self.test_hypothesis,
            outcome=TestOutcome.SUCCESS,
            actions_taken=actions,
            total_score_change=15.0,
            test_duration=5.5,
            evidence_collected={},
            confidence_progression=[0.8, 0.9],
            learning_insights=["Test insight"],
            failure_reasons=[],
            success_factors=["Test factor"],
            recommendations=["Test recommendation"],
            timestamp=datetime.now()
        )

        self.tester._save_experiment_result(experiment_result)

        # Verify data was saved
        cursor = self.db_connection.execute(
            "SELECT * FROM hypothesis_test_results WHERE experiment_id = ?",
            (experiment_result.experiment_id,)
        )
        result = cursor.fetchone()
        assert result is not None
        assert result['outcome'] == TestOutcome.SUCCESS.value

        cursor = self.db_connection.execute(
            "SELECT * FROM action_reasoning_log WHERE experiment_id = ?",
            (experiment_result.experiment_id,)
        )
        action_result = cursor.fetchone()
        assert action_result is not None
        assert action_result['action_type'] == "click"


class TestActionRecord:
    """Test ActionRecord dataclass."""

    def test_action_record_creation(self):
        """Test action record creation."""
        action_record = ActionRecord(
            action_id="test_action_record",
            action_type="drag",
            coordinates=(25, 30),
            timestamp=datetime.now(),
            reasoning="Test drag action reasoning",
            reason_category=ActionReason.PATTERN_EXPLORATION,
            hypothesis_context="test_hypothesis_context",
            expected_outcome="successful_drag",
            actual_outcome="successful_drag_execution",
            confidence_before=0.75,
            confidence_after=0.85,
            score_change=12.5,
            success=True,
            evidence_collected={"visual_changes": True}
        )

        assert action_record.action_id == "test_action_record"
        assert action_record.action_type == "drag"
        assert action_record.coordinates == (25, 30)
        assert action_record.reason_category == ActionReason.PATTERN_EXPLORATION
        assert action_record.success == True
        assert action_record.score_change == 12.5


class TestTestOutcome:
    """Test TestOutcome enum."""

    def test_test_outcome_values(self):
        """Test test outcome enum values."""
        for outcome in TestOutcome:
            assert isinstance(outcome.value, str)
            assert len(outcome.value) > 0

    def test_test_outcome_uniqueness(self):
        """Test that all test outcome values are unique."""
        values = [outcome.value for outcome in TestOutcome]
        assert len(values) == len(set(values))


class TestActionReason:
    """Test ActionReason enum."""

    def test_action_reason_values(self):
        """Test action reason enum values."""
        for reason in ActionReason:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0


class TestHypothesisTesterFactory:
    """Test factory functions."""

    def test_create_hypothesis_tester(self):
        """Test factory function."""
        tester = create_hypothesis_tester()
        assert isinstance(tester, HypothesisTester)

    def test_create_hypothesis_tester_with_db(self):
        """Test factory function with database."""
        db_connection = sqlite3.connect(':memory:')
        tester = create_hypothesis_tester(db_connection)
        assert isinstance(tester, HypothesisTester)
        assert tester.db_connection is db_connection

    def test_get_hypothesis_tester_singleton(self):
        """Test singleton pattern."""
        tester1 = get_hypothesis_tester()
        tester2 = get_hypothesis_tester()
        assert tester1 is tester2


class TestExperimentResult:
    """Test ExperimentResult dataclass."""

    def test_experiment_result_creation(self):
        """Test experiment result creation."""
        actions = [Mock()]

        experiment_result = ExperimentResult(
            experiment_id="test_experiment_result",
            hypothesis=self.test_hypothesis,
            outcome=TestOutcome.PARTIAL_SUCCESS,
            actions_taken=actions,
            total_score_change=8.5,
            test_duration=12.3,
            evidence_collected={"evidence": "data"},
            confidence_progression=[0.7, 0.8, 0.75],
            learning_insights=["Insight 1", "Insight 2"],
            failure_reasons=["Reason 1"],
            success_factors=["Factor 1"],
            recommendations=["Recommendation 1"],
            timestamp=datetime.now()
        )

        assert experiment_result.experiment_id == "test_experiment_result"
        assert experiment_result.outcome == TestOutcome.PARTIAL_SUCCESS
        assert len(experiment_result.actions_taken) == 1
        assert experiment_result.total_score_change == 8.5
        assert len(experiment_result.learning_insights) == 2


if __name__ == "__main__":
    pytest.main([__file__])
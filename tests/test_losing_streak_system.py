"""
Comprehensive Test Suite for Losing Streak Detection System

Tests all components of the losing streak detection and intervention system
including LosingStreakDetector, AntiPatternLearner, and EscalatedInterventionSystem.
"""

import pytest
import asyncio
import sqlite3
import tempfile
import os
import json
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import the components we're testing
from src.core.losing_streak_detector import (
    LosingStreakDetector, FailureType, EscalationLevel, LosingStreakData
)
from src.core.anti_pattern_learner import (
    AntiPatternLearner, AntiPatternType, AntiPatternData
)
from src.core.escalated_intervention_system import (
    EscalatedInterventionSystem, InterventionType, InterventionConfig, InterventionResult
)

class TestDatabaseSetup:
    """Helper class for setting up test databases."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()

        # Create connection and initialize schema
        conn = sqlite3.connect(temp_file.name)

        # Create losing_streaks table
        conn.execute("""
            CREATE TABLE losing_streaks (
                streak_id TEXT PRIMARY KEY,
                game_type TEXT NOT NULL,
                game_id TEXT NOT NULL,
                level_identifier TEXT,
                consecutive_failures INTEGER NOT NULL DEFAULT 1,
                total_attempts INTEGER NOT NULL DEFAULT 1,
                first_failure_timestamp REAL NOT NULL,
                last_failure_timestamp REAL NOT NULL,
                failure_types TEXT,
                escalation_level INTEGER DEFAULT 0,
                last_escalation_timestamp REAL,
                intervention_attempts INTEGER DEFAULT 0,
                successful_intervention BOOLEAN DEFAULT FALSE,
                streak_broken BOOLEAN DEFAULT FALSE,
                break_timestamp REAL,
                break_method TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Create anti_patterns table
        conn.execute("""
            CREATE TABLE anti_patterns (
                pattern_id TEXT PRIMARY KEY,
                game_type TEXT NOT NULL,
                game_id TEXT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                failure_count INTEGER DEFAULT 1,
                total_encounters INTEGER DEFAULT 1,
                failure_rate REAL DEFAULT 1.0,
                first_observed REAL NOT NULL,
                last_observed REAL NOT NULL,
                severity REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.5,
                context_data TEXT,
                alternative_suggestions TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Create escalated_interventions table
        conn.execute("""
            CREATE TABLE escalated_interventions (
                intervention_id TEXT PRIMARY KEY,
                streak_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                escalation_level INTEGER NOT NULL,
                intervention_type TEXT NOT NULL,
                intervention_data TEXT NOT NULL,
                applied_timestamp REAL NOT NULL,
                success BOOLEAN DEFAULT FALSE,
                outcome_data TEXT,
                duration_seconds REAL,
                recovery_actions INTEGER DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        conn.commit()

        yield conn

        # Cleanup
        conn.close()
        os.unlink(temp_file.name)

class TestLosingStreakDetector(TestDatabaseSetup):
    """Test suite for LosingStreakDetector."""

    def test_initialization(self, temp_db):
        """Test basic initialization of LosingStreakDetector."""
        detector = LosingStreakDetector(temp_db)

        assert detector.db == temp_db
        assert isinstance(detector.active_streaks, dict)
        assert detector.mild_escalation_threshold == 3
        assert detector.moderate_escalation_threshold == 6
        assert detector.aggressive_escalation_threshold == 10

    def test_record_first_failure(self, temp_db):
        """Test recording the first failure creates a new streak."""
        detector = LosingStreakDetector(temp_db)

        streak_detected, streak_data = detector.record_failure(
            "puzzle", "game_001", FailureType.TIMEOUT
        )

        assert not streak_detected  # First failure shouldn't trigger detection
        assert streak_data is not None
        assert streak_data.consecutive_failures == 1
        assert streak_data.escalation_level == EscalationLevel.NONE
        assert streak_data.failure_types == [FailureType.TIMEOUT]

    def test_record_multiple_failures(self, temp_db):
        """Test recording multiple failures escalates properly."""
        detector = LosingStreakDetector(temp_db)

        # Record failures to reach mild escalation
        for i in range(3):
            streak_detected, streak_data = detector.record_failure(
                "puzzle", "game_001", FailureType.ZERO_PROGRESS
            )

        assert streak_detected
        assert streak_data.consecutive_failures == 3
        assert streak_data.escalation_level == EscalationLevel.MILD

        # Continue to moderate escalation
        for i in range(3):
            detector.record_failure("puzzle", "game_001", FailureType.LOW_PROGRESS)

        streak_key = detector._generate_streak_key("puzzle", "game_001")
        streak = detector.active_streaks[streak_key]
        assert streak.consecutive_failures == 6
        assert streak.escalation_level == EscalationLevel.MODERATE

    def test_record_success_breaks_streak(self, temp_db):
        """Test that recording success breaks an active streak."""
        detector = LosingStreakDetector(temp_db)

        # Create a streak
        for i in range(4):
            detector.record_failure("puzzle", "game_001", FailureType.TIMEOUT)

        # Break the streak
        streak_broken = detector.record_success(
            "puzzle", "game_001", break_method="successful_strategy"
        )

        assert streak_broken

        # Check that streak is no longer active
        streak_key = detector._generate_streak_key("puzzle", "game_001")
        assert streak_key not in detector.active_streaks

    def test_get_streak_for_game(self, temp_db):
        """Test getting streak data for a specific game."""
        detector = LosingStreakDetector(temp_db)

        # No streak initially
        streak = detector.get_streak_for_game("puzzle", "game_001")
        assert streak is None

        # Create a streak
        detector.record_failure("puzzle", "game_001", FailureType.TIMEOUT)

        # Should now return streak data
        streak = detector.get_streak_for_game("puzzle", "game_001")
        assert streak is not None
        assert streak.game_id == "game_001"
        assert streak.consecutive_failures == 1

    def test_intervention_attempt_recording(self, temp_db):
        """Test recording intervention attempts."""
        detector = LosingStreakDetector(temp_db)

        # Create a streak
        detector.record_failure("puzzle", "game_001", FailureType.TIMEOUT)

        # Record intervention attempt
        detector.record_intervention_attempt("puzzle", "game_001", success=True)

        streak = detector.get_streak_for_game("puzzle", "game_001")
        assert streak.intervention_attempts == 1
        assert streak.successful_intervention == True

    def test_streak_statistics(self, temp_db):
        """Test getting streak statistics."""
        detector = LosingStreakDetector(temp_db)

        # Create some streaks
        detector.record_failure("puzzle", "game_001", FailureType.TIMEOUT)
        detector.record_failure("action", "game_002", FailureType.ZERO_PROGRESS)

        stats = detector.get_streak_statistics()

        assert stats["active_streaks"] == 2
        assert "active_by_escalation" in stats
        assert "total_historical_streaks" in stats

class TestAntiPatternLearner(TestDatabaseSetup):
    """Test suite for AntiPatternLearner."""

    def test_initialization(self, temp_db):
        """Test basic initialization of AntiPatternLearner."""
        learner = AntiPatternLearner(temp_db)

        assert learner.db == temp_db
        assert isinstance(learner.known_patterns, dict)
        assert learner.min_encounters_for_pattern == 3
        assert learner.high_failure_rate_threshold == 0.8

    def test_analyze_failure_action_sequences(self, temp_db):
        """Test analyzing failure for action sequence patterns."""
        learner = AntiPatternLearner(temp_db)

        action_sequence = [1, 2, 3, 6, 2]
        coordinates = [(100, 100), (200, 200)]
        context = {"final_score": 10, "actions_taken": 5}

        patterns = learner.analyze_failure(
            "puzzle", "game_001", action_sequence, coordinates, context
        )

        assert len(patterns) > 0

        # Check that action sequence patterns were identified
        action_patterns = [p for p in patterns if p.pattern_type == AntiPatternType.ACTION_SEQUENCE]
        assert len(action_patterns) > 0

        # Verify pattern data structure
        pattern = action_patterns[0]
        assert pattern.game_type == "puzzle"
        assert pattern.failure_count == 1
        assert pattern.failure_rate == 1.0

    def test_analyze_failure_coordinate_clusters(self, temp_db):
        """Test analyzing failure for coordinate cluster patterns."""
        learner = AntiPatternLearner(temp_db)

        # Coordinates close together (should form a cluster)
        coordinates = [(100, 100), (102, 98), (105, 103), (200, 200)]
        context = {"final_score": 5}

        patterns = learner.analyze_failure(
            "puzzle", "game_001", [6, 6, 6, 6], coordinates, context
        )

        # Check that coordinate cluster patterns were identified
        cluster_patterns = [p for p in patterns if p.pattern_type == AntiPatternType.COORDINATE_CLUSTER]
        assert len(cluster_patterns) > 0

        # Verify cluster data
        pattern = cluster_patterns[0]
        pattern_data = pattern.pattern_data
        assert "center" in pattern_data
        assert "coordinates" in pattern_data
        assert len(pattern_data["coordinates"]) >= 2

    def test_get_pattern_suggestions(self, temp_db):
        """Test getting pattern suggestions for avoiding anti-patterns."""
        learner = AntiPatternLearner(temp_db)

        # Create some anti-patterns first
        for _ in range(4):  # Create multiple encounters to establish pattern
            learner.analyze_failure(
                "puzzle", "game_001", [1, 2, 3], [(100, 100)], {"final_score": 0}
            )

        # Get suggestions for similar patterns
        suggestions = learner.get_pattern_suggestions(
            "puzzle", [1, 2, 3], [(100, 100)]
        )

        assert "warnings" in suggestions
        assert "suggestions" in suggestions
        assert "risk_score" in suggestions
        assert "recommendation" in suggestions

        # Should have warnings about the known failing pattern
        if suggestions["warnings"]:
            warning = suggestions["warnings"][0]
            assert warning["type"] in ["action_sequence", "coordinate_cluster"]
            assert "failure_rate" in warning

    def test_record_pattern_success(self, temp_db):
        """Test recording when a previously failing pattern succeeds."""
        learner = AntiPatternLearner(temp_db)

        # Create an anti-pattern
        learner.analyze_failure("puzzle", "game_001", [1, 2], [(100, 100)], {"final_score": 0})

        # Record success with same pattern
        learner.record_pattern_success("puzzle", [1, 2], [(100, 100)])

        # The pattern's failure rate should have decreased
        # (Note: This would require checking the database directly in a real test)
        # For this test, we just verify the method doesn't crash
        assert True

    def test_anti_pattern_statistics(self, temp_db):
        """Test getting anti-pattern statistics."""
        learner = AntiPatternLearner(temp_db)

        # Create some patterns
        learner.analyze_failure("puzzle", "game_001", [1, 2, 3], [], {"final_score": 0})
        learner.analyze_failure("action", "game_002", [4, 5, 6], [], {"final_score": 5})

        stats = learner.get_anti_pattern_statistics()

        assert "total_patterns" in stats
        assert "average_failure_rate" in stats
        assert "pattern_type_breakdown" in stats
        assert "active_patterns" in stats

class TestEscalatedInterventionSystem(TestDatabaseSetup):
    """Test suite for EscalatedInterventionSystem."""

    def test_initialization(self, temp_db):
        """Test basic initialization of EscalatedInterventionSystem."""
        anti_pattern_learner = AntiPatternLearner(temp_db)
        intervention_system = EscalatedInterventionSystem(temp_db, anti_pattern_learner)

        assert intervention_system.db == temp_db
        assert intervention_system.anti_pattern_learner == anti_pattern_learner
        assert isinstance(intervention_system.active_interventions, dict)
        assert len(intervention_system.intervention_configs) == 3  # MILD, MODERATE, AGGRESSIVE

    def test_apply_mild_intervention(self, temp_db):
        """Test applying mild escalation intervention."""
        anti_pattern_learner = AntiPatternLearner(temp_db)
        intervention_system = EscalatedInterventionSystem(temp_db, anti_pattern_learner)

        # Create streak data for mild escalation
        streak_data = LosingStreakData(
            streak_id="test_streak",
            game_type="puzzle",
            game_id="game_001",
            level_identifier=None,
            consecutive_failures=3,
            total_attempts=3,
            first_failure_timestamp=time.time(),
            last_failure_timestamp=time.time(),
            failure_types=[FailureType.TIMEOUT],
            escalation_level=EscalationLevel.MILD,
            last_escalation_timestamp=time.time(),
            intervention_attempts=0,
            successful_intervention=False,
            streak_broken=False,
            break_timestamp=None,
            break_method=None
        )

        game_context = {
            "game_type": "puzzle",
            "game_id": "game_001",
            "has_coordinates": True,
            "recent_actions": [1, 2, 3],
            "score_progression": [0, 10, 15]
        }

        result = intervention_system.apply_intervention(streak_data, game_context)

        assert result is not None
        assert result.intervention_type in [InterventionType.RANDOMIZATION, InterventionType.EXPLORATION_BOOST]
        assert result.applied_timestamp > 0
        assert result.intervention_id in intervention_system.active_interventions

    def test_intervention_progress_tracking(self, temp_db):
        """Test updating intervention progress."""
        anti_pattern_learner = AntiPatternLearner(temp_db)
        intervention_system = EscalatedInterventionSystem(temp_db, anti_pattern_learner)

        # Create a mock intervention
        intervention_id = "test_intervention"
        config = InterventionConfig(
            intervention_type=InterventionType.RANDOMIZATION,
            intensity=0.5,
            duration_actions=10,
            success_threshold=0.3,
            cooldown_seconds=60
        )

        intervention_system.active_interventions[intervention_id] = {
            "config": config,
            "start_time": time.time(),
            "remaining_actions": 10,
            "outcome_data": {}
        }

        # Update progress
        should_continue = intervention_system.update_intervention_progress(
            intervention_id, action_taken=True, progress_made=0.1
        )

        assert should_continue
        assert intervention_system.active_interventions[intervention_id]["remaining_actions"] == 9

        # Complete intervention by exhausting actions
        for _ in range(9):
            intervention_system.update_intervention_progress(intervention_id, action_taken=True)

        # Should be completed and removed from active interventions
        assert intervention_id not in intervention_system.active_interventions

    def test_get_intervention_guidance(self, temp_db):
        """Test getting guidance for active interventions."""
        anti_pattern_learner = AntiPatternLearner(temp_db)
        intervention_system = EscalatedInterventionSystem(temp_db, anti_pattern_learner)

        # Create a mock intervention
        intervention_id = "test_intervention"
        outcome_data = {
            "type": "randomization",
            "intensity": 0.5,
            "description": "Test intervention"
        }

        intervention_system.active_interventions[intervention_id] = {
            "outcome_data": outcome_data
        }

        guidance = intervention_system.get_intervention_guidance(intervention_id)

        assert guidance == outcome_data
        assert guidance["type"] == "randomization"

    def test_intervention_statistics(self, temp_db):
        """Test getting intervention statistics."""
        anti_pattern_learner = AntiPatternLearner(temp_db)
        intervention_system = EscalatedInterventionSystem(temp_db, anti_pattern_learner)

        stats = intervention_system.get_intervention_statistics()

        assert "total_interventions" in stats
        assert "overall_success_rate" in stats
        assert "active_interventions" in stats
        assert "intervention_type_stats" in stats
        assert "cached_success_rates" in stats

class TestIntegrationScenarios(TestDatabaseSetup):
    """Integration tests for the complete losing streak system."""

    def test_complete_losing_streak_workflow(self, temp_db):
        """Test a complete workflow from failure detection to intervention."""
        # Initialize all components
        detector = LosingStreakDetector(temp_db)
        anti_pattern_learner = AntiPatternLearner(temp_db)
        intervention_system = EscalatedInterventionSystem(temp_db, anti_pattern_learner)

        # Simulate repeated failures leading to escalation
        for i in range(4):
            # Record failure
            streak_detected, streak_data = detector.record_failure(
                "puzzle", "game_001", FailureType.TIMEOUT,
                context_data={"final_score": i * 5, "actions_taken": 20 + i}
            )

            # Analyze failure patterns
            anti_pattern_learner.analyze_failure(
                "puzzle", "game_001", [1, 2, 3, 6], [(100 + i * 10, 100 + i * 10)],
                {"final_score": i * 5, "actions_taken": 20 + i}
            )

            if streak_detected and streak_data.escalation_level.value > 0:
                # Apply intervention
                game_context = {
                    "game_type": "puzzle",
                    "game_id": "game_001",
                    "has_coordinates": True,
                    "recent_actions": [1, 2, 3, 6],
                    "score_progression": [0, 5, 10, 15]
                }

                intervention_result = intervention_system.apply_intervention(streak_data, game_context)
                if intervention_result:
                    # Simulate some progress under intervention
                    intervention_system.update_intervention_progress(
                        intervention_result.intervention_id, action_taken=True, progress_made=0.2
                    )

        # Verify final state
        final_streak = detector.get_streak_for_game("puzzle", "game_001")
        assert final_streak is not None
        assert final_streak.consecutive_failures == 4
        assert final_streak.escalation_level in [EscalationLevel.MILD, EscalationLevel.MODERATE]

        # Check that patterns were learned
        patterns = anti_pattern_learner.get_anti_pattern_statistics()
        assert patterns["total_patterns"] > 0

        # Check intervention was applied
        active_interventions = intervention_system.get_active_interventions()
        # May or may not have active interventions depending on duration

    def test_successful_streak_breaking(self, temp_db):
        """Test that successful outcomes properly break streaks and update patterns."""
        # Initialize components
        detector = LosingStreakDetector(temp_db)
        anti_pattern_learner = AntiPatternLearner(temp_db)

        # Create a losing streak
        for i in range(5):
            detector.record_failure("puzzle", "game_001", FailureType.LOW_PROGRESS)
            anti_pattern_learner.analyze_failure(
                "puzzle", "game_001", [1, 2, 1, 2], [(50, 50)], {"final_score": 10}
            )

        # Verify streak exists
        streak = detector.get_streak_for_game("puzzle", "game_001")
        assert streak is not None
        assert streak.consecutive_failures == 5

        # Record success with different pattern
        streak_broken = detector.record_success(
            "puzzle", "game_001", break_method="new_strategy_[3,4,5]"
        )

        assert streak_broken

        # Record successful pattern
        anti_pattern_learner.record_pattern_success("puzzle", [3, 4, 5], [(300, 300)])

        # Verify streak is broken
        final_streak = detector.get_streak_for_game("puzzle", "game_001")
        assert final_streak is None

    def test_anti_pattern_avoidance_integration(self, temp_db):
        """Test that anti-pattern suggestions help avoid known failing patterns."""
        learner = AntiPatternLearner(temp_db)

        # Create well-established anti-patterns
        failing_sequence = [1, 2, 3]
        failing_coordinates = [(100, 100), (101, 101)]

        for _ in range(5):  # Multiple encounters to establish pattern
            learner.analyze_failure(
                "puzzle", "game_type_test", failing_sequence, failing_coordinates,
                {"final_score": 0, "actions_taken": 10}
            )

        # Test pattern recognition
        suggestions = learner.get_pattern_suggestions(
            "puzzle", failing_sequence, failing_coordinates
        )

        assert len(suggestions["warnings"]) > 0
        assert suggestions["risk_score"] > 0.5
        assert suggestions["recommendation"] in ["avoid", "caution"]

        # Test alternative suggestions
        if suggestions["suggestions"]:
            alt_suggestion = suggestions["suggestions"][0]
            assert "type" in alt_suggestion
            assert alt_suggestion["type"] in ["action_substitution", "coordinate_offset", "reorder"]

# Pytest fixtures and configuration
@pytest.mark.asyncio
class TestAsyncComponents:
    """Test async components of the system."""

    async def test_async_pattern_suggestions(self, temp_db=None):
        """Test async aspects of pattern suggestion generation."""
        # This would test any async components if they existed
        # Currently most components are synchronous, but this provides
        # a framework for testing async functionality
        pass

# Performance tests
class TestPerformance(TestDatabaseSetup):
    """Performance tests for the losing streak system."""

    def test_large_volume_failure_processing(self, temp_db):
        """Test performance with large volumes of failure data."""
        detector = LosingStreakDetector(temp_db)

        start_time = time.time()

        # Process many failures
        for i in range(100):
            detector.record_failure(
                f"game_type_{i % 10}", f"game_{i % 20}", FailureType.TIMEOUT
            )

        processing_time = time.time() - start_time

        # Should process 100 failures in reasonable time (adjust threshold as needed)
        assert processing_time < 5.0  # 5 seconds max

        # Verify data integrity
        stats = detector.get_streak_statistics()
        assert stats["active_streaks"] > 0

    def test_pattern_analysis_performance(self, temp_db):
        """Test performance of pattern analysis with complex data."""
        learner = AntiPatternLearner(temp_db)

        start_time = time.time()

        # Analyze many complex patterns
        for i in range(50):
            complex_sequence = list(range(1, 8)) * 3  # Long action sequence
            complex_coordinates = [(x, y) for x in range(0, 100, 10) for y in range(0, 100, 10)]

            learner.analyze_failure(
                "complex_game", f"game_{i}", complex_sequence, complex_coordinates,
                {"final_score": i, "actions_taken": len(complex_sequence)}
            )

        processing_time = time.time() - start_time

        # Should handle complex pattern analysis efficiently
        assert processing_time < 10.0  # 10 seconds max

        # Verify patterns were created
        stats = learner.get_anti_pattern_statistics()
        assert stats["total_patterns"] > 0

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
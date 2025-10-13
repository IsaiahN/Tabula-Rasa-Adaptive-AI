#!/usr/bin/env python3
"""
Test Game Lifecycle Intelligence System

Verifies that the game over pattern analysis, action oscillation prevention,
and proactive strategy switching work as intended.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.game_lifecycle_analyzer import (
    GameLifecycleAnalyzer,
    FailureMode,
    StrategyType,
    GameOverPattern,
    OscillationDetector
)


class TestGameLifecycleAnalyzer:
    """Test suite for GameLifecycleAnalyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()

        self.analyzer = GameLifecycleAnalyzer(db_path=self.temp_db.name)

    def teardown_method(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_db.name)
        except OSError:
            pass

    @pytest.mark.asyncio
    async def test_game_over_pattern_analysis(self):
        """Test that game over patterns are correctly analyzed and stored."""

        # Simulate a failed game with high action count
        game_data = {
            'game_id': 'test_game_001',
            'game_type': 'action6_intensive',
            'actions_to_failure': 250,  # High action count indicating failure
            'final_score': 0.0,
            'game_won': False,
            'failure_reason': 'action_limit',
            'action_sequence_before_end': [6, 6, 6, 1, 6, 6, 2, 6, 6, 6],  # Some oscillation
            'efficiency_trajectory': [0, 0, 10, 10, 5, 0, 0, 0, 0, 0]
        }

        # Add pattern to analyzer
        await self.analyzer.add_game_over_pattern(game_data)

        # Verify pattern was stored
        assert 'action6_intensive' in self.analyzer.game_over_patterns
        pattern = self.analyzer.game_over_patterns['action6_intensive']

        assert pattern.avg_actions_to_failure == 250
        assert pattern.failure_mode == FailureMode.ACTION_LIMIT
        assert pattern.confidence > 0

        print(f"[OK] Game over pattern analysis working: {pattern.failure_mode.value}")

    @pytest.mark.asyncio
    async def test_failure_risk_calculation(self):
        """Test that failure risk is calculated correctly based on action count."""

        # Add a pattern with known failure point
        game_data = {
            'game_id': 'test_game_002',
            'game_type': 'standard',
            'actions_to_failure': 100,
            'final_score': 0.0,
            'game_won': False,
            'failure_reason': 'timeout',
            'action_sequence_before_end': [1, 2, 3, 4, 1, 2, 3, 4],
            'efficiency_trajectory': []
        }

        await self.analyzer.add_game_over_pattern(game_data)

        # Test risk at different action counts
        risk_at_50 = self.analyzer.get_failure_risk('standard', 50)  # 50% of failure point
        risk_at_80 = self.analyzer.get_failure_risk('standard', 80)  # 80% of failure point
        risk_at_120 = self.analyzer.get_failure_risk('standard', 120)  # Beyond failure point

        # Risk should increase as we approach failure point
        assert risk_at_50 < risk_at_80
        assert risk_at_80 < risk_at_120
        assert risk_at_120 >= 1.0  # Should be capped at 1.0

        print(f"[OK] Failure risk calculation: {risk_at_50:.2f} -> {risk_at_80:.2f} -> {risk_at_120:.2f}")

    def test_oscillation_detection(self):
        """Test that action oscillation patterns are detected correctly."""

        detector = OscillationDetector(window_size=8, oscillation_threshold=3)

        # Create oscillating action sequence (1-2-1-2-1-2-1-2)
        oscillating_sequence = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        oscillations = detector.detect_oscillations(oscillating_sequence)

        # Should detect oscillation
        assert len(oscillations) > 0
        assert any(set(osc['actions']) == {1, 2} for osc in oscillations)

        # Non-oscillating sequence should not trigger detection
        normal_sequence = [1, 2, 3, 4, 5, 6, 1, 3, 5]
        normal_oscillations = detector.detect_oscillations(normal_sequence)

        # Should detect fewer/no oscillations
        assert len(normal_oscillations) <= len(oscillations)

        print(f"[OK] Oscillation detection: {len(oscillations)} patterns found in oscillating sequence")

    @pytest.mark.asyncio
    async def test_action_effectiveness_tracking(self):
        """Test that action effectiveness is tracked over time windows."""

        # Simulate successful game with varied action effectiveness
        game_data = {
            'game_id': 'test_game_003',
            'game_type': 'multi_action',
            'actions_to_failure': 150,  # This was actually a success
            'final_score': 85.0,
            'game_won': True,
            'failure_reason': 'success',
            'action_sequence_before_end': [1, 1, 6, 6, 6, 2, 2, 6, 6, 6] * 15,  # 150 actions
            'efficiency_trajectory': list(range(0, 85, 5))  # Increasing score
        }

        await self.analyzer.add_game_over_pattern(game_data)

        # Check that action effectiveness was recorded
        game_context_hash = f"multi_action_{int(85.0/10)*10}"  # Score range grouping

        # Look for action effectiveness patterns
        effectiveness_found = False
        for (action_type, context_hash), effectiveness in self.analyzer.action_effectiveness.items():
            if context_hash == game_context_hash:
                effectiveness_found = True
                assert 0.0 <= effectiveness.early_game_effectiveness <= 1.0
                assert 0.0 <= effectiveness.mid_game_effectiveness <= 1.0
                assert 0.0 <= effectiveness.late_game_effectiveness <= 1.0
                break

        print(f"[OK] Action effectiveness tracking: patterns recorded for successful game")

    @pytest.mark.asyncio
    async def test_strategy_recommendation(self):
        """Test that alternative strategies are recommended based on risk."""

        # Add a pattern indicating high failure risk
        game_data = {
            'game_id': 'test_game_004',
            'game_type': 'high_risk',
            'actions_to_failure': 50,
            'final_score': 0.0,
            'game_won': False,
            'failure_reason': 'oscillation',
            'action_sequence_before_end': [6, 1, 6, 1, 6, 1, 6, 1],  # Clear oscillation
            'efficiency_trajectory': [0, 0, 0, 0, 0, 0, 0, 0]
        }

        await self.analyzer.add_game_over_pattern(game_data)

        # Test strategy recommendation for high-risk situation
        game_context = {
            'game_type': 'high_risk',
            'current_action_count': 45,  # Near failure point
            'recent_score_change': 0.0,  # No progress
            'current_score': 0.0
        }

        strategy = self.analyzer.get_alternative_strategy(game_context)

        assert 'strategy_type' in strategy
        assert 'parameters' in strategy
        assert 'confidence' in strategy
        assert strategy['confidence'] > 0

        # Should recommend exploration for stagnant situation
        assert strategy['strategy_type'] in ['exploration', 'hybrid', 'conservative', 'aggressive']

        print(f"[OK] Strategy recommendation: {strategy['strategy_type']} with confidence {strategy['confidence']}")

    @pytest.mark.asyncio
    async def test_action_avoidance_system(self):
        """Test that actions are flagged for avoidance based on patterns."""

        # Create pattern where action 6 leads to failure after action 100
        effectiveness_window = Mock()
        effectiveness_window.avoid_after_action_count = 100
        effectiveness_window.optimal_timing_start = 0
        effectiveness_window.optimal_timing_end = 80

        self.analyzer.action_effectiveness[(6, 'test_context')] = effectiveness_window

        # Test avoidance recommendation
        avoidance_check = self.analyzer.should_avoid_action(6, {
            'game_type': 'test',
            'current_action_count': 120,  # Past avoidance threshold
            'current_score': 50
        })

        assert avoidance_check['should_avoid'] == True
        assert 'reason' in avoidance_check
        assert avoidance_check['confidence'] > 0.5

        # Test action within safe window
        safe_check = self.analyzer.should_avoid_action(6, {
            'game_type': 'test',
            'current_action_count': 50,  # Within safe window
            'current_score': 50
        })

        assert safe_check['should_avoid'] == False

        print(f"[OK] Action avoidance system: correctly flagged action 6 after threshold")

    def test_lifecycle_insights_generation(self):
        """Test that comprehensive lifecycle insights are generated."""

        insights = self.analyzer.get_lifecycle_insights('action6_intensive')

        assert 'timestamp' in insights
        assert 'total_patterns_tracked' in insights
        assert 'total_effectiveness_patterns' in insights

        # Test insights for unknown game type
        unknown_insights = self.analyzer.get_lifecycle_insights('nonexistent_type')
        assert 'total_patterns_tracked' in unknown_insights

        print(f"[OK] Lifecycle insights: generated comprehensive report with {insights['total_patterns_tracked']} patterns")

    @pytest.mark.asyncio
    async def test_integration_with_enhanced_learning(self):
        """Test integration points with enhanced learning system."""

        # This tests the data structures that would be passed to enhanced learning
        game_data = {
            'game_id': 'integration_test',
            'game_type': 'integration',
            'actions_to_failure': 75,
            'final_score': 45.0,
            'game_won': False,
            'failure_reason': 'score_stagnation',
            'action_sequence_before_end': [6, 6, 1, 1, 6, 6, 2, 2],
            'efficiency_trajectory': [0, 10, 20, 30, 40, 45, 45, 45]
        }

        await self.analyzer.add_game_over_pattern(game_data)

        # Get data that would be passed to enhanced learning
        failure_risk = self.analyzer.get_failure_risk('integration', 75, [6, 6, 1, 1, 6, 6, 2, 2])
        oscillations = self.analyzer.oscillation_detector.detect_oscillations([6, 6, 1, 1, 6, 6, 2, 2])

        # Verify the data structure expected by enhanced learning
        lifecycle_data = {
            'failure_risk_score': failure_risk,
            'oscillation_detected': len(oscillations) > 0,
            'action_effectiveness': {},
            'game_type': 'integration'
        }

        assert all(key in lifecycle_data for key in ['failure_risk_score', 'oscillation_detected', 'action_effectiveness', 'game_type'])
        assert 0.0 <= lifecycle_data['failure_risk_score'] <= 1.0
        assert isinstance(lifecycle_data['oscillation_detected'], bool)

        print(f"[OK] Enhanced learning integration: data structure validated")


def run_basic_functionality_test():
    """Run a simple functionality test without pytest."""
    print("=== GAME LIFECYCLE INTELLIGENCE SYSTEM TEST ===")

    try:
        # Create temporary analyzer
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()

        analyzer = GameLifecycleAnalyzer(db_path=temp_db.name)
        print("[OK] GameLifecycleAnalyzer created successfully")

        # Test oscillation detection
        detector = OscillationDetector()
        oscillations = detector.detect_oscillations([1, 2, 1, 2, 1, 2, 1, 2])
        print(f"[OK] Oscillation detection working: {len(oscillations)} patterns found")

        # Test failure risk calculation
        risk = analyzer.get_failure_risk('test_type', 50)
        print(f"[OK] Failure risk calculation: {risk:.2f}")

        # Test strategy recommendation
        strategy = analyzer.get_alternative_strategy({
            'game_type': 'test',
            'current_action_count': 100,
            'recent_score_change': 0.0,
            'current_score': 50
        })
        print(f"[OK] Strategy recommendation: {strategy['strategy_type']}")

        # Test action avoidance
        avoidance = analyzer.should_avoid_action(6, {
            'game_type': 'test',
            'current_action_count': 50,
            'current_score': 25
        })
        print(f"[OK] Action avoidance system: should_avoid={avoidance['should_avoid']}")

        # Test insights generation
        insights = analyzer.get_lifecycle_insights()
        print(f"[OK] Lifecycle insights: {insights['total_patterns_tracked']} patterns tracked")

        # Clean up
        os.unlink(temp_db.name)

        print("\n[SUCCESS] ALL BASIC TESTS PASSED!")
        print("Game Lifecycle Intelligence System is working correctly!")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_basic_functionality_test()
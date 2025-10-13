"""
Tests for Game Pattern Analyzer

Tests the visual pattern detection and game mechanics analysis capabilities
of the GamePatternAnalyzer component.
"""

import pytest
import numpy as np
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Disable pycache
sys.dont_write_bytecode = True

# Add src to path for imports
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from intelligence.game_pattern_analyzer import (
    GamePatternAnalyzer,
    GameMechanic,
    VisualPattern,
    GameMechanicsProfile,
    get_game_pattern_analyzer,
    create_game_pattern_analyzer
)


class TestGamePatternAnalyzer:
    """Test suite for GamePatternAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = GamePatternAnalyzer()

        # Create test screenshot arrays
        self.simple_grid = np.array([
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0]
        ])

        self.complex_pattern = np.array([
            [1, 1, 2, 2, 3],
            [1, 1, 2, 2, 3],
            [4, 4, 5, 5, 6],
            [4, 4, 5, 5, 6],
            [7, 7, 8, 8, 9]
        ])

        self.empty_grid = np.zeros((10, 10), dtype=int)

    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'pattern_cache')
        assert hasattr(self.analyzer, 'mechanic_patterns')
        assert len(self.analyzer.mechanic_patterns) > 0

    def test_analyze_game_screenshot_simple_grid(self):
        """Test analysis of simple grid pattern."""
        profile = self.analyzer.analyze_game_screenshot(self.simple_grid, "test_game_1")

        assert isinstance(profile, GameMechanicsProfile)
        assert profile.primary_mechanic is not None
        assert isinstance(profile.secondary_mechanics, list)
        assert isinstance(profile.complexity_score, float)
        assert 0 <= profile.complexity_score <= 1
        assert profile.timestamp is not None

    def test_analyze_game_screenshot_complex_pattern(self):
        """Test analysis of complex pattern."""
        profile = self.analyzer.analyze_game_screenshot(self.complex_pattern, "test_game_2")

        assert isinstance(profile, GameMechanicsProfile)
        assert profile.complexity_score > 0
        assert len(profile.visual_patterns) >= 0
        assert len(profile.grid_features) > 0

    def test_analyze_game_screenshot_empty_grid(self):
        """Test analysis of empty grid."""
        profile = self.analyzer.analyze_game_screenshot(self.empty_grid, "test_game_3")

        assert isinstance(profile, GameMechanicsProfile)
        assert profile.primary_mechanic is not None
        # Empty grid should have low complexity
        assert profile.complexity_score < 0.5

    def test_caching_mechanism(self):
        """Test that caching works correctly."""
        game_id = "cache_test_game"

        # First analysis
        profile1 = self.analyzer.analyze_game_screenshot(self.simple_grid, game_id)

        # Second analysis should use cache
        profile2 = self.analyzer.analyze_game_screenshot(self.simple_grid, game_id)

        assert profile1 is profile2  # Should be same object from cache

    def test_extract_grid_features(self):
        """Test grid feature extraction."""
        features = self.analyzer._extract_grid_features(self.simple_grid)

        assert isinstance(features, dict)
        assert 'grid_detected' in features
        assert 'grid_size' in features
        assert 'cell_uniformity' in features
        assert 'color_count' in features
        assert 'shape_count' in features

    def test_detect_visual_patterns(self):
        """Test visual pattern detection."""
        patterns = self.analyzer._detect_visual_patterns(self.complex_pattern)

        assert isinstance(patterns, list)
        for pattern in patterns:
            assert isinstance(pattern, VisualPattern)
            assert hasattr(pattern, 'pattern_id')
            assert hasattr(pattern, 'pattern_type')
            assert hasattr(pattern, 'confidence')
            assert hasattr(pattern, 'location')

    def test_classify_game_mechanics(self):
        """Test game mechanics classification."""
        grid_features = {'grid_detected': True, 'shape_count': 5, 'color_count': 3}
        visual_patterns = []

        mechanic_scores = self.analyzer._classify_game_mechanics(grid_features, visual_patterns)

        assert isinstance(mechanic_scores, dict)
        assert len(mechanic_scores) == len(GameMechanic)

        for mechanic, score in mechanic_scores.items():
            assert isinstance(mechanic, GameMechanic)
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_get_hypothesis_indicators(self):
        """Test hypothesis indicator generation."""
        profile = self.analyzer.analyze_game_screenshot(self.complex_pattern, "indicator_test")
        indicators = self.analyzer.get_hypothesis_indicators(profile)

        assert isinstance(indicators, dict)
        assert 'primary_mechanic' in indicators
        assert 'complexity_level' in indicators
        assert 'suggested_approaches' in indicators
        assert 'visual_patterns' in indicators

    @patch('cv2.Canny')
    @patch('cv2.HoughLines')
    def test_detect_grid_structure_mock(self, mock_hough_lines, mock_canny):
        """Test grid structure detection with mocked CV2."""
        # Mock CV2 functions
        mock_canny.return_value = np.zeros((10, 10), dtype=np.uint8)
        mock_hough_lines.return_value = np.array([[[0, 0]], [[1, np.pi/2]]])

        result = self.analyzer._detect_grid_structure(self.simple_grid)
        assert isinstance(result, bool)

    def test_count_unique_colors(self):
        """Test unique color counting."""
        # Test grayscale image
        gray_count = self.analyzer._count_unique_colors(self.simple_grid)
        assert isinstance(gray_count, int)
        assert gray_count >= 1

        # Test RGB image
        rgb_image = np.stack([self.simple_grid, self.simple_grid, self.simple_grid], axis=-1)
        rgb_count = self.analyzer._count_unique_colors(rgb_image)
        assert isinstance(rgb_count, int)
        assert rgb_count >= 1

    def test_detect_symmetry(self):
        """Test symmetry detection."""
        symmetry = self.analyzer._detect_symmetry(self.simple_grid)

        assert isinstance(symmetry, dict)
        assert 'horizontal' in symmetry
        assert 'vertical' in symmetry
        assert 'diagonal' in symmetry

        for symmetry_type, detected in symmetry.items():
            assert isinstance(detected, bool)

    def test_pattern_supports_mechanic(self):
        """Test pattern-mechanic support checking."""
        pattern = VisualPattern(
            pattern_id="test_pattern",
            pattern_type="rectangle",
            confidence=0.8,
            location=(10, 10),
            size=(5, 5),
            features={},
            timestamp=datetime.now()
        )

        supports_spatial = self.analyzer._pattern_supports_mechanic(pattern, GameMechanic.SPATIAL_PUZZLE)
        assert isinstance(supports_spatial, bool)

        supports_navigation = self.analyzer._pattern_supports_mechanic(pattern, GameMechanic.NAVIGATION_PATHFINDING)
        assert isinstance(supports_navigation, bool)

    def test_calculate_complexity_score(self):
        """Test complexity score calculation."""
        grid_features = {
            'shape_complexity': 0.5,
            'shape_count': 8,
            'color_count': 4
        }
        visual_patterns = [Mock() for _ in range(3)]

        complexity = self.analyzer._calculate_complexity_score(grid_features, visual_patterns)

        assert isinstance(complexity, float)
        assert 0 <= complexity <= 1

    def test_suggest_approaches(self):
        """Test approach suggestion."""
        profile = Mock()
        profile.primary_mechanic = GameMechanic.PATTERN_COMPLETION
        profile.secondary_mechanics = [GameMechanic.COLOR_MATCHING]

        approaches = self.analyzer._suggest_approaches(profile)

        assert isinstance(approaches, list)
        assert len(approaches) > 0
        for approach in approaches:
            assert isinstance(approach, str)


class TestGamePatternAnalyzerFactory:
    """Test factory functions."""

    def test_create_game_pattern_analyzer(self):
        """Test factory function."""
        analyzer = create_game_pattern_analyzer()
        assert isinstance(analyzer, GamePatternAnalyzer)

    def test_get_game_pattern_analyzer_singleton(self):
        """Test singleton pattern."""
        analyzer1 = get_game_pattern_analyzer()
        analyzer2 = get_game_pattern_analyzer()
        assert analyzer1 is analyzer2


class TestGameMechanic:
    """Test GameMechanic enum."""

    def test_game_mechanic_values(self):
        """Test that all game mechanics have valid values."""
        for mechanic in GameMechanic:
            assert isinstance(mechanic.value, str)
            assert len(mechanic.value) > 0

    def test_game_mechanic_uniqueness(self):
        """Test that all game mechanic values are unique."""
        values = [mechanic.value for mechanic in GameMechanic]
        assert len(values) == len(set(values))


class TestVisualPattern:
    """Test VisualPattern dataclass."""

    def test_visual_pattern_creation(self):
        """Test visual pattern creation."""
        pattern = VisualPattern(
            pattern_id="test_pattern_1",
            pattern_type="circle",
            confidence=0.85,
            location=(15, 20),
            size=(10, 10),
            features={"radius": 5},
            timestamp=datetime.now()
        )

        assert pattern.pattern_id == "test_pattern_1"
        assert pattern.pattern_type == "circle"
        assert pattern.confidence == 0.85
        assert pattern.location == (15, 20)
        assert pattern.size == (10, 10)
        assert pattern.features["radius"] == 5


if __name__ == "__main__":
    pytest.main([__file__])
"""
Test Action 6 Exploration and Mapping Features

Tests the new exploration capabilities integrated into Action6Coordinator:
- Intelligent surveying system
- Boundary detection and mapping
- Strategic coordinate selection
- Quadrant exploration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.gameplay.action6_coordinator import Action6Coordinator


class TestAction6Exploration(unittest.TestCase):
    """Test Action 6 exploration and mapping features."""

    def setUp(self):
        """Set up test environment."""
        self.db_interface = Mock()
        self.vision_detector = Mock()
        self.coordinator = Action6Coordinator(
            db_interface=self.db_interface,
            vision_detector=self.vision_detector
        )

    def test_boundary_system_initialization(self):
        """Test boundary system initialization."""
        game_id = "test_game_001"

        # Initialize boundary system
        boundary_system = self.coordinator._ensure_boundary_system_initialized(game_id)

        # Check system structure
        self.assertIsInstance(boundary_system, dict)
        self.assertIn('boundary_data', boundary_system)
        self.assertIn('coordinate_attempts', boundary_system)
        self.assertIn('success_zone_mapping', boundary_system)
        self.assertIn('safe_regions', boundary_system)
        self.assertIn('directional_systems', boundary_system)

        # Check game-specific initialization
        self.assertIn(game_id, boundary_system['boundary_data'])
        self.assertIn(game_id, boundary_system['coordinate_attempts'])

        # Check directional system for Action 6
        self.assertIn(6, boundary_system['directional_systems'])
        self.assertIn(game_id, boundary_system['directional_systems'][6]['current_direction'])
        self.assertEqual(boundary_system['directional_systems'][6]['current_direction'][game_id], 'right')

    def test_strategic_coordinate_generation(self):
        """Test strategic coordinate generation for exploration."""
        game_id = "test_game_002"
        grid_dims = (20, 15)  # 20 width, 15 height

        # Test initial exploration (no safe regions)
        coords = self.coordinator._get_strategic_action6_coordinates(grid_dims, game_id)

        # Should return valid coordinates within grid bounds
        self.assertIsInstance(coords, tuple)
        self.assertEqual(len(coords), 2)
        x, y = coords
        self.assertGreaterEqual(x, 0)
        self.assertLess(x, grid_dims[0])
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, grid_dims[1])

    def test_survey_target_validation(self):
        """Test survey target validation logic."""
        known_boundaries = {(5, 5), (10, 10), (15, 5)}

        # Test good survey targets (far from boundaries)
        self.assertTrue(self.coordinator._is_good_survey_target((0, 0), known_boundaries, min_distance=3))
        self.assertTrue(self.coordinator._is_good_survey_target((20, 20), known_boundaries, min_distance=3))

        # Test bad survey targets (too close to boundaries)
        self.assertFalse(self.coordinator._is_good_survey_target((5, 7), known_boundaries, min_distance=3))
        self.assertFalse(self.coordinator._is_good_survey_target((12, 10), known_boundaries, min_distance=3))

    def test_intelligent_survey_target_selection(self):
        """Test intelligent survey target selection with safe regions."""
        game_id = "test_game_003"
        grid_dims = (30, 25)
        known_boundaries = {(5, 5)}

        # Mock safe regions
        safe_regions = {
            'region_1': {
                'coordinates': [(10, 10), (11, 10), (12, 10), (10, 11), (11, 11), (12, 11)],
                'center': (11, 10),
                'safety_score': 0.8
            }
        }

        # Test survey target selection
        target = self.coordinator._get_intelligent_survey_target(
            game_id, grid_dims, known_boundaries, safe_regions
        )

        if target:  # May return None if no good targets
            self.assertIsInstance(target, tuple)
            self.assertEqual(len(target), 3)  # x, y, reason
            x, y, reason = target
            self.assertIsInstance(x, int)
            self.assertIsInstance(y, int)
            self.assertIsInstance(reason, str)

            # Should be within grid bounds
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, grid_dims[0])
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, grid_dims[1])

    def test_stagnation_detection_and_recovery(self):
        """Test coordinate stagnation detection and emergency jumps."""
        game_id = "test_game_004"
        grid_dims = (20, 15)

        # Initialize boundary system
        self.coordinator._ensure_boundary_system_initialized(game_id)

        # Simulate coordinate stagnation
        coord_key = (10, 10)
        self.coordinator.boundary_system['coordinate_attempts'][game_id][coord_key] = {
            'attempts': 15,
            'consecutive_stuck': 15  # Above threshold
        }

        # This should trigger emergency jump
        coords = self.coordinator._get_strategic_action6_coordinates(grid_dims, game_id)

        # Should return valid coordinates (different from stuck coordinates)
        self.assertIsInstance(coords, tuple)
        self.assertNotEqual(coords, coord_key)  # Should jump away from stuck position

    def test_exploration_integration_in_main_method(self):
        """Test exploration integration in the main coordinate selection method."""
        game_id = "test_game_005"
        frame = [[1, 2, 3] * 10 for _ in range(8)]  # 10x8 grid
        context = {'available_actions': ['action6']}

        # Mock vision detector to return no pseudo-buttons
        self.vision_detector.detect_pseudo_buttons = MagicMock(return_value=[])

        # Mock penalty system and transfer learning to avoid dependencies
        async def mock_ensure_penalty_ready():
            pass

        async def mock_get_fallback_coords(frame, game_id, context):
            return (5, 5)

        with patch.object(self.coordinator, '_ensure_penalty_system_ready', side_effect=mock_ensure_penalty_ready):
            with patch.object(self.coordinator, '_get_fallback_coordinates', side_effect=mock_get_fallback_coords):
                # Test that exploration mode is used when no pseudo-buttons are found
                import asyncio
                coords = asyncio.run(
                    self.coordinator.get_optimal_action6_coordinates(frame, game_id, context)
                )

                # Should return valid coordinates
                self.assertIsInstance(coords, tuple)
                self.assertEqual(len(coords), 2)

    def test_exploration_statistics(self):
        """Test exploration statistics tracking."""
        # Simulate some exploration selections
        self.coordinator.stats['exploration_selections'] = 5
        self.coordinator.stats['action6_selections'] = 20

        # Mock some boundary data
        self.coordinator.boundary_system = {
            'boundary_data': {
                'game1': {(0, 0): {}, (5, 5): {}},
                'game2': {(10, 10): {}}
            }
        }
        self.coordinator.game_sessions = {
            'game1': {
                'tried_pseudo_buttons': [(1, 1), (2, 2)],
                'successful_sequences': [[(1, 1)]],
                'discovered_buttons': [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}]
            },
            'game2': {
                'tried_pseudo_buttons': [(3, 3)],
                'successful_sequences': [],
                'discovered_buttons': [{'x': 3, 'y': 3}]
            }
        }

        stats = self.coordinator.get_statistics()

        # Check exploration statistics
        self.assertEqual(stats['exploration_selections'], 5)
        self.assertEqual(stats['exploration_usage_rate'], 0.25)  # 5/20
        self.assertEqual(stats['total_boundaries_mapped'], 3)  # 2 + 1 boundaries

    def test_area_stagnation_detection(self):
        """Test detection of area-based stagnation (back and forth in small region)."""
        game_id = "test_game_006"

        # Simulate coordinates confined to a small area (2x2 area)
        session = self.coordinator._get_or_create_session(game_id)
        session['tried_pseudo_buttons'] = [
            (10, 10), (11, 10), (10, 11), (11, 11),  # 2x2 area
            (10, 10), (11, 10), (10, 11), (11, 11),  # Repeat same area
            (10, 10), (11, 10)  # More repetition
        ]

        # Should detect area stagnation
        area_stuck = self.coordinator._detect_area_stagnation(game_id)
        self.assertTrue(area_stuck)

        # Test with coordinates spread over larger area - should NOT detect stagnation
        session['tried_pseudo_buttons'] = [
            (5, 5), (15, 5), (5, 15), (15, 15),    # Corners of large area
            (10, 10), (20, 20), (0, 0), (25, 25)  # Spread out
        ]

        area_stuck = self.coordinator._detect_area_stagnation(game_id)
        self.assertFalse(area_stuck)

    def test_forced_exploration_activation(self):
        """Test that exploration is forced when area stagnation is detected."""
        game_id = "test_game_007"

        # Setup area stagnation scenario
        session = self.coordinator._get_or_create_session(game_id)
        session['tried_pseudo_buttons'] = [
            (5, 5), (6, 5), (5, 6), (6, 6),  # Small 2x2 area
            (5, 5), (6, 5), (5, 6), (6, 6),  # Repeat
            (5, 5), (6, 5)  # More repetition
        ]

        # Should force exploration
        should_force = self.coordinator._should_force_exploration(game_id)
        self.assertTrue(should_force)

        # Test coordinate repetition detection
        session['tried_pseudo_buttons'] = [
            (10, 10), (10, 10), (11, 11), (10, 10),  # Only 2 unique coords
            (11, 11), (10, 10), (11, 11), (10, 10)   # In 8 moves
        ]

        should_force = self.coordinator._should_force_exploration(game_id)
        self.assertTrue(should_force)

    def test_proactive_exploration_integration(self):
        """Test that exploration is used proactively in main coordinate selection."""
        game_id = "test_game_008"
        frame = [[1, 2, 3] * 10 for _ in range(8)]  # 10x8 grid
        context = {'available_actions': ['action6']}

        # Setup area stagnation
        session = self.coordinator._get_or_create_session(game_id)
        session['tried_pseudo_buttons'] = [
            (2, 2), (3, 2), (2, 3), (3, 3),  # Small area
            (2, 2), (3, 2), (2, 3), (3, 3),  # Repeat
            (2, 2), (3, 2)  # More repetition
        ]

        # Mock vision detector to return pseudo-buttons (which would normally be used)
        self.vision_detector.detect_pseudo_buttons = MagicMock(return_value=[
            {'x': 2, 'y': 2, 'confidence': 0.8, 'type': 'button'}
        ])

        # Mock penalty system and transfer learning to avoid dependencies
        async def mock_ensure_penalty_ready():
            pass

        with patch.object(self.coordinator, '_ensure_penalty_system_ready', side_effect=mock_ensure_penalty_ready):
            # Despite pseudo-buttons being available, should force exploration due to area stagnation
            import asyncio
            coords = asyncio.run(
                self.coordinator.get_optimal_action6_coordinates(frame, game_id, context)
            )

            # Should return valid coordinates (from exploration, not pseudo-buttons)
            self.assertIsInstance(coords, tuple)
            self.assertEqual(len(coords), 2)

            # Should have incremented forced exploration stats
            stats = self.coordinator.get_statistics()
            self.assertGreater(stats.get('forced_explorations', 0), 0)

    def tearDown(self):
        """Clean up test environment."""
        # Clear any instance variables that might affect other tests
        if hasattr(self.coordinator, 'boundary_system'):
            delattr(self.coordinator, 'boundary_system')


if __name__ == '__main__':
    # Ensure pycache is disabled
    import sys
    sys.dont_write_bytecode = True

    unittest.main()
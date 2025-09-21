#!/usr/bin/env python3
"""
Comprehensive test suite for the Coordinate Intelligence System.
"""

import unittest
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
import tempfile
import os
import shutil

# Import the coordinate intelligence system
from src.core.coordinate_intelligence_system import (
    CoordinateIntelligenceSystem, SuccessZoneMapper, CoordinateZone,
    ZoneType, ZoneConfidence, CoordinateIntelligence, create_coordinate_intelligence_system
)


class TestCoordinateZone(unittest.TestCase):
    """Test the CoordinateZone data class."""
    
    def test_coordinate_zone_creation(self):
        """Test creating a coordinate zone."""
        zone = CoordinateZone(
            zone_id="test_zone",
            center=(10, 15),
            coordinates={(10, 15), (11, 16)},
            zone_type=ZoneType.SUCCESS,
            confidence=ZoneConfidence.HIGH,
            success_rate=0.85,
            total_attempts=100,
            successful_actions={1, 2, 3},
            failure_actions={4, 5},
            last_updated=1234567890
        )
        
        self.assertEqual(zone.zone_id, "test_zone")
        self.assertEqual(zone.center, (10, 15))
        self.assertEqual(zone.zone_type, ZoneType.SUCCESS)
        self.assertEqual(zone.confidence, ZoneConfidence.HIGH)
        self.assertEqual(zone.success_rate, 0.85)
        self.assertEqual(zone.total_attempts, 100)
        self.assertEqual(zone.last_updated, 1234567890)
    
    def test_coordinate_zone_attributes(self):
        """Test coordinate zone attributes."""
        zone = CoordinateZone(
            zone_id="test_zone2",
            center=(5, 8),
            coordinates={(5, 8), (6, 9)},
            zone_type=ZoneType.FAILURE,
            confidence=ZoneConfidence.MEDIUM,
            success_rate=0.2,
            total_attempts=50,
            successful_actions={1},
            failure_actions={2, 3, 4},
            last_updated=1234567890
        )
        
        # Test that all attributes are accessible
        self.assertEqual(zone.zone_id, "test_zone2")
        self.assertEqual(zone.center, (5, 8))
        self.assertEqual(zone.zone_type, ZoneType.FAILURE)
        self.assertEqual(zone.confidence, ZoneConfidence.MEDIUM)
        self.assertEqual(zone.success_rate, 0.2)
        self.assertEqual(zone.total_attempts, 50)
        self.assertEqual(zone.last_updated, 1234567890)


class TestSuccessZoneMapper(unittest.TestCase):
    """Test the SuccessZoneMapper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mapper = SuccessZoneMapper()
    
    def test_initialization(self):
        """Test mapper initialization."""
        self.assertIsInstance(self.mapper.zones, dict)
        self.assertEqual(self.mapper.min_zone_size, 3)
        self.assertEqual(self.mapper.max_zone_radius, 5)
    
    def test_update_coordinate_intelligence(self):
        """Test updating coordinate intelligence."""
        # Update coordinate intelligence
        self.mapper.update_coordinate_intelligence(
            "test_game", 10, 15, 1, True, 0.9
        )
        self.mapper.update_coordinate_intelligence(
            "test_game", 11, 16, 2, True, 0.8
        )
        
        # Check that zones were created
        self.assertIn("test_game", self.mapper.zones)
        zones = self.mapper.zones["test_game"]
        self.assertGreater(len(zones), 0)
    
    def test_get_recommended_coordinates(self):
        """Test getting recommended coordinates."""
        # Add some coordinate intelligence first
        self.mapper.update_coordinate_intelligence(
            "test_game", 10, 15, 1, True, 0.9
        )
        self.mapper.update_coordinate_intelligence(
            "test_game", 20, 25, 2, True, 0.8
        )
        
        # Get recommendations
        recommendations = self.mapper.get_recommended_coordinates(
            "test_game", 1, (30, 30), preference="success"
        )
        
        self.assertIsInstance(recommendations, list)


class TestCoordinateIntelligenceSystem(unittest.TestCase):
    """Test the main CoordinateIntelligenceSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = CoordinateIntelligenceSystem()
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system.zone_mapper)
        self.assertIsInstance(self.system.coordinate_intelligence, dict)
        self.assertTrue(self.system.cross_game_learning)
        self.assertEqual(self.system.learning_rate, 0.1)
    
    def test_update_coordinate_intelligence(self):
        """Test updating coordinate intelligence."""
        # Update coordinate intelligence
        self.system.update_coordinate_intelligence(
            "test_game", 10, 15, 1, True, 0.9
        )
        self.system.update_coordinate_intelligence(
            "test_game", 11, 16, 2, True, 0.8
        )
        
        # Check that coordinate intelligence was updated
        self.assertGreater(len(self.system.coordinate_intelligence), 0)
        # Check that we have coordinate intelligence for the specific coordinates
        key1 = ("test_game", 10, 15)
        key2 = ("test_game", 11, 16)
        self.assertIn(key1, self.system.coordinate_intelligence)
        self.assertIn(key2, self.system.coordinate_intelligence)
    
    def test_get_recommended_coordinates(self):
        """Test getting coordinate recommendations."""
        # Add some coordinate intelligence first
        self.system.update_coordinate_intelligence(
            "test_game", 10, 15, 1, True, 0.9
        )
        self.system.update_coordinate_intelligence(
            "test_game", 20, 25, 2, True, 0.8
        )
        
        # Get recommendations
        recommendations = self.system.get_recommended_coordinates(
            "test_game", 1, (30, 30), preference="success"
        )
        
        self.assertIsInstance(recommendations, list)
    
    def test_get_zone_statistics(self):
        """Test getting zone statistics."""
        # Add some coordinate intelligence first
        self.system.update_coordinate_intelligence(
            "test_game", 10, 15, 1, True, 0.9
        )
        self.system.update_coordinate_intelligence(
            "test_game", 5, 8, 2, False, 0.1
        )
        
        # Get statistics
        stats = self.system.get_system_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_coordinates_tracked', stats)
        self.assertIn('total_updates', stats)
        self.assertIn('zone_statistics', stats)




if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
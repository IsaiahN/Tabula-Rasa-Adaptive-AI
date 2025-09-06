#!/usr/bin/env python3
"""
Test script for the coordinate-aware ARC integration system.
Validates that all components work together properly.
"""
import asyncio
import sys
import os
from pathlib import Path

import pytest
import numpy as np

from arc_integration.coordinate_aware_integration import CoordinateAwareTrainingManager
from vision.frame_analyzer import FrameAnalyzer
from learning.pathway_system import PathwayLearningSystem
from api.enhanced_client import ArcAgiApiClient


def test_coordinate_system_imports():
    """Test that all coordinate system components can be imported."""
    print("🧪 Testing coordinate system imports...")
    
    try:
        # Test frame analyzer
        frame_analyzer = FrameAnalyzer()
        print("✅ FrameAnalyzer imported successfully")
        
        # Test pathway system
        pathway_system = PathwayLearningSystem()
        print("✅ PathwayLearningSystem imported successfully")
        
        # Test API client (without API key)
        try:
            api_client = ArcAgiApiClient("test_key")
            print("✅ ArcAgiApiClient imported successfully")
        except Exception as e:
            print(f"⚠️ ArcAgiApiClient import warning: {e}")
        
        # Test training manager (without API key)
        try:
            manager = CoordinateAwareTrainingManager("test_key")
            print("✅ CoordinateAwareTrainingManager imported successfully")
        except Exception as e:
            print(f"⚠️ CoordinateAwareTrainingManager import warning: {e}")
        print("✅ All coordinate system imports successful!")
        assert True
    except Exception as e:
        pytest.fail(f"Import test failed: {e}")


def test_coordinate_system_components():
    """Test individual component functionality."""
    print("\n🧪 Testing coordinate system components...")
    
    try:
        # Test FrameAnalyzer
        frame_analyzer = FrameAnalyzer()
        # Use simple 2D integer grid (color indices) instead of 3-channel tuples
        test_frame = np.array([[1 for _ in range(10)] for _ in range(10)], dtype=int)
        analysis = frame_analyzer.analyze_frame(test_frame)
        print(f"✅ FrameAnalyzer analysis: {type(analysis)}")
        
        # Test PathwayLearningSystem
        pathway_system = PathwayLearningSystem()
        pathway_system.track_action(
            action=6,
            action_data={'coordinates': (32, 32)},
            score_before=0,
            score_after=10,
            win_score=100,
            game_id="test_game"
        )
        recommendations = pathway_system.get_pathway_recommendations([1, 2, 3, 4, 5, 6, 7], 0, "test_game")
        print(f"✅ PathwayLearningSystem recommendations: {type(recommendations)}")
        
        # Test coordinate manager
        from src.api.enhanced_client import CoordinateManager
        coord_manager = CoordinateManager()
        center_coords = coord_manager.get_center_coordinates()
        print(f"✅ CoordinateManager center coordinates: {center_coords}")
        
        print("✅ All component tests successful!")
        assert True
    except Exception as e:
        import traceback
        traceback.print_exc()
        pytest.fail(f"Component test failed: {e}")


@pytest.mark.asyncio
async def test_integration_without_api():
    """Test integration components without requiring API access."""
    print("\n🧪 Testing integration without API...")
    
    try:
        # Create mock frame data
        mock_frame_data = {
            'frame': np.array([[1 if i == j else 0 for j in range(10)] for i in range(10)], dtype=int),
            'score': 50,
            'state': 'PLAYING'
        }
        
        # Test frame analysis
        frame_analyzer = FrameAnalyzer()
        analysis = frame_analyzer.analyze_frame(mock_frame_data['frame'])
        print(f"✅ Frame analysis completed: movement_detected={analysis.get('movement_detected', False)}")
        
        # Test pathway learning
        pathway_system = PathwayLearningSystem()
        for i in range(5):
            pathway_system.track_action(
                action=i+1,
                action_data={'coordinates': (32 + i, 32 + i)},
                score_before=i * 10,
                score_after=(i + 1) * 10,
                win_score=100,
                game_id="integration_test"
            )
        
        recommendations = pathway_system.get_pathway_recommendations([1, 2, 3, 4, 5, 6, 7], 50, "integration_test")
        print(f"✅ Pathway recommendations generated: {len(recommendations)} actions")
        
        # Test coordinate selection
        from src.api.enhanced_client import CoordinateManager
        coord_manager = CoordinateManager()
        
        # Test center coordinates
        coords = coord_manager.get_center_coordinates()
        print(f"✅ center_start coordinates: {coords}")
        
        # Test corner coordinates
        coords = coord_manager.get_corner_coordinates()
        print(f"✅ corner_exploration coordinates: {coords}")
        
        # Test strategic positions
        coords = coord_manager.get_strategic_positions()
        print(f"✅ strategic positions coordinates: {coords[:3]}")  # Show first 3
        
        print("✅ Integration test successful!")
        assert True
    except Exception as e:
        import traceback
        traceback.print_exc()
        pytest.fail(f"Integration test failed: {e}")


def main():
    """Run all coordinate system tests."""
    print("🚀 Starting Coordinate System Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Imports
    try:
        test_coordinate_system_imports()
        tests_passed += 1
    except Exception:
        pass

    # Test 2: Components
    try:
        test_coordinate_system_components()
        tests_passed += 1
    except Exception:
        pass

    # Test 3: Integration
    try:
        asyncio.run(test_integration_without_api())
        tests_passed += 1
    except Exception:
        pass
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All coordinate system tests passed!")
        print("✅ System is ready for coordinate-aware training")
        assert True
    else:
        print("❌ Some tests failed. Check error messages above.")
        pytest.fail(f"{total_tests - tests_passed} coordinate system tests failed")


if __name__ == "__main__":
    main()

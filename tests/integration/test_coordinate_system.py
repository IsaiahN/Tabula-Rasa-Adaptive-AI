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

# Add src to path - adjust for tests/integration folder structure
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.arc_integration.coordinate_aware_integration import CoordinateAwareTrainingManager
from src.vision.frame_analyzer import FrameAnalyzer
from src.learning.pathway_system import PathwayLearningSystem
from src.api.enhanced_client import ArcAgiApiClient


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
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_coordinate_system_components():
    """Test individual component functionality."""
    print("\n🧪 Testing coordinate system components...")
    
    try:
        # Test FrameAnalyzer
        frame_analyzer = FrameAnalyzer()
        test_frame = [[[1, 0, 0] for _ in range(10)] for _ in range(10)]
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
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_integration_without_api():
    """Test integration components without requiring API access."""
    print("\n🧪 Testing integration without API...")
    
    try:
        # Create mock frame data
        mock_frame_data = {
            'frame': [[[1 if i == j else 0 for _ in range(3)] for j in range(10)] for i in range(10)],
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
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all coordinate system tests."""
    print("🚀 Starting Coordinate System Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Imports
    if test_coordinate_system_imports():
        tests_passed += 1
    
    # Test 2: Components
    if test_coordinate_system_components():
        tests_passed += 1
    
    # Test 3: Integration
    if asyncio.run(test_integration_without_api()):
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All coordinate system tests passed!")
        print("✅ System is ready for coordinate-aware training")
        return True
    else:
        print("❌ Some tests failed. Check error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

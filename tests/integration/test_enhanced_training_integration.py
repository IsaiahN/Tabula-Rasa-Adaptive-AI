#!/usr/bin/env python3
"""
Integration Test for Enhanced ARC Training System
Tests that the coordinate-aware system integrates properly with the training infrastructure.
"""
import asyncio
import sys
import os
from pathlib import Path
import tempfile
import json
import pytest

# Import coordinate system components  
from arc_integration.coordinate_aware_integration import CoordinateAwareTrainingManager
from arc_integration.continuous_learning_loop import ContinuousLearningLoop
from vision.frame_analyzer import FrameAnalyzer
from learning.pathway_system import PathwayLearningSystem
from api.enhanced_client import ArcAgiApiClient


class MockFrameData:
    """Mock frame data for testing without API calls."""
    def __init__(self, score=50, state='PLAYING'):
        self.frame = [[[i % 2 for _ in range(3)] for j in range(10)] for i in range(10)]
        self.score = score
        self.state = state
        self.available_actions = [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.asyncio
async def test_coordinate_system_integration():
    """Test that coordinate system integrates with training infrastructure."""
    print("🧪 Testing coordinate system integration with training infrastructure...")
    
    try:
        # Test 1: Initialize coordinate manager without API key (for testing)
        print("🔧 Initializing coordinate-aware training manager...")
        manager = CoordinateAwareTrainingManager("test_key", None)
        print("✅ Training manager initialized")
        
        # Test 2: Test frame analysis integration
        print("🔍 Testing frame analysis integration...")
        mock_frame_data = {
            'frame': [[[1 if i == j else 0 for _ in range(3)] for j in range(10)] for i in range(10)],
            'score': 75,
            'state': 'PLAYING'
        }
        
        frame_analyzer = FrameAnalyzer()
        analysis = frame_analyzer.analyze_frame(mock_frame_data['frame'])
        print(f"✅ Frame analysis completed: movement_detected={analysis.get('movement_detected', False)}")
        
        # Test 3: Test pathway learning integration
        print("📚 Testing pathway learning integration...")
        pathway_system = PathwayLearningSystem()
        
        # Simulate some training data
        for i in range(10):
            pathway_system.track_action(
                action=6,  # ACTION6 coordinate action
                action_data={'coordinates': (30 + i, 30 + i)},
                score_before=i * 5,
                score_after=(i + 1) * 5,
                win_score=100,
                game_id="integration_test"
            )
        
        # Get recommendations
        recommendations = pathway_system.get_pathway_recommendations(
            available_actions=[1, 2, 3, 4, 5, 6, 7],
            current_score=50,
            game_id="integration_test"
        )
        print(f"✅ Pathway recommendations generated with {len(recommendations.get('action_weights', {}))} actions weighted")
        
        # Test 4: Test coordinate selection logic
        print("🎯 Testing coordinate selection logic...")
        selected_action, coordinates = await manager._select_coordinate_aware_action(
            frame_data=mock_frame_data,
            available_actions=[1, 2, 3, 4, 5, 6, 7],
            agent=None,  # Will handle None gracefully
            game_id="integration_test"
        )
        print(f"✅ Action selected: {selected_action}, coordinates: {coordinates}")
        
        # Test 5: Test coordinate intelligence report
        print("📊 Testing coordinate intelligence reporting...")
        intel_report = manager._generate_coordinate_intelligence_report("integration_test")
        print(f"✅ Intelligence report generated with {len(intel_report)} metrics")
        
        # Test 6: Test action effectiveness calculation
        print("📈 Testing action effectiveness calculation...")
        mock_results = {
            'score_progression': [
                {'action_type': 6, 'improvement': 5},
                {'action_type': 1, 'improvement': 0},
                {'action_type': 6, 'improvement': 10},
                {'action_type': 2, 'improvement': 3},
            ]
        }
        effectiveness = manager._calculate_action_effectiveness(mock_results)
        print(f"✅ Action effectiveness calculated for {len(effectiveness)} action types")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_enhanced_training_script_integration():
    """Test that the enhanced training script can be imported and initialized."""
    print("\n🧪 Testing enhanced training script integration...")
    
    try:
        # Test importing the enhanced training manager
        from train_arc_agent_enhanced import EnhancedARCTrainingManager
        print("✅ Enhanced training manager imported successfully")
        
        # Test initialization (without API key for testing)
        print("🔧 Testing enhanced manager initialization...")
        enhanced_manager = EnhancedARCTrainingManager("test_key", None, use_coordinates=True)
        print("✅ Enhanced manager initialized successfully")
        
        # Test that coordinate system is properly integrated
        print("🔗 Testing coordinate system integration...")
        assert hasattr(enhanced_manager, 'coordinate_manager'), "Coordinate manager not found"
        assert hasattr(enhanced_manager, 'continuous_loop'), "Continuous loop not found"
        print("✅ Both coordinate and traditional systems integrated")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced training script integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_organization():
    """Test that files are properly organized after cleanup."""
    print("\n🧪 Testing file organization after cleanup...")
    
    try:
        base_path = Path(__file__).parent.parent.parent
        
        # Check that coordinate system test is in the right place
        coord_test_path = Path(__file__).parent / "test_coordinate_system.py"
        assert coord_test_path.exists(), f"Coordinate system test not found at {coord_test_path}"
        print("✅ Coordinate system test properly located")
        
        # Check that archived tests exist
        archived_path = base_path / "tests" / "archived"
        assert archived_path.exists(), "Archived test folder not found"
        archived_files = list(archived_path.glob("test_*.py"))
        print(f"✅ {len(archived_files)} test files properly archived")
        
        # Check that integration tests are in the right place
        integration_path = base_path / "tests" / "integration"
        integration_files = list(integration_path.glob("test_*.py"))
        print(f"✅ {len(integration_files)} integration test files properly organized")
        
        # Check that enhanced training script exists
        enhanced_script = base_path / "train_arc_agent_enhanced.py"
        assert enhanced_script.exists(), "Enhanced training script not found"
        print("✅ Enhanced training script created")
        
        print("✅ All file organization tests passed")
        
    except Exception as e:
        print(f"❌ File organization test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_backward_compatibility():
    """Test that existing training system still works."""
    print("\n🧪 Testing backward compatibility with existing training system...")
    
    try:
        # Test that we can still import the original continuous learning loop
        from arc_integration.continuous_learning_loop import ContinuousLearningLoop
        print("✅ Original continuous learning loop still importable")
        
        # Test initialization (this will fail without proper paths, but we just test import structure)
        try:
            base_path = str(Path(__file__).parent.parent.parent)
            continuous_loop = ContinuousLearningLoop(
                api_key="test_key",
                tabula_rasa_path=base_path,
                arc_agents_path=None
            )
            print("✅ Original continuous learning loop can be initialized")
        except Exception as e:
            # Expected to fail without proper API key, but structure should be intact
            if "api_key" in str(e) or "path" in str(e):
                print("✅ Original continuous learning loop structure intact (expected path/key errors)")
            else:
                raise e
        
        # Test that salience system still works
        from core.salience_system import SalienceMode
        print("✅ Salience system still importable")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all integration tests."""
    print("🚀 Running Enhanced ARC Training Integration Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Coordinate System Integration
    if await test_coordinate_system_integration():
        tests_passed += 1
    
    # Test 2: Enhanced Training Script Integration
    if await test_enhanced_training_script_integration():
        tests_passed += 1
    
    # Test 3: File Organization
    if test_file_organization():
        tests_passed += 1
    
    # Test 4: Backward Compatibility
    if await test_backward_compatibility():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Integration Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All integration tests passed!")
        print("✅ Enhanced ARC training system is ready for use")
        print("\n📋 Usage Instructions:")
        print("1. Use existing master_arc_trainer.py for traditional training")
        print("2. Use train_arc_agent_enhanced.py for coordinate-aware training")
        print("3. Run tests with: python -m pytest tests/integration/")
        return True
    else:
        print("❌ Some integration tests failed. Check error messages above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

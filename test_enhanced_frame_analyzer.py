#!/usr/bin/env python3
"""Test the enhanced frame analyzer system."""

from src.vision.frame_analyzer import FrameAnalyzer
import numpy as np

def test_enhanced_frame_analyzer():
    """Test the enhanced frame analyzer with multiple color objects."""
    
    # Test the enhanced frame analyzer
    analyzer = FrameAnalyzer()
    print('🎯 Enhanced Frame Analyzer Initialized')
    print('- Tracking systems: tried_coordinates, coordinate_results, color_object_tracker')
    print('- Exploration system: spiral coordinates, coordinate avoidance')
    print('- Movement analysis: object clustering, movement vectors')

    # Create a test frame with multiple color objects
    test_frame = np.zeros((64, 64), dtype=int)
    test_frame[10:15, 10:15] = 3  # Red object
    test_frame[30:35, 30:35] = 7  # Blue object  
    test_frame[50, 50] = 12       # Single yellow pixel
    test_frame[20:25, 40:45] = 5  # Green object

    # Wrap in list format expected by analyzer
    frame_wrapped = [test_frame.tolist()]

    # Test analysis
    analysis = analyzer.analyze_frame_for_action6_targets(frame_wrapped, 'test_game')
    print('\n📊 Analysis Results:')
    print(f'- Interactive targets found: {len(analysis["interactive_targets"])}')
    print(f'- Recommended coordinate: {analysis["recommended_action6_coord"]}')
    print(f'- Targeting reason: {analysis["targeting_reason"]}')
    print(f'- Confidence: {analysis["confidence"]:.2f}')

    if analysis['interactive_targets']:
        print('\n🎯 Top 3 Targets:')
        for i, target in enumerate(analysis['interactive_targets'][:3]):
            print(f'  {i+1}. ({target["x"]},{target["y"]}) - {target["type"]} - confidence: {target["confidence"]:.2f}')
            if 'color' in target:
                print(f'      Color: {target["color"]}, Size: {target.get("size", "unknown")}')

    # Test coordinate tracking
    print('\n🔄 Testing Coordinate Tracking:')
    analyzer.record_coordinate_attempt(12, 12, True, 10.0)  # Success
    analyzer.record_coordinate_attempt(30, 30, False, 0.0)  # Failure
    analyzer.record_coordinate_attempt(30, 30, False, 0.0)  # Another failure
    
    print(f'- Tried coordinates: {len(analyzer.tried_coordinates)}')
    print(f'- Coordinate results tracked: {len(analyzer.coordinate_results)}')
    
    # Test movement analysis
    movement_info = analyzer.get_movement_analysis()
    print(f'\n📈 Movement Analysis:')
    print(f'- Tracked objects: {movement_info["tracked_objects"]}')
    print(f'- Moving objects: {movement_info["moving_objects"]}')
    print(f'- Static objects: {movement_info["static_objects"]}')

    print('\n✅ Enhanced Frame Analyzer Test Complete')
    return True

if __name__ == "__main__":
    test_enhanced_frame_analyzer()

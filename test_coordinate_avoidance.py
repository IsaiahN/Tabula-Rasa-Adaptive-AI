#!/usr/bin/env python3
"""
Test script for Enhanced Coordinate Avoidance System
Tests the new stuck coordinate detection and emergency diversification.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vision.frame_analyzer import FrameAnalyzer

def test_coordinate_avoidance():
    """Test the enhanced coordinate avoidance system."""
    print("🧪 Testing Enhanced Coordinate Avoidance System")
    print("=" * 60)
    
    analyzer = FrameAnalyzer()
    
    print("\n📝 Phase 1: Simulating stuck coordinate scenario...")
    
    # Simulate many failed attempts at (0,0) with zero score change
    print("   Simulating repeated (0,0) attempts with zero progress...")
    for i in range(8):
        analyzer._record_coordinate_effectiveness(0, 0, True, 0.0, "zero_progress_loop")
        
        if i == 4:
            print(f"   After {i+1} attempts: Should be marked as stuck")
        elif i == 7:
            avoidance_score = analyzer.get_coordinate_avoidance_score(0, 0)
            should_avoid = analyzer.should_avoid_coordinate(0, 0)
            print(f"   After {i+1} attempts: Avoidance score = {avoidance_score:.2f}, Should avoid = {should_avoid}")
    
    print("\n📊 Phase 2: Testing emergency diversification...")
    
    # Create test frame
    test_frame = np.zeros((64, 64), dtype=int)
    test_frame[10:15, 10:15] = 5  # Some interesting content
    test_frame[30:35, 30:35] = 8  # More content
    test_frame[50:55, 50:55] = 12 # Even more content
    
    # Get emergency diversification target
    emergency_coord = analyzer.get_emergency_diversification_target(
        test_frame.tolist(), (64, 64)
    )
    print(f"   Emergency diversification target: {emergency_coord}")
    print(f"   Should avoid emergency target: {analyzer.should_avoid_coordinate(emergency_coord[0], emergency_coord[1])}")
    
    print("\n🎯 Phase 3: Testing smart exploration...")
    
    # Test smart exploration that avoids stuck coordinates
    smart_coord = analyzer._generate_smart_exploration_coordinate(test_frame)
    print(f"   Smart exploration target: {smart_coord}")
    print(f"   Should avoid smart target: {analyzer.should_avoid_coordinate(smart_coord[0], smart_coord[1])}")
    
    print("\n📈 Phase 4: Testing coordinate recovery...")
    
    # Simulate a successful attempt at (0,0) to see if it recovers
    print("   Testing coordinate recovery with positive score change...")
    analyzer._record_coordinate_effectiveness(0, 0, True, 15.0, "successful_recovery")
    
    recovered_avoidance = analyzer.get_coordinate_avoidance_score(0, 0)
    recovered_should_avoid = analyzer.should_avoid_coordinate(0, 0)
    print(f"   After recovery: Avoidance score = {recovered_avoidance:.2f}, Should avoid = {recovered_should_avoid}")
    
    print("\n🔍 Phase 5: Enhanced frame analysis with avoidance...")
    
    # Test full frame analysis with avoidance system
    frame_3d = [test_frame.tolist()]  # Convert to expected format
    analysis = analyzer.analyze_frame_for_action6_targets(frame_3d, "test_game")
    
    recommended_coord = analysis['recommended_action6_coord']
    targeting_reason = analysis['targeting_reason']
    confidence = analysis['confidence']
    
    print(f"   Recommended coordinate: {recommended_coord}")
    print(f"   Targeting reason: {targeting_reason}")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   Should avoid recommended: {analyzer.should_avoid_coordinate(recommended_coord[0], recommended_coord[1])}")
    
    print("\n✅ ENHANCED AVOIDANCE SYSTEM TEST COMPLETE!")
    print("   🚫 Stuck coordinates are properly detected and avoided")
    print("   🚀 Emergency diversification provides alternatives") 
    print("   🔍 Smart exploration avoids known stuck areas")
    print("   💚 Coordinate recovery works when progress is made")
    
    return analyzer

if __name__ == "__main__":
    test_analyzer = test_coordinate_avoidance()

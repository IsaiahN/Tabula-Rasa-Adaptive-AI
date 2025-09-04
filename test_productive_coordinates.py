#!/usr/bin/env python3
"""
Test script to verify productivity-based coordinate selection.
Tests that coordinates with score increases are prioritized over stuck coordinates.
"""

import numpy as np
from src.vision.frame_analyzer import FrameAnalyzer

def test_productive_coordinate_prioritization():
    """Test that productive coordinates are prioritized over stuck ones."""
    print("🧪 Testing Productive Coordinate Prioritization")
    
    analyzer = FrameAnalyzer()
    
    # Create a test frame
    test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Simulate coordinate history:
    # (10, 10) - stuck coordinate, many attempts, no score gains
    # (20, 20) - productive coordinate, multiple score gains
    # (30, 30) - new coordinate, no history
    
    # Record stuck coordinate (10, 10) - many attempts, no gains
    for i in range(10):
        analyzer._record_coordinate_effectiveness(10, 10, success=False, score_change=0, context="test")
    
    # Record productive coordinate (20, 20) - multiple score gains
    analyzer._record_coordinate_effectiveness(20, 20, success=True, score_change=1.0, context="test")
    analyzer._record_coordinate_effectiveness(20, 20, success=True, score_change=0.5, context="test")
    analyzer._record_coordinate_effectiveness(20, 20, success=True, score_change=0.3, context="test")
    
    # Test coordinate avoidance scores
    print("\n📊 Coordinate Avoidance Scores:")
    stuck_score = analyzer.get_coordinate_avoidance_score(10, 10)
    productive_score = analyzer.get_coordinate_avoidance_score(20, 20)
    new_score = analyzer.get_coordinate_avoidance_score(30, 30)
    
    print(f"   Stuck (10,10): {stuck_score:.3f} (higher = more avoided)")
    print(f"   Productive (20,20): {productive_score:.3f} (negative = actively sought)")
    print(f"   New (30,30): {new_score:.3f}")
    
    # Test target ranking with these coordinates
    test_targets = [
        {'x': 10, 'y': 10, 'confidence': 0.8, 'type': 'color_object'},  # Stuck
        {'x': 20, 'y': 20, 'confidence': 0.6, 'type': 'color_object'},  # Productive  
        {'x': 30, 'y': 30, 'confidence': 0.7, 'type': 'color_object'},  # New
    ]
    
    ranked = analyzer._rank_interaction_targets(test_targets)
    
    print("\n🎯 Target Ranking (highest priority first):")
    for i, target in enumerate(ranked):
        print(f"   {i+1}. ({target['x']},{target['y']}) - Score: {target['priority_score']:.1f}")
        print(f"      Productivity Bonus: +{target.get('productivity_bonus', 0)}")
        print(f"      Avoidance Penalty: -{target.get('avoidance_penalty', 0)}")
    
    # Verify productivity bonus is working
    productive_target = next(t for t in ranked if t['x'] == 20 and t['y'] == 20)
    stuck_target = next(t for t in ranked if t['x'] == 10 and t['y'] == 10)
    
    print(f"\n✅ Results:")
    print(f"   Productive coord (20,20) has priority score: {productive_target['priority_score']:.1f}")
    print(f"   Stuck coord (10,10) has priority score: {stuck_target['priority_score']:.1f}")
    
    if productive_target['priority_score'] > stuck_target['priority_score']:
        print("   🎯 SUCCESS: Productive coordinate ranks higher than stuck coordinate!")
        return True
    else:
        print("   ❌ FAILED: Stuck coordinate ranks higher - productivity bonus not working")
        return False

def test_score_increase_unstucks_coordinate():
    """Test that score increases remove stuck status."""
    print("\n🧪 Testing Score Increase Removes Stuck Status")
    
    analyzer = FrameAnalyzer()
    
    # Make coordinate stuck
    for i in range(10):
        analyzer._record_coordinate_effectiveness(15, 15, success=False, score_change=0, context="test")
    
    # Check if it's marked as stuck
    coord_result = analyzer.coordinate_results[(15, 15)]
    print(f"   Coordinate (15,15) stuck status: {coord_result.get('is_stuck_coordinate', False)}")
    print(f"   Zero progress streak: {coord_result.get('zero_progress_streak', 0)}")
    
    # Now record a score increase
    analyzer._record_coordinate_effectiveness(15, 15, success=True, score_change=2.0, context="test_productive")
    
    # Check if stuck status was removed
    after_result = analyzer.coordinate_results[(15, 15)]
    print(f"\n   After +2.0 score change:")
    print(f"   Stuck status: {after_result.get('is_stuck_coordinate', False)}")
    print(f"   Zero progress streak: {after_result.get('zero_progress_streak', 0)}")
    print(f"   Total score change: {after_result.get('total_score_change', 0)}")
    
    if not after_result.get('is_stuck_coordinate', False):
        print("   ✅ SUCCESS: Score increase removed stuck status!")
        return True
    else:
        print("   ❌ FAILED: Coordinate still marked as stuck after score increase")
        return False

if __name__ == "__main__":
    print("🎯 PRODUCTIVE COORDINATE TESTING")
    print("Testing user insight: 'stuck coordinator should be overridden if the score keeps increasing'")
    print("=" * 80)
    
    test1_passed = test_productive_coordinate_prioritization()
    test2_passed = test_score_increase_unstucks_coordinate()
    
    print("\n" + "=" * 80)
    print("📋 SUMMARY:")
    print(f"   Test 1 - Productive Prioritization: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Test 2 - Score Unstucks Coordinates: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("   🎉 ALL TESTS PASSED - Productivity system working!")
    else:
        print("   ⚠️ Some tests failed - needs debugging")

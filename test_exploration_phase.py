#!/usr/bin/env python3
"""
Test script for the initial exploration phase.
Tests that the system systematically explores every color object before moving to selective targeting.
"""

import numpy as np
from src.vision.frame_analyzer import FrameAnalyzer

def test_exploration_phase():
    """Test the initial exploration phase functionality."""
    print("🧪 Testing Initial Exploration Phase")
    print("User insight: 'initially you should click every color object at least once'")
    print("=" * 80)
    
    analyzer = FrameAnalyzer()
    
    # Create a test frame with multiple colors (simulating a game with various objects)
    test_frame = np.zeros((64, 64), dtype=int)
    
    # Add colored objects:
    # Color 1 - red objects (top left)
    test_frame[10:15, 10:15] = 1
    test_frame[20:25, 10:15] = 1
    
    # Color 2 - green objects (top right) 
    test_frame[10:15, 40:45] = 2
    
    # Color 3 - blue objects (bottom)
    test_frame[50:55, 20:25] = 3
    test_frame[50:55, 30:35] = 3
    
    # Color 4 - yellow object (center)
    test_frame[30:35, 30:35] = 4
    
    print(f"🎮 Test Frame Created:")
    print(f"   Colors present: {sorted(list(set(test_frame.flatten()) - {0}))}")
    print(f"   Frame size: {test_frame.shape}")
    
    # Test exploration phase target selection
    exploration_targets = []
    max_iterations = 10  # Safety limit
    
    print(f"\n🔍 EXPLORATION PHASE SIMULATION:")
    
    for iteration in range(max_iterations):
        if analyzer.exploration_complete:
            break
            
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Analyze frame for targets
        analysis = analyzer.analyze_frame_for_action6_targets(test_frame.tolist())
        
        if analysis.get('recommended_action6_coord'):
            x, y = analysis['recommended_action6_coord']
            color_at_target = test_frame[y, x]
            
            print(f"   🎯 Exploration target: ({x},{y}) - Color {color_at_target}")
            print(f"   Reason: {analysis.get('targeting_reason', 'unknown')}")
            
            exploration_targets.append({
                'iteration': iteration + 1,
                'coordinate': (x, y),
                'color': color_at_target,
                'reason': analysis.get('targeting_reason', 'unknown')
            })
            
            # Simulate clicking this coordinate (no effect for test)
            analyzer.mark_color_explored(
                color=color_at_target,
                coordinate=(x, y),
                success=True,  # Assume clicking worked
                score_change=0.0,  # No score change in test
                frame_changes={'changes_detected': False}
            )
            
            print(f"   ✅ Color {color_at_target} marked as explored")
            print(f"   📊 Explored colors so far: {sorted(list(analyzer.explored_color_objects))}")
        else:
            print(f"   ❌ No exploration target found")
            break
    
    print(f"\n📋 EXPLORATION RESULTS:")
    print(f"   Total iterations: {len(exploration_targets)}")
    print(f"   Exploration complete: {analyzer.exploration_complete}")
    print(f"   Colors explored: {sorted(list(analyzer.explored_color_objects))}")
    
    # Verify all colors were explored
    expected_colors = {1, 2, 3, 4}  # Colors we put in the frame
    explored_colors = analyzer.explored_color_objects
    
    print(f"\n✅ VERIFICATION:")
    print(f"   Expected colors: {sorted(list(expected_colors))}")
    print(f"   Explored colors: {sorted(list(explored_colors))}")
    
    missing_colors = expected_colors - explored_colors
    extra_colors = explored_colors - expected_colors
    
    if not missing_colors and not extra_colors:
        print(f"   🎉 SUCCESS: All colors explored exactly once!")
        success = True
    else:
        print(f"   ⚠️ Issues detected:")
        if missing_colors:
            print(f"      Missing colors: {sorted(list(missing_colors))}")
        if extra_colors:
            print(f"      Extra colors: {sorted(list(extra_colors))}")
        success = False
    
    # Test behavior patterns were recorded
    print(f"\n📊 BEHAVIOR PATTERNS RECORDED:")
    for color in expected_colors:
        if color in analyzer.color_behavior_patterns:
            pattern = analyzer.color_behavior_patterns[color]
            exploration_result = pattern.get('exploration_result', {})
            print(f"   Color {color}: Clickable={pattern.get('clickable', 'unknown')}, "
                 f"Effects={exploration_result.get('effects_observed', [])}")
        else:
            print(f"   Color {color}: No pattern recorded ❌")
    
    return success

def test_transition_to_normal_targeting():
    """Test that system transitions to normal targeting after exploration."""
    print("\n🧪 Testing Transition to Normal Targeting")
    print("=" * 50)
    
    analyzer = FrameAnalyzer()
    
    # Create frame and manually mark all colors as explored
    test_frame = np.zeros((64, 64), dtype=int)
    test_frame[10:15, 10:15] = 1
    test_frame[20:25, 20:25] = 2
    
    # Mark colors as explored
    analyzer.explored_color_objects = {1, 2}
    analyzer.current_game_colors = {0, 1, 2}
    analyzer.exploration_complete = True
    analyzer.exploration_phase = False
    
    print(f"   Setup: All colors marked as explored")
    print(f"   Exploration phase: {analyzer.exploration_phase}")
    print(f"   Exploration complete: {analyzer.exploration_complete}")
    
    # Test that normal targeting is used
    analysis = analyzer.analyze_frame_for_action6_targets(test_frame.tolist())
    
    if analysis.get('recommended_action6_coord'):
        reason = analysis.get('targeting_reason', '')
        is_exploration = 'exploration' in reason.lower()
        
        print(f"   🎯 Target selected with reason: {reason}")
        print(f"   Is exploration target: {is_exploration}")
        
        if not is_exploration:
            print(f"   ✅ SUCCESS: Normal targeting active after exploration")
            return True
        else:
            print(f"   ❌ FAILED: Still in exploration mode")
            return False
    else:
        print(f"   ⚠️ No target found - inconclusive")
        return False

if __name__ == "__main__":
    print("🔍 INITIAL EXPLORATION PHASE TESTING")
    print("Testing user insight: click every color object at least once initially")
    print("=" * 80)
    
    test1_passed = test_exploration_phase()
    test2_passed = test_transition_to_normal_targeting()
    
    print("\n" + "=" * 80)
    print("📋 SUMMARY:")
    print(f"   Test 1 - Exploration Phase: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Test 2 - Transition to Normal: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("   🎉 ALL TESTS PASSED - Initial exploration system working!")
    else:
        print("   ⚠️ Some tests failed - needs debugging")
    
    print("\n💡 EXPECTED BEHAVIOR IN TRAINING:")
    print("   1. Agent will systematically click each color object once")
    print("   2. Each click builds knowledge about what's clickable") 
    print("   3. After all colors tested, switches to intelligent targeting")
    print("   4. Sleep/hypothesis system uses exploration data for better predictions")

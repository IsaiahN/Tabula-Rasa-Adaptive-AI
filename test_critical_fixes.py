#!/usr/bin/env python3
"""
Test the action scoring fix and emergency override system.
"""

import sys
sys.path.append('src')

from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop

def test_action_scoring_fix():
    """Test that action scoring no longer crashes."""
    print("🧪 Testing action scoring fix...")
    
    loop = ContinuousLearningLoop('../Arc3-AGI', '.')
    
    # Test the fixed method
    try:
        score = loop._calculate_comprehensive_action_score(
            action=1, 
            game_id='test_game', 
            frame_analysis={'movement_detected': True}
        )
        print(f"✅ Action scoring working - Action 1 score: {score}")
        return True
        
    except Exception as e:
        print(f"❌ Action scoring still broken: {e}")
        return False

def test_emergency_override():
    """Test emergency override triggers correctly."""
    print("🧪 Testing emergency override system...")
    
    loop = ContinuousLearningLoop('../Arc3-AGI', '.')
    
    # Simulate stuck state
    loop._last_selected_actions = [1] * 15  # 15 consecutive Action 1s
    loop._actions_without_progress = 60  # 60 actions without progress
    
    # Test action selection with override
    available = [1, 2, 3, 4, 6]
    try:
        selected = loop._select_next_action({'available_actions': available}, 'test_game')
        if selected == 6:
            print("✅ Emergency override working - ACTION 6 selected")
            return True
        else:
            print(f"❌ Emergency override failed - selected action {selected}")
            return False
    except Exception as e:
        print(f"❌ Emergency override crashed: {e}")
        return False

def run_quick_fixes_test():
    """Run all quick fix tests."""
    print("🔧 TESTING CRITICAL FIXES")
    print("=" * 40)
    
    tests = [test_action_scoring_fix, test_emergency_override]
    passed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 40)
    print(f"📊 RESULTS: {passed}/{len(tests)} fixes working")
    
    if passed == len(tests):
        print("🎯 ALL CRITICAL FIXES WORKING - Ready to restart training!")
        return True
    else:
        print("⚠️ Some fixes still need attention")
        return False

if __name__ == "__main__":
    run_quick_fixes_test()

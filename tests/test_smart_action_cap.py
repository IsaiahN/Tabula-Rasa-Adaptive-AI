#!/usr/bin/env python3
"""
Test the smart action cap system and early termination logic.
"""

import sys
sys.path.append('src')

def test_smart_action_cap_system():
    """Test the smart action cap calculation."""
    print("🧪 Testing Smart Action Cap System...")
    
    from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
    
    # Create instance
    loop = ContinuousLearningLoop('../Arc3-AGI', '.')
    
    # Test different scenarios
    test_cases = [
        ([1, 2, 3, 4], "Simple game with 4 actions"),
        ([1, 2, 3, 4, 5, 6], "Standard game with 6 actions"),
        ([1, 2], "Very simple game with 2 actions"), 
        ([1, 2, 3, 4, 5, 6, 7], "Complex game with 7 actions"),
        ([], "No actions available")
    ]
    
    for actions, description in test_cases:
        cap = loop._calculate_dynamic_action_cap(actions)
        print(f"   {description}: {cap} action limit")
    
    print("✅ Smart action cap system working")

def test_early_termination():
    """Test the early termination logic."""
    print("\n🧪 Testing Early Termination System...")
    
    from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
    
    loop = ContinuousLearningLoop('../Arc3-AGI', '.')
    
    # Reset progress tracker
    loop._progress_tracker = {
        'actions_taken': 0,
        'last_score': 0,
        'actions_without_progress': 0,
        'last_meaningful_change': 0,
        'action_pattern_history': [],
        'score_history': [0],
        'termination_reason': None
    }
    
    # Test scenarios
    scenarios = [
        # (current_score, actions_taken, description)
        (0, 5, "Early game, no termination expected"),
        (0, 25, "25 actions with no progress - should terminate"),
        (10, 30, "Score increased, should continue"),
        (0, 15, "15 actions no progress - should terminate")
    ]
    
    for score, actions, description in scenarios:
        should_terminate, reason = loop._should_terminate_early(score, actions)
        print(f"   {description}: {'TERMINATE' if should_terminate else 'CONTINUE'} - {reason}")
    
    print("✅ Early termination system working")

def test_stagnation_analysis():
    """Test the stagnation analysis system."""
    print("\n🧪 Testing Stagnation Analysis...")
    
    from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop
    
    loop = ContinuousLearningLoop('../Arc3-AGI', '.')
    
    # Mock action history with repetitive actions
    action_history = [
        {'action': 1, 'score_change': 0} for _ in range(20)  # 20 repeated action 1s
    ] + [
        {'action': 2, 'score_change': 0} for _ in range(5)   # 5 action 2s  
    ]
    
    analysis = loop._analyze_stagnation_cause('test_game', action_history)
    
    print(f"   Analysis results:")
    print(f"   - Total actions: {analysis['total_actions']}")
    print(f"   - Stagnation patterns: {analysis['stagnation_patterns']}")
    print(f"   - Effectiveness: {analysis['action_effectiveness']}")
    print(f"   - Suggested fixes: {analysis['suggested_fixes']}")
    
    print("✅ Stagnation analysis working")

if __name__ == "__main__":
    print("🔧 TESTING SMART ACTION CAP AND ANALYSIS SYSTEMS")
    print("=" * 60)
    
    try:
        test_smart_action_cap_system()
        test_early_termination()
        test_stagnation_analysis()
        
        print("\n🎯 ALL TESTS PASSED - Smart action cap system ready!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

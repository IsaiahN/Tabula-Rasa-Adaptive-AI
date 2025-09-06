#!/usr/bin/env python3

"""
Test the learning feedback loop fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop

def test_action_effectiveness_tracking():
    """Test that action effectiveness is properly tracked and used for learning."""
    print("🧪 Testing action effectiveness tracking...")
    
    try:
        arc_agents_path = "C:\\Users\\Admin\\Documents\\GitHub\\ARC-AGI-3-Agents"
        tabula_rasa_path = "C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa"
        loop = ContinuousLearningLoop(arc_agents_path, tabula_rasa_path)
        
        # Simulate successful action 1
        print("   📈 Simulating successful action 1...")
        mock_response_success = {
            'state': 'IN_PROGRESS',
            'score': 10,
            'available_actions': [1, 2, 3, 4, 6],
            'frame': [[1, 2], [3, 4]]
        }
        loop._analyze_action_effectiveness(1, mock_response_success)
        
        # Simulate failed action 2
        print("   📉 Simulating failed action 2...")
        mock_response_fail = {
            'state': 'IN_PROGRESS',
            'score': 0,
            'available_actions': [],  # Game ended
            'error': 'Invalid action'
        }
        loop._analyze_action_effectiveness(2, mock_response_fail)
        
        # Check effectiveness data
        effectiveness = loop.available_actions_memory['action_effectiveness']
        
        action1_data = effectiveness.get(1, {})
        action2_data = effectiveness.get(2, {})
        
        print(f"   ✅ Action 1 effectiveness: {action1_data}")
        print(f"   ✅ Action 2 effectiveness: {action2_data}")
        
        # Verify action 1 was marked successful and action 2 as failed
        if action1_data.get('success_rate', 0) > 0.5 and action2_data.get('success_rate', 1) < 0.5:
            print("✅ Action effectiveness tracking working correctly!")
            return True
        else:
            print("❌ Action effectiveness not working as expected")
            return False
            
    except Exception as e:
        print(f"❌ Action effectiveness tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_recent_action_attempts():
    """Test that recent action attempts uses real data instead of placeholders."""
    print("\n🧪 Testing recent action attempts calculation...")
    
    try:
        arc_agents_path = "C:\\Users\\Admin\\Documents\\GitHub\\ARC-AGI-3-Agents"
        tabula_rasa_path = "C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa"
        loop = ContinuousLearningLoop(arc_agents_path, tabula_rasa_path)
        
        # Set up some action history
        loop.available_actions_memory['action_history'] = [1, 1, 2, 1, 3, 1, 1]
        
        # Set up effectiveness data for action 1
        loop.available_actions_memory['action_effectiveness'][1] = {
            'attempts': 5,
            'successes': 3,
            'success_rate': 0.6  # 60% success rate
        }
        
        # Test the fixed method
        recent_data = loop._get_recent_action_attempts(1, window=10)
        
        print(f"   📊 Recent action 1 data: {recent_data}")
        
        # Verify it's using real data
        if recent_data.get('overall_success_rate') == 0.6:  # Should use real 60% not placeholder 50%
            print("✅ Recent action attempts using real effectiveness data!")
            return True
        else:
            print(f"❌ Still using placeholder data - got {recent_data.get('overall_success_rate')}, expected 0.6")
            return False
            
    except Exception as e:
        print(f"❌ Recent action attempts calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action_relevance_update():
    """Test that action relevance scores are updated based on real effectiveness."""
    print("\n🧪 Testing action relevance score updates...")
    
    try:
        arc_agents_path = "C:\\Users\\Admin\\Documents\\GitHub\\ARC-AGI-3-Agents"
        tabula_rasa_path = "C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa"
        loop = ContinuousLearningLoop(arc_agents_path, tabula_rasa_path)
        
        # Set up initial relevance scores
        loop.available_actions_memory['action_relevance_scores'][1] = {
            'base_relevance': 0.5,
            'current_modifier': 1.0,
            'recent_success_rate': 0.0,
            'last_used': 0
        }
        
        # Set up effectiveness data showing high success
        loop.available_actions_memory['action_effectiveness'][1] = {
            'attempts': 10,
            'successes': 8,
            'success_rate': 0.8  # 80% success rate
        }
        
        # Set up action history
        loop.available_actions_memory['action_history'] = [1] * 5  # 5 recent action 1s
        
        initial_modifier = loop.available_actions_memory['action_relevance_scores'][1]['current_modifier']
        print(f"   📊 Initial modifier for action 1: {initial_modifier}")
        
        # Update relevance scores
        loop._update_action_relevance_scores()
        
        final_modifier = loop.available_actions_memory['action_relevance_scores'][1]['current_modifier']
        final_success_rate = loop.available_actions_memory['action_relevance_scores'][1]['recent_success_rate']
        
        print(f"   📈 Final modifier for action 1: {final_modifier}")
        print(f"   📈 Final success rate for action 1: {final_success_rate}")
        
        # Should increase modifier due to high success rate (80% > 70%)
        if final_modifier > initial_modifier and final_success_rate > 0.7:
            print("✅ Action relevance scores updating based on real effectiveness!")
            return True
        else:
            print(f"❌ Action relevance not updating properly")
            return False
            
    except Exception as e:
        print(f"❌ Action relevance update failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 TESTING LEARNING FEEDBACK LOOP FIX")
    print("=" * 60)
    
    tests = [
        test_action_effectiveness_tracking,
        test_recent_action_attempts, 
        test_action_relevance_update
    ]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed}/{len(tests)} learning feedback loop fixes working")
    
    if passed == len(tests):
        print("🎯 LEARNING FEEDBACK LOOP FIXED - Actions will now improve decision-making!")
    else:
        print("⚠️ LEARNING FEEDBACK LOOP STILL HAS ISSUES")

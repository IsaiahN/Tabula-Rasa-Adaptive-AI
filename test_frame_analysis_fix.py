#!/usr/bin/env python3

"""
Test the frame analysis integration fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.arc_integration.continuous_learning_loop import ContinuousLearningLoop

def test_frame_analysis_integration():
    """Test that frame analysis is properly integrated and working."""
    print("🧪 Testing frame analysis integration...")
    
    try:
        # Create a test continuous learning loop
        arc_agents_path = "C:\\Users\\Admin\\Documents\\GitHub\\ARC-AGI-3-Agents"
        tabula_rasa_path = "C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa"
        loop = ContinuousLearningLoop(arc_agents_path, tabula_rasa_path)
        
        # Create mock response data with frame
        mock_response_data = {
            'frame': [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 0], 
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3]
            ],
            'available_actions': [1, 2, 3, 4, 6],
            'score': 0
        }
        
        # Test frame analysis
        frame_analysis = loop._analyze_frame_for_action_selection(mock_response_data, "test_game")
        
        if frame_analysis:
            print(f"✅ Frame analysis working - Analysis keys: {list(frame_analysis.keys())}")
            if 'interactive_targets' in frame_analysis:
                print(f"   Interactive targets: {len(frame_analysis.get('interactive_targets', []))}")
            if 'movement_detected' in frame_analysis:
                print(f"   Movement detected: {frame_analysis.get('movement_detected', False)}")
        else:
            print(f"❌ Frame analysis returned None or empty")
            
        return frame_analysis is not None
        
    except Exception as e:
        print(f"❌ Frame analysis integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action6_with_frame_analysis():
    """Test ACTION6 coordinate selection with frame analysis."""
    print("\n🧪 Testing ACTION6 coordinate selection with frame analysis...")
    
    try:
        arc_agents_path = "C:\\Users\\Admin\\Documents\\GitHub\\ARC-AGI-3-Agents"
        tabula_rasa_path = "C:\\Users\\Admin\\Documents\\GitHub\\tabula-rasa"
        loop = ContinuousLearningLoop(arc_agents_path, tabula_rasa_path)
        
        # Mock frame analysis with targets
        mock_frame_analysis = {
            'interactive_targets': [
                {'coordinate': (2, 3), 'confidence': 0.8, 'reason': 'high_contrast'},
                {'coordinate': (4, 1), 'confidence': 0.6, 'reason': 'isolated_pixel'}
            ],
            'movement_detected': True,
            'recommended_coordinate': (2, 3),
            'confidence': 0.8
        }
        
        # Test coordinate selection
        context = {
            'available_actions': [1, 2, 3, 4, 6],
            'game_id': 'test_game',
            'frame_analysis': mock_frame_analysis,
            'response_data': {'frame': [[1,2],[3,4]]}
        }
        
        action = loop._select_intelligent_action_with_relevance([1,2,3,4,6], context)
        
        print(f"✅ ACTION6 coordinate selection working - Selected action: {action}")
        return True
        
    except Exception as e:
        print(f"❌ ACTION6 coordinate selection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 TESTING FRAME ANALYSIS INTEGRATION FIX")
    print("=" * 50)
    
    tests = [test_frame_analysis_integration, test_action6_with_frame_analysis]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{len(tests)} frame analysis fixes working")
    
    if passed == len(tests):
        print("🎯 FRAME ANALYSIS INTEGRATION WORKING - Ready for next fix!")
    else:
        print("⚠️ FRAME ANALYSIS INTEGRATION STILL HAS ISSUES")

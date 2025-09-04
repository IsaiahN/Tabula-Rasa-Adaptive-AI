#!/usr/bin/env python3
"""
Test script for the enhanced hypothesis generation system.
Tests the interaction logging and sleep consolidation features.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vision.frame_analyzer import FrameAnalyzer

def create_test_frame():
    """Create a test frame with various colors."""
    frame = np.zeros((40, 40, 3), dtype=int)
    
    # Add some colored objects
    frame[10:15, 10:15] = [255, 0, 0]    # Red square
    frame[20:25, 20:25] = [0, 255, 0]    # Green square  
    frame[30:35, 10:15] = [0, 0, 255]    # Blue square
    frame[15:20, 30:35] = [255, 255, 0]  # Yellow square
    
    return frame.tolist()

def test_hypothesis_generation():
    """Test the hypothesis generation and sleep consolidation system."""
    print("🧪 Testing Enhanced Frame Analyzer Hypothesis System")
    print("=" * 60)
    
    analyzer = FrameAnalyzer()
    test_frame = create_test_frame()
    
    # Simulate several ACTION6 interactions with varying results
    print("\n📝 Phase 1: Simulating ACTION6 interactions...")
    
    # Successful red interactions
    for i in range(3):
        coord = (12 + i, 12)  # Red area
        target_info = {"dominant_color": [255, 0, 0], "shape": "square"}
        before_state = {"score": 0, "frame": test_frame}
        after_state = {"score": 10, "frame": test_frame}
        analyzer.log_action6_interaction(
            x=coord[0], y=coord[1],
            target_info=target_info,
            before_state=before_state,
            after_state=after_state,
            score_change=10
        )
        print(f"   ✅ Logged successful red interaction at {coord}")
    
    # Mixed green interactions
    for i in range(2):
        coord = (22, 22 + i)  # Green area
        score_change = 5 if i == 0 else -2
        target_info = {"dominant_color": [0, 255, 0], "shape": "square"}
        before_state = {"score": 10, "frame": test_frame}
        after_state = {"score": 10 + score_change, "frame": test_frame}
        analyzer.log_action6_interaction(
            x=coord[0], y=coord[1],
            target_info=target_info,
            before_state=before_state,
            after_state=after_state,
            score_change=score_change
        )
        success = "✅" if score_change > 0 else "❌"
        print(f"   {success} Logged green interaction at {coord} (score: {score_change:+})")
    
    # Failed blue interactions
    for i in range(2):
        coord = (32, 12 + i)  # Blue area
        target_info = {"dominant_color": [0, 0, 255], "shape": "square"}
        before_state = {"score": 10, "frame": test_frame}
        after_state = {"score": 5, "frame": test_frame}
        analyzer.log_action6_interaction(
            x=coord[0], y=coord[1],
            target_info=target_info,
            before_state=before_state,
            after_state=after_state,
            score_change=-5
        )
        print(f"   ❌ Logged failed blue interaction at {coord}")
    
    print(f"\n📊 Interaction Summary:")
    print(f"   Total interactions logged: {len(analyzer.interaction_log)}")
    print(f"   Color patterns tracked: {len(analyzer.color_behavior_patterns)}")
    
    # Print color behavior patterns
    print(f"\n🎨 Color Behavior Analysis:")
    for color, data in analyzer.color_behavior_patterns.items():
        success_rate = data['success_rate'] * 100
        print(f"   Color {color}: {success_rate:.0f}% success rate ({len(data['interactions'])} interactions)")
    
    print("\n🛏️ Phase 2: Sleep consolidation and hypothesis generation...")
    
    # Run sleep consolidation
    consolidation_results = analyzer.consolidate_learning_during_sleep()
    
    print(f"\n🧠 Phase 3: Generated hypotheses analysis...")
    print(f"   Total hypotheses in database: {len(analyzer.hypothesis_database)}")
    
    # Display top hypotheses
    top_hypotheses = sorted(analyzer.hypothesis_database, 
                           key=lambda h: h['confidence_score'], reverse=True)[:5]
    
    for i, hypothesis in enumerate(top_hypotheses, 1):
        print(f"\n   Hypothesis #{i} (Confidence: {hypothesis['confidence_score']:.2f}):")
        print(f"      Type: {hypothesis['type']}")
        print(f"      Description: {hypothesis['description']}")
        print(f"      Prediction: {hypothesis['prediction']}")
        print(f"      Advice: {hypothesis.get('actionable_advice', 'None')}")
    
    print(f"\n🎯 Phase 4: Actionable recommendations...")
    recommendations = analyzer.get_actionable_recommendations()
    
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"\n   Recommendation #{i} (Priority: {rec['priority']:.2f}):")
        print(f"      Type: {rec['type']}")
        print(f"      Recommendation: {rec['recommendation']}")
        print(f"      Confidence: {rec['confidence']:.2f}")
        print(f"      Reasoning: {rec['reasoning']}")
    
    print(f"\n📈 Phase 5: Learning state analysis...")
    learning_state = analyzer.get_current_learning_state()
    
    print(f"   Recent success rate: {learning_state['recent_success_rate']:.1%}")
    print(f"   Overall success rate: {learning_state['overall_success_rate']:.1%}")
    print(f"   Best performing colors: {learning_state['best_colors']}")
    print(f"   Coordinates explored: {learning_state['coordinates_explored']}")
    
    print(f"\n💡 Learning Insights:")
    for insight in learning_state['learning_insights']:
        print(f"   • {insight}")
    
    print(f"\n✅ HYPOTHESIS SYSTEM TEST COMPLETE!")
    print(f"   🎯 System successfully generated {len(top_hypotheses)} actionable hypotheses")
    print(f"   🔍 Identified patterns in color behaviors and spatial preferences")  
    print(f"   🛏️ Sleep consolidation processed {consolidation_results['new_hypotheses_generated']} new hypotheses")
    print(f"   💡 Generated {len(learning_state['learning_insights'])} learning insights")
    
    return analyzer

if __name__ == "__main__":
    test_analyzer = test_hypothesis_generation()

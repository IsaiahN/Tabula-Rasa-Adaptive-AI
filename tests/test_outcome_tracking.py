#!/usr/bin/env python3
"""
Test script for the comprehensive outcome tracking system
"""

import sys
import json
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.outcome_tracker import OutcomeTracker, PerformanceMetrics, OutcomeStatus
from src.core.meta_cognitive_governor import MetaCognitiveGovernor

def test_outcome_tracking_system():
    """Test the comprehensive outcome tracking system."""
    print("📊 Testing Comprehensive Outcome Tracking System...")
    
    # Create test directories
    test_dir = Path("tests/tmp/test_outcome_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Governor with outcome tracking
    governor = MetaCognitiveGovernor(
        log_file="tests/tmp/test_governor.log",
        outcome_tracking_dir=str(test_dir)
    )
    
    print(f"\n🧠 Governor initialized with outcome tracking: {governor.outcome_tracker is not None}")
    
    # Simulate a training session with outcome tracking
    print("\n🎮 Simulating Training Session with Outcome Tracking...")
    
    # Test scenario 1: Governor makes a recommendation
    current_performance = {
        'win_rate': 0.2,
        'average_score': 15.0,
        'learning_efficiency': 0.6,
        'computational_efficiency': 0.8
    }
    
    current_config = {
        'max_actions_per_game': 500,
        'salience_mode': 'decay_compression'
    }
    
    # Get recommendation
    recommendation = governor.get_recommended_configuration(
        puzzle_type="transformation",
        current_performance=current_performance,
        current_config=current_config
    )
    
    if recommendation:
        print(f"   🎯 Recommendation: {recommendation.type.value}")
        print(f"   📋 Changes: {recommendation.configuration_changes}")
        print(f"   🎭 Confidence: {recommendation.confidence:.2f}")
        
        # Simulate applying the recommendation and measuring outcomes
        time.sleep(0.1)  # Simulate processing time
        
        # Find the decision ID from the last decision
        last_decision = list(governor.decision_history)[-1]
        decision_id = last_decision['decision_id']
        
        print(f"   📍 Decision ID: {decision_id}")
        
        # Simulate improved performance after applying recommendation
        improved_performance = {
            'win_rate': 0.35,  # Improved from 0.2
            'average_score': 22.0,  # Improved from 15.0
            'learning_efficiency': 0.7,  # Improved from 0.6
            'computational_efficiency': 0.8
        }
        
        # Complete outcome measurement
        governor.complete_outcome_measurement(
            decision_id=decision_id,
            post_performance=improved_performance,
            sample_size=10,
            notes="Test simulation with positive improvement"
        )
        
        print("   ✅ Outcome measurement completed")
    
    # Test scenario 2: Multiple recommendations with different outcomes
    print("\n🔄 Testing Multiple Outcomes...")
    
    scenarios = [
        # Scenario with good outcome
        {
            'pre': {'win_rate': 0.3, 'average_score': 20.0, 'learning_efficiency': 0.5},
            'post': {'win_rate': 0.5, 'average_score': 30.0, 'learning_efficiency': 0.7},
            'expected': 'SUCCESS'
        },
        # Scenario with poor outcome
        {
            'pre': {'win_rate': 0.4, 'average_score': 25.0, 'learning_efficiency': 0.6},
            'post': {'win_rate': 0.35, 'average_score': 20.0, 'learning_efficiency': 0.5},
            'expected': 'FAILURE'
        },
        # Scenario with mixed outcome
        {
            'pre': {'win_rate': 0.2, 'average_score': 15.0, 'learning_efficiency': 0.4},
            'post': {'win_rate': 0.35, 'average_score': 12.0, 'learning_efficiency': 0.6},
            'expected': 'PARTIAL'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n   🎯 Scenario {i} (Expected: {scenario['expected']}):")
        
        rec = governor.get_recommended_configuration(
            puzzle_type="pattern",
            current_performance=scenario['pre'],
            current_config=current_config
        )
        
        if rec and governor.decision_history:
            decision_id = list(governor.decision_history)[-1]['decision_id']
            
            # Complete measurement
            governor.complete_outcome_measurement(
                decision_id=decision_id,
                post_performance=scenario['post'],
                sample_size=5,
                notes=f"Test scenario {i}"
            )
            
            # Get the outcome record
            if governor.outcome_tracker and governor.outcome_tracker.outcome_history:
                outcome = list(governor.outcome_tracker.outcome_history)[-1]
                print(f"      📊 Actual: {outcome.status.value} (Score: {outcome.success_score:.2f})")
                print(f"      📈 Performance Delta: {outcome.performance_deltas.get('total_performance_delta', 0):.3f}")
    
    # Test scenario 3: Effectiveness analysis
    print("\n📈 Testing Effectiveness Analysis...")
    
    insights = governor.get_effectiveness_insights()
    
    if insights.get('insights_available'):
        gov_insights = insights.get('governor_specific', {})
        
        print(f"   📋 Total Decisions: {gov_insights.get('total_decisions', 0)}")
        print(f"   🎯 Tracked Outcomes: {gov_insights.get('tracked_outcomes', 0)}")
        print(f"   ⏳ Pending Measurements: {gov_insights.get('pending_measurements', 0)}")
        print(f"   📊 Estimated Success Rate: {gov_insights.get('estimated_success_rate', 0):.1%}")
        
        # Show effectiveness by recommendation type
        for key, value in gov_insights.items():
            if key.endswith('_effectiveness') and isinstance(value, dict):
                rec_type = key.replace('_effectiveness', '').replace('governor_', '')
                if value.get('sample_count', 0) > 0:
                    print(f"   🔧 {rec_type.title()}: {value['success_rate']:.1%} success "
                          f"({value['sample_count']} samples)")
        
        # Show learning insights
        main_insights = insights
        if main_insights.get('most_effective_interventions'):
            print("\n   🏆 Most Effective Interventions:")
            for intervention, stats in main_insights['most_effective_interventions'][:2]:
                print(f"      • {intervention}: {stats['success_rate']:.1%} success rate")
        
        if main_insights.get('recommendations'):
            print("\n   💡 System Recommendations:")
            for rec in main_insights['recommendations'][:2]:
                print(f"      • {rec}")
    
    # Test direct outcome tracker functionality
    print("\n🔬 Testing Direct Outcome Tracker...")
    
    if governor.outcome_tracker:
        tracker = governor.outcome_tracker
        
        # Test intervention effectiveness
        param_effectiveness = tracker.get_intervention_effectiveness('governor_parameter_adjustment')
        print(f"   📊 Parameter Adjustment Effectiveness: {param_effectiveness}")
        
        # Test performance trends
        trends = tracker.get_performance_trends('governor_parameter_adjustment', days=1)
        print(f"   📈 Recent Performance Trends: {len(trends)} data points")
        
        # Show recent outcomes
        if tracker.outcome_history:
            recent_outcomes = list(tracker.outcome_history)[-3:]
            print("\n   📋 Recent Outcomes:")
            for outcome in recent_outcomes:
                print(f"      • {outcome.intervention_type}: {outcome.status.value} "
                      f"(Score: {outcome.success_score:.2f})")
    
    print("\n✅ Outcome tracking system tests completed!")
    print("🔍 Key capabilities demonstrated:")
    print("   • Automatic outcome measurement initiation")
    print("   • Performance delta calculation and assessment")
    print("   • Success/failure classification with confidence")
    print("   • Historical effectiveness analysis")
    print("   • Learning insights and recommendations")
    print("   • Integration with Governor decision-making")
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    if Path("tests/tmp/test_governor.log").exists():
        Path("tests/tmp/test_governor.log").unlink()

if __name__ == "__main__":
    test_outcome_tracking_system()

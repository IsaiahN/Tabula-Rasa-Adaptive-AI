#!/usr/bin/env python3
"""
Test script for enhanced Governor algorithms
"""

import sys
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.meta_cognitive_governor import MetaCognitiveGovernor, GovernorRecommendationType

def test_enhanced_governor():
    """Test the enhanced Governor recommendation algorithms."""
    print("🧠 Testing Enhanced Governor Algorithms...")
    
    # Initialize Governor
    governor = MetaCognitiveGovernor()
    
    # Test scenario 1: Low efficiency should trigger diverse parameter adjustments
    print("\n📊 Test 1: Parameter Adjustment Diversity")
    system_analysis = {
        'average_efficiency': 0.6,
        'win_rate': 0.3
    }
    
    # Make multiple recommendations to see diversity
    recommendations = []
    for i in range(5):
        rec = governor._evaluate_parameter_adjustments(system_analysis)
        if rec:
            recommendations.append({
                'iteration': i + 1,
                'config': rec.configuration_changes,
                'confidence': rec.confidence,
                'rationale': rec.rationale
            })
            
            # Simulate adding to decision history
            governor.decision_history.append({
                'recommendation_type': GovernorRecommendationType.PARAMETER_ADJUSTMENT.value,
                'configuration_changes': rec.configuration_changes,
                'outcome_metrics': {'win_rate_change': 0.05, 'efficiency_change': 0.1}  # Simulate success
            })
    
    for rec in recommendations:
        print(f"   Iteration {rec['iteration']}: {rec['config']} (confidence: {rec['confidence']:.2f})")
        print(f"   Rationale: {rec['rationale']}")
        print()
    
    # Test scenario 2: Mode switching with different win rates
    print("🔄 Test 2: Adaptive Mode Switching")
    test_cases = [
        {'win_rate': 0.05, 'puzzle_type': 'transformation'},
        {'win_rate': 0.15, 'puzzle_type': 'pattern'},
        {'win_rate': 0.25, 'puzzle_type': 'spatial'},
    ]
    
    for case in test_cases:
        performance = {
            'current_win_rate': case['win_rate'],
            'average_score': 8.0 if case['win_rate'] > 0.2 else 3.0
        }
        
        rec = governor._evaluate_mode_switching(case['puzzle_type'], performance)
        if rec:
            print(f"   Win Rate {case['win_rate']:.2f}: {rec.configuration_changes}")
            print(f"   Strategy: {rec.rationale}")
            print(f"   Confidence: {rec.confidence:.2f}, Urgency: {rec.urgency:.1f}")
            print()
    
    # Test scenario 3: Adaptive confidence calculation
    print("🎯 Test 3: Adaptive Confidence Calculation")
    
    # Test with successful history
    successful_adjustments = [
        {'outcome_metrics': {'win_rate_change': 0.1, 'efficiency_change': 0.05}},
        {'outcome_metrics': {'win_rate_change': 0.08, 'efficiency_change': 0.03}},
    ]
    success_rate = governor._calculate_adjustment_success_rate(successful_adjustments)
    print(f"   Successful history success rate: {success_rate:.2f}")
    
    # Test with failed history  
    failed_adjustments = [
        {'outcome_metrics': {'win_rate_change': -0.02, 'efficiency_change': 0.0}},
        {'outcome_metrics': {'win_rate_change': 0.0, 'efficiency_change': -0.01}},
    ]
    fail_rate = governor._calculate_adjustment_success_rate(failed_adjustments)
    print(f"   Failed history success rate: {fail_rate:.2f}")
    
    print("\n✅ Enhanced Governor algorithm tests completed!")
    print("🔍 Key improvements:")
    print("   • Parameter adjustment diversity based on history")
    print("   • Adaptive confidence scoring based on past success")
    print("   • Context-aware mode switching strategies")
    print("   • Prevention of repetitive recommendations")

if __name__ == "__main__":
    test_enhanced_governor()

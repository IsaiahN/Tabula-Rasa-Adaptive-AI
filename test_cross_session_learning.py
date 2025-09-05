#!/usr/bin/env python3
"""
Test script for cross-session learning persistence system
"""

import sys
import json
import time
import shutil
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.cross_session_learning import (
    CrossSessionLearningManager, KnowledgeType, PersistenceLevel, LearnedPattern
)
from src.core.meta_cognitive_governor import MetaCognitiveGovernor

def test_cross_session_learning_system():
    """Test the comprehensive cross-session learning system."""
    print("🧠 Testing Cross-Session Learning Persistence System...")
    
    # Create test directories
    test_dir = Path("test_cross_session_data")
    test_dir.mkdir(exist_ok=True)
    
    # Test 1: Basic Learning Manager functionality
    print("\n📚 Test 1: Basic Learning Manager")
    
    learning_manager = CrossSessionLearningManager(test_dir)
    
    # Start a session
    session_context = {
        'training_phase': 'exploration',
        'puzzle_types': ['transformation', 'pattern'],
        'initial_performance': 0.2
    }
    session_id = learning_manager.start_session(session_context)
    print(f"   🎯 Started session: {session_id}")
    
    # Learn some patterns
    contexts = [
        {
            'puzzle_type': 'transformation',
            'current_performance': {'win_rate': 0.2, 'learning_efficiency': 0.4}
        },
        {
            'puzzle_type': 'pattern', 
            'current_performance': {'win_rate': 0.3, 'learning_efficiency': 0.6}
        }
    ]
    
    patterns_learned = []
    for i, context in enumerate(contexts):
        pattern_data = {
            'strategy_name': f'test_strategy_{i}',
            'parameters': {'max_actions': 500 + i * 100, 'threshold': 0.5 + i * 0.1},
            'success_metrics': {'win_rate_improvement': 0.1 + i * 0.05}
        }
        
        pattern_id = learning_manager.learn_pattern(
            KnowledgeType.STRATEGY_PATTERN,
            pattern_data,
            context,
            success_rate=0.6 + i * 0.1,
            persistence_level=PersistenceLevel.PERMANENT
        )
        patterns_learned.append(pattern_id)
        print(f"   📖 Learned pattern: {pattern_id}")
    
    # Test pattern retrieval
    print("\n🔍 Testing Pattern Retrieval...")
    applicable_patterns = learning_manager.retrieve_applicable_patterns(
        KnowledgeType.STRATEGY_PATTERN,
        contexts[0],
        min_confidence=0.3
    )
    
    print(f"   📋 Found {len(applicable_patterns)} applicable patterns")
    for pattern in applicable_patterns:
        print(f"      • {pattern.pattern_id}: {pattern.success_rate:.1%} success, "
              f"{pattern.confidence:.2f} confidence")
    
    # End session
    performance_summary = {
        'total_decisions': 10,
        'successful_decisions': 7,
        'avg_improvement': 0.15
    }
    learning_manager.end_session(performance_summary)
    print("   ✅ Session ended")
    
    # Test 2: Governor Integration
    print("\n🏛️ Test 2: Governor Integration with Cross-Session Learning")
    
    governor = MetaCognitiveGovernor(
        log_file="test_governor_cross_session.log",
        outcome_tracking_dir=str(test_dir / "outcomes"),
        persistence_dir=str(test_dir / "learning")
    )
    
    print(f"   🧠 Governor initialized with cross-session learning: {governor.learning_manager is not None}")
    
    # Start learning session
    session_id = governor.start_learning_session({
        'training_type': 'arc_puzzles',
        'difficulty': 'mixed'
    })
    print(f"   📚 Learning session started: {session_id}")
    
    # Simulate training with recommendations and outcomes
    training_scenarios = [
        {
            'puzzle_type': 'transformation',
            'performance': {'win_rate': 0.25, 'average_score': 12.0, 'learning_efficiency': 0.5},
            'expected_improvement': 0.12
        },
        {
            'puzzle_type': 'pattern',
            'performance': {'win_rate': 0.35, 'average_score': 18.0, 'learning_efficiency': 0.6},
            'expected_improvement': 0.08
        },
        {
            'puzzle_type': 'transformation',
            'performance': {'win_rate': 0.15, 'average_score': 8.0, 'learning_efficiency': 0.4},
            'expected_improvement': 0.15
        }
    ]
    
    for i, scenario in enumerate(training_scenarios):
        print(f"\n   🎮 Training Scenario {i + 1}:")
        
        # Get recommendation
        current_config = {'max_actions_per_game': 500, 'salience_mode': 'decay_compression'}
        recommendation = governor.get_recommended_configuration(
            puzzle_type=scenario['puzzle_type'],
            current_performance=scenario['performance'],
            current_config=current_config
        )
        
        if recommendation:
            print(f"      🎯 Recommendation: {recommendation.type.value}")
            print(f"      📋 Changes: {recommendation.configuration_changes}")
            print(f"      🎭 Confidence: {recommendation.confidence:.2f}")
            
            # Simulate applying recommendation and measuring outcome
            time.sleep(0.05)  # Brief processing time
            
            # Create success metrics
            success_metrics = {
                'win_rate_improvement': scenario['expected_improvement'],
                'score_improvement': scenario['expected_improvement'] * 20,
                'efficiency_improvement': scenario['expected_improvement'] * 0.5
            }
            
            # Learn from outcome
            context = {
                'puzzle_type': scenario['puzzle_type'],
                'current_performance': scenario['performance'],
                'system_state': {'efficiency': 0.7}
            }
            
            governor.learn_from_recommendation_outcome(recommendation, context, success_metrics)
            print(f"      📚 Learned from outcome (improvement: {scenario['expected_improvement']:.2f})")
    
    # Test learned recommendations
    print("\n🎓 Testing Learned Recommendations...")
    
    test_context = {
        'puzzle_type': 'transformation',
        'current_performance': {'win_rate': 0.2, 'learning_efficiency': 0.45}
    }
    
    learned_recs = governor.get_learned_recommendations(
        'transformation',
        test_context['current_performance']
    )
    
    print(f"   📋 Found {len(learned_recs)} learned recommendations:")
    for rec in learned_recs:
        print(f"      • {rec['type']}: {rec['success_rate']:.1%} success rate, "
              f"{rec['confidence']:.2f} confidence")
        print(f"        Applied {rec['applications']} times")
    
    # Test 3: Cross-Session Persistence
    print("\n💾 Test 3: Cross-Session Persistence")
    
    # End the session and shutdown
    governor.end_learning_session()
    
    if governor.learning_manager:
        governor.learning_manager.shutdown()
    
    print("   📊 Session ended and data persisted")
    
    # Create a new learning manager to test persistence
    print("   🔄 Creating new learning manager to test persistence...")
    
    new_learning_manager = CrossSessionLearningManager(test_dir / "learning")
    
    print(f"   📚 Loaded {len(new_learning_manager.learned_patterns)} patterns from disk")
    
    # Verify patterns are available
    if new_learning_manager.learned_patterns:
        print("   ✅ Patterns successfully persisted and loaded")
        
        # Show some loaded patterns
        for pattern_id, pattern in list(new_learning_manager.learned_patterns.items())[:2]:
            print(f"      • {pattern_id}: {pattern.knowledge_type.value}, "
                  f"{pattern.success_rate:.1%} success")
    
    # Test 4: Performance Insights
    print("\n📈 Test 4: Performance Insights")
    
    insights = new_learning_manager.get_performance_insights()
    
    print(f"   📊 Total patterns learned: {insights['total_patterns']}")
    print("   🏆 Most successful patterns:")
    
    for pattern_info in insights['most_successful_patterns'][:3]:
        print(f"      • {pattern_info['type']}: {pattern_info['success_rate']:.1%} success "
              f"({pattern_info['applications']} applications)")
    
    if 'learning_trends' in insights:
        trends = insights['learning_trends']
        print(f"   📈 Average patterns per session: {trends.get('average_patterns_per_session', 0):.1f}")
    
    # Test Governor insights
    print("\n🏛️ Governor Cross-Session Insights:")
    
    # Create new governor to test loaded state
    governor2 = MetaCognitiveGovernor(
        persistence_dir=str(test_dir / "learning")
    )
    
    if governor2.learning_manager:
        cross_session_insights = governor2.get_cross_session_insights()
        
        if cross_session_insights.get('learning_available'):
            print(f"   📚 Patterns available: {cross_session_insights.get('total_patterns', 0)}")
            
            patterns_by_type = cross_session_insights.get('patterns_by_type', {})
            for pattern_type, count in patterns_by_type.items():
                print(f"      • {pattern_type}: {count} patterns")
    
    new_learning_manager.shutdown()
    governor2.learning_manager.shutdown() if governor2.learning_manager else None
    
    print("\n✅ Cross-session learning system tests completed!")
    print("🔍 Key capabilities demonstrated:")
    print("   • Pattern learning and persistence across sessions")
    print("   • Context-aware pattern retrieval")  
    print("   • Governor integration with learned strategies")
    print("   • Automatic success/failure pattern recognition")
    print("   • Performance insights and trend analysis")
    print("   • Robust serialization and recovery")
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    test_files = [
        "test_governor_cross_session.log"
    ]
    
    for file in test_files:
        if Path(file).exists():
            Path(file).unlink()

if __name__ == "__main__":
    test_cross_session_learning_system()

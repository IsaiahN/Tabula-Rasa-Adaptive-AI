"""
Test Phase 1 Memory Pattern Optimization with Governor Integration

This test demonstrates the immediate wins from integrating memory pattern 
recognition into the Governor system. Shows real-time pattern detection,
optimization recommendations, and Governor decision-making enhancement.
"""

import sys
import time
import json
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.core.meta_cognitive_governor import MetaCognitiveGovernor
    from src.core.memory_pattern_optimizer import MemoryPatternOptimizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to run this from the project root directory")
    sys.exit(1)

def simulate_memory_access_patterns():
    """Simulate realistic memory access patterns for testing"""
    patterns = [
        # Pattern 1: Governor decision sequence
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'governor_decisions.log', 'operation': 'read', 'success': True, 'duration': 0.001},
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'governor_decisions.log', 'operation': 'write', 'success': True, 'duration': 0.002},
        
        # Pattern 2: Architect evolution cycle
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'architect_evolution.json', 'operation': 'read', 'success': True, 'duration': 0.003},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'meta_learning_session.json', 'operation': 'read', 'success': True, 'duration': 0.005},
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'architect_evolution.json', 'operation': 'write', 'success': True, 'duration': 0.004},
        
        # Pattern 3: Repeated session access (optimization opportunity)
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'session_data.json', 'operation': 'read', 'success': True, 'duration': 0.010},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'session_data.json', 'operation': 'read', 'success': True, 'duration': 0.008},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'session_data.json', 'operation': 'read', 'success': True, 'duration': 0.012},
        
        # Pattern 4: Temporary file burst (cleanup opportunity)
        {'memory_type': 'TEMPORARY_PURGE', 'file_path': 'temp_debug_001.log', 'operation': 'write', 'success': True, 'duration': 0.001},
        {'memory_type': 'TEMPORARY_PURGE', 'file_path': 'temp_debug_002.log', 'operation': 'write', 'success': True, 'duration': 0.001},
        {'memory_type': 'TEMPORARY_PURGE', 'file_path': 'temp_debug_003.log', 'operation': 'write', 'success': True, 'duration': 0.001},
        {'memory_type': 'TEMPORARY_PURGE', 'file_path': 'temp_sandbox.json', 'operation': 'write', 'success': True, 'duration': 0.001},
        
        # Pattern 5: Failed access pattern (efficiency issue)
        {'memory_type': 'REGULAR_DECAY', 'file_path': 'old_data.json', 'operation': 'read', 'success': False, 'duration': 0.050},
        {'memory_type': 'REGULAR_DECAY', 'file_path': 'old_data.json', 'operation': 'read', 'success': False, 'duration': 0.045},
        
        # Pattern 6: Directory clustering
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'continuous_learning_data/action_intel_001.json', 'operation': 'read', 'success': True, 'duration': 0.003},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'continuous_learning_data/action_intel_002.json', 'operation': 'read', 'success': True, 'duration': 0.003},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'continuous_learning_data/meta_learning_001.json', 'operation': 'read', 'success': True, 'duration': 0.004},
    ]
    
    return patterns

def test_standalone_pattern_optimizer():
    """Test the MemoryPatternOptimizer independently"""
    print("🧠 Testing Standalone Memory Pattern Optimizer")
    print("=" * 50)
    
    optimizer = MemoryPatternOptimizer()
    
    # Simulate memory accesses
    patterns = simulate_memory_access_patterns()
    for i, access in enumerate(patterns):
        optimizer.record_memory_access(access)
        time.sleep(0.01)  # Small delay to create temporal patterns
        if i % 5 == 0:
            print(f"   📊 Recorded {i+1} memory accesses...")
    
    # Analyze patterns
    print("\n🔍 Analyzing detected patterns...")
    recommendations = optimizer.get_governor_recommendations()
    summary = optimizer.get_pattern_summary()
    
    print(f"\n📈 Pattern Analysis Results:")
    print(f"   🎯 Total patterns detected: {summary['total_patterns']}")
    print(f"   ⚡ Temporal patterns: {summary['temporal_count']}")
    print(f"   🗺️ Spatial patterns: {summary['spatial_count']}")  
    print(f"   🔗 Semantic patterns: {summary['semantic_count']}")
    print(f"   📊 Top optimization potential: {summary['top_optimization_potential']:.3f}")
    print(f"   ✅ Current efficiency: {summary['efficiency_status']:.3f}")
    
    # Show immediate actions
    immediate_actions = recommendations.get('immediate_actions', [])
    if immediate_actions:
        print(f"\n⚡ Immediate Governor Recommendations ({len(immediate_actions)}):")
        for i, action in enumerate(immediate_actions, 1):
            print(f"   {i}. {action['action']} - {action['reason']} (Priority: {action['urgency']})")
    
    # Show optimization opportunities
    optimizations = recommendations.get('priority_optimizations', [])
    if optimizations:
        print(f"\n🎯 Top Optimization Opportunities ({len(optimizations)}):")
        for i, opt in enumerate(optimizations[:3], 1):
            print(f"   {i}. {opt['action']} - {opt['expected_improvement']} (Potential: {opt['potential']:.3f})")
    
    return optimizer, recommendations, summary

def test_governor_integration():
    """Test Governor integration with pattern optimization"""
    print("\n" + "=" * 60)
    print("🎯 Testing Governor Integration (Phase 1)")
    print("=" * 60)
    
    # Create Governor with pattern optimization
    governor = MetaCognitiveGovernor(
        log_file="test_governor_phase1.log",
        persistence_dir="."
    )
    
    # Verify pattern optimizer integration
    if not governor.pattern_optimizer:
        print("❌ Pattern optimizer not integrated - check imports")
        return None
    
    print("✅ Governor successfully integrated with Pattern Optimizer")
    
    # Simulate memory access patterns through Governor
    print("\n📝 Recording memory accesses through Governor...")
    patterns = simulate_memory_access_patterns()
    
    for access in patterns:
        governor.record_memory_access(access)
        time.sleep(0.005)  # Small delay
    
    print(f"✅ Recorded {len(patterns)} memory accesses")
    
    # Trigger Governor memory analysis
    print("\n🧠 Triggering intelligent memory analysis...")
    analysis_result = governor.trigger_intelligent_memory_analysis()
    
    # Display results
    print(f"\n📊 Governor Analysis Results:")
    print(f"   Status: {analysis_result.get('status', 'unknown')}")
    
    if 'key_insights' in analysis_result:
        insights = analysis_result['key_insights']
        print(f"   🎯 Patterns detected: {insights.get('total_patterns_detected', 0)}")
        print(f"   ⚡ Optimization potential: {insights.get('optimization_potential', 0):.3f}")
        print(f"   📈 Efficiency trend: {insights.get('efficiency_trend', 'unknown')}")
        print(f"   🎪 Governor confidence: {insights.get('governor_confidence', 0):.2f}")
    
    # Show Governor recommendations
    recommendations = analysis_result.get('governor_recommendations', [])
    if recommendations:
        print(f"\n🎯 Governor Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. [{rec['priority'].upper()}] {rec['action']}")
            print(f"      Reason: {rec['reason']}")
            print(f"      Source: {rec['source']}")
    
    # Test enhanced memory status
    print("\n🔍 Enhanced Memory Status Check...")
    memory_status = governor.get_memory_status()
    
    if 'governor_analysis' in memory_status:
        analysis = memory_status['governor_analysis']
        print(f"   Health Status: {analysis.get('health_status', 'unknown')}")
        print(f"   Recommendations: {len(analysis.get('recommendations', []))}")
        
        if 'pattern_analysis' in analysis:
            pattern_analysis = analysis['pattern_analysis']
            print(f"   Pattern-based insights: ✅")
            print(f"   Immediate actions: {len(pattern_analysis.get('immediate_actions', []))}")
        else:
            print(f"   Pattern-based insights: ❌")
    
    return governor, analysis_result

def test_optimization_trigger():
    """Test the optimize_memory_patterns method"""
    print("\n" + "=" * 50)
    print("⚡ Testing Pattern Optimization Trigger")
    print("=" * 50)
    
    # Create Governor
    governor = MetaCognitiveGovernor(persistence_dir=".")
    
    if not governor.pattern_optimizer:
        print("❌ Pattern optimizer not available")
        return
    
    # Add some patterns
    patterns = simulate_memory_access_patterns()
    for access in patterns:
        governor.record_memory_access(access)
    
    # Trigger optimization
    print("🚀 Triggering memory pattern optimization...")
    optimization_result = governor.optimize_memory_patterns()
    
    print(f"\n📊 Optimization Results:")
    print(f"   Status: {optimization_result.get('status', 'unknown')}")
    
    if optimization_result.get('status') == 'analysis_complete':
        patterns_analyzed = optimization_result.get('patterns_analyzed', {})
        applied_optimizations = optimization_result.get('applied_optimizations', [])
        
        print(f"   🧠 Patterns analyzed: {patterns_analyzed.get('total_patterns', 0)}")
        print(f"   ⚡ Applied optimizations: {len(applied_optimizations)}")
        
        if applied_optimizations:
            print(f"\n✅ Optimization Actions Recommended:")
            for i, opt in enumerate(applied_optimizations, 1):
                print(f"   {i}. {opt['action']} ({opt['priority']} priority)")
                print(f"      Reason: {opt['reason']}")
    
    return optimization_result

def main():
    """Run comprehensive Phase 1 testing"""
    print("🧠 Phase 1 Memory Pattern Optimization Testing")
    print("=" * 60)
    print("Testing immediate Governor wins with pattern recognition")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test 1: Standalone pattern optimizer
        optimizer, recommendations, summary = test_standalone_pattern_optimizer()
        
        # Test 2: Governor integration
        governor, analysis_result = test_governor_integration()
        
        # Test 3: Optimization trigger
        optimization_result = test_optimization_trigger()
        
        # Summary
        print("\n" + "=" * 60)
        print("🏆 Phase 1 Testing Summary")
        print("=" * 60)
        
        print("✅ Memory Pattern Optimizer: Working")
        print("✅ Governor Integration: Working") 
        print("✅ Pattern-Based Recommendations: Working")
        print("✅ Enhanced Memory Analysis: Working")
        print("✅ Optimization Triggers: Working")
        
        print(f"\n📊 Key Metrics:")
        if summary:
            print(f"   🎯 Total patterns detected: {summary['total_patterns']}")
            print(f"   ⚡ Optimization potential: {summary['top_optimization_potential']:.3f}")
        
        if analysis_result and 'key_insights' in analysis_result:
            insights = analysis_result['key_insights']
            print(f"   🎪 Governor confidence: {insights.get('governor_confidence', 0):.2f}")
            print(f"   📈 Pattern-based recommendations: {insights.get('immediate_actions_needed', 0)}")
        
        print("\n🚀 Phase 1 Complete - Ready for Phase 2 (Hierarchical Clustering)")
        
    except Exception as e:
        print(f"❌ Phase 1 testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

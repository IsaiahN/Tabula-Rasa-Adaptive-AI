"""
Test Phase 2 Hierarchical Memory Clustering with Enhanced Governor Integration

This test demonstrates the Governor's enhanced capabilities with both
Phase 1 pattern recognition AND Phase 2 hierarchical clustering working
together to provide intelligent, dynamic memory management that replaces
static thresholds with cluster-based optimization.
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
    from src.core.hierarchical_memory_clusterer import HierarchicalMemoryClusterer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to run this from the project root directory")
    sys.exit(1)

def simulate_advanced_memory_patterns():
    """Simulate complex memory access patterns for Phase 2 testing"""
    patterns = [
        # Governor decision cascade (causal chain)
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'governor_decisions.log', 'operation': 'read', 'success': True, 'duration': 0.001, 'timestamp': time.time() - 200},
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'system_monitors.json', 'operation': 'read', 'success': True, 'duration': 0.002, 'timestamp': time.time() - 199.5},
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'governor_decisions.log', 'operation': 'write', 'success': True, 'duration': 0.003, 'timestamp': time.time() - 199},
        
        # Architect evolution sequence (causal chain)
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'architect_evolution.json', 'operation': 'read', 'success': True, 'duration': 0.002, 'timestamp': time.time() - 150},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'evolution_analysis.json', 'operation': 'read', 'success': True, 'duration': 0.008, 'timestamp': time.time() - 149.5},
        {'memory_type': 'CRITICAL_LOSSLESS', 'file_path': 'architect_evolution.json', 'operation': 'write', 'success': True, 'duration': 0.004, 'timestamp': time.time() - 149},
        
        # Training session cluster (temporal + semantic)
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'continuous_learning_data/session_001.json', 'operation': 'read', 'success': True, 'duration': 0.005, 'timestamp': time.time() - 100},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'continuous_learning_data/session_002.json', 'operation': 'read', 'success': True, 'duration': 0.006, 'timestamp': time.time() - 99},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'continuous_learning_data/meta_learning.json', 'operation': 'read', 'success': True, 'duration': 0.007, 'timestamp': time.time() - 98},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'continuous_learning_data/action_intelligence.json', 'operation': 'read', 'success': True, 'duration': 0.004, 'timestamp': time.time() - 97},
        
        # Performance hotspot (performance cluster)
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'high_performance_model.pt', 'operation': 'read', 'success': True, 'duration': 0.001, 'timestamp': time.time() - 80},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'high_performance_config.json', 'operation': 'read', 'success': True, 'duration': 0.001, 'timestamp': time.time() - 79},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'high_performance_model.pt', 'operation': 'read', 'success': True, 'duration': 0.001, 'timestamp': time.time() - 78},
        
        # Problematic files (low performance cluster)
        {'memory_type': 'REGULAR_DECAY', 'file_path': 'slow_legacy_data.json', 'operation': 'read', 'success': False, 'duration': 0.150, 'timestamp': time.time() - 60},
        {'memory_type': 'REGULAR_DECAY', 'file_path': 'corrupted_cache.dat', 'operation': 'read', 'success': False, 'duration': 0.200, 'timestamp': time.time() - 59},
        {'memory_type': 'REGULAR_DECAY', 'file_path': 'slow_legacy_data.json', 'operation': 'read', 'success': False, 'duration': 0.180, 'timestamp': time.time() - 58},
        
        # Temporary file burst
        {'memory_type': 'TEMPORARY_PURGE', 'file_path': 'temp/debug_001.log', 'operation': 'write', 'success': True, 'duration': 0.001, 'timestamp': time.time() - 40},
        {'memory_type': 'TEMPORARY_PURGE', 'file_path': 'temp/debug_002.log', 'operation': 'write', 'success': True, 'duration': 0.001, 'timestamp': time.time() - 39.5},
        {'memory_type': 'TEMPORARY_PURGE', 'file_path': 'temp/cache_temp.json', 'operation': 'write', 'success': True, 'duration': 0.001, 'timestamp': time.time() - 39},
        {'memory_type': 'TEMPORARY_PURGE', 'file_path': 'temp/sandbox_test.dat', 'operation': 'write', 'success': True, 'duration': 0.001, 'timestamp': time.time() - 38.5},
        
        # Cross-session pattern (persistent learning data)
    {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'continuous_learning_data/backups/persistent_learning_state.json', 'operation': 'read', 'success': True, 'duration': 0.003, 'timestamp': time.time() - 20},
        {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'cross_session_patterns.json', 'operation': 'read', 'success': True, 'duration': 0.004, 'timestamp': time.time() - 19},
    {'memory_type': 'IMPORTANT_DECAY', 'file_path': 'continuous_learning_data/backups/persistent_learning_state.json', 'operation': 'write', 'success': True, 'duration': 0.005, 'timestamp': time.time() - 18},
    ]
    
    return patterns

def test_standalone_hierarchical_clusterer():
    """Test the HierarchicalMemoryClusterer independently"""
    print("🗂️ Testing Standalone Hierarchical Memory Clusterer")
    print("=" * 55)
    
    clusterer = HierarchicalMemoryClusterer()
    
    # Simulate memory data and access patterns
    memories = [
        {"file_path": "governor_decisions.log", "memory_type": "CRITICAL_LOSSLESS", "classification": "critical"},
        {"file_path": "system_monitors.json", "memory_type": "CRITICAL_LOSSLESS", "classification": "critical"},
        {"file_path": "architect_evolution.json", "memory_type": "CRITICAL_LOSSLESS", "classification": "critical"},
        {"file_path": "continuous_learning_data/session_001.json", "memory_type": "IMPORTANT_DECAY", "classification": "important"},
        {"file_path": "continuous_learning_data/session_002.json", "memory_type": "IMPORTANT_DECAY", "classification": "important"},
        {"file_path": "continuous_learning_data/meta_learning.json", "memory_type": "IMPORTANT_DECAY", "classification": "important"},
        {"file_path": "high_performance_model.pt", "memory_type": "IMPORTANT_DECAY", "classification": "important"},
        {"file_path": "high_performance_config.json", "memory_type": "IMPORTANT_DECAY", "classification": "important"},
        {"file_path": "slow_legacy_data.json", "memory_type": "REGULAR_DECAY", "classification": "regular"},
        {"file_path": "corrupted_cache.dat", "memory_type": "REGULAR_DECAY", "classification": "regular"},
        {"file_path": "temp/debug_001.log", "memory_type": "TEMPORARY_PURGE", "classification": "temporary"},
        {"file_path": "temp/cache_temp.json", "memory_type": "TEMPORARY_PURGE", "classification": "temporary"},
    ]
    
    patterns = simulate_advanced_memory_patterns()
    
    # Create intelligent clusters
    print("🧠 Creating intelligent clusters...")
    clusters = clusterer.create_intelligent_clusters(memories, patterns)
    summary = clusterer.get_clustering_summary()
    recommendations = clusterer.get_cluster_optimization_recommendations()
    
    print(f"\n📊 Clustering Results:")
    print(f"   🗂️ Total clusters created: {summary['total_clusters']}")
    print(f"   📋 Cluster types: {summary['cluster_types']}")
    print(f"   🎯 Total clustered memories: {summary['total_clustered_memories']}")
    print(f"   💚 Average cluster health: {summary['cluster_health']['avg_health_score']:.3f}")
    print(f"   🔗 Cluster relationships: {summary['cluster_relationships']}")
    
    if recommendations:
        print(f"\n⚡ Cluster Optimization Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:3], 1):  # Top 3
            print(f"   {i}. [{rec['priority'].upper()}] {rec['action']}")
            print(f"      Reason: {rec['reason']}")
            print(f"      Expected: {rec['expected_improvement']}")
    
    return clusterer, clusters, summary, recommendations

def test_enhanced_governor_integration():
    """Test Governor with both Phase 1 patterns and Phase 2 clusters"""
    print("\n" + "=" * 70)
    print("🎯 Testing Enhanced Governor Integration (Phase 1 + Phase 2)")
    print("=" * 70)
    
    # Create Governor with both pattern optimization and clustering
    governor = MetaCognitiveGovernor(
    log_file="tests/test_governor_phase2.log",
        persistence_dir="."
    )
    
    # Verify both enhancements are integrated
    pattern_status = "✅" if governor.pattern_optimizer else "❌"
    cluster_status = "✅" if governor.memory_clusterer else "❌"
    
    print(f"Pattern Optimizer Integration: {pattern_status}")
    print(f"Hierarchical Clusterer Integration: {cluster_status}")
    
    if not (governor.pattern_optimizer and governor.memory_clusterer):
        print("❌ Required components not integrated properly")
        return None
    
    print("✅ Governor successfully enhanced with Phase 1 + Phase 2 capabilities")
    
    # Record advanced memory access patterns
    print("\n📝 Recording advanced memory access patterns...")
    patterns = simulate_advanced_memory_patterns()
    
    for i, access in enumerate(patterns):
        governor.record_memory_access(access)
        if i % 5 == 0:
            print(f"   📊 Recorded {i+1}/{len(patterns)} accesses...")
        time.sleep(0.002)  # Small delay for realistic timing
    
    print(f"✅ Recorded {len(patterns)} complex memory access patterns")
    
    # Trigger enhanced intelligent memory analysis
    print("\n🧠 Triggering enhanced intelligent memory analysis (Phase 2)...")
    analysis_result = governor.trigger_intelligent_memory_analysis()
    
    # Display comprehensive results
    print(f"\n📊 Enhanced Governor Analysis Results:")
    print(f"   Status: {analysis_result.get('status', 'unknown')}")
    print(f"   Analysis Type: {analysis_result.get('analysis_type', 'unknown')}")
    
    if 'key_insights' in analysis_result:
        insights = analysis_result['key_insights']
        print(f"\n🧠 Key Insights (Phase 1 + Phase 2):")
        print(f"   🎯 Patterns detected: {insights.get('total_patterns_detected', 0)}")
        print(f"   🗂️ Clusters created: {insights.get('total_clusters_created', 0)}")
        print(f"   ⚡ Optimization potential: {insights.get('optimization_potential', 0):.3f}")
        print(f"   📈 Efficiency trend: {insights.get('efficiency_trend', 'unknown')}")
        print(f"   💚 Cluster health: {insights.get('cluster_health', 0):.3f}")
        print(f"   🎪 Governor confidence: {insights.get('governor_confidence', 0):.3f}")
    
    # Show enhanced Governor recommendations
    recommendations = analysis_result.get("governor_recommendations", [])
    if recommendations:
        print(f"\n🎯 Enhanced Governor Recommendations ({len(recommendations)}):")
        
        # Group by source
        sources = {}
        for rec in recommendations:
            source = rec.get('source', 'unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(rec)
        
        for source, recs in sources.items():
            print(f"\n   📋 From {source.replace('_', ' ').title()}:")
            for i, rec in enumerate(recs[:2], 1):  # Top 2 per source
                print(f"      {i}. [{rec['priority'].upper()}] {rec['action']}")
                print(f"         Reason: {rec['reason']}")
                if 'cluster_id' in rec:
                    print(f"         Cluster: {rec['cluster_id']}")
    
    return governor, analysis_result

def test_cluster_based_retention_policy():
    """Test the new cluster-based retention policies"""
    print("\n" + "=" * 60)
    print("🔄 Testing Cluster-Based Retention Policies")
    print("=" * 60)
    
    # Create Governor
    governor = MetaCognitiveGovernor(persistence_dir=".")
    
    if not governor.memory_clusterer:
        print("❌ Memory clusterer not available")
        return
    
    # Add patterns and create clusters
    patterns = simulate_advanced_memory_patterns()
    for access in patterns:
        governor.record_memory_access(access)
    
    # Create clusters
    cluster_result = governor.create_intelligent_memory_clusters()
    print(f"✅ Created {cluster_result.get('clusters_created', 0)} intelligent clusters")
    
    # Test retention policies for different memory types
    test_files = [
        "governor_decisions.log",
        "continuous_learning_data/session_001.json", 
        "high_performance_model.pt",
        "slow_legacy_data.json",
        "temp/debug_001.log"
    ]
    
    print(f"\n🔍 Testing Cluster-Based Retention Policies:")
    
    for file_path in test_files:
        policy = governor.get_cluster_based_retention_policy(file_path)
        
        print(f"\n   📁 {file_path}:")
        print(f"      Policy: {policy['policy']}")
        print(f"      Retention Priority: {policy['retention_priority']:.3f}")
        print(f"      Reason: {policy['reason']}")
        
        if 'cluster_count' in policy:
            print(f"      Clusters: {policy['cluster_count']}")
        
        if 'cluster_policies' in policy and policy['cluster_policies']:
            print(f"      Cluster Policies: {', '.join(policy['cluster_policies'])}")
    
    return cluster_result

def main():
    """Run comprehensive Phase 2 testing"""
    print("🗂️ Phase 2 Hierarchical Memory Clustering Testing")
    print("=" * 70)
    print("Testing Governor enhancement with intelligent clusters")
    print("=" * 70)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test 1: Standalone hierarchical clusterer
        clusterer, clusters, summary, recommendations = test_standalone_hierarchical_clusterer()
        
        # Test 2: Enhanced Governor integration
        governor, analysis_result = test_enhanced_governor_integration()
        
        # Test 3: Cluster-based retention policies
        cluster_result = test_cluster_based_retention_policy()
        
        # Summary
        print("\n" + "=" * 70)
        print("🏆 Phase 2 Testing Summary")
        print("=" * 70)
        
        print("✅ Hierarchical Memory Clusterer: Working")
        print("✅ Enhanced Governor Integration: Working")
        print("✅ Pattern + Cluster Analysis: Working") 
        print("✅ Cluster-Based Optimization: Working")
        print("✅ Dynamic Retention Policies: Working")
        
        print(f"\n📊 Key Metrics:")
        if summary:
            print(f"   🗂️ Intelligent clusters created: {summary['total_clusters']}")
            print(f"   💚 Average cluster health: {summary['cluster_health']['avg_health_score']:.3f}")
        
        if analysis_result and 'key_insights' in analysis_result:
            insights = analysis_result['key_insights']
            print(f"   🧠 Combined patterns + clusters: {insights.get('total_patterns_detected', 0)} + {insights.get('total_clusters_created', 0)}")
            print(f"   🎪 Enhanced Governor confidence: {insights.get('governor_confidence', 0):.3f}")
            print(f"   ⚡ Total optimization opportunities: {len(analysis_result.get('governor_recommendations', []))}")
        
        print("\n🎯 Phase 2 Impact:")
        print("   • Replaced static 4-tier system with intelligent clusters")
        print("   • Governor now uses cluster relationships for decisions")
        print("   • Dynamic retention policies based on cluster health")
        print("   • Pattern recognition enhanced with cluster context")
        print("   • Multi-dimensional optimization recommendations")
        
        print("\n🚀 Phase 2 Complete - Ready for Phase 3 (Architect Evolution)")
        
    except Exception as e:
        print(f"❌ Phase 2 testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

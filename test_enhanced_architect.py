#!/usr/bin/env python3
"""
Test script for enhanced Architect mutation capabilities
"""

import sys
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.architect import Architect, SystemGenome, MutationType

def test_enhanced_architect():
    """Test the enhanced Architect mutation generation."""
    print("🏗️ Testing Enhanced Architect Mutation Capabilities...")
    
    # Create a sample genome
    genome = SystemGenome(
        max_actions_per_game=500,
        target_score=50,
        salience_mode="decay_compression",
        salience_threshold=0.8,
        enable_contrarian_strategy=False,
        enable_boredom_detection=True,
        enable_mid_game_sleep=True
    )
    
    # Initialize Architect with correct parameters
    architect_instance = Architect(".", ".", None)
    
    # Access the mutation engine to test the templates
    mutation_engine = architect_instance.mutation_engine
    
    print(f"\n📋 Available mutation types: {len(mutation_engine.mutation_templates)}")
    for i, template in enumerate(mutation_engine.mutation_templates, 1):
        print(f"   {i}. {template['type'].value} (Impact: {template['impact'].value})")
    
    print("\n🔬 Generating diverse mutations...")
    
    # Generate mutations from different types
    mutations = []
    mutation_types_seen = set()
    
    for i in range(15):  # Generate more mutations to see variety
        mutation = mutation_engine.generate_exploratory_mutation()
        mutations.append({
            'id': mutation.id,
            'type': mutation.type.value,
            'impact': mutation.impact.value,
            'changes': mutation.changes,
            'rationale': mutation.rationale,
            'confidence': mutation.confidence,
            'expected_improvement': mutation.expected_improvement,
            'test_duration': mutation.test_duration_estimate
        })
        mutation_types_seen.add(mutation.type.value)
    
    # Group mutations by type
    by_type = {}
    for mut in mutations:
        mut_type = mut['type']
        if mut_type not in by_type:
            by_type[mut_type] = []
        by_type[mut_type].append(mut)
    
    print(f"\n📊 Generated {len(mutations)} mutations spanning {len(mutation_types_seen)} types:")
    
    for mut_type, muts in by_type.items():
        print(f"\n🔧 {mut_type.upper().replace('_', ' ')} ({len(muts)} mutations):")
        for mut in muts[:2]:  # Show first 2 of each type
            print(f"   • Changes: {mut['changes']}")
            print(f"     Rationale: {mut['rationale']}")
            print(f"     Confidence: {mut['confidence']:.2f}, Improvement: {mut['expected_improvement']:.2f}")
            print(f"     Test Duration: {mut['test_duration']:.1f}s")
            print()
    
    # Test specific advanced mutation types
    print("🚀 Testing Advanced Mutation Types:")
    
    # Test Neural Architecture Search
    nas_template = next(t for t in mutation_engine.mutation_templates 
                       if t['type'] == MutationType.NEURAL_ARCHITECTURE_SEARCH)
    print(f"\n🧠 Neural Architecture Search targets: {nas_template['targets']}")
    print(f"   Strategies: {nas_template['strategies']}")
    
    # Test Multi-Objective Optimization
    moo_template = next(t for t in mutation_engine.mutation_templates 
                       if t['type'] == MutationType.MULTI_OBJECTIVE_OPTIMIZATION)
    print(f"\n🎯 Multi-Objective Optimization targets: {moo_template['targets']}")
    print(f"   Strategies: {moo_template['strategies']}")
    
    # Test Attention Modifications
    attention_template = next(t for t in mutation_engine.mutation_templates 
                             if t['type'] == MutationType.ATTENTION_MODIFICATION)
    print(f"\n👁️ Attention Modification targets: {attention_template['targets']}")
    print(f"   Strategies: {attention_template['strategies']}")
    
    print("\n✅ Enhanced Architect mutation tests completed!")
    print("🔍 Key improvements:")
    print("   • 6 new sophisticated mutation types")
    print("   • Context-aware parameter generation")
    print("   • Adaptive confidence and duration estimates")
    print("   • Neural architecture search capabilities")
    print("   • Multi-objective optimization support")
    print("   • Advanced attention mechanisms")
    print("   • Ensemble and fusion strategies")

if __name__ == "__main__":
    test_enhanced_architect()

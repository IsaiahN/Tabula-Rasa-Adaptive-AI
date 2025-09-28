#!/usr/bin/env python3
"""
Rapid Stress Test for Tier 3 Systems

Tests rapid creation of hypotheses and graph operations to verify
the ID uniqueness fix and overall system stability.
"""

import sys
import os
import asyncio
import time
import sqlite3
from datetime import datetime

# Add paths
sys.path.append('.')
sys.path.append('./core')

async def rapid_bayesian_test():
    """Rapid creation of hypotheses to test ID uniqueness."""
    print("RAPID BAYESIAN STRESS TEST")
    print("=" * 40)

    # Create database
    db = sqlite3.connect(':memory:')
    with open('database/tier3_schema_extension.sql', 'r') as f:
        schema_sql = f.read()

    cursor = db.cursor()
    for statement in schema_sql.split(';'):
        if statement.strip():
            cursor.execute(statement)
    db.commit()

    # Import and initialize
    import importlib.util
    spec = importlib.util.spec_from_file_location('bayesian_inference_engine', 'core/bayesian_inference_engine.py')
    bayesian_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bayesian_module)

    BayesianInferenceEngine = bayesian_module.BayesianInferenceEngine
    HypothesisType = bayesian_module.HypothesisType
    EvidenceType = bayesian_module.EvidenceType

    bayesian = BayesianInferenceEngine(db)
    print("[OK] Bayesian system initialized")

    # Rapid hypothesis creation
    print("Creating 100 hypotheses rapidly...")
    start_time = time.time()
    hypotheses_created = 0
    errors = 0

    for i in range(100):
        try:
            hypothesis_id = await bayesian.create_hypothesis(
                hypothesis_type=HypothesisType.ACTION_OUTCOME,
                description=f"Rapid test hypothesis {i}",
                prior_probability=0.5,
                context_conditions={'test_index': i, 'rapid_test': True},
                game_id='rapid_test',
                session_id=f'rapid_session_{i}'
            )
            hypotheses_created += 1

            # Add evidence rapidly
            await bayesian.add_evidence(
                hypothesis_id=hypothesis_id,
                evidence_type=EvidenceType.DIRECT_OBSERVATION,
                supports_hypothesis=(i % 2 == 0),
                strength=0.8,
                context={'rapid_test': True, 'index': i},
                game_id='rapid_test',
                session_id=f'rapid_session_{i}'
            )

        except Exception as e:
            errors += 1
            print(f"[ERROR] Hypothesis {i}: {e}")

    elapsed = time.time() - start_time

    print(f"Results:")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Hypotheses created: {hypotheses_created}")
    print(f"  Rate: {hypotheses_created/elapsed:.1f} hypotheses/second")
    print(f"  Errors: {errors}")

    # Test prediction generation
    action_candidates = [{'id': 1}, {'id': 2}, {'id': 3}]
    prediction = await bayesian.generate_prediction(
        action_candidates=action_candidates,
        current_context={'rapid_test': True},
        game_id='rapid_test',
        session_id='final_session'
    )

    if prediction:
        print(f"  Generated prediction: Action {prediction.predicted_action.get('id')}")
    else:
        print("  No prediction generated")

    db.close()
    return errors == 0 and hypotheses_created == 100

async def rapid_graph_test():
    """Rapid creation of graphs to test performance."""
    print("\nRAPID GRAPH STRESS TEST")
    print("=" * 40)

    # Create database
    db = sqlite3.connect(':memory:')
    with open('database/tier3_schema_extension.sql', 'r') as f:
        schema_sql = f.read()

    cursor = db.cursor()
    for statement in schema_sql.split(';'):
        if statement.strip():
            cursor.execute(statement)
    db.commit()

    # Import and initialize
    import importlib.util
    spec = importlib.util.spec_from_file_location('enhanced_graph_traversal', 'core/enhanced_graph_traversal.py')
    graph_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_module)

    EnhancedGraphTraversal = graph_module.EnhancedGraphTraversal
    GraphType = graph_module.GraphType
    GraphNode = graph_module.GraphNode
    GraphEdge = graph_module.GraphEdge
    NodeType = graph_module.NodeType
    TraversalAlgorithm = graph_module.TraversalAlgorithm

    graph_traversal = EnhancedGraphTraversal(db)
    print("[OK] Graph traversal system initialized")

    # Rapid graph creation
    print("Creating 20 graphs rapidly...")
    start_time = time.time()
    graphs_created = 0
    traversals_performed = 0
    errors = 0

    for i in range(20):
        try:
            # Create nodes
            nodes = []
            for j in range(5):
                node = GraphNode(
                    node_id=f'rapid_node_{i}_{j}',
                    node_type=NodeType.STATE_NODE,
                    properties={'graph_index': i, 'node_index': j},
                    coordinates=(j * 1.0, i * 0.5)
                )
                nodes.append(node)

            # Create edges
            edges = []
            for j in range(4):
                edge = GraphEdge(
                    edge_id=f'rapid_edge_{i}_{j}',
                    from_node=f'rapid_node_{i}_{j}',
                    to_node=f'rapid_node_{i}_{j+1}',
                    weight=1.0,
                    properties={'graph_index': i}
                )
                edges.append(edge)

            # Create graph
            graph_id = await graph_traversal.create_graph(
                graph_type=GraphType.GAME_STATE_GRAPH,
                initial_nodes=nodes,
                initial_edges=edges,
                game_id=f'rapid_test_{i}'
            )
            graphs_created += 1

            # Perform traversal
            result = await graph_traversal.traverse_graph(
                graph_id=graph_id,
                start_node='rapid_node_{}_0'.format(i),
                end_node='rapid_node_{}_4'.format(i),
                algorithm=TraversalAlgorithm.DIJKSTRA
            )

            if result.success:
                traversals_performed += 1

        except Exception as e:
            errors += 1
            print(f"[ERROR] Graph {i}: {e}")

    elapsed = time.time() - start_time

    print(f"Results:")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Graphs created: {graphs_created}")
    print(f"  Traversals performed: {traversals_performed}")
    print(f"  Rate: {graphs_created/elapsed:.1f} graphs/second")
    print(f"  Errors: {errors}")

    db.close()
    return errors == 0 and graphs_created == 20 and traversals_performed == 20

async def main():
    """Main rapid stress test."""
    print("RAPID TIER 3 STRESS TEST")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

    # Run tests
    bayesian_success = await rapid_bayesian_test()
    graph_success = await rapid_graph_test()

    print(f"\nRAPID STRESS TEST RESULTS")
    print("=" * 50)
    print(f"Bayesian System: {'PASS' if bayesian_success else 'FAIL'}")
    print(f"Graph System: {'PASS' if graph_success else 'FAIL'}")

    overall_success = bayesian_success and graph_success

    print(f"\nOverall Result: {'SUCCESS' if overall_success else 'FAILURE'}")

    if overall_success:
        print("\n✅ RAPID STRESS TEST PASSED!")
        print("- ID uniqueness fix working correctly")
        print("- Systems handle rapid operations well")
        print("- No database constraint violations")
        print("- Performance is good under rapid load")
    else:
        print("\n❌ Issues detected in rapid stress test")

    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())
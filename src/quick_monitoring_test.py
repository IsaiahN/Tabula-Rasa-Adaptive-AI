#!/usr/bin/env python3
"""
Quick Tier 3 Monitoring Test

A faster test to verify the systems are working and provide immediate feedback.
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

async def quick_tier3_test():
    """Quick test of both Tier 3 systems."""
    print("QUICK TIER 3 MONITORING TEST")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

    # Create database
    print("Creating test database...")
    db = sqlite3.connect(':memory:')

    try:
        with open('database/tier3_schema_extension.sql', 'r') as f:
            schema_sql = f.read()

        cursor = db.cursor()
        for statement in schema_sql.split(';'):
            if statement.strip():
                cursor.execute(statement)
        db.commit()
        print("[OK] Database initialized")
    except Exception as e:
        print(f"[ERROR] Database failed: {e}")
        return False

    # Test Bayesian system
    print("\nTesting Bayesian Inference Engine...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('bayesian_inference_engine', 'core/bayesian_inference_engine.py')
        bayesian_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bayesian_module)

        BayesianInferenceEngine = bayesian_module.BayesianInferenceEngine
        HypothesisType = bayesian_module.HypothesisType
        EvidenceType = bayesian_module.EvidenceType

        bayesian = BayesianInferenceEngine(db)
        print("[OK] Bayesian system initialized")

        # Quick test operations
        for i in range(10):
            hypothesis_id = await bayesian.create_hypothesis(
                hypothesis_type=HypothesisType.ACTION_OUTCOME,
                description=f"Quick test hypothesis {i}",
                prior_probability=0.5,
                context_conditions={'test': True, 'iteration': i},
                game_id='quick_test',
                session_id='quick_session'
            )

            await bayesian.add_evidence(
                hypothesis_id=hypothesis_id,
                evidence_type=EvidenceType.DIRECT_OBSERVATION,
                supports_hypothesis=(i % 2 == 0),
                strength=0.8,
                context={'score_change': i * 5},
                game_id='quick_test',
                session_id='quick_session'
            )

        print(f"[OK] Created and tested 10 hypotheses")

        # Test prediction
        prediction = await bayesian.generate_prediction(
            action_candidates=[{'id': 1}, {'id': 2}, {'id': 3}],
            current_context={'test': True},
            game_id='quick_test',
            session_id='quick_session'
        )

        if prediction:
            print(f"[OK] Generated prediction: Action {prediction.predicted_action.get('id')}")
        else:
            print("[INFO] No prediction generated (normal for small dataset)")

    except Exception as e:
        print(f"[ERROR] Bayesian test failed: {e}")
        return False

    # Test Graph system
    print("\nTesting Enhanced Graph Traversal...")
    try:
        spec2 = importlib.util.spec_from_file_location('enhanced_graph_traversal', 'core/enhanced_graph_traversal.py')
        graph_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(graph_module)

        EnhancedGraphTraversal = graph_module.EnhancedGraphTraversal
        GraphType = graph_module.GraphType
        GraphNode = graph_module.GraphNode
        GraphEdge = graph_module.GraphEdge
        NodeType = graph_module.NodeType
        TraversalAlgorithm = graph_module.TraversalAlgorithm

        graph_traversal = EnhancedGraphTraversal(db)
        print("[OK] Graph traversal system initialized")

        # Create test graph
        nodes = []
        for i in range(5):
            node = GraphNode(
                node_id=f'quick_node_{i}',
                node_type=NodeType.STATE_NODE,
                properties={'score': i * 10},
                coordinates=(i * 1.0, i * 0.5)
            )
            nodes.append(node)

        edges = []
        for i in range(4):
            edge = GraphEdge(
                edge_id=f'quick_edge_{i}',
                from_node=f'quick_node_{i}',
                to_node=f'quick_node_{i+1}',
                weight=1.0 + i * 0.5,
                properties={'test': True}
            )
            edges.append(edge)

        graph_id = await graph_traversal.create_graph(
            graph_type=GraphType.GAME_STATE_GRAPH,
            initial_nodes=nodes,
            initial_edges=edges,
            game_id='quick_test'
        )

        print(f"[OK] Created test graph: {graph_id}")

        # Test traversal
        result = await graph_traversal.traverse_graph(
            graph_id=graph_id,
            start_node='quick_node_0',
            end_node='quick_node_4',
            algorithm=TraversalAlgorithm.DIJKSTRA
        )

        if result.success:
            print(f"[OK] Traversal successful: {len(result.primary_path.nodes)} nodes")
        else:
            print("[ERROR] Traversal failed")
            return False

    except Exception as e:
        print(f"[ERROR] Graph test failed: {e}")
        return False

    print(f"\nQUICK TEST COMPLETED")
    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")
    print("[SUCCESS] All Tier 3 systems working correctly!")

    db.close()
    return True

if __name__ == "__main__":
    result = asyncio.run(quick_tier3_test())
    sys.exit(0 if result else 1)
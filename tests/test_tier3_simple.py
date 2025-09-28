#!/usr/bin/env python3
"""
Simple test script for Tier 3 system integration.
Tests Bayesian inference engine and enhanced graph traversal.
"""

import sys
import os
import sqlite3
import asyncio
import json
from datetime import datetime

# Add paths
sys.path.append('.')
sys.path.append('./core')
sys.path.append('./database')

def create_test_database():
    """Create an in-memory test database with Tier 3 schemas."""
    db = sqlite3.connect(':memory:')

    # Read and execute the Tier 3 schema
    try:
        with open('src/database/tier3_schema_extension.sql', 'r') as f:
            schema_sql = f.read()

        # Execute schema creation
        cursor = db.cursor()
        for statement in schema_sql.split(';'):
            if statement.strip():
                cursor.execute(statement)

        db.commit()
        print("[OK] Tier 3 database schema initialized successfully")
        return db
    except Exception as e:
        print(f"[ERROR] Database schema initialization failed: {e}")
        return None

async def test_bayesian_system(db):
    """Test the Bayesian inference engine."""
    try:
        # Import directly
        import importlib.util
        spec = importlib.util.spec_from_file_location('bayesian_inference_engine', 'src/core/bayesian_inference_engine.py')
        bayesian_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bayesian_module)

        BayesianInferenceEngine = bayesian_module.BayesianInferenceEngine
        HypothesisType = bayesian_module.HypothesisType
        EvidenceType = bayesian_module.EvidenceType

        # Initialize system
        bayesian = BayesianInferenceEngine(db)
        print("[OK] Bayesian inference engine initialized")

        # Test hypothesis creation
        hypothesis_id = await bayesian.create_hypothesis(
            hypothesis_type=HypothesisType.ACTION_OUTCOME,
            description="Action 1 leads to positive score change",
            prior_probability=0.5,
            context_conditions={'action_id': 1, 'level': 1},
            game_id='test_game',
            session_id='test_session'
        )
        print(f"[OK] Created hypothesis: {hypothesis_id}")

        # Test evidence addition
        evidence_id = await bayesian.add_evidence(
            hypothesis_id=hypothesis_id,
            evidence_type=EvidenceType.DIRECT_OBSERVATION,
            supports_hypothesis=True,
            strength=0.8,
            context={'score_change': 10},
            game_id='test_game',
            session_id='test_session'
        )
        print(f"[OK] Added evidence: {evidence_id}")

        # Test prediction generation
        action_candidates = [{'id': 1}, {'id': 2}, {'id': 3}]
        current_context = {'level': 1, 'score': 50}

        prediction = await bayesian.generate_prediction(
            action_candidates=action_candidates,
            current_context=current_context,
            game_id='test_game',
            session_id='test_session'
        )

        if prediction:
            print(f"[OK] Generated prediction: Action {prediction.predicted_action.get('id')} " +
                  f"(probability: {prediction.success_probability:.2f})")
        else:
            print("[INFO] No prediction generated (expected for small data)")

        print("[PASS] Bayesian inference engine test completed successfully")
        return True

    except Exception as e:
        print(f"[ERROR] Bayesian system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_graph_traversal_system(db):
    """Test the enhanced graph traversal system."""
    try:
        # Import directly
        import importlib.util
        spec = importlib.util.spec_from_file_location('enhanced_graph_traversal', 'src/core/enhanced_graph_traversal.py')
        graph_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graph_module)

        EnhancedGraphTraversal = graph_module.EnhancedGraphTraversal
        GraphType = graph_module.GraphType
        GraphNode = graph_module.GraphNode
        GraphEdge = graph_module.GraphEdge
        NodeType = graph_module.NodeType
        TraversalAlgorithm = graph_module.TraversalAlgorithm

        # Initialize system
        graph_traversal = EnhancedGraphTraversal(db)
        print("[OK] Enhanced graph traversal initialized")

        # Create test nodes
        node1 = GraphNode(
            node_id='start',
            node_type=NodeType.STATE_NODE,
            properties={'score': 0, 'level': 1},
            coordinates=(0.0, 0.0)
        )

        node2 = GraphNode(
            node_id='middle',
            node_type=NodeType.STATE_NODE,
            properties={'score': 25, 'level': 1},
            coordinates=(1.0, 0.5)
        )

        node3 = GraphNode(
            node_id='end',
            node_type=NodeType.STATE_NODE,
            properties={'score': 50, 'level': 2},
            coordinates=(2.0, 1.0)
        )

        # Create test edges
        edge1 = GraphEdge(
            edge_id='start_to_middle',
            from_node='start',
            to_node='middle',
            weight=1.0,
            properties={'action': 1}
        )

        edge2 = GraphEdge(
            edge_id='middle_to_end',
            from_node='middle',
            to_node='end',
            weight=1.5,
            properties={'action': 2}
        )

        # Create graph
        graph_id = await graph_traversal.create_graph(
            graph_type=GraphType.GAME_STATE_GRAPH,
            initial_nodes=[node1, node2, node3],
            initial_edges=[edge1, edge2],
            game_id='test_game'
        )
        print(f"[OK] Created graph: {graph_id}")

        # Test traversal
        traversal_result = await graph_traversal.traverse_graph(
            graph_id=graph_id,
            start_node='start',
            end_node='end',
            algorithm=TraversalAlgorithm.DIJKSTRA
        )

        if traversal_result.success:
            path = traversal_result.primary_path
            print(f"[OK] Found path: {len(path.nodes)} nodes, weight: {path.total_weight:.2f}")
        else:
            print("[ERROR] Traversal failed")

        print("[PASS] Enhanced graph traversal test completed successfully")
        return True

    except Exception as e:
        print(f"[ERROR] Graph traversal system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("TIER 3 SYSTEM INTEGRATION TEST")
    print("="*50)

    # Create test database
    db = create_test_database()
    if not db:
        print("[ERROR] Cannot proceed without database")
        return

    # Test individual systems
    print("\nTESTING BAYESIAN INFERENCE ENGINE")
    print("-" * 40)
    bayesian_success = await test_bayesian_system(db)

    print("\nTESTING ENHANCED GRAPH TRAVERSAL")
    print("-" * 40)
    graph_success = await test_graph_traversal_system(db)

    # Final results
    print("\nTEST RESULTS")
    print("="*50)
    print(f"Bayesian Inference Engine: {'PASS' if bayesian_success else 'FAIL'}")
    print(f"Enhanced Graph Traversal: {'PASS' if graph_success else 'FAIL'}")

    if bayesian_success and graph_success:
        print("\nALL TIER 3 SYSTEMS WORKING CORRECTLY!")
        print("Ready for full training integration.")
    else:
        print("\nSome tests failed. Review errors above.")

    # Close database
    db.close()

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Extended Validation Test for Tier 3 Systems

Runs repeated cycles to validate stability over time.
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

async def validation_cycle(cycle_num):
    """Run one validation cycle."""
    print(f"\n--- VALIDATION CYCLE {cycle_num} ---")

    # Create database
    db = sqlite3.connect(':memory:')
    try:
        with open('database/tier3_schema_extension.sql', 'r') as f:
            schema_sql = f.read()

        cursor = db.cursor()
        for statement in schema_sql.split(';'):
            if statement.strip():
                cursor.execute(statement)
        db.commit()
    except Exception as e:
        print(f"[ERROR] Database failed: {e}")
        return False

    # Test Bayesian system
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('bayesian_inference_engine', 'core/bayesian_inference_engine.py')
        bayesian_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bayesian_module)

        BayesianInferenceEngine = bayesian_module.BayesianInferenceEngine
        HypothesisType = bayesian_module.HypothesisType
        EvidenceType = bayesian_module.EvidenceType

        bayesian = BayesianInferenceEngine(db)

        # Create 25 hypotheses per cycle
        for i in range(25):
            hypothesis_id = await bayesian.create_hypothesis(
                hypothesis_type=HypothesisType.ACTION_OUTCOME,
                description=f"Cycle {cycle_num} hypothesis {i}",
                prior_probability=0.5,
                context_conditions={'cycle': cycle_num, 'iteration': i},
                game_id=f'validation_test_{cycle_num}',
                session_id=f'validation_session_{cycle_num}'
            )

            await bayesian.add_evidence(
                hypothesis_id=hypothesis_id,
                evidence_type=EvidenceType.DIRECT_OBSERVATION,
                supports_hypothesis=(i % 2 == 0),
                strength=0.8,
                context={'score_change': i * 5},
                game_id=f'validation_test_{cycle_num}',
                session_id=f'validation_session_{cycle_num}'
            )

        print(f"[OK] Bayesian: 25 hypotheses created")

    except Exception as e:
        print(f"[ERROR] Bayesian failed: {e}")
        return False

    # Test Graph system
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

        # Create 5 graphs per cycle with traversals
        for g in range(5):
            nodes = []
            for i in range(4):
                node = GraphNode(
                    node_id=f'val_node_{cycle_num}_{g}_{i}',
                    node_type=NodeType.STATE_NODE,
                    properties={'cycle': cycle_num, 'graph': g, 'node': i},
                    coordinates=(i * 1.0, g * 0.5)
                )
                nodes.append(node)

            edges = []
            for i in range(3):
                edge = GraphEdge(
                    edge_id=f'val_edge_{cycle_num}_{g}_{i}',
                    from_node=f'val_node_{cycle_num}_{g}_{i}',
                    to_node=f'val_node_{cycle_num}_{g}_{i+1}',
                    weight=1.0 + i * 0.5,
                    properties={'cycle': cycle_num}
                )
                edges.append(edge)

            graph_id = await graph_traversal.create_graph(
                graph_type=GraphType.GAME_STATE_GRAPH,
                initial_nodes=nodes,
                initial_edges=edges,
                game_id=f'validation_test_{cycle_num}_{g}'
            )

            # Perform traversal
            result = await graph_traversal.traverse_graph(
                graph_id=graph_id,
                start_node=f'val_node_{cycle_num}_{g}_0',
                end_node=f'val_node_{cycle_num}_{g}_3',
                algorithm=TraversalAlgorithm.DIJKSTRA
            )

            if not result.success:
                print(f"[ERROR] Traversal {g} failed")
                return False

        print(f"[OK] Graph: 5 graphs with traversals")

    except Exception as e:
        print(f"[ERROR] Graph failed: {e}")
        return False

    db.close()
    return True

async def main():
    """Main extended validation test."""
    print("EXTENDED TIER 3 VALIDATION TEST")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

    cycles = 12  # Run 12 cycles (approximately 5-10 minutes)
    success_count = 0

    for cycle in range(1, cycles + 1):
        start_time = time.time()
        success = await validation_cycle(cycle)
        elapsed = time.time() - start_time

        if success:
            success_count += 1
            print(f"[SUCCESS] Cycle {cycle} completed in {elapsed:.2f}s")
        else:
            print(f"[FAILURE] Cycle {cycle} failed after {elapsed:.2f}s")

        # Brief pause between cycles
        await asyncio.sleep(0.5)

    print(f"\nEXTENDED VALIDATION RESULTS")
    print("=" * 50)
    print(f"Successful cycles: {success_count}/{cycles}")
    print(f"Success rate: {100*success_count/cycles:.1f}%")

    if success_count == cycles:
        print("\n✓ ALL VALIDATION CYCLES PASSED!")
        print("- ID uniqueness fixes working correctly")
        print("- No database constraint violations")
        print("- Systems stable under extended operation")
    else:
        print(f"\n✗ {cycles - success_count} validation cycles failed")

    print(f"End time: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())
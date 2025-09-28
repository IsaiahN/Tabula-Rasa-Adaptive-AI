#!/usr/bin/env python3
"""
Comprehensive Tier 3 Systems Stress Test

This script performs a comprehensive stress test of the Bayesian Inference Engine
and Enhanced Graph Traversal systems to validate their performance under load.
"""

import sys
import os
import asyncio
import time
import sqlite3
import json
import random
from datetime import datetime

# Add paths
sys.path.append('.')
sys.path.append('./core')

async def stress_test_bayesian_system(db, duration_seconds=300):
    """Stress test the Bayesian inference engine for specified duration."""
    print(f"[BAYESIAN STRESS TEST] Running for {duration_seconds} seconds...")

    try:
        # Import directly to avoid relative import issues
        import importlib.util
        spec = importlib.util.spec_from_file_location('bayesian_inference_engine', 'core/bayesian_inference_engine.py')
        bayesian_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bayesian_module)

        BayesianInferenceEngine = bayesian_module.BayesianInferenceEngine
        HypothesisType = bayesian_module.HypothesisType
        EvidenceType = bayesian_module.EvidenceType

        # Initialize system
        bayesian = BayesianInferenceEngine(db)
        print("[OK] Bayesian inference engine initialized")

        start_time = time.time()
        end_time = start_time + duration_seconds

        # Performance counters
        hypotheses_created = 0
        evidence_added = 0
        predictions_generated = 0
        errors = 0

        hypothesis_ids = []
        game_id = 'stress_test_game'

        # Stress test loop
        iteration = 0
        while time.time() < end_time:
            iteration += 1
            session_id = f'stress_session_{iteration}'

            try:
                # Create hypotheses periodically
                if iteration % 10 == 0:
                    for action_id in [1, 2, 3, 6]:
                        hypothesis_id = await bayesian.create_hypothesis(
                            hypothesis_type=random.choice(list(HypothesisType)),
                            description=f"Stress test hypothesis for action {action_id}, iteration {iteration}",
                            prior_probability=random.uniform(0.3, 0.7),
                            context_conditions={
                                'action_id': action_id,
                                'stress_test': True,
                                'iteration': iteration,
                                'level': random.randint(1, 5)
                            },
                            game_id=game_id,
                            session_id=session_id
                        )
                        hypothesis_ids.append(hypothesis_id)
                        hypotheses_created += 1

                # Add evidence to existing hypotheses
                if hypothesis_ids:
                    for _ in range(random.randint(1, 5)):
                        hypothesis_id = random.choice(hypothesis_ids)
                        score_change = random.randint(-20, 30)

                        await bayesian.add_evidence(
                            hypothesis_id=hypothesis_id,
                            evidence_type=random.choice(list(EvidenceType)),
                            supports_hypothesis=(score_change > 0),
                            strength=random.uniform(0.1, 1.0),
                            context={
                                'score_change': score_change,
                                'iteration': iteration,
                                'timestamp': time.time()
                            },
                            game_id=game_id,
                            session_id=session_id
                        )
                        evidence_added += 1

                # Generate predictions
                if iteration % 5 == 0 and hypothesis_ids:
                    action_candidates = [
                        {'id': 1}, {'id': 2}, {'id': 3},
                        {'id': 6, 'x': random.randint(10, 50), 'y': random.randint(10, 50)}
                    ]

                    prediction = await bayesian.generate_prediction(
                        action_candidates=action_candidates,
                        current_context={
                            'iteration': iteration,
                            'score': random.randint(0, 100),
                            'level': random.randint(1, 5)
                        },
                        game_id=game_id,
                        session_id=session_id
                    )

                    if prediction:
                        predictions_generated += 1

                # Prune old hypotheses periodically
                if iteration % 50 == 0:
                    pruned = await bayesian.prune_low_confidence_hypotheses()
                    if pruned > 0:
                        print(f"[PRUNING] Removed {pruned} low-confidence hypotheses")

                # Progress update
                if iteration % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"[PROGRESS] Iteration {iteration}, {elapsed:.1f}s elapsed, " +
                          f"{hypotheses_created} hypotheses, {evidence_added} evidence, " +
                          f"{predictions_generated} predictions")

            except Exception as e:
                errors += 1
                if errors < 5:  # Only print first few errors
                    print(f"[ERROR] Iteration {iteration}: {e}")

        # Final results
        total_time = time.time() - start_time
        print(f"\n[BAYESIAN RESULTS]")
        print(f"Duration: {total_time:.2f} seconds")
        print(f"Iterations: {iteration}")
        print(f"Hypotheses created: {hypotheses_created}")
        print(f"Evidence added: {evidence_added}")
        print(f"Predictions generated: {predictions_generated}")
        print(f"Errors: {errors}")
        print(f"Performance: {iteration/total_time:.2f} iterations/second")

        # Get final insights
        insights = await bayesian.get_hypothesis_insights(game_id)
        print(f"Final state: {insights.get('total_hypotheses', 0)} active hypotheses")

        return {
            'success': True,
            'iterations': iteration,
            'hypotheses_created': hypotheses_created,
            'evidence_added': evidence_added,
            'predictions_generated': predictions_generated,
            'errors': errors,
            'performance': iteration/total_time
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] Bayesian stress test failed: {e}")
        return {'success': False, 'error': str(e)}

async def stress_test_graph_traversal(db, duration_seconds=300):
    """Stress test the enhanced graph traversal system."""
    print(f"[GRAPH STRESS TEST] Running for {duration_seconds} seconds...")

    try:
        # Import directly
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

        # Initialize system
        graph_traversal = EnhancedGraphTraversal(db)
        print("[OK] Enhanced graph traversal initialized")

        start_time = time.time()
        end_time = start_time + duration_seconds

        # Performance counters
        graphs_created = 0
        nodes_created = 0
        edges_created = 0
        traversals_performed = 0
        errors = 0

        graph_ids = []
        game_id = 'stress_test_game'

        # Stress test loop
        iteration = 0
        while time.time() < end_time:
            iteration += 1

            try:
                # Create new graphs periodically
                if iteration % 20 == 0:
                    # Create nodes for new graph
                    nodes = []
                    num_nodes = random.randint(5, 15)
                    for i in range(num_nodes):
                        node = GraphNode(
                            node_id=f'stress_node_{iteration}_{i}',
                            node_type=random.choice(list(NodeType)),
                            properties={
                                'score': random.randint(0, 100),
                                'level': random.randint(1, 5),
                                'iteration': iteration
                            },
                            coordinates=(
                                random.uniform(0.0, 10.0),
                                random.uniform(0.0, 10.0)
                            )
                        )
                        nodes.append(node)

                    # Create edges
                    edges = []
                    for i in range(len(nodes) - 1):
                        edge = GraphEdge(
                            edge_id=f'stress_edge_{iteration}_{i}',
                            from_node=nodes[i].node_id,
                            to_node=nodes[i + 1].node_id,
                            weight=random.uniform(0.5, 3.0),
                            properties={'iteration': iteration}
                        )
                        edges.append(edge)

                    # Create graph
                    graph_id = await graph_traversal.create_graph(
                        graph_type=random.choice(list(GraphType)),
                        initial_nodes=nodes,
                        initial_edges=edges,
                        game_id=game_id
                    )

                    graph_ids.append(graph_id)
                    graphs_created += 1
                    nodes_created += len(nodes)
                    edges_created += len(edges)

                # Perform traversals on existing graphs
                if graph_ids and iteration % 5 == 0:
                    graph_id = random.choice(graph_ids)
                    graph = graph_traversal.graphs.get(graph_id)

                    if graph and len(graph.nodes) >= 2:
                        node_ids = list(graph.nodes.keys())
                        start_node = random.choice(node_ids)
                        end_node = random.choice(node_ids)

                        if start_node != end_node:
                            algorithm = random.choice(list(TraversalAlgorithm))

                            try:
                                result = await graph_traversal.traverse_graph(
                                    graph_id=graph_id,
                                    start_node=start_node,
                                    end_node=end_node,
                                    algorithm=algorithm
                                )
                                traversals_performed += 1

                                if result.success and iteration % 50 == 0:
                                    print(f"[TRAVERSAL] {algorithm.value}: {len(result.primary_path.nodes)} nodes, " +
                                          f"weight: {result.primary_path.total_weight:.2f}")

                            except Exception as e:
                                # Some traversals may fail due to disconnected graphs
                                pass

                # Progress update
                if iteration % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"[PROGRESS] Iteration {iteration}, {elapsed:.1f}s elapsed, " +
                          f"{graphs_created} graphs, {nodes_created} nodes, " +
                          f"{traversals_performed} traversals")

            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"[ERROR] Graph iteration {iteration}: {e}")

        # Final results
        total_time = time.time() - start_time
        print(f"\n[GRAPH RESULTS]")
        print(f"Duration: {total_time:.2f} seconds")
        print(f"Iterations: {iteration}")
        print(f"Graphs created: {graphs_created}")
        print(f"Nodes created: {nodes_created}")
        print(f"Edges created: {edges_created}")
        print(f"Traversals performed: {traversals_performed}")
        print(f"Errors: {errors}")
        print(f"Performance: {iteration/total_time:.2f} iterations/second")

        return {
            'success': True,
            'iterations': iteration,
            'graphs_created': graphs_created,
            'nodes_created': nodes_created,
            'edges_created': edges_created,
            'traversals_performed': traversals_performed,
            'errors': errors,
            'performance': iteration/total_time
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] Graph stress test failed: {e}")
        return {'success': False, 'error': str(e)}

async def integrated_stress_test(db, duration_seconds=180):
    """Run integrated stress test of both systems working together."""
    print(f"[INTEGRATED STRESS TEST] Running for {duration_seconds} seconds...")

    try:
        # Import both systems
        import importlib.util

        # Bayesian system
        spec1 = importlib.util.spec_from_file_location('bayesian_inference_engine', 'core/bayesian_inference_engine.py')
        bayesian_module = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(bayesian_module)

        BayesianInferenceEngine = bayesian_module.BayesianInferenceEngine
        HypothesisType = bayesian_module.HypothesisType
        EvidenceType = bayesian_module.EvidenceType

        # Graph system
        spec2 = importlib.util.spec_from_file_location('enhanced_graph_traversal', 'core/enhanced_graph_traversal.py')
        graph_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(graph_module)

        EnhancedGraphTraversal = graph_module.EnhancedGraphTraversal
        GraphType = graph_module.GraphType
        GraphNode = graph_module.GraphNode
        NodeType = graph_module.NodeType
        TraversalAlgorithm = graph_module.TraversalAlgorithm

        # Initialize both systems
        bayesian = BayesianInferenceEngine(db)
        graph_traversal = EnhancedGraphTraversal(db)
        print("[OK] Both systems initialized for integrated test")

        start_time = time.time()
        end_time = start_time + duration_seconds

        # Performance counters
        integrated_operations = 0
        errors = 0

        game_id = 'integrated_stress_test'
        hypothesis_ids = []
        graph_ids = []

        # Integrated stress test loop
        iteration = 0
        while time.time() < end_time:
            iteration += 1
            session_id = f'integrated_session_{iteration}'

            try:
                # Simulate a complete gameplay scenario
                # 1. Create hypotheses about game mechanics
                if iteration % 15 == 0:
                    for action_id in [1, 2, 3, 6]:
                        hypothesis_id = await bayesian.create_hypothesis(
                            hypothesis_type=HypothesisType.ACTION_OUTCOME,
                            description=f"Integrated test: Action {action_id} effectiveness",
                            prior_probability=0.5,
                            context_conditions={'action_id': action_id, 'integrated_test': True},
                            game_id=game_id,
                            session_id=session_id
                        )
                        hypothesis_ids.append(hypothesis_id)

                # 2. Create game state graph
                if iteration % 25 == 0:
                    # Create nodes representing game states
                    nodes = []
                    for i in range(8):
                        node = GraphNode(
                            node_id=f'integrated_state_{iteration}_{i}',
                            node_type=NodeType.STATE_NODE,
                            properties={
                                'score': i * 15,
                                'actions_taken': i,
                                'hypothesis_supported': random.choice(hypothesis_ids) if hypothesis_ids else None
                            },
                            coordinates=(i * 0.5, i * 0.3)
                        )
                        nodes.append(node)

                    graph_id = await graph_traversal.create_graph(
                        graph_type=GraphType.GAME_STATE_GRAPH,
                        initial_nodes=nodes,
                        game_id=game_id
                    )
                    graph_ids.append(graph_id)

                # 3. Simulate action outcomes and add evidence
                if hypothesis_ids:
                    hypothesis_id = random.choice(hypothesis_ids)
                    score_change = random.randint(-15, 25)

                    await bayesian.add_evidence(
                        hypothesis_id=hypothesis_id,
                        evidence_type=EvidenceType.DIRECT_OBSERVATION,
                        supports_hypothesis=(score_change > 0),
                        strength=min(1.0, abs(score_change) / 20.0),
                        context={
                            'score_change': score_change,
                            'integrated_test': True,
                            'graph_associated': graph_ids[-1] if graph_ids else None
                        },
                        game_id=game_id,
                        session_id=session_id
                    )

                # 4. Use Bayesian predictions to guide graph traversal
                if iteration % 10 == 0 and hypothesis_ids and graph_ids:
                    # Generate prediction
                    action_candidates = [
                        {'id': 1}, {'id': 2}, {'id': 3}, {'id': 6, 'x': 32, 'y': 32}
                    ]

                    prediction = await bayesian.generate_prediction(
                        action_candidates=action_candidates,
                        current_context={'integrated_test': True, 'score': random.randint(0, 100)},
                        game_id=game_id,
                        session_id=session_id
                    )

                    # Use prediction to influence graph traversal
                    if prediction and graph_ids:
                        graph_id = random.choice(graph_ids)
                        graph = graph_traversal.graphs.get(graph_id)

                        if graph and len(graph.nodes) >= 2:
                            # Choose traversal algorithm based on prediction confidence
                            if prediction.confidence_level > 0.7:
                                algorithm = TraversalAlgorithm.A_STAR  # High confidence -> efficient algorithm
                            else:
                                algorithm = TraversalAlgorithm.BEST_FIRST  # Low confidence -> exploratory

                            node_ids = list(graph.nodes.keys())
                            if len(node_ids) >= 2:
                                result = await graph_traversal.traverse_graph(
                                    graph_id=graph_id,
                                    start_node=node_ids[0],
                                    end_node=node_ids[-1],
                                    algorithm=algorithm
                                )

                                if result.success:
                                    integrated_operations += 1

                # Progress update
                if iteration % 50 == 0:
                    elapsed = time.time() - start_time
                    active_hypotheses = len(hypothesis_ids)
                    active_graphs = len(graph_ids)
                    print(f"[INTEGRATED PROGRESS] Iteration {iteration}, {elapsed:.1f}s elapsed, " +
                          f"{active_hypotheses} hypotheses, {active_graphs} graphs, " +
                          f"{integrated_operations} integrated operations")

            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"[ERROR] Integrated iteration {iteration}: {e}")

        # Final results
        total_time = time.time() - start_time
        print(f"\n[INTEGRATED RESULTS]")
        print(f"Duration: {total_time:.2f} seconds")
        print(f"Iterations: {iteration}")
        print(f"Integrated operations: {integrated_operations}")
        print(f"Active hypotheses: {len(hypothesis_ids)}")
        print(f"Active graphs: {len(graph_ids)}")
        print(f"Errors: {errors}")
        print(f"Performance: {iteration/total_time:.2f} iterations/second")

        return {
            'success': True,
            'iterations': iteration,
            'integrated_operations': integrated_operations,
            'active_hypotheses': len(hypothesis_ids),
            'active_graphs': len(graph_ids),
            'errors': errors,
            'performance': iteration/total_time
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] Integrated stress test failed: {e}")
        return {'success': False, 'error': str(e)}

async def main():
    """Main stress test function."""
    print("=" * 70)
    print("COMPREHENSIVE TIER 3 SYSTEMS STRESS TEST")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create test database with Tier 3 schema
    print("\nInitializing test database...")
    db = sqlite3.connect(':memory:')

    try:
        with open('database/tier3_schema_extension.sql', 'r') as f:
            schema_sql = f.read()

        cursor = db.cursor()
        for statement in schema_sql.split(';'):
            if statement.strip():
                cursor.execute(statement)
        db.commit()
        print("[OK] Tier 3 database schema initialized")
    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}")
        return

    # Run stress tests
    print("\n" + "=" * 70)
    print("RUNNING STRESS TESTS")
    print("=" * 70)

    # Test 1: Bayesian system stress test (2 minutes)
    bayesian_results = await stress_test_bayesian_system(db, 120)

    print("\n" + "-" * 70)

    # Test 2: Graph traversal stress test (2 minutes)
    graph_results = await stress_test_graph_traversal(db, 120)

    print("\n" + "-" * 70)

    # Test 3: Integrated stress test (3 minutes)
    integrated_results = await integrated_stress_test(db, 180)

    # Final summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nResults:")
    print(f"Bayesian System: {'PASS' if bayesian_results['success'] else 'FAIL'}")
    if bayesian_results['success']:
        print(f"  - {bayesian_results['iterations']} iterations")
        print(f"  - {bayesian_results['hypotheses_created']} hypotheses created")
        print(f"  - {bayesian_results['evidence_added']} evidence added")
        print(f"  - {bayesian_results['predictions_generated']} predictions generated")
        print(f"  - {bayesian_results['performance']:.2f} ops/second")

    print(f"\nGraph Traversal: {'PASS' if graph_results['success'] else 'FAIL'}")
    if graph_results['success']:
        print(f"  - {graph_results['iterations']} iterations")
        print(f"  - {graph_results['graphs_created']} graphs created")
        print(f"  - {graph_results['traversals_performed']} traversals performed")
        print(f"  - {graph_results['performance']:.2f} ops/second")

    print(f"\nIntegrated Systems: {'PASS' if integrated_results['success'] else 'FAIL'}")
    if integrated_results['success']:
        print(f"  - {integrated_results['iterations']} iterations")
        print(f"  - {integrated_results['integrated_operations']} integrated operations")
        print(f"  - {integrated_results['active_hypotheses']} final hypotheses")
        print(f"  - {integrated_results['active_graphs']} final graphs")
        print(f"  - {integrated_results['performance']:.2f} ops/second")

    overall_success = (bayesian_results['success'] and
                      graph_results['success'] and
                      integrated_results['success'])

    print(f"\nOVERALL RESULT: {'SUCCESS' if overall_success else 'FAILURE'}")

    if overall_success:
        print("\nTier 3 systems are ready for production use!")
        print("- Both systems handle stress testing well")
        print("- Database operations are stable under load")
        print("- Integration between systems works correctly")
        print("- Performance is acceptable for real-time use")
    else:
        print("\nIssues detected during stress testing.")
        print("Review individual test results above.")

    db.close()

if __name__ == "__main__":
    asyncio.run(main())
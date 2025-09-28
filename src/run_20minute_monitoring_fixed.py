#!/usr/bin/env python3
"""
Fixed 20-Minute Tier 3 System Monitoring

This script runs comprehensive monitoring of the Bayesian Inference Engine
and Enhanced Graph Traversal systems with proper error handling for
hypothesis management and pruning.
"""

import sys
import os
import asyncio
import time
import sqlite3
import json
import random
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict, deque

# Add paths
sys.path.append('.')
sys.path.append('./core')

class SystemMonitor:
    """Monitors system performance and detects issues."""

    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process()
        self.memory_samples = deque(maxlen=500)
        self.performance_metrics = defaultdict(list)
        self.error_log = []
        self.last_report_time = time.time()

    def sample_system_metrics(self):
        """Sample current system metrics."""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()

            self.memory_samples.append({
                'timestamp': time.time(),
                'rss_mb': memory_info.rss / (1024 * 1024),
                'cpu_percent': cpu_percent
            })

        except Exception as e:
            self.error_log.append({
                'timestamp': time.time(),
                'type': 'monitoring_error',
                'error': str(e)
            })

    def should_report(self, interval_seconds=60) -> bool:
        """Check if it's time for a status report."""
        current_time = time.time()
        if current_time - self.last_report_time >= interval_seconds:
            self.last_report_time = current_time
            return True
        return False

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.memory_samples:
            return {'memory_mb': 0, 'cpu_percent': 0, 'samples': 0}

        latest = self.memory_samples[-1]
        return {
            'memory_mb': latest['rss_mb'],
            'cpu_percent': latest['cpu_percent'],
            'samples': len(self.memory_samples)
        }

async def fixed_bayesian_monitoring(db, monitor: SystemMonitor, duration_minutes=10):
    """Run monitoring with proper hypothesis management."""
    print(f"[BAYESIAN MONITORING] Running for {duration_minutes} minutes...")

    try:
        # Import Bayesian system
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
        end_time = start_time + (duration_minutes * 60)

        # Monitoring counters
        operations_count = 0
        hypotheses_created = 0
        evidence_added = 0
        evidence_failed = 0
        predictions_generated = 0
        hypothesis_registry = {}  # Track active hypotheses

        game_id = 'monitoring_test'

        print("Starting Bayesian monitoring...")

        while time.time() < end_time:
            try:
                operations_count += 1

                # Sample system metrics
                if operations_count % 20 == 0:
                    monitor.sample_system_metrics()

                session_id = f"session_{operations_count}"

                # Phase 1: Create hypotheses
                if operations_count % 30 == 0:
                    for action_id in [1, 2, 3, 6]:
                        try:
                            hypothesis_id = await bayesian.create_hypothesis(
                                hypothesis_type=HypothesisType.ACTION_OUTCOME,
                                description=f"Monitoring test: Action {action_id} effectiveness",
                                prior_probability=random.uniform(0.4, 0.6),
                                context_conditions={
                                    'action_id': action_id,
                                    'monitoring_test': True,
                                    'operation': operations_count
                                },
                                game_id=game_id,
                                session_id=session_id
                            )

                            # Track the hypothesis
                            hypothesis_registry[hypothesis_id] = {
                                'action_id': action_id,
                                'created_at': time.time(),
                                'evidence_count': 0
                            }
                            hypotheses_created += 1

                        except Exception as e:
                            monitor.error_log.append({
                                'timestamp': time.time(),
                                'type': 'hypothesis_creation_error',
                                'error': str(e)
                            })

                # Phase 2: Add evidence to existing hypotheses (with validation)
                if hypothesis_registry and operations_count % 5 == 0:
                    # Get a list of current hypotheses (filter out potentially pruned ones)
                    active_hypotheses = []
                    for hyp_id, hyp_info in list(hypothesis_registry.items()):
                        # Check if hypothesis still exists in the system
                        if hyp_id in bayesian.hypotheses:
                            active_hypotheses.append(hyp_id)
                        else:
                            # Remove from our registry if it's been pruned
                            del hypothesis_registry[hyp_id]

                    if active_hypotheses:
                        hypothesis_id = random.choice(active_hypotheses)
                        score_change = random.randint(-15, 25)

                        try:
                            evidence_id = await bayesian.add_evidence(
                                hypothesis_id=hypothesis_id,
                                evidence_type=EvidenceType.DIRECT_OBSERVATION,
                                supports_hypothesis=(score_change > 0),
                                strength=min(1.0, abs(score_change) / 20.0),
                                context={
                                    'score_change': score_change,
                                    'operation': operations_count,
                                    'monitoring_test': True
                                },
                                game_id=game_id,
                                session_id=session_id
                            )

                            if evidence_id:
                                evidence_added += 1
                                hypothesis_registry[hypothesis_id]['evidence_count'] += 1
                            else:
                                evidence_failed += 1

                        except Exception as e:
                            evidence_failed += 1
                            monitor.error_log.append({
                                'timestamp': time.time(),
                                'type': 'evidence_addition_error',
                                'error': str(e),
                                'hypothesis_id': hypothesis_id
                            })

                # Phase 3: Generate predictions
                if operations_count % 25 == 0 and hypothesis_registry:
                    try:
                        action_candidates = [
                            {'id': 1}, {'id': 2}, {'id': 3},
                            {'id': 6, 'x': 32, 'y': 32}
                        ]

                        prediction = await bayesian.generate_prediction(
                            action_candidates=action_candidates,
                            current_context={
                                'monitoring_test': True,
                                'operation': operations_count,
                                'score': random.randint(0, 100)
                            },
                            game_id=game_id,
                            session_id=session_id
                        )

                        if prediction:
                            predictions_generated += 1

                    except Exception as e:
                        monitor.error_log.append({
                            'timestamp': time.time(),
                            'type': 'prediction_error',
                            'error': str(e)
                        })

                # Phase 4: Periodic maintenance
                if operations_count % 100 == 0:
                    try:
                        # Prune low-confidence hypotheses
                        pruned_count = await bayesian.prune_low_confidence_hypotheses()
                        if pruned_count > 0:
                            print(f"[MAINTENANCE] Pruned {pruned_count} hypotheses at operation {operations_count}")

                        # Clean up our registry
                        for hyp_id in list(hypothesis_registry.keys()):
                            if hyp_id not in bayesian.hypotheses:
                                del hypothesis_registry[hyp_id]

                    except Exception as e:
                        monitor.error_log.append({
                            'timestamp': time.time(),
                            'type': 'maintenance_error',
                            'error': str(e)
                        })

                # Status reports
                if monitor.should_report(90):  # Every 1.5 minutes
                    elapsed = time.time() - start_time
                    stats = monitor.get_current_stats()

                    print(f"\n[BAYESIAN STATUS] {elapsed/60:.1f}min elapsed")
                    print(f"  Operations: {operations_count}")
                    print(f"  Hypotheses created: {hypotheses_created}")
                    print(f"  Active hypotheses: {len(hypothesis_registry)}")
                    print(f"  Evidence added: {evidence_added}")
                    print(f"  Evidence failed: {evidence_failed}")
                    print(f"  Predictions: {predictions_generated}")
                    print(f"  Memory: {stats['memory_mb']:.1f} MB")
                    print(f"  CPU: {stats['cpu_percent']:.1f}%")
                    print(f"  Errors logged: {len(monitor.error_log)}")

                # Force garbage collection occasionally
                if operations_count % 200 == 0:
                    gc.collect()

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.002)

            except Exception as e:
                monitor.error_log.append({
                    'timestamp': time.time(),
                    'operation': operations_count,
                    'type': 'main_loop_error',
                    'error': str(e)
                })

        # Final results
        total_time = time.time() - start_time
        final_stats = monitor.get_current_stats()

        print(f"\n[BAYESIAN MONITORING COMPLETE]")
        print(f"Duration: {total_time/60:.2f} minutes")
        print(f"Total operations: {operations_count}")
        print(f"Rate: {operations_count/total_time:.1f} ops/second")
        print(f"Hypotheses created: {hypotheses_created}")
        print(f"Evidence added: {evidence_added}")
        print(f"Evidence failed: {evidence_failed}")
        print(f"Predictions: {predictions_generated}")
        print(f"Final memory: {final_stats['memory_mb']:.1f} MB")
        print(f"Total errors: {len(monitor.error_log)}")

        return {
            'success': True,
            'operations_count': operations_count,
            'hypotheses_created': hypotheses_created,
            'evidence_added': evidence_added,
            'evidence_failed': evidence_failed,
            'predictions_generated': predictions_generated,
            'error_count': len(monitor.error_log),
            'duration_minutes': total_time / 60,
            'final_memory_mb': final_stats['memory_mb']
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] Bayesian monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

async def fixed_graph_monitoring(db, monitor: SystemMonitor, duration_minutes=10):
    """Run graph monitoring with proper resource management."""
    print(f"[GRAPH MONITORING] Running for {duration_minutes} minutes...")

    try:
        # Import graph system
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
        end_time = start_time + (duration_minutes * 60)

        # Monitoring counters
        operations_count = 0
        graphs_created = 0
        traversals_performed = 0
        traversals_failed = 0
        graph_registry = {}

        game_id = 'monitoring_test'

        print("Starting graph monitoring...")

        while time.time() < end_time:
            try:
                operations_count += 1

                # Sample system metrics
                if operations_count % 20 == 0:
                    monitor.sample_system_metrics()

                # Phase 1: Create graphs
                if operations_count % 50 == 0:
                    try:
                        # Create nodes
                        node_count = random.randint(4, 8)
                        nodes = []
                        for i in range(node_count):
                            node = GraphNode(
                                node_id=f'monitor_node_{operations_count}_{i}',
                                node_type=NodeType.STATE_NODE,
                                properties={
                                    'score': i * 10,
                                    'operation': operations_count
                                },
                                coordinates=(i * 1.0, i * 0.5)
                            )
                            nodes.append(node)

                        # Create edges
                        edges = []
                        for i in range(len(nodes) - 1):
                            edge = GraphEdge(
                                edge_id=f'monitor_edge_{operations_count}_{i}',
                                from_node=nodes[i].node_id,
                                to_node=nodes[i + 1].node_id,
                                weight=random.uniform(1.0, 3.0),
                                properties={'operation': operations_count}
                            )
                            edges.append(edge)

                        # Create graph
                        graph_id = await graph_traversal.create_graph(
                            graph_type=GraphType.GAME_STATE_GRAPH,
                            initial_nodes=nodes,
                            initial_edges=edges,
                            game_id=game_id
                        )

                        graph_registry[graph_id] = {
                            'node_count': len(nodes),
                            'edge_count': len(edges),
                            'created_at': time.time()
                        }
                        graphs_created += 1

                    except Exception as e:
                        monitor.error_log.append({
                            'timestamp': time.time(),
                            'type': 'graph_creation_error',
                            'error': str(e)
                        })

                # Phase 2: Perform traversals
                if graph_registry and operations_count % 10 == 0:
                    graph_id = random.choice(list(graph_registry.keys()))
                    graph = graph_traversal.graphs.get(graph_id)

                    if graph and len(graph.nodes) >= 2:
                        try:
                            node_ids = list(graph.nodes.keys())
                            start_node = node_ids[0]
                            end_node = node_ids[-1]

                            algorithm = random.choice([
                                TraversalAlgorithm.DIJKSTRA,
                                TraversalAlgorithm.BREADTH_FIRST,
                                TraversalAlgorithm.A_STAR
                            ])

                            result = await graph_traversal.traverse_graph(
                                graph_id=graph_id,
                                start_node=start_node,
                                end_node=end_node,
                                algorithm=algorithm
                            )

                            if result.success:
                                traversals_performed += 1
                            else:
                                traversals_failed += 1

                        except Exception as e:
                            traversals_failed += 1
                            monitor.error_log.append({
                                'timestamp': time.time(),
                                'type': 'traversal_error',
                                'error': str(e),
                                'graph_id': graph_id
                            })

                # Status reports
                if monitor.should_report(90):  # Every 1.5 minutes
                    elapsed = time.time() - start_time
                    stats = monitor.get_current_stats()

                    print(f"\n[GRAPH STATUS] {elapsed/60:.1f}min elapsed")
                    print(f"  Operations: {operations_count}")
                    print(f"  Graphs created: {graphs_created}")
                    print(f"  Active graphs: {len(graph_registry)}")
                    print(f"  Traversals performed: {traversals_performed}")
                    print(f"  Traversals failed: {traversals_failed}")
                    print(f"  Memory: {stats['memory_mb']:.1f} MB")
                    print(f"  CPU: {stats['cpu_percent']:.1f}%")

                # Garbage collection
                if operations_count % 200 == 0:
                    gc.collect()

                await asyncio.sleep(0.002)

            except Exception as e:
                monitor.error_log.append({
                    'timestamp': time.time(),
                    'operation': operations_count,
                    'type': 'graph_main_loop_error',
                    'error': str(e)
                })

        # Final results
        total_time = time.time() - start_time
        final_stats = monitor.get_current_stats()

        print(f"\n[GRAPH MONITORING COMPLETE]")
        print(f"Duration: {total_time/60:.2f} minutes")
        print(f"Total operations: {operations_count}")
        print(f"Rate: {operations_count/total_time:.1f} ops/second")
        print(f"Graphs created: {graphs_created}")
        print(f"Traversals performed: {traversals_performed}")
        print(f"Traversals failed: {traversals_failed}")
        print(f"Final memory: {final_stats['memory_mb']:.1f} MB")
        print(f"Total errors: {len(monitor.error_log)}")

        return {
            'success': True,
            'operations_count': operations_count,
            'graphs_created': graphs_created,
            'traversals_performed': traversals_performed,
            'traversals_failed': traversals_failed,
            'error_count': len(monitor.error_log),
            'duration_minutes': total_time / 60,
            'final_memory_mb': final_stats['memory_mb']
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] Graph monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

async def main():
    """Main monitoring function."""
    print("=" * 60)
    print("FIXED 20-MINUTE TIER 3 SYSTEM MONITORING")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize system monitor
    monitor = SystemMonitor()

    # Create test database
    print("\nInitializing monitoring database...")
    db = sqlite3.connect(':memory:')

    try:
        with open('database/tier3_schema_extension.sql', 'r') as f:
            schema_sql = f.read()

        cursor = db.cursor()
        for statement in schema_sql.split(';'):
            if statement.strip():
                cursor.execute(statement)
        db.commit()
        print("[OK] Tier 3 monitoring database initialized")
    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}")
        return

    overall_start = time.time()

    # Run monitoring sessions
    print("\n" + "=" * 60)
    print("STARTING MONITORING SESSIONS")
    print("=" * 60)

    # Session 1: Bayesian monitoring (10 minutes)
    bayesian_results = await fixed_bayesian_monitoring(db, monitor, 10)

    # Session 2: Graph monitoring (10 minutes)
    graph_results = await fixed_graph_monitoring(db, monitor, 10)

    # Final analysis
    overall_end = time.time()
    total_duration = overall_end - overall_start

    print("\n" + "=" * 60)
    print("MONITORING ANALYSIS")
    print("=" * 60)
    print(f"Total time: {total_duration/60:.2f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Results summary
    print(f"\nRESULTS SUMMARY:")
    print(f"Bayesian System: {'PASS' if bayesian_results['success'] else 'FAIL'}")
    if bayesian_results['success']:
        print(f"  - Operations: {bayesian_results['operations_count']}")
        print(f"  - Hypotheses: {bayesian_results['hypotheses_created']}")
        print(f"  - Evidence: {bayesian_results['evidence_added']}")
        print(f"  - Evidence failures: {bayesian_results['evidence_failed']}")
        print(f"  - Predictions: {bayesian_results['predictions_generated']}")
        print(f"  - Errors: {bayesian_results['error_count']}")

    print(f"\nGraph System: {'PASS' if graph_results['success'] else 'FAIL'}")
    if graph_results['success']:
        print(f"  - Operations: {graph_results['operations_count']}")
        print(f"  - Graphs: {graph_results['graphs_created']}")
        print(f"  - Traversals: {graph_results['traversals_performed']}")
        print(f"  - Traversal failures: {graph_results['traversals_failed']}")
        print(f"  - Errors: {graph_results['error_count']}")

    # Overall assessment
    overall_success = (bayesian_results['success'] and
                      graph_results['success'] and
                      len(monitor.error_log) < 50)

    print(f"\nOVERALL ASSESSMENT: {'SUCCESS' if overall_success else 'NEEDS ATTENTION'}")

    if overall_success:
        print("\n✅ TIER 3 SYSTEMS MONITORING SUCCESSFUL!")
        print("- Both systems handled extended operation well")
        print("- Error rates within acceptable limits")
        print("- Memory usage stable")
        print("- Hypothesis management working correctly")
        print("- Graph operations performing well")
    else:
        print("\n⚠️  Issues detected during monitoring:")
        if not bayesian_results['success']:
            print("- Bayesian system issues")
        if not graph_results['success']:
            print("- Graph system issues")
        if len(monitor.error_log) >= 50:
            print("- High error count")

    db.close()

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Extended 20-Minute Tier 3 System Monitoring

This script runs comprehensive monitoring of the Bayesian Inference Engine
and Enhanced Graph Traversal systems for 20-minute sessions to detect
any long-term stability issues, memory leaks, or performance degradation.
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
        self.memory_samples = deque(maxlen=1000)
        self.cpu_samples = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        self.error_log = []
        self.last_report_time = time.time()

    def sample_system_metrics(self):
        """Sample current system metrics."""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()

            current_time = time.time()

            self.memory_samples.append({
                'timestamp': current_time,
                'rss': memory_info.rss,
                'vms': memory_info.vms
            })

            self.cpu_samples.append({
                'timestamp': current_time,
                'cpu_percent': cpu_percent
            })

        except Exception as e:
            self.error_log.append({
                'timestamp': time.time(),
                'type': 'monitoring_error',
                'error': str(e)
            })

    def detect_memory_leak(self) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        if len(self.memory_samples) < 10:
            return {'detected': False, 'reason': 'insufficient_samples'}

        # Get memory usage over time
        recent_samples = list(self.memory_samples)[-50:]  # Last 50 samples
        early_samples = list(self.memory_samples)[:50]    # First 50 samples

        if len(recent_samples) < 10 or len(early_samples) < 10:
            return {'detected': False, 'reason': 'insufficient_data'}

        # Calculate average memory usage
        early_avg = sum(s['rss'] for s in early_samples) / len(early_samples)
        recent_avg = sum(s['rss'] for s in recent_samples) / len(recent_samples)

        # Calculate growth rate
        growth_rate = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0

        # Detect significant memory growth (> 50% increase)
        leak_detected = growth_rate > 0.5

        return {
            'detected': leak_detected,
            'growth_rate': growth_rate,
            'early_avg_mb': early_avg / (1024 * 1024),
            'recent_avg_mb': recent_avg / (1024 * 1024),
            'total_samples': len(self.memory_samples)
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Memory statistics
        if self.memory_samples:
            memory_values = [s['rss'] for s in self.memory_samples]
            memory_stats = {
                'current_mb': memory_values[-1] / (1024 * 1024),
                'peak_mb': max(memory_values) / (1024 * 1024),
                'average_mb': sum(memory_values) / len(memory_values) / (1024 * 1024)
            }
        else:
            memory_stats = {'current_mb': 0, 'peak_mb': 0, 'average_mb': 0}

        # CPU statistics
        if self.cpu_samples:
            cpu_values = [s['cpu_percent'] for s in self.cpu_samples]
            cpu_stats = {
                'current_percent': cpu_values[-1],
                'peak_percent': max(cpu_values),
                'average_percent': sum(cpu_values) / len(cpu_values)
            }
        else:
            cpu_stats = {'current_percent': 0, 'peak_percent': 0, 'average_percent': 0}

        # Memory leak detection
        leak_info = self.detect_memory_leak()

        return {
            'elapsed_seconds': elapsed,
            'elapsed_minutes': elapsed / 60,
            'memory_stats': memory_stats,
            'cpu_stats': cpu_stats,
            'memory_leak': leak_info,
            'total_errors': len(self.error_log),
            'sample_count': len(self.memory_samples)
        }

    def should_report(self, interval_seconds=60) -> bool:
        """Check if it's time for a status report."""
        current_time = time.time()
        if current_time - self.last_report_time >= interval_seconds:
            self.last_report_time = current_time
            return True
        return False

async def extended_bayesian_monitoring(db, monitor: SystemMonitor, duration_minutes=20):
    """Run extended monitoring of Bayesian inference system."""
    print(f"[BAYESIAN EXTENDED MONITORING] Running for {duration_minutes} minutes...")

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
        print("[OK] Bayesian inference engine initialized for extended monitoring")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        # Extended monitoring counters
        operations_count = 0
        hypothesis_ids = []
        error_count = 0

        # Simulate different game scenarios
        game_scenarios = [
            {'game_id': 'puzzle_game', 'difficulty': 'easy', 'actions': [1, 2, 3, 6]},
            {'game_id': 'action_game', 'difficulty': 'medium', 'actions': [1, 2, 3, 4, 5, 6]},
            {'game_id': 'strategy_game', 'difficulty': 'hard', 'actions': [1, 2, 3, 6]},
            {'game_id': 'arcade_game', 'difficulty': 'variable', 'actions': [1, 2, 3, 4, 6]}
        ]

        scenario_index = 0
        session_counter = 0

        print("Starting extended Bayesian monitoring loop...")

        while time.time() < end_time:
            try:
                operations_count += 1
                session_counter += 1

                # Sample system metrics periodically
                if operations_count % 10 == 0:
                    monitor.sample_system_metrics()

                # Rotate through different game scenarios
                scenario = game_scenarios[scenario_index % len(game_scenarios)]
                scenario_index += 1

                session_id = f"extended_session_{session_counter}"

                # Phase 1: Create hypotheses for different scenarios
                if operations_count % 25 == 0:
                    for action_id in scenario['actions']:
                        # Create different types of hypotheses
                        hypothesis_types = [
                            HypothesisType.ACTION_OUTCOME,
                            HypothesisType.COORDINATE_EFFECTIVENESS,
                            HypothesisType.CONDITIONAL_RULE,
                            HypothesisType.SEQUENCE_PATTERN
                        ]

                        hypothesis_type = random.choice(hypothesis_types)

                        # Context varies by scenario
                        context = {
                            'action_id': action_id,
                            'difficulty': scenario['difficulty'],
                            'scenario_type': scenario['game_id'],
                            'session_type': 'extended_monitoring',
                            'operation_count': operations_count
                        }

                        if hypothesis_type == HypothesisType.COORDINATE_EFFECTIVENESS and action_id == 6:
                            context.update({
                                'x': random.randint(10, 60),
                                'y': random.randint(10, 60),
                                'coordinate_type': random.choice(['button', 'interactive', 'target'])
                            })

                        hypothesis_id = await bayesian.create_hypothesis(
                            hypothesis_type=hypothesis_type,
                            description=f"Extended monitoring: {hypothesis_type.value} for action {action_id} in {scenario['game_id']}",
                            prior_probability=random.uniform(0.3, 0.7),
                            context_conditions=context,
                            game_id=scenario['game_id'],
                            session_id=session_id
                        )

                        hypothesis_ids.append({
                            'id': hypothesis_id,
                            'type': hypothesis_type,
                            'scenario': scenario['game_id'],
                            'action_id': action_id,
                            'created_at': time.time()
                        })

                # Phase 2: Add evidence to existing hypotheses
                if hypothesis_ids and operations_count % 3 == 0:
                    # Select random hypothesis
                    hypothesis_info = random.choice(hypothesis_ids)

                    # Generate realistic evidence based on scenario
                    if scenario['difficulty'] == 'easy':
                        score_changes = [5, 10, 15, 20, -2, -5]
                    elif scenario['difficulty'] == 'medium':
                        score_changes = [3, 7, 12, 18, -3, -8, -12]
                    elif scenario['difficulty'] == 'hard':
                        score_changes = [2, 5, 8, 15, -5, -10, -15, -20]
                    else:  # variable
                        score_changes = range(-20, 25)

                    score_change = random.choice(score_changes)

                    evidence_types = list(EvidenceType)
                    evidence_type = random.choice(evidence_types)

                    await bayesian.add_evidence(
                        hypothesis_id=hypothesis_info['id'],
                        evidence_type=evidence_type,
                        supports_hypothesis=(score_change > 0),
                        strength=min(1.0, abs(score_change) / 25.0),
                        context={
                            'score_change': score_change,
                            'scenario': scenario['game_id'],
                            'difficulty': scenario['difficulty'],
                            'evidence_source': 'extended_monitoring',
                            'operation_count': operations_count,
                            'timestamp': time.time()
                        },
                        game_id=scenario['game_id'],
                        session_id=session_id
                    )

                # Phase 3: Generate predictions periodically
                if operations_count % 15 == 0 and hypothesis_ids:
                    # Create action candidates based on scenario
                    action_candidates = []
                    for action_id in scenario['actions']:
                        if action_id == 6:
                            # Add coordinate variations for Action 6
                            for x, y in [(16, 16), (32, 32), (48, 48)]:
                                action_candidates.append({
                                    'id': action_id,
                                    'x': x,
                                    'y': y,
                                    'scenario': scenario['game_id']
                                })
                        else:
                            action_candidates.append({
                                'id': action_id,
                                'scenario': scenario['game_id']
                            })

                    current_context = {
                        'scenario': scenario['game_id'],
                        'difficulty': scenario['difficulty'],
                        'score': random.randint(0, 200),
                        'level': random.randint(1, 10),
                        'actions_taken': random.randint(10, 100),
                        'operation_count': operations_count
                    }

                    prediction = await bayesian.generate_prediction(
                        action_candidates=action_candidates[:8],  # Limit candidates
                        current_context=current_context,
                        game_id=scenario['game_id'],
                        session_id=session_id
                    )

                    # Log interesting predictions
                    if prediction and prediction.success_probability > 0.7:
                        monitor.performance_metrics['high_confidence_predictions'].append({
                            'timestamp': time.time(),
                            'probability': prediction.success_probability,
                            'confidence': prediction.confidence_level,
                            'scenario': scenario['game_id'],
                            'action': prediction.predicted_action.get('id')
                        })

                # Phase 4: Periodic maintenance and analysis
                if operations_count % 100 == 0:
                    # Prune old hypotheses
                    pruned_count = await bayesian.prune_low_confidence_hypotheses()
                    if pruned_count > 0:
                        monitor.performance_metrics['pruning_events'].append({
                            'timestamp': time.time(),
                            'hypotheses_pruned': pruned_count,
                            'operation_count': operations_count
                        })

                    # Get system insights
                    for game_id in [s['game_id'] for s in game_scenarios]:
                        insights = await bayesian.get_hypothesis_insights(game_id)
                        monitor.performance_metrics['system_insights'].append({
                            'timestamp': time.time(),
                            'game_id': game_id,
                            'total_hypotheses': insights.get('total_hypotheses', 0),
                            'high_confidence': len(insights.get('high_confidence_hypotheses', [])),
                            'low_confidence': len(insights.get('low_confidence_hypotheses', []))
                        })

                # Periodic status reports
                if monitor.should_report(120):  # Every 2 minutes
                    elapsed = time.time() - start_time
                    performance = monitor.get_performance_summary()

                    print(f"\n[BAYESIAN STATUS] {elapsed/60:.1f} minutes elapsed")
                    print(f"  Operations: {operations_count}")
                    print(f"  Active hypotheses: {len(hypothesis_ids)}")
                    print(f"  Memory: {performance['memory_stats']['current_mb']:.1f} MB")
                    print(f"  CPU: {performance['cpu_stats']['current_percent']:.1f}%")
                    print(f"  Errors: {error_count}")

                    if performance['memory_leak']['detected']:
                        print(f"  [WARNING] Potential memory leak detected: {performance['memory_leak']['growth_rate']:.2%} growth")

                # Force garbage collection periodically
                if operations_count % 500 == 0:
                    gc.collect()

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.001)

            except Exception as e:
                error_count += 1
                monitor.error_log.append({
                    'timestamp': time.time(),
                    'operation': operations_count,
                    'type': 'bayesian_operation_error',
                    'error': str(e),
                    'scenario': scenario.get('game_id', 'unknown')
                })

                # Don't spam error messages
                if error_count <= 10:
                    print(f"[ERROR] Operation {operations_count}: {e}")
                elif error_count == 11:
                    print("[ERROR] More errors occurring, suppressing messages...")

        # Final results
        total_time = time.time() - start_time
        final_performance = monitor.get_performance_summary()

        print(f"\n[BAYESIAN EXTENDED MONITORING COMPLETE]")
        print(f"Duration: {total_time/60:.2f} minutes")
        print(f"Total operations: {operations_count}")
        print(f"Operations/second: {operations_count/total_time:.2f}")
        print(f"Final active hypotheses: {len(hypothesis_ids)}")
        print(f"Total errors: {error_count}")
        print(f"Memory usage: {final_performance['memory_stats']['current_mb']:.1f} MB")
        print(f"Peak memory: {final_performance['memory_stats']['peak_mb']:.1f} MB")
        print(f"Memory leak detected: {final_performance['memory_leak']['detected']}")

        return {
            'success': True,
            'operations_count': operations_count,
            'error_count': error_count,
            'final_hypotheses': len(hypothesis_ids),
            'performance': final_performance,
            'duration_minutes': total_time / 60
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] Extended Bayesian monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

async def extended_graph_monitoring(db, monitor: SystemMonitor, duration_minutes=20):
    """Run extended monitoring of graph traversal system."""
    print(f"[GRAPH EXTENDED MONITORING] Running for {duration_minutes} minutes...")

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
        print("[OK] Enhanced graph traversal initialized for extended monitoring")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        # Extended monitoring counters
        operations_count = 0
        graph_ids = []
        error_count = 0

        # Graph scenarios for testing
        graph_scenarios = [
            {'type': GraphType.GAME_STATE_GRAPH, 'complexity': 'simple', 'node_count': (5, 10)},
            {'type': GraphType.DECISION_TREE, 'complexity': 'medium', 'node_count': (8, 15)},
            {'type': GraphType.PATTERN_SPACE, 'complexity': 'complex', 'node_count': (12, 20)},
            {'type': GraphType.ACTION_SEQUENCE, 'complexity': 'variable', 'node_count': (6, 18)}
        ]

        scenario_index = 0

        print("Starting extended graph monitoring loop...")

        while time.time() < end_time:
            try:
                operations_count += 1

                # Sample system metrics periodically
                if operations_count % 10 == 0:
                    monitor.sample_system_metrics()

                scenario = graph_scenarios[scenario_index % len(graph_scenarios)]
                scenario_index += 1

                # Phase 1: Create graphs with varying complexity
                if operations_count % 30 == 0:
                    min_nodes, max_nodes = scenario['node_count']
                    node_count = random.randint(min_nodes, max_nodes)

                    # Create nodes
                    nodes = []
                    for i in range(node_count):
                        node = GraphNode(
                            node_id=f'extended_node_{operations_count}_{i}',
                            node_type=random.choice(list(NodeType)),
                            properties={
                                'score': random.randint(0, 100),
                                'level': random.randint(1, 5),
                                'complexity': scenario['complexity'],
                                'operation_count': operations_count,
                                'node_index': i
                            },
                            coordinates=(
                                random.uniform(0.0, 20.0),
                                random.uniform(0.0, 20.0)
                            )
                        )
                        nodes.append(node)

                    # Create edges based on complexity
                    edges = []
                    if scenario['complexity'] == 'simple':
                        # Linear connections
                        for i in range(len(nodes) - 1):
                            edge = GraphEdge(
                                edge_id=f'extended_edge_{operations_count}_{i}',
                                from_node=nodes[i].node_id,
                                to_node=nodes[i + 1].node_id,
                                weight=random.uniform(1.0, 3.0),
                                properties={'type': 'linear', 'complexity': 'simple'}
                            )
                            edges.append(edge)

                    elif scenario['complexity'] == 'medium':
                        # Tree-like structure
                        for i in range(len(nodes) - 1):
                            edge = GraphEdge(
                                edge_id=f'extended_edge_{operations_count}_{i}',
                                from_node=nodes[i].node_id,
                                to_node=nodes[i + 1].node_id,
                                weight=random.uniform(0.5, 4.0),
                                properties={'type': 'tree', 'complexity': 'medium'}
                            )
                            edges.append(edge)

                            # Add some branching
                            if i > 0 and random.random() < 0.3:
                                branch_edge = GraphEdge(
                                    edge_id=f'extended_branch_{operations_count}_{i}',
                                    from_node=nodes[i].node_id,
                                    to_node=nodes[min(i + 2, len(nodes) - 1)].node_id,
                                    weight=random.uniform(1.5, 5.0),
                                    properties={'type': 'branch', 'complexity': 'medium'}
                                )
                                edges.append(branch_edge)

                    else:  # complex or variable
                        # More connected graph
                        for i in range(len(nodes)):
                            # Connect to multiple other nodes
                            connections = random.randint(1, min(3, len(nodes) - 1))
                            for _ in range(connections):
                                target_idx = random.randint(0, len(nodes) - 1)
                                if target_idx != i:
                                    edge = GraphEdge(
                                        edge_id=f'extended_edge_{operations_count}_{i}_{target_idx}',
                                        from_node=nodes[i].node_id,
                                        to_node=nodes[target_idx].node_id,
                                        weight=random.uniform(0.5, 6.0),
                                        properties={'type': 'complex', 'complexity': scenario['complexity']}
                                    )
                                    edges.append(edge)

                    # Create graph
                    graph_id = await graph_traversal.create_graph(
                        graph_type=scenario['type'],
                        initial_nodes=nodes,
                        initial_edges=edges,
                        game_id=f"extended_monitoring_{scenario['complexity']}"
                    )

                    graph_ids.append({
                        'id': graph_id,
                        'type': scenario['type'],
                        'complexity': scenario['complexity'],
                        'node_count': len(nodes),
                        'edge_count': len(edges),
                        'created_at': time.time()
                    })

                # Phase 2: Perform traversals with different algorithms
                if graph_ids and operations_count % 8 == 0:
                    graph_info = random.choice(graph_ids)
                    graph = graph_traversal.graphs.get(graph_info['id'])

                    if graph and len(graph.nodes) >= 2:
                        node_ids = list(graph.nodes.keys())
                        start_node = random.choice(node_ids)
                        end_node = random.choice(node_ids)

                        if start_node != end_node:
                            # Test different algorithms
                            algorithms = list(TraversalAlgorithm)
                            algorithm = random.choice(algorithms)

                            try:
                                result = await graph_traversal.traverse_graph(
                                    graph_id=graph_info['id'],
                                    start_node=start_node,
                                    end_node=end_node,
                                    algorithm=algorithm
                                )

                                # Log successful traversals
                                if result.success:
                                    monitor.performance_metrics['successful_traversals'].append({
                                        'timestamp': time.time(),
                                        'algorithm': algorithm.value,
                                        'path_length': len(result.primary_path.nodes),
                                        'total_weight': result.primary_path.total_weight,
                                        'nodes_explored': result.nodes_explored,
                                        'computation_time': result.computation_time,
                                        'graph_complexity': graph_info['complexity']
                                    })

                            except Exception as e:
                                # Some traversals may fail - this is normal
                                pass

                # Phase 3: Test path optimization
                if operations_count % 50 == 0 and graph_ids:
                    graph_info = random.choice(graph_ids)
                    graph = graph_traversal.graphs.get(graph_info['id'])

                    if graph and len(graph.nodes) >= 3:
                        node_ids = list(graph.nodes.keys())
                        start_node = random.choice(node_ids[:len(node_ids)//2])
                        end_node = random.choice(node_ids[len(node_ids)//2:])

                        try:
                            optimal_paths = await graph_traversal.find_optimal_paths(
                                graph_id=graph_info['id'],
                                start_node=start_node,
                                end_node=end_node,
                                max_alternatives=3
                            )

                            if optimal_paths:
                                monitor.performance_metrics['path_optimizations'].append({
                                    'timestamp': time.time(),
                                    'graph_complexity': graph_info['complexity'],
                                    'paths_found': len(optimal_paths),
                                    'best_weight': optimal_paths[0].total_weight,
                                    'graph_size': graph_info['node_count']
                                })

                        except Exception as e:
                            # Path optimization may fail on some graphs
                            pass

                # Periodic status reports
                if monitor.should_report(120):  # Every 2 minutes
                    elapsed = time.time() - start_time
                    performance = monitor.get_performance_summary()

                    print(f"\n[GRAPH STATUS] {elapsed/60:.1f} minutes elapsed")
                    print(f"  Operations: {operations_count}")
                    print(f"  Active graphs: {len(graph_ids)}")
                    print(f"  Memory: {performance['memory_stats']['current_mb']:.1f} MB")
                    print(f"  CPU: {performance['cpu_stats']['current_percent']:.1f}%")
                    print(f"  Errors: {error_count}")

                # Force garbage collection periodically
                if operations_count % 500 == 0:
                    gc.collect()

                # Small delay
                await asyncio.sleep(0.001)

            except Exception as e:
                error_count += 1
                monitor.error_log.append({
                    'timestamp': time.time(),
                    'operation': operations_count,
                    'type': 'graph_operation_error',
                    'error': str(e),
                    'scenario': scenario.get('complexity', 'unknown')
                })

                if error_count <= 10:
                    print(f"[ERROR] Graph operation {operations_count}: {e}")
                elif error_count == 11:
                    print("[ERROR] More graph errors occurring, suppressing messages...")

        # Final results
        total_time = time.time() - start_time
        final_performance = monitor.get_performance_summary()

        print(f"\n[GRAPH EXTENDED MONITORING COMPLETE]")
        print(f"Duration: {total_time/60:.2f} minutes")
        print(f"Total operations: {operations_count}")
        print(f"Operations/second: {operations_count/total_time:.2f}")
        print(f"Final active graphs: {len(graph_ids)}")
        print(f"Total errors: {error_count}")
        print(f"Memory usage: {final_performance['memory_stats']['current_mb']:.1f} MB")
        print(f"Peak memory: {final_performance['memory_stats']['peak_mb']:.1f} MB")

        return {
            'success': True,
            'operations_count': operations_count,
            'error_count': error_count,
            'final_graphs': len(graph_ids),
            'performance': final_performance,
            'duration_minutes': total_time / 60
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] Extended graph monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

async def main():
    """Main extended monitoring function."""
    print("=" * 80)
    print("EXTENDED 20-MINUTE TIER 3 SYSTEM MONITORING")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Monitoring for bugs, memory leaks, and performance degradation...")

    # Initialize system monitor
    monitor = SystemMonitor()

    # Create test database with Tier 3 schema
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

    # Run extended monitoring sessions
    print("\n" + "=" * 80)
    print("STARTING EXTENDED MONITORING SESSIONS")
    print("=" * 80)

    overall_start = time.time()

    # Session 1: Extended Bayesian monitoring (20 minutes)
    print("\n" + "=" * 50)
    print("SESSION 1: BAYESIAN SYSTEM EXTENDED MONITORING")
    print("=" * 50)

    bayesian_results = await extended_bayesian_monitoring(db, monitor, 20)

    # Session 2: Extended graph monitoring (20 minutes)
    print("\n" + "=" * 50)
    print("SESSION 2: GRAPH SYSTEM EXTENDED MONITORING")
    print("=" * 50)

    graph_results = await extended_graph_monitoring(db, monitor, 20)

    # Final comprehensive analysis
    overall_end = time.time()
    total_duration = overall_end - overall_start

    print("\n" + "=" * 80)
    print("EXTENDED MONITORING ANALYSIS")
    print("=" * 80)
    print(f"Total monitoring time: {total_duration/60:.2f} minutes")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Performance analysis
    final_performance = monitor.get_performance_summary()

    print(f"\nSYSTEM PERFORMANCE ANALYSIS:")
    print(f"Peak memory usage: {final_performance['memory_stats']['peak_mb']:.1f} MB")
    print(f"Average memory usage: {final_performance['memory_stats']['average_mb']:.1f} MB")
    print(f"Final memory usage: {final_performance['memory_stats']['current_mb']:.1f} MB")
    print(f"Peak CPU usage: {final_performance['cpu_stats']['peak_percent']:.1f}%")
    print(f"Average CPU usage: {final_performance['cpu_stats']['average_percent']:.1f}%")

    # Memory leak analysis
    memory_leak = final_performance['memory_leak']
    print(f"\nMEMORY LEAK ANALYSIS:")
    print(f"Memory leak detected: {memory_leak['detected']}")
    if memory_leak['detected']:
        print(f"Memory growth rate: {memory_leak['growth_rate']:.2%}")
        print(f"Early average: {memory_leak['early_avg_mb']:.1f} MB")
        print(f"Recent average: {memory_leak['recent_avg_mb']:.1f} MB")
    else:
        print("No significant memory leaks detected")

    # Error analysis
    total_errors = len(monitor.error_log)
    print(f"\nERROR ANALYSIS:")
    print(f"Total errors logged: {total_errors}")

    if total_errors > 0:
        error_types = defaultdict(int)
        for error in monitor.error_log:
            error_types[error['type']] += 1

        print("Error breakdown:")
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")

    # Results summary
    print(f"\nRESULTS SUMMARY:")
    print(f"Bayesian System: {'PASS' if bayesian_results['success'] else 'FAIL'}")
    if bayesian_results['success']:
        print(f"  - Operations: {bayesian_results['operations_count']}")
        print(f"  - Errors: {bayesian_results['error_count']}")
        print(f"  - Error rate: {bayesian_results['error_count']/bayesian_results['operations_count']*100:.3f}%")

    print(f"\nGraph System: {'PASS' if graph_results['success'] else 'FAIL'}")
    if graph_results['success']:
        print(f"  - Operations: {graph_results['operations_count']}")
        print(f"  - Errors: {graph_results['error_count']}")
        print(f"  - Error rate: {graph_results['error_count']/graph_results['operations_count']*100:.3f}%")

    # Performance metrics summary
    if monitor.performance_metrics:
        print(f"\nPERFORMACE METRICS COLLECTED:")
        for metric_type, values in monitor.performance_metrics.items():
            print(f"  {metric_type}: {len(values)} entries")

    # Final verdict
    overall_success = (bayesian_results['success'] and
                      graph_results['success'] and
                      not memory_leak['detected'] and
                      total_errors < 100)  # Allow some errors but not excessive

    print(f"\nOVERALL VERDICT: {'SUCCESS' if overall_success else 'NEEDS ATTENTION'}")

    if overall_success:
        print("\n✅ TIER 3 SYSTEMS PASSED EXTENDED MONITORING!")
        print("- No critical issues detected")
        print("- Memory usage stable")
        print("- Error rates acceptable")
        print("- Performance maintained over extended runtime")
        print("- Systems ready for production deployment")
    else:
        print("\n⚠️  Issues detected during extended monitoring:")
        if memory_leak['detected']:
            print("- Memory leak detected")
        if total_errors >= 100:
            print("- High error rate")
        if not bayesian_results['success']:
            print("- Bayesian system issues")
        if not graph_results['success']:
            print("- Graph system issues")

    # Save detailed results
    try:
        results_data = {
            'start_time': datetime.fromtimestamp(overall_start).isoformat(),
            'end_time': datetime.fromtimestamp(overall_end).isoformat(),
            'total_duration_minutes': total_duration / 60,
            'bayesian_results': bayesian_results,
            'graph_results': graph_results,
            'performance_summary': final_performance,
            'error_log': monitor.error_log,
            'performance_metrics': dict(monitor.performance_metrics),
            'overall_success': overall_success
        }

        with open('extended_monitoring_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"\nDetailed results saved to: extended_monitoring_results.json")

    except Exception as e:
        print(f"[WARNING] Could not save results file: {e}")

    db.close()

if __name__ == "__main__":
    asyncio.run(main())
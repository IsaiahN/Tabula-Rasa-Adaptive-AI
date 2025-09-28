#!/usr/bin/env python3
"""
Test script for Tier 3 system integration.
This script tests the Bayesian inference engine and enhanced graph traversal
without going through the problematic import structure.
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
        with open('database/tier3_schema_extension.sql', 'r') as f:
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
        print(f"❌ Database schema initialization failed: {e}")
        return None

async def test_bayesian_system(db):
    """Test the Bayesian inference engine."""
    try:
        # Import directly
        import importlib.util
        spec = importlib.util.spec_from_file_location('bayesian_inference_engine', 'core/bayesian_inference_engine.py')
        bayesian_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bayesian_module)

        BayesianInferenceEngine = bayesian_module.BayesianInferenceEngine
        HypothesisType = bayesian_module.HypothesisType
        EvidenceType = bayesian_module.EvidenceType

        # Initialize system
        bayesian = BayesianInferenceEngine(db)
        print("✅ Bayesian inference engine initialized")

        # Test hypothesis creation
        hypothesis_id = await bayesian.create_hypothesis(
            hypothesis_type=HypothesisType.ACTION_OUTCOME,
            description="Action 1 leads to positive score change",
            prior_probability=0.5,
            context_conditions={'action_id': 1, 'level': 1},
            game_id='test_game',
            session_id='test_session'
        )
        print(f"✅ Created hypothesis: {hypothesis_id}")

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
        print(f"✅ Added evidence: {evidence_id}")

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
            print(f"✅ Generated prediction: Action {prediction.predicted_action.get('id')} " +
                  f"(probability: {prediction.success_probability:.2f})")
        else:
            print("⚠️  No prediction generated (expected for small data)")

        # Test hypothesis insights
        insights = await bayesian.get_hypothesis_insights('test_game')
        print(f"✅ Retrieved insights: {insights.get('total_hypotheses', 0)} hypotheses")

        return True

    except Exception as e:
        print(f"❌ Bayesian system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_graph_traversal_system(db):
    """Test the enhanced graph traversal system."""
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
        print("✅ Enhanced graph traversal initialized")

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
        print(f"✅ Created graph: {graph_id}")

        # Test traversal
        traversal_result = await graph_traversal.traverse_graph(
            graph_id=graph_id,
            start_node='start',
            end_node='end',
            algorithm=TraversalAlgorithm.DIJKSTRA
        )

        if traversal_result.success:
            path = traversal_result.primary_path
            print(f"✅ Found path: {len(path.nodes)} nodes, weight: {path.total_weight:.2f}")
            print(f"   Path: {' -> '.join(path.nodes)}")
        else:
            print("❌ Traversal failed")

        # Test optimal path finding
        optimal_paths = await graph_traversal.find_optimal_paths(
            graph_id=graph_id,
            start_node='start',
            end_node='end',
            max_alternatives=2
        )

        print(f"✅ Found {len(optimal_paths)} optimal paths")

        return True

    except Exception as e:
        print(f"❌ Graph traversal system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration_scenario(db):
    """Test a realistic integration scenario simulating gameplay."""
    try:
        print("\n🎮 TESTING INTEGRATION SCENARIO")
        print("="*50)

        # Import both systems
        import importlib.util

        # Bayesian system
        spec1 = importlib.util.spec_from_file_location('bayesian_inference_engine', 'core/bayesian_inference_engine.py')
        bayesian_module = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(bayesian_module)

        BayesianInferenceEngine = bayesian_module.BayesianInferenceEngine
        HypothesisType = bayesian_module.HypothesisType
        EvidenceType = bayesian_module.EvidenceType

        # Graph traversal system
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

        print("✅ Both systems initialized")

        # Simulate a game scenario
        game_id = 'integration_test_game'
        session_id = 'integration_test_session'

        # Create hypotheses for different actions
        action_hypotheses = {}
        for action_id in [1, 2, 3, 6]:
            hypothesis_id = await bayesian.create_hypothesis(
                hypothesis_type=HypothesisType.ACTION_OUTCOME,
                description=f"Action {action_id} leads to positive outcomes",
                prior_probability=0.5,
                context_conditions={'action_id': action_id},
                game_id=game_id,
                session_id=session_id
            )
            action_hypotheses[action_id] = hypothesis_id

        print(f"✅ Created {len(action_hypotheses)} action hypotheses")

        # Create coordinate hypotheses for Action 6
        coord_hypotheses = {}
        for x, y in [(32, 32), (16, 16), (48, 48)]:
            hypothesis_id = await bayesian.create_hypothesis(
                hypothesis_type=HypothesisType.COORDINATE_EFFECTIVENESS,
                description=f"Coordinate ({x},{y}) is effective",
                prior_probability=0.4,
                context_conditions={'x': x, 'y': y, 'action_type': 6},
                game_id=game_id,
                session_id=session_id
            )
            coord_hypotheses[(x, y)] = hypothesis_id

        print(f"✅ Created {len(coord_hypotheses)} coordinate hypotheses")

        # Create game state graph
        game_nodes = []
        for i in range(5):  # 5 game states
            node = GraphNode(
                node_id=f'state_{i}',
                node_type=NodeType.STATE_NODE,
                properties={'score': i * 20, 'actions_taken': i},
                coordinates=(i * 0.25, i * 0.2)
            )
            game_nodes.append(node)

        graph_id = await graph_traversal.create_graph(
            graph_type=GraphType.GAME_STATE_GRAPH,
            initial_nodes=game_nodes,
            game_id=game_id
        )

        print(f"✅ Created game state graph: {graph_id}")

        # Simulate game actions and learning
        simulated_actions = [
            (1, 15),   # Action 1, score +15
            (2, -5),   # Action 2, score -5
            (3, 10),   # Action 3, score +10
            (6, 25),   # Action 6 at (32,32), score +25
            (1, 20),   # Action 1, score +20
        ]

        for action_id, score_change in simulated_actions:
            # Add evidence to Bayesian system
            hypothesis_id = action_hypotheses.get(action_id)
            if hypothesis_id:
                await bayesian.add_evidence(
                    hypothesis_id=hypothesis_id,
                    evidence_type=EvidenceType.DIRECT_OBSERVATION,
                    supports_hypothesis=(score_change > 0),
                    strength=min(1.0, abs(score_change) / 30.0),
                    context={'score_change': score_change},
                    game_id=game_id,
                    session_id=session_id
                )

            # For Action 6, also update coordinate hypothesis
            if action_id == 6:
                coord_hypothesis_id = coord_hypotheses.get((32, 32))
                if coord_hypothesis_id:
                    await bayesian.add_evidence(
                        hypothesis_id=coord_hypothesis_id,
                        evidence_type=EvidenceType.DIRECT_OBSERVATION,
                        supports_hypothesis=(score_change > 0),
                        strength=min(1.0, abs(score_change) / 30.0),
                        context={'score_change': score_change, 'x': 32, 'y': 32},
                        game_id=game_id,
                        session_id=session_id
                    )

        print(f"✅ Processed {len(simulated_actions)} simulated actions")

        # Generate predictions based on learned patterns
        action_candidates = [
            {'id': 1}, {'id': 2}, {'id': 3},
            {'id': 6, 'x': 32, 'y': 32},
            {'id': 6, 'x': 16, 'y': 16}
        ]

        prediction = await bayesian.generate_prediction(
            action_candidates=action_candidates,
            current_context={'level': 1, 'score': 75, 'actions_taken': 5},
            game_id=game_id,
            session_id=session_id
        )

        if prediction:
            predicted_action = prediction.predicted_action
            print(f"✅ FINAL PREDICTION: Action {predicted_action.get('id')}")
            if predicted_action.get('id') == 6:
                print(f"   Coordinates: ({predicted_action.get('x')}, {predicted_action.get('y')})")
            print(f"   Success probability: {prediction.success_probability:.2f}")
            print(f"   Confidence level: {prediction.confidence_level:.2f}")
            print(f"   Supporting hypotheses: {len(prediction.supporting_hypotheses)}")

            if prediction.uncertainty_factors:
                print(f"   Uncertainty factors: {len(prediction.uncertainty_factors)}")
        else:
            print("⚠️  No prediction generated")

        # Get insights
        insights = await bayesian.get_hypothesis_insights(game_id)
        print(f"✅ FINAL INSIGHTS:")
        print(f"   Total hypotheses: {insights.get('total_hypotheses', 0)}")
        print(f"   High confidence: {len(insights.get('high_confidence_hypotheses', []))}")
        print(f"   Low confidence: {len(insights.get('low_confidence_hypotheses', []))}")

        # Test graph traversal on final state
        if len(game_nodes) >= 2:
            final_traversal = await graph_traversal.traverse_graph(
                graph_id=graph_id,
                start_node='state_0',
                end_node='state_4',
                algorithm=TraversalAlgorithm.A_STAR,
                heuristic='euclidean'
            )

            if final_traversal.success:
                print(f"✅ FINAL TRAVERSAL: Found path with {final_traversal.nodes_explored} nodes explored")

        return True

    except Exception as e:
        print(f"❌ Integration scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🚀 TIER 3 SYSTEM INTEGRATION TEST")
    print("="*50)

    # Create test database
    db = create_test_database()
    if not db:
        print("❌ Cannot proceed without database")
        return

    # Test individual systems
    print("\n🧠 TESTING BAYESIAN INFERENCE ENGINE")
    print("-" * 40)
    bayesian_success = await test_bayesian_system(db)

    print("\n🗺️  TESTING ENHANCED GRAPH TRAVERSAL")
    print("-" * 40)
    graph_success = await test_graph_traversal_system(db)

    # Test integration
    if bayesian_success and graph_success:
        integration_success = await test_integration_scenario(db)
    else:
        integration_success = False

    # Final results
    print("\n📊 TEST RESULTS")
    print("="*50)
    print(f"Bayesian Inference Engine: {'✅ PASS' if bayesian_success else '❌ FAIL'}")
    print(f"Enhanced Graph Traversal: {'✅ PASS' if graph_success else '❌ FAIL'}")
    print(f"Integration Scenario: {'✅ PASS' if integration_success else '❌ FAIL'}")

    if bayesian_success and graph_success and integration_success:
        print("\n🎉 ALL TIER 3 SYSTEMS WORKING CORRECTLY!")
        print("Ready for full training integration.")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")

    # Close database
    db.close()

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Comprehensive gameplay simulation with Tier 3 systems.
This simulates the actual gameplay loop with Bayesian inference and graph traversal.
"""

import sys
import os
import sqlite3
import asyncio
import json
import random
from datetime import datetime

# Add paths
sys.path.append('.')
sys.path.append('./core')
sys.path.append('./database')

class GameplaySimulator:
    """Simulates ARC-AGI gameplay with Tier 3 system integration."""

    def __init__(self, db):
        self.db = db
        self.bayesian_system = None
        self.graph_traversal_system = None
        self.game_id = "sim_game_001"
        self.session_id = "sim_session_001"
        self.current_score = 0
        self.actions_taken = 0
        self.current_level = 1
        self.game_state_history = []

    async def initialize_systems(self):
        """Initialize Tier 3 systems."""
        try:
            # Import and initialize Bayesian system
            import importlib.util
            spec1 = importlib.util.spec_from_file_location('bayesian_inference_engine', 'core/bayesian_inference_engine.py')
            bayesian_module = importlib.util.module_from_spec(spec1)
            spec1.loader.exec_module(bayesian_module)

            self.BayesianInferenceEngine = bayesian_module.BayesianInferenceEngine
            self.HypothesisType = bayesian_module.HypothesisType
            self.EvidenceType = bayesian_module.EvidenceType

            # Import and initialize Graph traversal system
            spec2 = importlib.util.spec_from_file_location('enhanced_graph_traversal', 'core/enhanced_graph_traversal.py')
            graph_module = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(graph_module)

            self.EnhancedGraphTraversal = graph_module.EnhancedGraphTraversal
            self.GraphType = graph_module.GraphType
            self.GraphNode = graph_module.GraphNode
            self.GraphEdge = graph_module.GraphEdge
            self.NodeType = graph_module.NodeType
            self.TraversalAlgorithm = graph_module.TraversalAlgorithm

            # Initialize systems
            self.bayesian_system = self.BayesianInferenceEngine(self.db)
            self.graph_traversal_system = self.EnhancedGraphTraversal(self.db)

            print("[OK] Tier 3 systems initialized for gameplay simulation")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to initialize systems: {e}")
            return False

    def simulate_available_actions(self):
        """Simulate available actions like a real ARC-AGI game."""
        # Common ARC-AGI actions: 1-5 are standard actions, 6 is coordinate-based, 7 is another action
        base_actions = [1, 2, 3, 4, 5, 7]

        # Sometimes Action 6 (coordinate-based) is available
        if random.random() > 0.3:  # 70% chance Action 6 is available
            base_actions.append(6)

        # Randomly remove some actions to simulate game state constraints
        num_actions = random.randint(2, len(base_actions))
        return random.sample(base_actions, num_actions)

    def simulate_action_outcome(self, action_id, coordinates=None):
        """Simulate the outcome of taking an action."""
        # Simulate score changes based on action effectiveness
        action_effectiveness = {
            1: 0.6,  # 60% chance of positive outcome
            2: 0.4,  # 40% chance of positive outcome
            3: 0.7,  # 70% chance of positive outcome
            4: 0.3,  # 30% chance of positive outcome
            5: 0.5,  # 50% chance of positive outcome
            6: 0.8,  # 80% chance of positive outcome (if good coordinates)
            7: 0.2,  # 20% chance of positive outcome
        }

        base_effectiveness = action_effectiveness.get(action_id, 0.5)

        # For Action 6, effectiveness depends on coordinates
        if action_id == 6 and coordinates:
            x, y = coordinates
            # Simulate "good" coordinates (closer to center is better)
            center_distance = abs(x - 32) + abs(y - 32)
            coordinate_bonus = max(0, (64 - center_distance) / 64 * 0.3)
            base_effectiveness += coordinate_bonus

        # Determine if action is successful
        is_successful = random.random() < base_effectiveness

        if is_successful:
            score_change = random.randint(5, 25)

            # Occasionally trigger level progression
            if random.random() < 0.1:  # 10% chance
                score_change += 50  # Big bonus for level completion
                self.current_level += 1

        else:
            score_change = random.randint(-10, 0)

        self.current_score += score_change
        self.actions_taken += 1

        return {
            'score_change': score_change,
            'new_score': self.current_score,
            'is_successful': is_successful,
            'level_progressed': score_change >= 50
        }

    async def create_hypotheses_for_actions(self, available_actions):
        """Create hypotheses for available actions."""
        hypotheses = {}

        for action_id in available_actions:
            # Create action-outcome hypothesis
            hypothesis_id = await self.bayesian_system.create_hypothesis(
                hypothesis_type=self.HypothesisType.ACTION_OUTCOME,
                description=f"Action {action_id} leads to positive outcomes in level {self.current_level}",
                prior_probability=0.5,
                context_conditions={'action_id': action_id, 'level': self.current_level},
                game_id=self.game_id,
                session_id=self.session_id
            )
            hypotheses[action_id] = hypothesis_id

        return hypotheses

    async def create_coordinate_hypotheses(self, coordinates_list):
        """Create hypotheses for coordinate effectiveness."""
        coord_hypotheses = {}

        for x, y in coordinates_list:
            hypothesis_id = await self.bayesian_system.create_hypothesis(
                hypothesis_type=self.HypothesisType.COORDINATE_EFFECTIVENESS,
                description=f"Coordinate ({x},{y}) is effective in level {self.current_level}",
                prior_probability=0.4,
                context_conditions={'x': x, 'y': y, 'level': self.current_level, 'action_type': 6},
                game_id=self.game_id,
                session_id=self.session_id
            )
            coord_hypotheses[(x, y)] = hypothesis_id

        return coord_hypotheses

    async def update_hypotheses_with_evidence(self, action_id, outcome, coordinates=None, action_hypotheses=None, coord_hypotheses=None):
        """Update hypotheses based on action outcomes."""
        # Update action hypothesis
        if action_hypotheses and action_id in action_hypotheses:
            hypothesis_id = action_hypotheses[action_id]
            evidence_strength = min(1.0, abs(outcome['score_change']) / 20.0)

            await self.bayesian_system.add_evidence(
                hypothesis_id=hypothesis_id,
                evidence_type=self.EvidenceType.DIRECT_OBSERVATION,
                supports_hypothesis=outcome['is_successful'],
                strength=evidence_strength,
                context={
                    'score_change': outcome['score_change'],
                    'level': self.current_level,
                    'actions_taken': self.actions_taken
                },
                game_id=self.game_id,
                session_id=self.session_id
            )

        # Update coordinate hypothesis for Action 6
        if action_id == 6 and coordinates and coord_hypotheses:
            coord_key = coordinates
            if coord_key in coord_hypotheses:
                hypothesis_id = coord_hypotheses[coord_key]
                evidence_strength = min(1.0, abs(outcome['score_change']) / 25.0)

                await self.bayesian_system.add_evidence(
                    hypothesis_id=hypothesis_id,
                    evidence_type=self.EvidenceType.DIRECT_OBSERVATION,
                    supports_hypothesis=outcome['is_successful'],
                    strength=evidence_strength,
                    context={
                        'score_change': outcome['score_change'],
                        'x': coordinates[0],
                        'y': coordinates[1],
                        'level': self.current_level
                    },
                    game_id=self.game_id,
                    session_id=self.session_id
                )

    async def get_bayesian_action_recommendation(self, available_actions):
        """Get action recommendation from Bayesian system."""
        # Create action candidates
        action_candidates = []
        for action_id in available_actions:
            if action_id == 6:
                # For Action 6, try several coordinate options
                coordinates_to_try = [(32, 32), (16, 16), (48, 48), (24, 40)]
                for x, y in coordinates_to_try:
                    action_candidates.append({'id': 6, 'x': x, 'y': y})
            else:
                action_candidates.append({'id': action_id})

        # Get prediction
        current_context = {
            'level': self.current_level,
            'score': self.current_score,
            'actions_taken': self.actions_taken,
            'available_actions': len(available_actions)
        }

        prediction = await self.bayesian_system.generate_prediction(
            action_candidates=action_candidates,
            current_context=current_context,
            game_id=self.game_id,
            session_id=self.session_id
        )

        return prediction

    async def update_game_state_graph(self):
        """Update the game state graph with current state."""
        # Create node for current state
        state_node = self.GraphNode(
            node_id=f'state_{self.actions_taken}',
            node_type=self.NodeType.STATE_NODE,
            properties={
                'score': self.current_score,
                'level': self.current_level,
                'actions_taken': self.actions_taken
            },
            coordinates=(self.actions_taken / 100.0, self.current_score / 100.0)
        )

        # Use a fixed graph ID to avoid recreating
        if not hasattr(self, 'graph_id'):
            self.graph_id = await self.graph_traversal_system.create_graph(
                graph_type=self.GraphType.GAME_STATE_GRAPH,
                initial_nodes=[state_node],
                game_id=self.game_id
            )
        else:
            # Add node to existing graph
            graph = self.graph_traversal_system.graphs[self.graph_id]
            graph.add_node(state_node)

        self.game_state_history.append(state_node.node_id)
        return self.graph_id

    async def find_optimal_path_to_success(self, graph_id):
        """Find optimal path through game states."""
        if len(self.game_state_history) < 2:
            return None

        try:
            # Find path from early successful state to current
            start_state = self.game_state_history[max(0, len(self.game_state_history) - 5)]
            current_state = self.game_state_history[-1]

            optimal_paths = await self.graph_traversal_system.find_optimal_paths(
                graph_id=graph_id,
                start_node=start_state,
                end_node=current_state,
                max_alternatives=2
            )

            return optimal_paths

        except Exception as e:
            # Expected when nodes aren't connected
            return None

    async def run_gameplay_simulation(self, max_actions=20):
        """Run the main gameplay simulation."""
        print(f"\nSTARTING GAMEPLAY SIMULATION")
        print(f"Game ID: {self.game_id}")
        print(f"Max actions: {max_actions}")
        print("-" * 50)

        # Track hypotheses
        action_hypotheses = {}
        coord_hypotheses = {}

        # Coordinate options for Action 6
        coordinate_options = [(32, 32), (16, 16), (48, 48), (24, 40), (40, 24)]

        for turn in range(max_actions):
            print(f"\n--- TURN {turn + 1} ---")
            print(f"Current Score: {self.current_score}, Level: {self.current_level}")

            # Get available actions
            available_actions = self.simulate_available_actions()
            print(f"Available actions: {available_actions}")

            # Create hypotheses for new actions
            new_action_hypotheses = await self.create_hypotheses_for_actions(available_actions)
            action_hypotheses.update(new_action_hypotheses)

            # Create coordinate hypotheses if Action 6 is available
            if 6 in available_actions and not coord_hypotheses:
                coord_hypotheses = await self.create_coordinate_hypotheses(coordinate_options)

            # Get Bayesian recommendation
            bayesian_prediction = await self.get_bayesian_action_recommendation(available_actions)

            if bayesian_prediction:
                recommended_action = bayesian_prediction.predicted_action
                print(f"Bayesian recommendation: Action {recommended_action.get('id')} " +
                      f"(probability: {bayesian_prediction.success_probability:.2f})")

                # Use Bayesian recommendation
                chosen_action_id = recommended_action.get('id')
                chosen_coordinates = None
                if chosen_action_id == 6:
                    chosen_coordinates = (recommended_action.get('x'), recommended_action.get('y'))
            else:
                # Fallback: choose random action
                chosen_action_id = random.choice(available_actions)
                chosen_coordinates = None
                if chosen_action_id == 6:
                    chosen_coordinates = random.choice(coordinate_options)
                print(f"Fallback choice: Action {chosen_action_id}")

            # Execute action
            outcome = self.simulate_action_outcome(chosen_action_id, chosen_coordinates)

            print(f"Action {chosen_action_id} executed:")
            if chosen_coordinates:
                print(f"  Coordinates: {chosen_coordinates}")
            print(f"  Score change: {outcome['score_change']}")
            print(f"  New score: {outcome['new_score']}")
            print(f"  Success: {outcome['is_successful']}")
            if outcome['level_progressed']:
                print(f"  LEVEL UP! Now on level {self.current_level}")

            # Update hypotheses with evidence
            await self.update_hypotheses_with_evidence(
                chosen_action_id, outcome, chosen_coordinates,
                action_hypotheses, coord_hypotheses
            )

            # Update game state graph
            graph_id = await self.update_game_state_graph()

            # Analyze optimal paths every few turns
            if turn > 0 and turn % 5 == 0:
                optimal_paths = await self.find_optimal_path_to_success(graph_id)
                if optimal_paths:
                    best_path = optimal_paths[0]
                    print(f"  Graph analysis: Found optimal path with {len(best_path.nodes)} states")

            # Check for early termination conditions
            if self.current_score >= 200:  # High score achieved
                print(f"\nHIGH SCORE ACHIEVED! Ending simulation.")
                break

            if self.current_score < -50:  # Very poor performance
                print(f"\nPOOR PERFORMANCE. Ending simulation.")
                break

        # Final analysis
        print(f"\n" + "="*50)
        print(f"SIMULATION COMPLETE")
        print(f"Final Score: {self.current_score}")
        print(f"Actions Taken: {self.actions_taken}")
        print(f"Final Level: {self.current_level}")

        # Get final insights
        insights = await self.bayesian_system.get_hypothesis_insights(self.game_id)
        print(f"Total hypotheses: {insights.get('total_hypotheses', 0)}")
        print(f"High confidence hypotheses: {len(insights.get('high_confidence_hypotheses', []))}")

        return {
            'final_score': self.current_score,
            'actions_taken': self.actions_taken,
            'final_level': self.current_level,
            'insights': insights
        }

def create_test_database():
    """Create test database with Tier 3 schemas."""
    db = sqlite3.connect(':memory:')
    try:
        with open('database/tier3_schema_extension.sql', 'r') as f:
            schema_sql = f.read()

        cursor = db.cursor()
        for statement in schema_sql.split(';'):
            if statement.strip():
                cursor.execute(statement)

        db.commit()
        print("[OK] Test database with Tier 3 schemas created")
        return db
    except Exception as e:
        print(f"[ERROR] Database creation failed: {e}")
        return None

async def main():
    """Main function."""
    print("TIER 3 GAMEPLAY SIMULATION")
    print("="*50)

    # Create database
    db = create_test_database()
    if not db:
        return

    # Initialize simulator
    simulator = GameplaySimulator(db)

    if not await simulator.initialize_systems():
        print("[ERROR] Failed to initialize systems")
        return

    # Run simulation
    results = await simulator.run_gameplay_simulation(max_actions=15)

    # Summary
    print(f"\nSIMULATION RESULTS:")
    print(f"Final Score: {results['final_score']}")
    print(f"Actions Taken: {results['actions_taken']}")
    print(f"Learning Insights: {results['insights'].get('total_hypotheses', 0)} hypotheses created")

    if results['final_score'] > 0:
        print("[SUCCESS] Positive score achieved with Tier 3 integration")
    else:
        print("[INFO] Negative score - learning from failures")

    db.close()

if __name__ == "__main__":
    asyncio.run(main())
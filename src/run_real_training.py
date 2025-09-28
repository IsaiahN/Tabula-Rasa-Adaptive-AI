#!/usr/bin/env python3
"""
Real Training Runner for Tier 3 System Testing

This script bypasses import issues and runs actual training to find real runtime errors.
"""

import sys
import os
import asyncio
import sqlite3
import logging
from datetime import datetime
import json

# Set up paths
sys.path.append('.')
sys.path.append('./core')
sys.path.append('./database')
sys.path.append('./training')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinimalAPIManager:
    """Minimal API manager for testing without real API calls."""

    def __init__(self):
        self.is_initialized_flag = False
        self.scorecard_counter = 0
        self.action_counter = 0

    async def initialize(self):
        self.is_initialized_flag = True
        return True

    def is_initialized(self):
        return self.is_initialized_flag

    async def get_available_games(self):
        # Return fake game data
        return [{'game_id': 'test_game_001', 'name': 'Test Game'}]

    async def create_scorecard(self, name, description):
        self.scorecard_counter += 1
        return f"scorecard_{self.scorecard_counter}"

    async def reset_game(self, game_id, scorecard_id=None):
        # Return fake GameState-like object
        class FakeGameState:
            def __init__(self, manager):
                self.guid = f"guid_{manager.scorecard_counter}"
                self.state = 'NOT_FINISHED'
                self.score = 0.0
                self.available_actions = [1, 2, 3, 6, 7]
                self.frame = [[0 for _ in range(64)] for _ in range(64)]

        return FakeGameState(self)

    async def get_game_state(self, game_id, card_id=None, guid=None):
        class FakeGameState:
            def __init__(self, manager):
                self.guid = guid or f"guid_{manager.action_counter}"
                self.state = 'NOT_FINISHED' if manager.action_counter < 20 else 'WIN'
                self.score = float(manager.action_counter * 5)
                self.available_actions = [1, 2, 3, 6, 7]
                self.frame = [[1 if (i + j) % 10 == 0 else 0 for j in range(64)] for i in range(64)]

        return FakeGameState(self)

    async def take_action(self, game_id, action, scorecard_id=None, guid=None):
        self.action_counter += 1

        class FakeActionResult:
            def __init__(self, manager):
                self.score = float(manager.action_counter * 5)
                self.state = 'NOT_FINISHED' if manager.action_counter < 20 else 'WIN'
                self.available_actions = [1, 2, 3, 6, 7]
                self.frame = [[1 if (i + j + manager.action_counter) % 10 == 0 else 0 for j in range(64)] for i in range(64)]

        return FakeActionResult(self)

    async def close_scorecard(self, scorecard_id):
        return True

    async def close(self):
        return True


class MinimalTrainingLoop:
    """Minimal training loop that focuses on testing Tier 3 systems."""

    def __init__(self):
        self.api_manager = MinimalAPIManager()
        self.setup_database()
        self.initialize_tier3_systems()

    def setup_database(self):
        """Set up in-memory database with Tier 3 schemas."""
        try:
            self.db_connection = sqlite3.connect(':memory:')

            # Load and execute Tier 3 schema
            try:
                with open('database/tier3_schema_extension.sql', 'r') as f:
                    schema_sql = f.read()

                cursor = self.db_connection.cursor()
                for statement in schema_sql.split(';'):
                    if statement.strip():
                        cursor.execute(statement)

                self.db_connection.commit()
                logger.info("✅ Database schema initialized")

            except FileNotFoundError:
                # Create minimal schema manually if file not found
                logger.warning("Schema file not found, creating minimal schema")
                self.create_minimal_schema()

        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise

    def create_minimal_schema(self):
        """Create minimal schema for testing."""
        cursor = self.db_connection.cursor()

        # Minimal Bayesian tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bayesian_hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                hypothesis_type TEXT NOT NULL,
                description TEXT NOT NULL,
                prior_probability REAL NOT NULL,
                posterior_probability REAL NOT NULL,
                evidence_count INTEGER DEFAULT 0,
                supporting_evidence INTEGER DEFAULT 0,
                refuting_evidence INTEGER DEFAULT 0,
                confidence_lower REAL DEFAULT 0.0,
                confidence_upper REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                context_conditions TEXT,
                validation_threshold REAL DEFAULT 0.8,
                is_active BOOLEAN DEFAULT 1
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bayesian_evidence (
                evidence_id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                supports_hypothesis BOOLEAN NOT NULL,
                strength REAL NOT NULL,
                context TEXT,
                observed_at TEXT NOT NULL,
                game_id TEXT,
                session_id TEXT
            )
        """)

        # Minimal Graph tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_structures (
                graph_id TEXT PRIMARY KEY,
                graph_type TEXT NOT NULL,
                properties TEXT,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                node_count INTEGER DEFAULT 0,
                edge_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL,
                node_type TEXT NOT NULL,
                properties TEXT,
                coordinates_x REAL,
                coordinates_y REAL,
                created_at TEXT NOT NULL,
                last_visited TEXT,
                visit_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0
            )
        """)

        self.db_connection.commit()
        logger.info("✅ Minimal schema created")

    def initialize_tier3_systems(self):
        """Initialize Tier 3 systems."""
        try:
            # Import and initialize Bayesian system
            import importlib.util

            # Bayesian inference engine
            spec1 = importlib.util.spec_from_file_location('bayesian_inference_engine', 'core/bayesian_inference_engine.py')
            bayesian_module = importlib.util.module_from_spec(spec1)
            spec1.loader.exec_module(bayesian_module)

            BayesianInferenceEngine = bayesian_module.BayesianInferenceEngine
            self.HypothesisType = bayesian_module.HypothesisType
            self.EvidenceType = bayesian_module.EvidenceType

            self.bayesian_system = BayesianInferenceEngine(self.db_connection)
            logger.info("✅ Bayesian inference engine initialized")

            # Enhanced graph traversal
            spec2 = importlib.util.spec_from_file_location('enhanced_graph_traversal', 'core/enhanced_graph_traversal.py')
            graph_module = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(graph_module)

            EnhancedGraphTraversal = graph_module.EnhancedGraphTraversal
            self.GraphType = graph_module.GraphType
            self.GraphNode = graph_module.GraphNode
            self.GraphEdge = graph_module.GraphEdge
            self.NodeType = graph_module.NodeType
            self.TraversalAlgorithm = graph_module.TraversalAlgorithm

            self.graph_system = EnhancedGraphTraversal(self.db_connection)
            logger.info("✅ Enhanced graph traversal initialized")

            # Track existing hypotheses to avoid duplicates
            self.action_hypotheses = {}  # action_id -> hypothesis_id
            self.coordinate_hypotheses = {}  # (x, y) -> hypothesis_id

            self.tier3_initialized = True

        except Exception as e:
            logger.error(f"Tier 3 initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.tier3_initialized = False

    async def run_training_session(self, max_actions=50):
        """Run a real training session with Tier 3 systems."""
        try:
            logger.info("🚀 Starting real training session with Tier 3 systems")

            # Initialize API
            await self.api_manager.initialize()

            # Get available games
            games = await self.api_manager.get_available_games()
            if not games:
                raise Exception("No games available")

            game_id = games[0]['game_id']
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            logger.info(f"Starting game: {game_id}")

            # Create scorecard
            scorecard_id = await self.api_manager.create_scorecard(
                f"Tier 3 Test Session",
                f"Testing Bayesian + Graph systems"
            )

            # Reset game
            reset_response = await self.api_manager.reset_game(game_id, scorecard_id)
            game_guid = reset_response.guid

            # Game loop
            actions_taken = 0
            total_score = 0
            action_history = []
            score_history = [0]

            while actions_taken < max_actions:
                try:
                    # Get current state
                    current_response = await self.api_manager.get_game_state(game_id, scorecard_id, game_guid)

                    if current_response.state != 'NOT_FINISHED':
                        if current_response.state == 'WIN':
                            logger.info(f"🎉 GAME WON after {actions_taken} actions!")
                        break

                    current_score = current_response.score
                    available_actions = current_response.available_actions

                    # TIER 3 INTEGRATION TEST
                    if self.tier3_initialized:
                        await self.test_tier3_during_gameplay(
                            game_id, session_id, actions_taken, current_score,
                            available_actions, action_history, score_history
                        )

                    # Choose action (simple strategy for testing)
                    if 6 in available_actions:
                        # Test Action 6 with coordinates
                        chosen_action = {'id': 6, 'x': 32, 'y': 32}
                    else:
                        # Choose first available action
                        chosen_action = {'id': available_actions[0]}

                    # Execute action
                    action_result = await self.api_manager.take_action(
                        game_id, chosen_action, scorecard_id, game_guid
                    )

                    if action_result:
                        actions_taken += 1
                        new_score = action_result.score
                        score_change = new_score - current_score
                        total_score = new_score

                        action_history.append(chosen_action)
                        score_history.append(new_score)

                        logger.info(f"Action {actions_taken}: {chosen_action} -> Score: {new_score} (+{score_change})")

                        # Test Tier 3 systems with action result
                        if self.tier3_initialized:
                            await self.test_tier3_with_action_result(
                                game_id, session_id, chosen_action, score_change,
                                actions_taken, new_score
                            )

                    # Small delay
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error during action {actions_taken}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            # Close scorecard
            await self.api_manager.close_scorecard(scorecard_id)

            # Final results
            logger.info(f"Training session completed:")
            logger.info(f"  Actions taken: {actions_taken}")
            logger.info(f"  Final score: {total_score}")
            logger.info(f"  Game state: {current_response.state}")

            if self.tier3_initialized:
                await self.final_tier3_analysis(game_id, session_id)

            return {
                'success': True,
                'actions_taken': actions_taken,
                'final_score': total_score,
                'game_state': current_response.state
            }

        except Exception as e:
            logger.error(f"Training session failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    async def test_tier3_during_gameplay(self, game_id, session_id, actions_taken,
                                       current_score, available_actions, action_history, score_history):
        """Test Tier 3 systems during active gameplay."""
        try:
            # Test Bayesian predictions
            if len(action_history) > 3:  # Need some history
                action_candidates = []
                for action_id in available_actions:
                    if action_id == 6:
                        action_candidates.append({'id': 6, 'x': 32, 'y': 32})
                        action_candidates.append({'id': 6, 'x': 16, 'y': 16})
                    else:
                        action_candidates.append({'id': action_id})

                current_context = {
                    'actions_taken': actions_taken,
                    'current_score': current_score,
                    'available_actions': len(available_actions)
                }

                prediction = await self.bayesian_system.generate_prediction(
                    action_candidates=action_candidates,
                    current_context=current_context,
                    game_id=game_id,
                    session_id=session_id
                )

                if prediction:
                    logger.info(f"🎯 Bayesian prediction: Action {prediction.predicted_action.get('id')} " +
                              f"(prob: {prediction.success_probability:.2f})")

            # Test graph traversal
            if actions_taken % 10 == 0 and actions_taken > 0:  # Every 10 actions
                # Create game state nodes
                node_id = f"state_{actions_taken}"
                node = self.GraphNode(
                    node_id=node_id,
                    node_type=self.NodeType.STATE_NODE,
                    properties={'score': current_score, 'actions': actions_taken},
                    coordinates=(actions_taken / 100.0, current_score / 100.0)
                )

                # Try to create or update graph
                try:
                    graph_id = f"game_graph_{game_id}"
                    if graph_id not in self.graph_system.graphs:
                        await self.graph_system.create_graph(
                            graph_type=self.GraphType.GAME_STATE_GRAPH,
                            initial_nodes=[node],
                            game_id=game_id
                        )
                        logger.info(f"🗺️  Created game state graph: {graph_id}")

                except Exception as e:
                    logger.debug(f"Graph creation error: {e}")

        except Exception as e:
            logger.error(f"Tier 3 gameplay test error: {e}")

    async def test_tier3_with_action_result(self, game_id, session_id, action, score_change,
                                          actions_taken, new_score):
        """Test Tier 3 systems with action results."""
        try:
            action_id = action.get('id')

            # Create/update hypotheses based on action results
            if score_change != 0:  # Only for meaningful results

                # Action outcome hypothesis - reuse existing or create new
                if action_id not in self.action_hypotheses:
                    hypothesis_desc = f"Action {action_id} leads to positive outcomes"
                    hypothesis_id = await self.bayesian_system.create_hypothesis(
                        hypothesis_type=self.HypothesisType.ACTION_OUTCOME,
                        description=hypothesis_desc,
                        prior_probability=0.5,
                        context_conditions={'action_id': action_id},
                        game_id=game_id,
                        session_id=session_id
                    )
                    self.action_hypotheses[action_id] = hypothesis_id
                    logger.debug(f"📝 Created new action hypothesis for action {action_id}")
                else:
                    hypothesis_id = self.action_hypotheses[action_id]

                # Add evidence
                evidence_strength = min(1.0, abs(score_change) / 20.0)
                await self.bayesian_system.add_evidence(
                    hypothesis_id=hypothesis_id,
                    evidence_type=self.EvidenceType.DIRECT_OBSERVATION,
                    supports_hypothesis=(score_change > 0),
                    strength=evidence_strength,
                    context={'score_change': score_change, 'action_id': action_id},
                    game_id=game_id,
                    session_id=session_id
                )

                logger.debug(f"📊 Added evidence for action {action_id}: " +
                           f"{'positive' if score_change > 0 else 'negative'} " +
                           f"(strength: {evidence_strength:.2f})")

                # Coordinate hypothesis for Action 6 - reuse existing or create new
                if action_id == 6 and 'x' in action and 'y' in action:
                    coord_key = (action['x'], action['y'])
                    if coord_key not in self.coordinate_hypotheses:
                        coord_hypothesis_desc = f"Coordinate ({action['x']},{action['y']}) is effective"
                        coord_hypothesis_id = await self.bayesian_system.create_hypothesis(
                            hypothesis_type=self.HypothesisType.COORDINATE_EFFECTIVENESS,
                            description=coord_hypothesis_desc,
                            prior_probability=0.4,
                            context_conditions={'x': action['x'], 'y': action['y'], 'action_type': 6},
                            game_id=game_id,
                            session_id=session_id
                        )
                        self.coordinate_hypotheses[coord_key] = coord_hypothesis_id
                        logger.debug(f"📍 Created new coordinate hypothesis for {coord_key}")
                    else:
                        coord_hypothesis_id = self.coordinate_hypotheses[coord_key]

                    await self.bayesian_system.add_evidence(
                        hypothesis_id=coord_hypothesis_id,
                        evidence_type=self.EvidenceType.DIRECT_OBSERVATION,
                        supports_hypothesis=(score_change > 0),
                        strength=evidence_strength,
                        context={'score_change': score_change, 'x': action['x'], 'y': action['y']},
                        game_id=game_id,
                        session_id=session_id
                    )

                    logger.debug(f"📍 Added coordinate evidence for ({action['x']},{action['y']})")

        except Exception as e:
            logger.error(f"Tier 3 action result test error: {e}")
            import traceback
            traceback.print_exc()

    async def final_tier3_analysis(self, game_id, session_id):
        """Perform final analysis using Tier 3 systems."""
        try:
            logger.info("🔍 Performing final Tier 3 analysis...")

            # Get Bayesian insights
            insights = await self.bayesian_system.get_hypothesis_insights(game_id)

            logger.info(f"📊 Bayesian Analysis Results:")
            logger.info(f"  Total hypotheses: {insights.get('total_hypotheses', 0)}")
            logger.info(f"  High confidence: {len(insights.get('high_confidence_hypotheses', []))}")
            logger.info(f"  Low confidence: {len(insights.get('low_confidence_hypotheses', []))}")

            # Show high confidence hypotheses
            for hyp in insights.get('high_confidence_hypotheses', [])[:3]:
                logger.info(f"  ✅ {hyp['description']} (prob: {hyp['probability']:.2f})")

            # Prune low confidence hypotheses
            pruned_count = await self.bayesian_system.prune_low_confidence_hypotheses()
            if pruned_count > 0:
                logger.info(f"🧹 Pruned {pruned_count} low-confidence hypotheses")

            logger.info("✅ Final Tier 3 analysis completed")

        except Exception as e:
            logger.error(f"Final Tier 3 analysis error: {e}")


async def main():
    """Main function to run real training."""
    try:
        logger.info("🎮 STARTING REAL TIER 3 TRAINING TEST")
        logger.info("="*60)

        # Create minimal training loop
        training_loop = MinimalTrainingLoop()

        if not training_loop.tier3_initialized:
            logger.error("❌ Tier 3 systems failed to initialize")
            return

        # Run training session
        result = await training_loop.run_training_session(max_actions=30)

        if result['success']:
            logger.info("🎉 TRAINING SESSION COMPLETED SUCCESSFULLY")
            logger.info(f"  Actions: {result['actions_taken']}")
            logger.info(f"  Score: {result['final_score']}")
            logger.info(f"  State: {result['game_state']}")
        else:
            logger.error(f"❌ Training session failed: {result.get('error', 'Unknown error')}")

        # Close database
        training_loop.db_connection.close()

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
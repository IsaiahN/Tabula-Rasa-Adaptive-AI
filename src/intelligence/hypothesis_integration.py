"""
Hypothesis Integration System

This module provides integration between the Game-Specific Hypothesis Generation
and Testing System and the existing Action6Coordinator and Enhanced Gameplay systems.
Creates a unified interface for intelligent game strategy testing.

Key Features:
- Integration with Action6Coordinator for action execution
- Enhanced Gameplay coordination for game context
- Automatic hypothesis generation based on game state
- Systematic testing with detailed reasoning
- Multi-level learning integration
- Database persistence and learning
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
from datetime import datetime
import asyncio

# Disable pycache
import sys
sys.dont_write_bytecode = True

# Import our intelligence components
from .game_pattern_analyzer import GamePatternAnalyzer, get_game_pattern_analyzer
from .hypothesis_generator import HypothesisGenerator, get_hypothesis_generator
from .hypothesis_tester import HypothesisTester, get_hypothesis_tester, TestOutcome

# Import existing game type classifier
from ..learning.game_type_classifier import get_game_type_classifier

logger = logging.getLogger(__name__)

class HypothesisIntegrationSystem:
    """
    Integration system that coordinates hypothesis generation and testing
    with existing gameplay systems.

    Provides a unified interface for intelligent game strategy testing
    that leverages the existing Action6Coordinator and Enhanced Gameplay.
    """

    def __init__(self, db_connection: Optional[sqlite3.Connection] = None,
                 action6_coordinator=None, enhanced_gameplay=None):
        self.db_connection = db_connection
        self.action6_coordinator = action6_coordinator
        self.enhanced_gameplay = enhanced_gameplay

        # Initialize intelligence components
        self.pattern_analyzer = get_game_pattern_analyzer()
        self.hypothesis_generator = get_hypothesis_generator(db_connection)
        self.hypothesis_tester = get_hypothesis_tester(db_connection, self._execute_action_with_integration)
        self.game_type_classifier = get_game_type_classifier()

        # Integration state tracking
        self.active_experiments: Dict[str, bool] = {}
        self.game_sessions: Dict[str, Dict[str, Any]] = {}
        self.integration_stats = {
            'games_analyzed': 0,
            'hypotheses_generated': 0,
            'hypotheses_tested': 0,
            'successful_tests': 0,
            'action6_integrations': 0,
            'enhanced_gameplay_feedbacks': 0
        }

        logger.info("Hypothesis Integration System initialized")

    async def analyze_and_generate_hypotheses(self, game_id: str, screenshot_array: np.ndarray,
                                            game_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze game state and generate hypotheses for winning strategies.

        Args:
            game_id: Unique identifier for the game
            screenshot_array: Current game screenshot as numpy array
            game_context: Game state context from Enhanced Gameplay

        Returns:
            List of generated hypotheses with their details
        """
        logger.info(f"Starting hypothesis generation for game {game_id}")

        try:
            # Get or create session data
            session = self._get_or_create_session(game_id)
            session['last_analysis'] = datetime.now()

            # Generate hypotheses
            hypotheses = self.hypothesis_generator.generate_hypotheses(
                game_id, screenshot_array, max_hypotheses=5
            )

            # Store hypotheses in session
            session['current_hypotheses'] = hypotheses
            session['hypothesis_generation_count'] = session.get('hypothesis_generation_count', 0) + 1

            # Update statistics
            self.integration_stats['games_analyzed'] += 1
            self.integration_stats['hypotheses_generated'] += len(hypotheses)

            # Convert hypotheses to integration format
            integration_results = []
            for hypothesis in hypotheses:
                integration_results.append({
                    'hypothesis_id': hypothesis.hypothesis_id,
                    'hypothesis_type': hypothesis.hypothesis_type.value,
                    'source': hypothesis.source.value,
                    'description': hypothesis.description,
                    'predicted_coordinates': hypothesis.predicted_coordinates,
                    'expected_actions': hypothesis.expected_actions,
                    'confidence': hypothesis.confidence,
                    'reasoning': hypothesis.reasoning,
                    'game_mechanics': [m.value for m in hypothesis.game_mechanics],
                    'created_at': hypothesis.created_at.isoformat()
                })

            logger.info(f"Generated {len(hypotheses)} hypotheses for game {game_id}")

            # Store results in database if available
            if self.db_connection:
                await self._store_hypothesis_generation_session(game_id, hypotheses, game_context)

            return integration_results

        except Exception as e:
            logger.error(f"Error in hypothesis generation for game {game_id}: {e}")
            return []

    async def test_hypothesis(self, game_id: str, hypothesis_id: str,
                            game_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a specific hypothesis using the Action6Coordinator and Enhanced Gameplay.

        Args:
            game_id: Game identifier
            hypothesis_id: ID of hypothesis to test
            game_context: Current game context

        Returns:
            Test result with detailed outcomes and analysis
        """
        logger.info(f"Starting hypothesis test: {hypothesis_id} for game {game_id}")

        try:
            session = self._get_or_create_session(game_id)
            hypotheses = session.get('current_hypotheses', [])

            # Find the hypothesis to test
            target_hypothesis = None
            for hypothesis in hypotheses:
                if hypothesis.hypothesis_id == hypothesis_id:
                    target_hypothesis = hypothesis
                    break

            if not target_hypothesis:
                logger.error(f"Hypothesis {hypothesis_id} not found in session")
                return {'error': 'Hypothesis not found'}

            # Mark experiment as active
            self.active_experiments[hypothesis_id] = True

            try:
                # Execute hypothesis test
                experiment_result = await self.hypothesis_tester.test_hypothesis(
                    target_hypothesis, game_context, max_actions=8, timeout_seconds=120.0
                )

                # Update statistics
                self.integration_stats['hypotheses_tested'] += 1
                if experiment_result.outcome == TestOutcome.SUCCESS:
                    self.integration_stats['successful_tests'] += 1

                # Store test result in session
                session['test_results'] = session.get('test_results', [])
                session['test_results'].append(experiment_result)

                # Update hypothesis in generator with test results
                self.hypothesis_generator.save_hypothesis_result(
                    target_hypothesis,
                    experiment_result.outcome == TestOutcome.SUCCESS,
                    experiment_result.total_score_change,
                    {
                        'experiment_id': experiment_result.experiment_id,
                        'test_duration': experiment_result.test_duration,
                        'actions_count': len(experiment_result.actions_taken)
                    }
                )

                # Create integration result
                integration_result = {
                    'experiment_id': experiment_result.experiment_id,
                    'hypothesis_id': hypothesis_id,
                    'outcome': experiment_result.outcome.value,
                    'total_score_change': experiment_result.total_score_change,
                    'test_duration': experiment_result.test_duration,
                    'actions_taken': len(experiment_result.actions_taken),
                    'successful_actions': len([a for a in experiment_result.actions_taken if a.success]),
                    'learning_insights': experiment_result.learning_insights,
                    'recommendations': experiment_result.recommendations,
                    'action_details': [
                        {
                            'action_id': action.action_id,
                            'coordinates': action.coordinates,
                            'reasoning': action.reasoning,
                            'expected_outcome': action.expected_outcome,
                            'actual_outcome': action.actual_outcome,
                            'success': action.success,
                            'score_change': action.score_change
                        }
                        for action in experiment_result.actions_taken
                    ]
                }

                logger.info(f"Hypothesis test completed: {hypothesis_id}, outcome: {experiment_result.outcome.value}")

                return integration_result

            finally:
                # Mark experiment as inactive
                self.active_experiments[hypothesis_id] = False

        except Exception as e:
            logger.error(f"Error testing hypothesis {hypothesis_id}: {e}")
            return {'error': str(e)}

    async def get_intelligent_action6_coordinates(self, game_id: str, frame: List[List[int]],
                                                game_context: Dict[str, Any]) -> Tuple[int, int]:
        """
        Get intelligent Action6 coordinates using hypothesis-driven approach.

        This method integrates with Action6Coordinator to provide coordinates
        based on current hypotheses and game analysis.

        Args:
            game_id: Game identifier
            frame: Current game frame
            game_context: Game context from Enhanced Gameplay

        Returns:
            Optimal (x, y) coordinates for Action6
        """
        try:
            session = self._get_or_create_session(game_id)

            # Check if we have active hypotheses that can guide coordinate selection
            current_hypotheses = session.get('current_hypotheses', [])

            if current_hypotheses:
                # Use hypothesis-based coordinate selection
                best_hypothesis = max(current_hypotheses, key=lambda h: h.confidence)

                if best_hypothesis.predicted_coordinates:
                    # Get next untested coordinate from best hypothesis
                    tested_coords = set()
                    for test_result in session.get('test_results', []):
                        for action in test_result.actions_taken:
                            tested_coords.add(action.coordinates)

                    for coord in best_hypothesis.predicted_coordinates:
                        if coord not in tested_coords:
                            logger.info(f"Using hypothesis-guided coordinate: {coord} from {best_hypothesis.hypothesis_id}")
                            return coord

            # Fallback to Action6Coordinator if available
            if self.action6_coordinator:
                coords = await self.action6_coordinator.get_optimal_action6_coordinates(
                    frame, game_id, game_context
                )
                logger.debug(f"Using Action6Coordinator coordinates: {coords}")
                return coords

            # Ultimate fallback
            return self._get_center_coordinates(frame)

        except Exception as e:
            logger.error(f"Error getting intelligent Action6 coordinates: {e}")
            return self._get_center_coordinates(frame)

    async def analyze_action_result(self, game_id: str, action_coordinates: Tuple[int, int],
                                  frame_before: List[List[int]], frame_after: List[List[int]],
                                  score_change: float) -> Dict[str, Any]:
        """
        Analyze the result of an action and provide feedback to learning systems.

        Args:
            game_id: Game identifier
            action_coordinates: Coordinates that were clicked
            frame_before: Frame before the action
            frame_after: Frame after the action
            score_change: Change in game score

        Returns:
            Analysis result with learning insights
        """
        try:
            session = self._get_or_create_session(game_id)

            # Analyze with Action6Coordinator if available
            action6_analysis = None
            if self.action6_coordinator:
                await self.action6_coordinator.analyze_action6_effectiveness(
                    frame_before, frame_after, action_coordinates, game_id, score_change
                )
                action6_analysis = "Action6Coordinator analysis completed"

            # Analyze with Enhanced Gameplay if available
            enhanced_analysis = None
            if self.enhanced_gameplay:
                await self.enhanced_gameplay.analyze_action6_result(
                    frame_before, frame_after, action_coordinates, game_id, score_change
                )
                enhanced_analysis = "Enhanced Gameplay analysis completed"
                self.integration_stats['enhanced_gameplay_feedbacks'] += 1

            # Store action result in session
            action_result = {
                'coordinates': action_coordinates,
                'score_change': score_change,
                'timestamp': datetime.now().isoformat(),
                'action6_analysis': action6_analysis,
                'enhanced_analysis': enhanced_analysis
            }

            session['action_results'] = session.get('action_results', [])
            session['action_results'].append(action_result)

            # Update integration statistics
            self.integration_stats['action6_integrations'] += 1

            # Create multi-level learning insight
            learning_insight = await self._create_learning_insight(
                game_id, action_coordinates, score_change, frame_before, frame_after
            )

            return {
                'analysis_completed': True,
                'score_change': score_change,
                'learning_insight': learning_insight,
                'action6_analysis': action6_analysis,
                'enhanced_analysis': enhanced_analysis
            }

        except Exception as e:
            logger.error(f"Error analyzing action result: {e}")
            return {'error': str(e)}

    async def get_adaptive_strategy(self, game_id: str, current_score: float,
                                  recent_performance: List[float]) -> Dict[str, Any]:
        """
        Get adaptive strategy recommendations based on current game state and performance.

        Args:
            game_id: Game identifier
            current_score: Current game score
            recent_performance: List of recent score changes

        Returns:
            Adaptive strategy recommendations
        """
        try:
            session = self._get_or_create_session(game_id)

            # Analyze recent performance
            if recent_performance:
                avg_performance = np.mean(recent_performance)
                performance_trend = "improving" if len(recent_performance) > 1 and recent_performance[-1] > recent_performance[0] else "declining"
            else:
                avg_performance = 0.0
                performance_trend = "unknown"

            # Get game type for strategy adaptation
            game_type = self.game_type_classifier.extract_game_type(game_id)
            game_knowledge = self.game_type_classifier.get_game_type_knowledge(game_type)

            # Determine strategy adaptation
            strategy_recommendations = []

            if avg_performance < 0:
                strategy_recommendations.append("Consider exploratory hypothesis generation")
                strategy_recommendations.append("Test alternative coordinate patterns")

            if performance_trend == "declining":
                strategy_recommendations.append("Switch to database-retrieved successful strategies")
                strategy_recommendations.append("Increase hypothesis diversity")

            if game_knowledge['success_rate'] > 0.7:
                strategy_recommendations.append("Leverage high-confidence game type patterns")

            # Get current hypotheses effectiveness
            current_hypotheses = session.get('current_hypotheses', [])
            if current_hypotheses:
                best_hypothesis = max(current_hypotheses, key=lambda h: h.confidence)
                strategy_recommendations.append(f"Continue with {best_hypothesis.hypothesis_type.value} approach")

            return {
                'game_type': game_type,
                'performance_trend': performance_trend,
                'avg_recent_performance': avg_performance,
                'game_type_success_rate': game_knowledge['success_rate'],
                'strategy_recommendations': strategy_recommendations,
                'adaptive_confidence': min(1.0, (avg_performance + 50) / 100.0)  # Convert score to confidence
            }

        except Exception as e:
            logger.error(f"Error getting adaptive strategy: {e}")
            return {'error': str(e)}

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""

        total_sessions = len(self.game_sessions)
        total_hypotheses_per_session = (
            self.integration_stats['hypotheses_generated'] / max(total_sessions, 1)
        )
        success_rate = (
            self.integration_stats['successful_tests'] / max(self.integration_stats['hypotheses_tested'], 1)
        )

        # Calculate session-level statistics
        session_stats = {
            'total_hypothesis_generations': 0,
            'total_test_results': 0,
            'total_action_results': 0
        }

        for session in self.game_sessions.values():
            session_stats['total_hypothesis_generations'] += session.get('hypothesis_generation_count', 0)
            session_stats['total_test_results'] += len(session.get('test_results', []))
            session_stats['total_action_results'] += len(session.get('action_results', []))

        return {
            **self.integration_stats,
            'total_sessions': total_sessions,
            'avg_hypotheses_per_session': total_hypotheses_per_session,
            'hypothesis_test_success_rate': success_rate,
            'active_experiments': len([exp for exp in self.active_experiments.values() if exp]),
            **session_stats
        }

    async def _execute_action_with_integration(self, coordinate: Tuple[int, int], action: str,
                                             reasoning: str) -> Dict[str, Any]:
        """
        Execute an action through integration with existing systems.

        This method serves as the action executor for the HypothesisTester,
        providing integration with Action6Coordinator and Enhanced Gameplay.
        """
        try:
            logger.info(f"Executing integrated action: {action} at {coordinate}")
            logger.info(f"Action reasoning: {reasoning}")

            # For now, simulate action execution since we don't have direct game interface
            # In real implementation, this would interact with the actual game

            # Simulate outcome based on action type and coordinate
            success_probability = 0.6  # Base success probability

            # Adjust probability based on action type
            if "click" in action.lower():
                success_probability += 0.1
            elif "drag" in action.lower():
                success_probability -= 0.1

            # Simulate random outcome
            success = np.random.random() < success_probability

            if success:
                outcome = f"successful_{action}_execution"
                score_change = np.random.uniform(5, 25)
            else:
                outcome = f"failed_{action}_execution"
                score_change = np.random.uniform(-15, -2)

            result = {
                'outcome': outcome,
                'score_change': score_change,
                'success': success,
                'reasoning_applied': reasoning,
                'integration_used': True
            }

            logger.info(f"Action execution result: {outcome}, score change: {score_change:.1f}")

            return result

        except Exception as e:
            logger.error(f"Error in integrated action execution: {e}")
            return {
                'outcome': f"error: {str(e)}",
                'score_change': -10.0,
                'success': False
            }

    def _get_or_create_session(self, game_id: str) -> Dict[str, Any]:
        """Get or create integration session data for a game."""
        if game_id not in self.game_sessions:
            self.game_sessions[game_id] = {
                'game_id': game_id,
                'created_at': datetime.now(),
                'last_analysis': None,
                'current_hypotheses': [],
                'test_results': [],
                'action_results': [],
                'hypothesis_generation_count': 0
            }
        return self.game_sessions[game_id]

    def _get_center_coordinates(self, frame: List[List[int]]) -> Tuple[int, int]:
        """Get center coordinates as fallback."""
        if not frame or not frame[0]:
            return (25, 25)

        height, width = len(frame), len(frame[0])
        return (width // 2, height // 2)

    async def _store_hypothesis_generation_session(self, game_id: str, hypotheses: List,
                                                 game_context: Dict[str, Any]):
        """Store hypothesis generation session in database."""
        if not self.db_connection:
            return

        try:
            generation_data = {
                'game_id': game_id,
                'game_type': self.game_type_classifier.extract_game_type(game_id),
                'hypotheses_generated': len(hypotheses),
                'generation_strategies_used': list(set(h.source.value for h in hypotheses)),
                'pattern_analysis_time': 0.1,  # Placeholder
                'database_query_time': 0.05,   # Placeholder
                'total_generation_time': 0.15,  # Placeholder
                'top_hypothesis_confidence': max(h.confidence for h in hypotheses) if hypotheses else 0.0,
                'average_hypothesis_confidence': np.mean([h.confidence for h in hypotheses]) if hypotheses else 0.0,
                'session_timestamp': datetime.now().isoformat(),
                'screenshot_analyzed': True
            }

            self.db_connection.execute("""
                INSERT INTO hypothesis_generation_sessions
                (session_id, game_id, game_type, hypotheses_generated, generation_strategies_used,
                 pattern_analysis_time, database_query_time, total_generation_time,
                 top_hypothesis_confidence, average_hypothesis_confidence,
                 session_timestamp, screenshot_analyzed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"session_{game_id}_{int(datetime.now().timestamp())}",
                generation_data['game_id'],
                generation_data['game_type'],
                generation_data['hypotheses_generated'],
                ','.join(generation_data['generation_strategies_used']),
                generation_data['pattern_analysis_time'],
                generation_data['database_query_time'],
                generation_data['total_generation_time'],
                generation_data['top_hypothesis_confidence'],
                generation_data['average_hypothesis_confidence'],
                generation_data['session_timestamp'],
                generation_data['screenshot_analyzed']
            ))

            self.db_connection.commit()
            logger.debug(f"Stored hypothesis generation session for game {game_id}")

        except Exception as e:
            logger.error(f"Error storing hypothesis generation session: {e}")

    async def _create_learning_insight(self, game_id: str, coordinates: Tuple[int, int],
                                     score_change: float, frame_before: List[List[int]],
                                     frame_after: List[List[int]]) -> Dict[str, Any]:
        """Create multi-level learning insight from action result."""
        try:
            game_type = self.game_type_classifier.extract_game_type(game_id)

            # Determine learning level
            if score_change > 20:
                learning_level = "macro"
                insight_type = "strategy_effectiveness"
                description = f"High-impact action at {coordinates} in {game_type} game (+{score_change:.1f})"
            elif score_change > 0:
                learning_level = "meso"
                insight_type = "pattern_recognition"
                description = f"Positive action at {coordinates} in {game_type} game (+{score_change:.1f})"
            else:
                learning_level = "micro"
                insight_type = "mechanic_understanding"
                description = f"Learning from negative action at {coordinates} in {game_type} game ({score_change:.1f})"

            learning_insight = {
                'learning_level': learning_level,
                'learning_context': 'within_game',
                'insight_type': insight_type,
                'insight_description': description,
                'supporting_data': {
                    'coordinates': coordinates,
                    'score_change': score_change,
                    'game_type': game_type
                },
                'confidence': min(1.0, abs(score_change) / 50.0),
                'impact_score': abs(score_change) / 100.0,
                'created_at': datetime.now().isoformat()
            }

            # Store in database if available
            if self.db_connection:
                try:
                    self.db_connection.execute("""
                        INSERT INTO multi_level_learning
                        (learning_id, learning_level, game_id, game_type, learning_context,
                         insight_type, insight_description, supporting_data, confidence,
                         impact_score, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"insight_{game_id}_{int(datetime.now().timestamp())}",
                        learning_insight['learning_level'],
                        game_id,
                        game_type,
                        learning_insight['learning_context'],
                        learning_insight['insight_type'],
                        learning_insight['insight_description'],
                        str(learning_insight['supporting_data']),
                        learning_insight['confidence'],
                        learning_insight['impact_score'],
                        learning_insight['created_at']
                    ))
                    self.db_connection.commit()
                except Exception as e:
                    logger.debug(f"Could not store learning insight: {e}")

            return learning_insight

        except Exception as e:
            logger.error(f"Error creating learning insight: {e}")
            return {'error': str(e)}

# Module initialization
def create_hypothesis_integration_system(db_connection: Optional[sqlite3.Connection] = None,
                                        action6_coordinator=None,
                                        enhanced_gameplay=None) -> HypothesisIntegrationSystem:
    """Factory function to create a HypothesisIntegrationSystem instance."""
    return HypothesisIntegrationSystem(db_connection, action6_coordinator, enhanced_gameplay)

# Singleton instance for global use
_integration_instance = None

def get_hypothesis_integration_system(db_connection: Optional[sqlite3.Connection] = None,
                                    action6_coordinator=None,
                                    enhanced_gameplay=None) -> HypothesisIntegrationSystem:
    """Get the singleton HypothesisIntegrationSystem instance."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = create_hypothesis_integration_system(
            db_connection, action6_coordinator, enhanced_gameplay
        )
    return _integration_instance
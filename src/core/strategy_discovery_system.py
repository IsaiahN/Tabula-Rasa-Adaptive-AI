#!/usr/bin/env python3
"""
Strategy Discovery & Replication System

This module implements the discovery, refinement, and replication of winning
action sequences for improved game performance.
"""

import logging
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.database.system_integration import get_system_integration
from src.learning.game_type_classifier import get_game_type_classifier

logger = logging.getLogger(__name__)

@dataclass
class WinningStrategy:
    """Represents a discovered winning strategy."""
    strategy_id: str
    game_type: str
    game_id: str
    action_sequence: List[int]
    score_progression: List[float]
    total_score_increase: float
    efficiency: float  # Score per action
    discovery_timestamp: float
    replication_attempts: int = 0
    successful_replications: int = 0
    refinement_level: int = 0
    is_active: bool = True

@dataclass
class StrategyRefinement:
    """Represents a strategy refinement attempt."""
    strategy_id: str
    refinement_attempt: int
    original_efficiency: float
    new_efficiency: float
    improvement: float
    action_sequence: List[int]
    refinement_timestamp: float

class StrategyDiscoverySystem:
    """
    Strategy Discovery & Replication System
    
    Discovers winning action sequences, refines them for efficiency,
    and replicates them across similar games.
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        self.game_type_classifier = get_game_type_classifier()
        self.active_strategies: Dict[str, List[WinningStrategy]] = {}
        self.refinement_history: Dict[str, List[StrategyRefinement]] = {}
        
    async def discover_winning_strategy(self, 
                                      game_id: str,
                                      action_sequence: List[int],
                                      score_progression: List[float]) -> Optional[WinningStrategy]:
        """
        Discover and record a winning strategy from successful action sequence.
        
        Args:
            game_id: Game identifier
            action_sequence: Sequence of actions that led to success
            score_progression: Score values throughout the sequence
            
        Returns:
            WinningStrategy if strategy is worth recording, None otherwise
        """
        try:
            # Only record if we achieved significant score increase
            if len(score_progression) < 3 or len(action_sequence) < 3:
                return None
            
            score_increase = score_progression[-1] - score_progression[0]
            if score_increase < 5.0:  # Minimum score increase to consider
                return None
            
            # Calculate efficiency (score per action)
            efficiency = score_increase / len(action_sequence)
            
            # Get game type
            game_type = self.game_type_classifier.extract_game_type(game_id)
            
            # Create strategy
            strategy_id = f"{game_type}_{uuid.uuid4().hex[:8]}"
            strategy = WinningStrategy(
                strategy_id=strategy_id,
                game_type=game_type,
                game_id=game_id,
                action_sequence=action_sequence.copy(),
                score_progression=score_progression.copy(),
                total_score_increase=score_increase,
                efficiency=efficiency,
                discovery_timestamp=time.time()
            )
            
            # Store in database
            await self._store_winning_strategy(strategy)

            # Analyze win conditions for this successful strategy
            try:
                win_conditions = await self.analyze_win_conditions(game_id, action_sequence, score_progression)
                logger.info(f" WIN CONDITIONS DISCOVERED: {len(win_conditions)} conditions analyzed for strategy {strategy_id}")
            except Exception as e:
                logger.warning(f"Failed to analyze win conditions for strategy {strategy_id}: {e}")

            # Update local cache
            if game_type not in self.active_strategies:
                self.active_strategies[game_type] = []
            self.active_strategies[game_type].append(strategy)

            logger.info(f" WINNING STRATEGY DISCOVERED: {strategy_id} - "
                       f"{len(action_sequence)} actions, +{score_increase:.1f} score, "
                       f"efficiency: {efficiency:.2f}")

            return strategy
            
        except Exception as e:
            logger.error(f"Error discovering winning strategy: {e}")
            return None
    
    async def refine_strategy(self, 
                            strategy: WinningStrategy,
                            new_attempt: Dict[str, Any]) -> Optional[StrategyRefinement]:
        """
        Refine a winning strategy to achieve the same result with fewer actions.
        
        Args:
            strategy: Strategy to refine
            new_attempt: New attempt data with potentially better efficiency
            
        Returns:
            StrategyRefinement if improvement found, None otherwise
        """
        try:
            new_efficiency = new_attempt.get('efficiency', 0)
            new_action_sequence = new_attempt.get('action_sequence', [])
            
            if new_efficiency <= strategy.efficiency:
                return None
            
            # Create refinement record
            refinement = StrategyRefinement(
                strategy_id=strategy.strategy_id,
                refinement_attempt=strategy.refinement_level + 1,
                original_efficiency=strategy.efficiency,
                new_efficiency=new_efficiency,
                improvement=new_efficiency - strategy.efficiency,
                action_sequence=new_action_sequence,
                refinement_timestamp=time.time()
            )
            
            # Store refinement in database
            await self._store_strategy_refinement(refinement)
            
            # Update strategy
            strategy.efficiency = new_efficiency
            strategy.action_sequence = new_action_sequence
            strategy.refinement_level += 1
            
            # Update in database
            await self._update_strategy_efficiency(strategy)
            
            # Update local cache
            if strategy.strategy_id not in self.refinement_history:
                self.refinement_history[strategy.strategy_id] = []
            self.refinement_history[strategy.strategy_id].append(refinement)
            
            logger.info(f" STRATEGY REFINED: {strategy.strategy_id} - "
                       f"New efficiency: {new_efficiency:.2f} "
                       f"(was {refinement.original_efficiency:.2f}, "
                       f"improvement: +{refinement.improvement:.2f})")
            
            return refinement
            
        except Exception as e:
            logger.error(f"Error refining strategy: {strategy.strategy_id}: {e}")
            return None
    
    async def get_best_strategy_for_game(self, game_id: str) -> Optional[WinningStrategy]:
        """Get the best known strategy for a game type."""
        try:
            game_type = self.game_type_classifier.extract_game_type(game_id)
            
            # Load strategies for this game type
            strategies = await self._load_strategies_for_game_type(game_type)
            if not strategies:
                return None
            
            # Return the most efficient strategy
            best_strategy = max(strategies, key=lambda s: s.efficiency)
            return best_strategy
            
        except Exception as e:
            logger.error(f"Error getting best strategy for game {game_id}: {e}")
            return None
    
    async def should_attempt_strategy_replication(self, game_id: str) -> bool:
        """Determine if we should attempt to replicate a known winning strategy."""
        try:
            best_strategy = await self.get_best_strategy_for_game(game_id)
            if not best_strategy:
                return False
            
            # Only attempt replication if we have a reasonably efficient strategy
            if best_strategy.efficiency < 0.5:  # Less than 0.5 score per action
                return False
            
            # Check recent replication attempts
            recent_attempts = await self._get_recent_replication_attempts(best_strategy.strategy_id)
            if len(recent_attempts) > 0:
                last_attempt = max(recent_attempts, key=lambda a: a['replication_timestamp'])
                if time.time() - last_attempt['replication_timestamp'] < 300:  # 5 minutes between attempts
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking strategy replication eligibility: {e}")
            return False
    
    async def execute_strategy_replication(self, 
                                         game_id: str,
                                         strategy: WinningStrategy) -> Dict[str, Any]:
        """
        Execute a known winning strategy to test its replicability.
        
        Args:
            game_id: Current game identifier
            strategy: Strategy to replicate
            
        Returns:
            Replication execution data
        """
        try:
            # Record replication attempt
            replication_data = {
                'strategy_id': strategy.strategy_id,
                'game_id': game_id,
                'replication_attempt': strategy.replication_attempts + 1,
                'expected_efficiency': strategy.efficiency,
                'action_sequence': strategy.action_sequence,
                'replication_timestamp': time.time()
            }
            
            # Store replication attempt
            await self._store_strategy_replication(replication_data)
            
            # Update strategy replication count
            strategy.replication_attempts += 1
            await self._update_strategy_replication_count(strategy)
            
            logger.info(f" REPLICATING STRATEGY: {strategy.strategy_id} - "
                       f"Attempt {strategy.replication_attempts}")
            
            return replication_data
            
        except Exception as e:
            logger.error(f"Error executing strategy replication: {e}")
            return {}
    
    async def record_replication_result(self, 
                                      replication_data: Dict[str, Any],
                                      actual_efficiency: float,
                                      success: bool):
        """Record the result of a strategy replication attempt."""
        try:
            strategy_id = replication_data['strategy_id']
            game_id = replication_data['game_id']
            
            # Update replication record
            await self.integration.db.execute("""
                UPDATE strategy_replications
                SET actual_efficiency = ?, success = ?
                WHERE strategy_id = ? AND game_id = ? AND replication_attempt = ?
            """, (actual_efficiency, success, strategy_id, game_id, 
                  replication_data['replication_attempt']))
            
            # Update strategy success count
            if success:
                await self.integration.db.execute("""
                    UPDATE winning_strategies
                    SET successful_replications = successful_replications + 1
                    WHERE strategy_id = ?
                """, (strategy_id,))
                
                # Update local cache
                for game_type, strategies in self.active_strategies.items():
                    for strategy in strategies:
                        if strategy.strategy_id == strategy_id:
                            strategy.successful_replications += 1
                            break
            
            logger.info(f"Strategy replication {'successful' if success else 'failed'}: "
                       f"{strategy_id} - efficiency: {actual_efficiency:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording replication result: {e}")
    
    async def get_strategy_recommendations(self, 
                                         game_id: str,
                                         current_actions: List[int]) -> List[Dict[str, Any]]:
        """Get strategy recommendations for current game state."""
        try:
            game_type = self.game_type_classifier.extract_game_type(game_id)
            
            # Load strategies for this game type
            strategies = await self._load_strategies_for_game_type(game_type)
            if not strategies:
                return []
            
            # Filter strategies that can be applied with current actions
            applicable_strategies = []
            for strategy in strategies:
                if strategy.is_active and self._can_apply_strategy(strategy, current_actions):
                    applicable_strategies.append(strategy)
            
            # Sort by efficiency and success rate
            applicable_strategies.sort(key=lambda s: (
                s.efficiency * (1 + s.successful_replications / max(1, s.replication_attempts)),
                s.efficiency
            ), reverse=True)
            
            # Return top recommendations
            recommendations = []
            for strategy in applicable_strategies[:3]:  # Top 3 strategies
                recommendations.append({
                    'strategy_id': strategy.strategy_id,
                    'action_sequence': strategy.action_sequence,
                    'efficiency': strategy.efficiency,
                    'success_rate': (strategy.successful_replications / 
                                   max(1, strategy.replication_attempts)),
                    'confidence': min(1.0, strategy.efficiency * 0.5 + 
                                    (strategy.successful_replications / max(1, strategy.replication_attempts)) * 0.5)
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting strategy recommendations: {e}")
            return []
    
    def _can_apply_strategy(self, strategy: WinningStrategy, current_actions: List[int]) -> bool:
        """Check if a strategy can be applied with current available actions."""
        try:
            # Check if all actions in strategy are available
            strategy_actions = set(strategy.action_sequence)
            available_actions = set(current_actions)
            
            return strategy_actions.issubset(available_actions)
            
        except Exception as e:
            logger.error(f"Error checking strategy applicability: {e}")
            return False
    
    async def _store_winning_strategy(self, strategy: WinningStrategy):
        """Store winning strategy in database."""
        try:
            await self.integration.db.execute("""
                INSERT INTO winning_strategies
                (strategy_id, game_type, game_id, action_sequence, score_progression,
                 total_score_increase, efficiency, discovery_timestamp, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy.strategy_id, strategy.game_type, strategy.game_id,
                json.dumps(strategy.action_sequence), json.dumps(strategy.score_progression),
                strategy.total_score_increase, strategy.efficiency,
                strategy.discovery_timestamp, strategy.is_active
            ))
            
        except Exception as e:
            logger.error(f"Error storing winning strategy: {e}")
    
    async def _store_strategy_refinement(self, refinement: StrategyRefinement):
        """Store strategy refinement in database."""
        try:
            await self.integration.db.execute("""
                INSERT INTO strategy_refinements
                (strategy_id, refinement_attempt, original_efficiency, new_efficiency,
                 improvement, action_sequence, refinement_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                refinement.strategy_id, refinement.refinement_attempt,
                refinement.original_efficiency, refinement.new_efficiency,
                refinement.improvement, json.dumps(refinement.action_sequence),
                refinement.refinement_timestamp
            ))
            
        except Exception as e:
            logger.error(f"Error storing strategy refinement: {e}")
    
    async def _store_strategy_replication(self, replication_data: Dict[str, Any]):
        """Store strategy replication attempt in database."""
        try:
            await self.integration.db.execute("""
                INSERT INTO strategy_replications
                (strategy_id, game_id, replication_attempt, expected_efficiency,
                 replication_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                replication_data['strategy_id'], replication_data['game_id'],
                replication_data['replication_attempt'], replication_data['expected_efficiency'],
                replication_data['replication_timestamp']
            ))
            
        except Exception as e:
            logger.error(f"Error storing strategy replication: {e}")
    
    async def _load_strategies_for_game_type(self, game_type: str) -> List[WinningStrategy]:
        """Load strategies for a specific game type."""
        try:
            query = """
                SELECT strategy_id, game_type, game_id, action_sequence, score_progression,
                       total_score_increase, efficiency, discovery_timestamp,
                       replication_attempts, successful_replications, refinement_level, is_active
                FROM winning_strategies
                WHERE game_type = ? AND is_active = 1
                ORDER BY efficiency DESC
            """
            
            results = await self.integration.db.fetch_all(query, (game_type,))
            
            strategies = []
            for row in results:
                strategy = WinningStrategy(
                    strategy_id=row['strategy_id'],
                    game_type=row['game_type'],
                    game_id=row['game_id'],
                    action_sequence=json.loads(row['action_sequence']),
                    score_progression=json.loads(row['score_progression']),
                    total_score_increase=row['total_score_increase'],
                    efficiency=row['efficiency'],
                    discovery_timestamp=row['discovery_timestamp'],
                    replication_attempts=row['replication_attempts'],
                    successful_replications=row['successful_replications'],
                    refinement_level=row['refinement_level'],
                    is_active=bool(row['is_active'])
                )
                strategies.append(strategy)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error loading strategies for game type {game_type}: {e}")
            return []
    
    async def _get_recent_replication_attempts(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get recent replication attempts for a strategy."""
        try:
            query = """
                SELECT game_id, replication_attempt, replication_timestamp
                FROM strategy_replications
                WHERE strategy_id = ?
                ORDER BY replication_timestamp DESC
                LIMIT 10
            """
            
            results = await self.integration.db.fetch_all(query, (strategy_id,))
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting recent replication attempts: {e}")
            return []
    
    async def _update_strategy_efficiency(self, strategy: WinningStrategy):
        """Update strategy efficiency in database."""
        try:
            await self.integration.db.execute("""
                UPDATE winning_strategies
                SET efficiency = ?, action_sequence = ?, refinement_level = ?
                WHERE strategy_id = ?
            """, (strategy.efficiency, json.dumps(strategy.action_sequence),
                  strategy.refinement_level, strategy.strategy_id))
            
        except Exception as e:
            logger.error(f"Error updating strategy efficiency: {e}")
    
    async def _update_strategy_replication_count(self, strategy: WinningStrategy):
        """Update strategy replication count in database."""
        try:
            await self.integration.db.execute("""
                UPDATE winning_strategies
                SET replication_attempts = ?
                WHERE strategy_id = ?
            """, (strategy.replication_attempts, strategy.strategy_id))
            
        except Exception as e:
            logger.error(f"Error updating strategy replication count: {e}")
    
    async def get_strategy_statistics(self, game_type: str) -> Dict[str, Any]:
        """Get strategy statistics for a game type."""
        try:
            strategies = await self._load_strategies_for_game_type(game_type)
            if not strategies:
                return {
                    'total_strategies': 0,
                    'average_efficiency': 0.0,
                    'total_replications': 0,
                    'success_rate': 0.0
                }
            
            total_strategies = len(strategies)
            average_efficiency = sum(s.efficiency for s in strategies) / total_strategies
            total_replications = sum(s.replication_attempts for s in strategies)
            successful_replications = sum(s.successful_replications for s in strategies)
            success_rate = (successful_replications / total_replications) if total_replications > 0 else 0.0
            
            return {
                'total_strategies': total_strategies,
                'average_efficiency': average_efficiency,
                'total_replications': total_replications,
                'success_rate': success_rate,
                'top_strategies': [
                    {
                        'strategy_id': s.strategy_id,
                        'efficiency': s.efficiency,
                        'replications': s.replication_attempts,
                        'successes': s.successful_replications
                    } for s in strategies[:5]  # Top 5 strategies
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy statistics: {e}")
            return {}

    # ============================================================================
    # WIN CONDITION ANALYSIS METHODS
    # ============================================================================

    async def analyze_win_conditions(self,
                                   game_id: str,
                                   action_sequence: List[int],
                                   score_progression: List[float]) -> List[Dict[str, Any]]:
        """
        Analyze and extract win conditions from successful game completion.

        Args:
            game_id: Game identifier
            action_sequence: Sequence of actions that led to success
            score_progression: Score values throughout the sequence

        Returns:
            List of extracted win conditions
        """
        try:
            game_type = self.game_type_classifier.extract_game_type(game_id)
            win_conditions = []

            # 1. Action Pattern Analysis
            pattern_conditions = self._extract_action_patterns(action_sequence, game_type, game_id)
            win_conditions.extend(pattern_conditions)

            # 2. Score Threshold Analysis
            threshold_conditions = self._extract_score_thresholds(score_progression, game_type, game_id)
            win_conditions.extend(threshold_conditions)

            # 3. Sequence Timing Analysis
            timing_conditions = self._extract_sequence_timing(action_sequence, score_progression, game_type, game_id)
            win_conditions.extend(timing_conditions)

            # 4. Level Completion Analysis (if applicable)
            if "_level_" in game_id:
                level_conditions = self._extract_level_completion_patterns(action_sequence, score_progression, game_type, game_id)
                win_conditions.extend(level_conditions)

            # Store discovered conditions
            for condition in win_conditions:
                await self.store_win_condition(game_type, condition['type'], condition['data'])

            logger.info(f" WIN CONDITIONS EXTRACTED: {len(win_conditions)} conditions from {game_id}")
            return win_conditions

        except Exception as e:
            logger.error(f"Error analyzing win conditions for {game_id}: {e}")
            return []

    async def store_win_condition(self,
                                game_type: str,
                                condition_type: str,
                                condition_data: Dict[str, Any],
                                game_id: str = None,
                                strategy_id: str = None) -> str:
        """
        Store a new win condition in the database.

        Args:
            game_type: Type of game
            condition_type: Type of condition ('action_pattern', 'score_threshold', etc.)
            condition_data: Data describing the condition
            game_id: Optional specific game ID
            strategy_id: Optional associated strategy ID

        Returns:
            The condition_id of the stored condition
        """
        try:
            condition_id = f"{game_type}_{condition_type}_{uuid.uuid4().hex[:8]}"

            query = """
                INSERT OR REPLACE INTO win_conditions
                (condition_id, game_type, game_id, condition_type, condition_data,
                 first_observed, last_observed, strategy_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            current_time = time.time()
            await self.integration.db.execute(
                query,
                (condition_id, game_type, game_id, condition_type,
                 json.dumps(condition_data), current_time, current_time, strategy_id)
            )

            logger.debug(f"Stored win condition: {condition_id}")
            return condition_id

        except Exception as e:
            logger.error(f"Error storing win condition: {e}")
            return ""

    async def get_win_conditions_for_game_type(self, game_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve all known win conditions for a specific game type.

        Args:
            game_type: Type of game to get conditions for

        Returns:
            List of win conditions with their data
        """
        try:
            query = """
                SELECT condition_id, condition_type, condition_data, frequency,
                       success_rate, first_observed, last_observed, total_games_observed
                FROM win_conditions
                WHERE game_type = ?
                ORDER BY success_rate DESC, frequency DESC
            """

            results = await self.integration.db.fetch_all(query, (game_type,))

            conditions = []
            for row in results:
                condition = {
                    'condition_id': row['condition_id'],
                    'condition_type': row['condition_type'],
                    'condition_data': json.loads(row['condition_data']),
                    'frequency': row['frequency'],
                    'success_rate': row['success_rate'],
                    'first_observed': row['first_observed'],
                    'last_observed': row['last_observed'],
                    'total_games_observed': row['total_games_observed']
                }
                conditions.append(condition)

            return conditions

        except Exception as e:
            logger.error(f"Error getting win conditions for {game_type}: {e}")
            return []

    async def compare_multiple_wins(self, game_type: str) -> Dict[str, Any]:
        """
        Analyze patterns across multiple wins for the same game type.

        Args:
            game_type: Type of game to analyze

        Returns:
            Analysis of common patterns across wins
        """
        try:
            # Get all strategies for this game type
            strategies = await self._load_strategies_for_game_type(game_type)
            if len(strategies) < 2:
                return {'error': 'Not enough strategies to compare', 'count': len(strategies)}

            # Get all win conditions for this game type
            conditions = await self.get_win_conditions_for_game_type(game_type)

            # Extract common patterns
            common_patterns = await self.extract_common_patterns(strategies)

            analysis = {
                'game_type': game_type,
                'total_strategies': len(strategies),
                'total_conditions': len(conditions),
                'common_patterns': common_patterns,
                'most_successful_conditions': [
                    c for c in conditions if c['success_rate'] > 0.7
                ][:5],  # Top 5 most successful
                'pattern_statistics': {
                    'action_patterns': len([c for c in conditions if c['condition_type'] == 'action_pattern']),
                    'score_thresholds': len([c for c in conditions if c['condition_type'] == 'score_threshold']),
                    'timing_patterns': len([c for c in conditions if c['condition_type'] == 'sequence_timing']),
                    'level_patterns': len([c for c in conditions if c['condition_type'] == 'level_completion'])
                }
            }

            return analysis

        except Exception as e:
            logger.error(f"Error comparing multiple wins for {game_type}: {e}")
            return {}

    async def extract_common_patterns(self, strategies: List[WinningStrategy]) -> List[Dict[str, Any]]:
        """
        Extract common winning patterns from multiple strategies.

        Args:
            strategies: List of winning strategies to analyze

        Returns:
            List of common patterns found
        """
        try:
            if len(strategies) < 2:
                return []

            common_patterns = []

            # 1. Common Action Sequences (2+ actions in a row)
            action_sequences = {}
            for strategy in strategies:
                seq = strategy.action_sequence
                for i in range(len(seq) - 1):
                    pattern = tuple(seq[i:i+2])
                    if pattern not in action_sequences:
                        action_sequences[pattern] = 0
                    action_sequences[pattern] += 1

            # Find sequences that appear in multiple strategies
            common_threshold = max(2, len(strategies) // 2)
            for pattern, count in action_sequences.items():
                if count >= common_threshold:
                    common_patterns.append({
                        'type': 'common_action_sequence',
                        'pattern': list(pattern),
                        'frequency': count,
                        'percentage': count / len(strategies)
                    })

            # 2. Common Efficiency Ranges
            efficiencies = [s.efficiency for s in strategies]
            avg_efficiency = np.mean(efficiencies)
            std_efficiency = np.std(efficiencies)

            if std_efficiency < avg_efficiency * 0.3:  # Low variance
                common_patterns.append({
                    'type': 'efficiency_consistency',
                    'avg_efficiency': avg_efficiency,
                    'std_deviation': std_efficiency,
                    'range': [min(efficiencies), max(efficiencies)]
                })

            # 3. Common Score Progression Patterns
            score_increases = [s.total_score_increase for s in strategies]
            if len(set(score_increases)) < len(score_increases) * 0.5:  # Many similar scores
                common_patterns.append({
                    'type': 'score_convergence',
                    'common_scores': list(set(score_increases)),
                    'frequency_distribution': {score: score_increases.count(score) for score in set(score_increases)}
                })

            return common_patterns

        except Exception as e:
            logger.error(f"Error extracting common patterns: {e}")
            return []

    async def update_win_condition_frequency(self, condition_id: str, success: bool) -> None:
        """
        Update frequency and success rate of a win condition.

        Args:
            condition_id: ID of the condition to update
            success: Whether the condition led to success
        """
        try:
            # Get current values
            query = "SELECT frequency, success_rate, total_games_observed FROM win_conditions WHERE condition_id = ?"
            result = await self.integration.db.fetch_one(query, (condition_id,))

            if result:
                current_freq = result['frequency']
                current_success_rate = result['success_rate']
                current_total = result['total_games_observed']

                # Update values
                new_total = current_total + 1
                if success:
                    new_freq = current_freq + 1
                else:
                    new_freq = current_freq

                new_success_rate = new_freq / new_total

                # Update database
                update_query = """
                    UPDATE win_conditions
                    SET frequency = ?, success_rate = ?, total_games_observed = ?, last_observed = ?
                    WHERE condition_id = ?
                """

                await self.integration.db.execute(
                    update_query,
                    (new_freq, new_success_rate, new_total, time.time(), condition_id)
                )

        except Exception as e:
            logger.error(f"Error updating win condition frequency: {e}")

    # ============================================================================
    # PRIVATE WIN CONDITION EXTRACTION HELPERS
    # ============================================================================

    def _extract_action_patterns(self, action_sequence: List[int], game_type: str, game_id: str) -> List[Dict[str, Any]]:
        """Extract action pattern conditions from sequence."""
        patterns = []

        try:
            # Repeated action sequences
            for length in [2, 3, 4]:
                if len(action_sequence) >= length:
                    for i in range(len(action_sequence) - length + 1):
                        pattern = action_sequence[i:i+length]

                        # Check if this pattern repeats
                        pattern_count = 0
                        for j in range(len(action_sequence) - length + 1):
                            if action_sequence[j:j+length] == pattern:
                                pattern_count += 1

                        if pattern_count >= 2:  # Pattern repeats at least twice
                            patterns.append({
                                'type': 'action_pattern',
                                'data': {
                                    'pattern': pattern,
                                    'length': length,
                                    'repetitions': pattern_count,
                                    'positions': [j for j in range(len(action_sequence) - length + 1)
                                                if action_sequence[j:j+length] == pattern]
                                }
                            })

            # Action frequency distribution
            action_counts = {}
            for action in action_sequence:
                action_counts[action] = action_counts.get(action, 0) + 1

            # Find dominant actions (>30% of sequence)
            total_actions = len(action_sequence)
            for action, count in action_counts.items():
                if count / total_actions > 0.3:
                    patterns.append({
                        'type': 'action_pattern',
                        'data': {
                            'dominant_action': action,
                            'frequency': count,
                            'percentage': count / total_actions
                        }
                    })

        except Exception as e:
            logger.error(f"Error extracting action patterns: {e}")

        return patterns

    def _extract_score_thresholds(self, score_progression: List[float], game_type: str, game_id: str) -> List[Dict[str, Any]]:
        """Extract score threshold conditions."""
        thresholds = []

        try:
            if len(score_progression) < 3:
                return thresholds

            # Find significant score jumps
            for i in range(1, len(score_progression)):
                score_change = score_progression[i] - score_progression[i-1]
                if score_change > 10:  # Significant increase
                    thresholds.append({
                        'type': 'score_threshold',
                        'data': {
                            'threshold_score': score_progression[i-1],
                            'target_score': score_progression[i],
                            'score_jump': score_change,
                            'position': i
                        }
                    })

            # Final score threshold
            final_score = score_progression[-1]
            if final_score > 50:  # Arbitrary threshold for "good" score
                thresholds.append({
                    'type': 'score_threshold',
                    'data': {
                        'final_score_threshold': final_score,
                        'total_increase': final_score - score_progression[0]
                    }
                })

        except Exception as e:
            logger.error(f"Error extracting score thresholds: {e}")

        return thresholds

    def _extract_sequence_timing(self, action_sequence: List[int], score_progression: List[float],
                                game_type: str, game_id: str) -> List[Dict[str, Any]]:
        """Extract sequence timing conditions."""
        timing_patterns = []

        try:
            # Actions per score increase
            if len(action_sequence) > 0 and len(score_progression) > 1:
                total_score_increase = score_progression[-1] - score_progression[0]
                if total_score_increase > 0:
                    actions_per_score = len(action_sequence) / total_score_increase

                    timing_patterns.append({
                        'type': 'sequence_timing',
                        'data': {
                            'actions_per_score_point': actions_per_score,
                            'total_actions': len(action_sequence),
                            'total_score_increase': total_score_increase,
                            'efficiency_ratio': total_score_increase / len(action_sequence)
                        }
                    })

            # Early vs late game patterns
            if len(action_sequence) >= 6:
                early_actions = action_sequence[:len(action_sequence)//3]
                late_actions = action_sequence[-len(action_sequence)//3:]

                if early_actions != late_actions:
                    timing_patterns.append({
                        'type': 'sequence_timing',
                        'data': {
                            'early_pattern': early_actions,
                            'late_pattern': late_actions,
                            'pattern_shift': True
                        }
                    })

        except Exception as e:
            logger.error(f"Error extracting sequence timing: {e}")

        return timing_patterns

    def _extract_level_completion_patterns(self, action_sequence: List[int], score_progression: List[float],
                                         game_type: str, game_id: str) -> List[Dict[str, Any]]:
        """Extract level completion patterns."""
        level_patterns = []

        try:
            # Extract level number from game_id
            level_num = None
            if "_level_" in game_id:
                try:
                    level_num = int(game_id.split("_level_")[1])
                except (IndexError, ValueError):
                    level_num = None

            if level_num:
                level_patterns.append({
                    'type': 'level_completion',
                    'data': {
                        'level_number': level_num,
                        'completion_actions': len(action_sequence),
                        'completion_score': score_progression[-1] if score_progression else 0,
                        'action_sequence': action_sequence,
                        'score_progression': score_progression
                    }
                })

        except Exception as e:
            logger.error(f"Error extracting level completion patterns: {e}")

        return level_patterns

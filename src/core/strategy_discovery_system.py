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

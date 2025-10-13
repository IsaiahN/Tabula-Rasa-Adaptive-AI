"""
Strategic Learning Loop System for ARC Games

This module implements a reinforcement learning system that evolves strategies
based on outcomes and updates hypothesis weights. It integrates with the
hypothesis generator and causal analysis systems to create a complete
learning loop.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import sqlite3
import logging
from datetime import datetime
import json

# Disable pycache
import sys
sys.dont_write_bytecode = True

logger = logging.getLogger(__name__)

@dataclass
class Strategy:
    """Represents a game-playing strategy."""
    strategy_id: str
    name: str
    description: str
    components: Dict[str, Any]
    state_conditions: Dict[str, Any]
    action_weights: Dict[int, float]
    success_rate: float
    confidence: float
    usage_count: int

@dataclass
class StrategyOutcome:
    """Represents the outcome of applying a strategy."""
    strategy_id: str
    success: bool
    score_change: float
    action_sequence: List[int]
    state_changes: Dict[str, Any]
    duration: float
    timestamp: datetime

class StrategicLearningSystem:
    """Core system for strategic learning and evolution."""

    def __init__(self, db_connection: Optional[sqlite3.Connection] = None):
        self.db_connection = db_connection
        self.strategies: Dict[str, Strategy] = {}
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.confidence_threshold = 0.6

    def select_strategy(self,
                       game_id: str,
                       game_state: Dict[str, Any],
                       available_actions: List[int]) -> Optional[Strategy]:
        """Select best strategy for current game state."""
        
        applicable_strategies = self._find_applicable_strategies(game_state)
        
        if not applicable_strategies:
            return None
            
        # Decide between exploration and exploitation
        if np.random.random() < self.exploration_rate:
            # Exploration: try less-used applicable strategy
            return self._select_exploration_strategy(applicable_strategies)
        else:
            # Exploitation: use best performing applicable strategy
            return self._select_best_strategy(applicable_strategies, game_state)

    def update_strategy(self,
                       strategy: Strategy,
                       outcome: StrategyOutcome) -> None:
        """Update strategy based on its outcome."""
        
        try:
            # Update usage statistics
            strategy.usage_count += 1
            
            # Update success rate with exponential moving average
            old_rate = strategy.success_rate
            new_rate = outcome.success
            strategy.success_rate = (old_rate * (1 - self.learning_rate) +
                                   new_rate * self.learning_rate)
            
            # Update action weights based on outcome
            self._update_action_weights(strategy, outcome)
            
            # Update confidence based on consistency
            strategy.confidence = self._calculate_strategy_confidence(strategy)
            
            # Store updates in database
            self._store_strategy_update(strategy, outcome)
            
        except Exception as e:
            logger.error(f"Error updating strategy: {e}")

    def generate_new_strategy(self,
                            game_id: str,
                            successful_sequences: List[Dict[str, Any]]) -> Optional[Strategy]:
        """Generate new strategy from successful action sequences."""
        
        try:
            if not successful_sequences:
                return None
                
            # Extract common patterns from successful sequences
            common_patterns = self._extract_common_patterns(successful_sequences)
            
            if not common_patterns:
                return None
                
            # Create new strategy
            strategy_id = self._generate_strategy_id(common_patterns)
            
            strategy = Strategy(
                strategy_id=strategy_id,
                name=f"Generated Strategy {strategy_id[-8:]}",
                description=self._generate_strategy_description(common_patterns),
                components=common_patterns,
                state_conditions=self._extract_state_conditions(successful_sequences),
                action_weights=self._initialize_action_weights(successful_sequences),
                success_rate=0.0,
                confidence=self.confidence_threshold,
                usage_count=0
            )
            
            # Store new strategy
            self.strategies[strategy_id] = strategy
            self._store_new_strategy(strategy)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return None

    def evolve_strategies(self, game_id: str) -> None:
        """Evolve strategy population based on performance."""
        
        try:
            # Get all strategies for this game
            game_strategies = self._get_game_strategies(game_id)
            
            if not game_strategies:
                return
                
            # Remove poorly performing strategies
            self._remove_poor_strategies(game_strategies)
            
            # Combine successful strategies
            new_strategies = self._combine_successful_strategies(game_strategies)
            
            # Add variations of successful strategies
            variations = self._create_strategy_variations(game_strategies)
            
            # Add new strategies to population
            for strategy in new_strategies + variations:
                if strategy.strategy_id not in self.strategies:
                    self.strategies[strategy.strategy_id] = strategy
                    self._store_new_strategy(strategy)
                    
        except Exception as e:
            logger.error(f"Error evolving strategies: {e}")

    def _find_applicable_strategies(self, game_state: Dict[str, Any]) -> List[Strategy]:
        """Find strategies applicable to current game state."""
        
        applicable = []
        
        for strategy in self.strategies.values():
            if self._check_state_conditions(strategy.state_conditions, game_state):
                applicable.append(strategy)
                
        return applicable

    def _select_exploration_strategy(self, strategies: List[Strategy]) -> Strategy:
        """Select strategy for exploration."""
        
        # Weight strategies inversely by usage count
        weights = [1.0 / (s.usage_count + 1) for s in strategies]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return np.random.choice(strategies)
            
        probs = [w / total_weight for w in weights]
        return np.random.choice(strategies, p=probs)

    def _select_best_strategy(self,
                            strategies: List[Strategy],
                            game_state: Dict[str, Any]) -> Strategy:
        """Select best performing applicable strategy."""
        
        # Calculate scores considering success rate and confidence
        scores = [
            strategy.success_rate * strategy.confidence +
            self._calculate_state_match_bonus(strategy, game_state)
            for strategy in strategies
        ]
        
        return strategies[np.argmax(scores)]

    def _calculate_state_match_bonus(self,
                                   strategy: Strategy,
                                   game_state: Dict[str, Any]) -> float:
        """Calculate bonus score based on how well strategy matches game state."""
        
        match_score = 0.0
        total_conditions = len(strategy.state_conditions)
        
        if total_conditions == 0:
            return 0.0
            
        for condition, value in strategy.state_conditions.items():
            if condition in game_state and game_state[condition] == value:
                match_score += 1.0
                
        return 0.1 * (match_score / total_conditions)

    def _update_action_weights(self,
                             strategy: Strategy,
                             outcome: StrategyOutcome) -> None:
        """Update action weights based on outcome."""
        
        if not outcome.action_sequence:
            return
            
        # Calculate weight updates
        weight_changes = {}
        for action in outcome.action_sequence:
            if action in strategy.action_weights:
                if outcome.success:
                    # Increase weights for successful actions
                    weight_changes[action] = self.learning_rate
                else:
                    # Decrease weights for failed actions
                    weight_changes[action] = -self.learning_rate
                    
        # Apply updates
        for action, change in weight_changes.items():
            old_weight = strategy.action_weights[action]
            strategy.action_weights[action] = max(0.0, min(1.0, old_weight + change))

    def _calculate_strategy_confidence(self, strategy: Strategy) -> float:
        """Calculate confidence score for a strategy."""
        
        if strategy.usage_count == 0:
            return self.confidence_threshold
            
        # Consider factors:
        # 1. Usage count (more usage = more confidence)
        usage_factor = min(1.0, strategy.usage_count / 10.0)
        
        # 2. Success rate consistency
        consistency = 1.0 - np.std([
            strategy.success_rate for _ in range(strategy.usage_count)
        ])
        
        # 3. Action weight stability
        weight_stability = 1.0 - np.std(list(strategy.action_weights.values()))
        
        # Combine factors
        confidence = (0.4 * usage_factor +
                     0.4 * consistency +
                     0.2 * weight_stability)
                     
        return min(1.0, max(0.0, confidence))

    def _extract_common_patterns(self,
                               sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common patterns from successful sequences."""
        
        if not sequences:
            return {}
            
        patterns = {}
        
        # Extract action patterns
        action_sequences = [seq['actions'] for seq in sequences if 'actions' in seq]
        if action_sequences:
            patterns['action_patterns'] = self._find_action_patterns(action_sequences)
            
        # Extract state transition patterns
        state_changes = [seq['state_changes'] for seq in sequences if 'state_changes' in seq]
        if state_changes:
            patterns['state_patterns'] = self._find_state_patterns(state_changes)
            
        return patterns

    def _find_action_patterns(self, sequences: List[List[int]]) -> Dict[str, Any]:
        """Find patterns in action sequences."""
        
        patterns = {
            'common_subsequences': [],
            'action_frequencies': {},
            'action_transitions': {}
        }
        
        # Calculate action frequencies
        for sequence in sequences:
            for action in sequence:
                if action not in patterns['action_frequencies']:
                    patterns['action_frequencies'][action] = 0
                patterns['action_frequencies'][action] += 1
                
        # Calculate action transitions
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                action1, action2 = sequence[i], sequence[i + 1]
                key = f"{action1}->{action2}"
                if key not in patterns['action_transitions']:
                    patterns['action_transitions'][key] = 0
                patterns['action_transitions'][key] += 1
                
        # Find common subsequences
        patterns['common_subsequences'] = self._find_common_subsequences(sequences)
        
        return patterns

    def _find_state_patterns(self,
                            state_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find patterns in state changes."""
        
        patterns = {
            'common_changes': {},
            'change_sequences': [],
            'stable_properties': set()
        }
        
        # Find common state changes
        for changes in state_changes:
            for key, value in changes.items():
                if key not in patterns['common_changes']:
                    patterns['common_changes'][key] = []
                patterns['common_changes'][key].append(value)
                
        # Identify stable properties (those that don't change)
        all_properties = set().union(*[set(changes.keys()) for changes in state_changes])
        for prop in all_properties:
            if all(prop not in changes for changes in state_changes):
                patterns['stable_properties'].add(prop)
                
        return patterns

    def _find_common_subsequences(self,
                                sequences: List[List[int]],
                                min_length: int = 2) -> List[List[int]]:
        """Find common subsequences in action sequences."""
        
        if not sequences:
            return []
            
        common_sequences = []
        sequence_counts = {}
        
        # Generate all subsequences of minimum length
        for sequence in sequences:
            for i in range(len(sequence) - min_length + 1):
                for length in range(min_length, len(sequence) - i + 1):
                    subsequence = tuple(sequence[i:i + length])
                    if subsequence not in sequence_counts:
                        sequence_counts[subsequence] = 0
                    sequence_counts[subsequence] += 1
                    
        # Filter for common subsequences
        min_occurrences = len(sequences) // 2  # At least half the sequences
        common_sequences = [
            list(subsequence)
            for subsequence, count in sequence_counts.items()
            if count >= min_occurrences
        ]
        
        return common_sequences

    def _extract_state_conditions(self,
                                sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common state conditions from successful sequences."""
        
        conditions = {}
        
        # Extract initial states
        initial_states = [
            seq.get('initial_state', {})
            for seq in sequences
        ]
        
        if not initial_states:
            return conditions
            
        # Find common conditions across all sequences
        first_state = initial_states[0]
        for key, value in first_state.items():
            if all(state.get(key) == value for state in initial_states[1:]):
                conditions[key] = value
                
        return conditions

    def _initialize_action_weights(self,
                                 sequences: List[Dict[str, Any]]) -> Dict[int, float]:
        """Initialize action weights based on successful sequences."""
        
        weights = {}
        total_actions = 0
        
        # Count action occurrences
        for sequence in sequences:
            actions = sequence.get('actions', [])
            for action in actions:
                if action not in weights:
                    weights[action] = 0
                weights[action] += 1
                total_actions += 1
                
        # Normalize weights
        if total_actions > 0:
            for action in weights:
                weights[action] = weights[action] / total_actions
                
        return weights

    def _generate_strategy_id(self, patterns: Dict[str, Any]) -> str:
        """Generate unique ID for a strategy."""
        import hashlib
        pattern_str = json.dumps(patterns, sort_keys=True)
        hash_obj = hashlib.md5(pattern_str.encode())
        return f"strategy_{hash_obj.hexdigest()[:8]}"

    def _generate_strategy_description(self, patterns: Dict[str, Any]) -> str:
        """Generate human-readable description of strategy."""
        
        description_parts = []
        
        # Describe action patterns
        if 'action_patterns' in patterns:
            action_patterns = patterns['action_patterns']
            
            # Most common actions
            if 'action_frequencies' in action_patterns:
                top_actions = sorted(
                    action_patterns['action_frequencies'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                if top_actions:
                    description_parts.append(
                        "Frequently uses actions: " +
                        ", ".join(str(action) for action, _ in top_actions)
                    )
                    
            # Common sequences
            if 'common_subsequences' in action_patterns:
                if action_patterns['common_subsequences']:
                    description_parts.append(
                        "Uses action sequences: " +
                        ", ".join(
                            "->".join(str(a) for a in seq)
                            for seq in action_patterns['common_subsequences'][:2]
                        )
                    )
                    
        # Describe state patterns
        if 'state_patterns' in patterns:
            state_patterns = patterns['state_patterns']
            
            # Common state changes
            if 'common_changes' in state_patterns:
                common_changes = state_patterns['common_changes']
                if common_changes:
                    description_parts.append(
                        "Focuses on changing: " +
                        ", ".join(list(common_changes.keys())[:3])
                    )
                    
            # Stable properties
            if 'stable_properties' in state_patterns:
                stable_props = state_patterns['stable_properties']
                if stable_props:
                    description_parts.append(
                        "Maintains stable: " +
                        ", ".join(list(stable_props)[:3])
                    )
                    
        return " | ".join(description_parts) if description_parts else "Generated strategy"

    def _store_strategy_update(self, strategy: Strategy, outcome: StrategyOutcome) -> None:
        """Store strategy update in database."""
        
        if not self.db_connection:
            return
            
        try:
            cursor = self.db_connection.cursor()
            
            # Update strategy
            cursor.execute("""
                UPDATE strategies
                SET success_rate = ?, confidence = ?, usage_count = ?,
                    action_weights = ?, last_updated = ?
                WHERE strategy_id = ?
            """, (
                strategy.success_rate,
                strategy.confidence,
                strategy.usage_count,
                json.dumps(strategy.action_weights),
                datetime.now().isoformat(),
                strategy.strategy_id
            ))
            
            # Store outcome
            cursor.execute("""
                INSERT INTO strategy_outcomes
                (strategy_id, success, score_change, action_sequence,
                 state_changes, duration, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.strategy_id,
                outcome.success,
                outcome.score_change,
                json.dumps(outcome.action_sequence),
                json.dumps(outcome.state_changes),
                outcome.duration,
                outcome.timestamp.isoformat()
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing strategy update: {e}")

    def _store_new_strategy(self, strategy: Strategy) -> None:
        """Store new strategy in database."""
        
        if not self.db_connection:
            return
            
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT INTO strategies
                (strategy_id, name, description, components,
                 state_conditions, action_weights, success_rate,
                 confidence, usage_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy.strategy_id,
                strategy.name,
                strategy.description,
                json.dumps(strategy.components),
                json.dumps(strategy.state_conditions),
                json.dumps(strategy.action_weights),
                strategy.success_rate,
                strategy.confidence,
                strategy.usage_count,
                datetime.now().isoformat()
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing new strategy: {e}")

    def _get_game_strategies(self, game_id: str) -> List[Strategy]:
        """Get all strategies for a game."""
        return list(self.strategies.values())

    def _remove_poor_strategies(self, strategies: List[Strategy]) -> None:
        """Remove poorly performing strategies."""
        
        for strategy in strategies:
            if (strategy.usage_count >= 5 and
                strategy.success_rate < 0.3 and
                strategy.confidence < 0.4):
                    
                if strategy.strategy_id in self.strategies:
                    del self.strategies[strategy.strategy_id]
                    
                if self.db_connection:
                    try:
                        cursor = self.db_connection.cursor()
                        cursor.execute(
                            "DELETE FROM strategies WHERE strategy_id = ?",
                            (strategy.strategy_id,)
                        )
                        self.db_connection.commit()
                    except Exception as e:
                        logger.error(f"Error removing strategy: {e}")

    def _combine_successful_strategies(self, strategies: List[Strategy]) -> List[Strategy]:
        """Combine successful strategies to create new ones."""
        
        new_strategies = []
        successful = [
            s for s in strategies
            if s.success_rate >= 0.7 and s.confidence >= 0.6
        ]
        
        for i in range(len(successful)):
            for j in range(i + 1, len(successful)):
                combined = self._combine_strategies(successful[i], successful[j])
                if combined:
                    new_strategies.append(combined)
                    
        return new_strategies

    def _combine_strategies(self, strategy1: Strategy, strategy2: Strategy) -> Optional[Strategy]:
        """Combine two strategies into a new one."""
        
        try:
            # Combine components
            combined_components = {
                k: v for k, v in strategy1.components.items()
            }
            for k, v in strategy2.components.items():
                if k not in combined_components:
                    combined_components[k] = v
                else:
                    # Merge component data
                    if isinstance(combined_components[k], dict):
                        combined_components[k].update(v)
                        
            # Combine state conditions
            combined_conditions = {
                k: v for k, v in strategy1.state_conditions.items()
                if k in strategy2.state_conditions and
                   strategy2.state_conditions[k] == v
            }
            
            # Combine action weights
            combined_weights = {}
            all_actions = set(strategy1.action_weights.keys()) | set(strategy2.action_weights.keys())
            for action in all_actions:
                weight1 = strategy1.action_weights.get(action, 0.0)
                weight2 = strategy2.action_weights.get(action, 0.0)
                combined_weights[action] = (weight1 + weight2) / 2
                
            # Create combined strategy
            strategy_id = self._generate_strategy_id(combined_components)
            
            return Strategy(
                strategy_id=strategy_id,
                name=f"Combined Strategy {strategy_id[-8:]}",
                description=f"Combination of {strategy1.name} and {strategy2.name}",
                components=combined_components,
                state_conditions=combined_conditions,
                action_weights=combined_weights,
                success_rate=0.0,
                confidence=self.confidence_threshold,
                usage_count=0
            )
            
        except Exception as e:
            logger.error(f"Error combining strategies: {e}")
            return None

    def _create_strategy_variations(self, strategies: List[Strategy]) -> List[Strategy]:
        """Create variations of successful strategies."""
        
        variations = []
        successful = [
            s for s in strategies
            if s.success_rate >= 0.7 and s.confidence >= 0.6
        ]
        
        for strategy in successful:
            # Create variation with modified action weights
            weight_variation = self._create_weight_variation(strategy)
            if weight_variation:
                variations.append(weight_variation)
                
            # Create variation with relaxed state conditions
            condition_variation = self._create_condition_variation(strategy)
            if condition_variation:
                variations.append(condition_variation)
                
        return variations

    def _create_weight_variation(self, strategy: Strategy) -> Optional[Strategy]:
        """Create variation with modified action weights."""
        
        try:
            # Modify weights randomly within small range
            varied_weights = {}
            for action, weight in strategy.action_weights.items():
                variation = np.random.uniform(-0.1, 0.1)
                varied_weights[action] = max(0.0, min(1.0, weight + variation))
                
            # Create variation
            strategy_id = f"{strategy.strategy_id}_weight_var"
            
            return Strategy(
                strategy_id=strategy_id,
                name=f"Weight Variation {strategy_id[-8:]}",
                description=f"Weight variation of {strategy.name}",
                components=strategy.components.copy(),
                state_conditions=strategy.state_conditions.copy(),
                action_weights=varied_weights,
                success_rate=0.0,
                confidence=self.confidence_threshold,
                usage_count=0
            )
            
        except Exception as e:
            logger.error(f"Error creating weight variation: {e}")
            return None

    def _create_condition_variation(self, strategy: Strategy) -> Optional[Strategy]:
        """Create variation with relaxed state conditions."""
        
        try:
            # Remove some conditions randomly
            varied_conditions = strategy.state_conditions.copy()
            if varied_conditions:
                num_to_remove = max(1, len(varied_conditions) // 3)
                remove_keys = np.random.choice(
                    list(varied_conditions.keys()),
                    size=num_to_remove,
                    replace=False
                )
                for key in remove_keys:
                    del varied_conditions[key]
                    
            # Create variation
            strategy_id = f"{strategy.strategy_id}_cond_var"
            
            return Strategy(
                strategy_id=strategy_id,
                name=f"Condition Variation {strategy_id[-8:]}",
                description=f"Condition variation of {strategy.name}",
                components=strategy.components.copy(),
                state_conditions=varied_conditions,
                action_weights=strategy.action_weights.copy(),
                success_rate=0.0,
                confidence=self.confidence_threshold,
                usage_count=0
            )
            
        except Exception as e:
            logger.error(f"Error creating condition variation: {e}")
            return None

# Create singleton instance
_strategic_learning_system = None

def get_strategic_learning_system(db_connection: Optional[sqlite3.Connection] = None) -> StrategicLearningSystem:
    """Get or create the strategic learning system singleton."""
    global _strategic_learning_system
    if _strategic_learning_system is None:
        _strategic_learning_system = StrategicLearningSystem(db_connection)
    return _strategic_learning_system
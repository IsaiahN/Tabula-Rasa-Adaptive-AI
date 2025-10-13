#!/usr/bin/env python3
"""
Game Lifecycle Intelligence System

This system analyzes game over patterns, tracks action effectiveness over time,
and provides proactive recommendations to prevent action oscillation and improve
strategy switching before historical failure points.
"""

import json
import logging
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import asyncio
import sqlite3
import os

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of game failure modes."""
    TIMEOUT = "timeout"
    STUCK_LOOP = "stuck_loop"
    ACTION_LIMIT = "action_limit"
    SCORE_STAGNATION = "score_stagnation"
    OSCILLATION = "oscillation"
    UNKNOWN = "unknown"


class StrategyType(Enum):
    """Available strategy types for switching."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    HYBRID = "hybrid"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class GameOverPattern:
    """Represents a pattern detected in game over scenarios."""
    game_type: str
    avg_actions_to_failure: float
    failure_mode: FailureMode
    failure_action_window: List[int]  # Last N actions before failure
    score_trajectory: List[float]  # Score progression before failure
    common_oscillation_patterns: List[Tuple[int, int]]  # (action1, action2) oscillations
    success_threshold_actions: Optional[int] = None  # Actions needed for success
    confidence: float = 0.0


@dataclass
class ActionEffectivenessWindow:
    """Tracks action effectiveness in temporal windows."""
    action_type: int
    game_context_hash: str
    early_game_effectiveness: float  # Actions 1-33% of game
    mid_game_effectiveness: float    # Actions 34-66% of game
    late_game_effectiveness: float   # Actions 67-100% of game
    optimal_timing_start: int
    optimal_timing_end: int
    avoid_after_action_count: Optional[int] = None


class OscillationDetector:
    """Detects action oscillation patterns that lead to failure."""

    def __init__(self, window_size: int = 10, oscillation_threshold: int = 3):
        self.window_size = window_size
        self.oscillation_threshold = oscillation_threshold

    def detect_oscillations(self, action_sequence: List[int]) -> List[Dict[str, Any]]:
        """Detect oscillation patterns in action sequence."""
        oscillations = []

        if len(action_sequence) < self.window_size:
            return oscillations

        # Look for back-and-forth patterns
        for i in range(len(action_sequence) - self.window_size + 1):
            window = action_sequence[i:i + self.window_size]

            # Count consecutive pairs
            pair_counts = defaultdict(int)
            for j in range(len(window) - 1):
                pair = tuple(sorted([window[j], window[j+1]]))
                pair_counts[pair] += 1

            # Find oscillating pairs
            for pair, count in pair_counts.items():
                if count >= self.oscillation_threshold:
                    oscillations.append({
                        'actions': pair,
                        'start_position': i,
                        'frequency': count,
                        'window_size': self.window_size,
                        'severity': min(count / self.window_size, 1.0)
                    })

        return oscillations


class GameLifecycleAnalyzer:
    """Core system for analyzing game lifecycle patterns and preventing failures."""

    def __init__(self, db_path: str = "data/game_lifecycle.db"):
        self.db_path = db_path
        self.game_over_patterns = {}  # game_type -> GameOverPattern
        self.action_effectiveness = {}  # (action, context) -> ActionEffectivenessWindow
        self.oscillation_detector = OscillationDetector()
        self.strategy_transition_points = defaultdict(list)
        self.recent_games = deque(maxlen=1000)  # Keep last 1000 games for analysis

        # Risk assessment parameters
        self.failure_risk_threshold = 0.7  # Switch strategy at 70% of avg failure point
        self.oscillation_risk_weight = 0.3
        self.efficiency_decay_weight = 0.4
        self.historical_failure_weight = 0.3

        self._init_database()
        self._load_existing_patterns()

    def _init_database(self):
        """Initialize SQLite database for storing lifecycle patterns."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS game_lifecycle_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_type TEXT NOT NULL,
                        avg_actions_to_failure REAL,
                        avg_actions_to_success REAL,
                        failure_mode TEXT,
                        failure_action_window TEXT,  -- JSON array
                        score_trajectory TEXT,       -- JSON array
                        oscillation_patterns TEXT,   -- JSON array
                        success_threshold_actions INTEGER,
                        confidence REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.execute('''
                    CREATE TABLE IF NOT EXISTS action_effectiveness_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action_type INTEGER NOT NULL,
                        game_context_hash TEXT NOT NULL,
                        early_game_effectiveness REAL,
                        mid_game_effectiveness REAL,
                        late_game_effectiveness REAL,
                        optimal_timing_start INTEGER,
                        optimal_timing_end INTEGER,
                        avoid_after_action_count INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(action_type, game_context_hash)
                    )
                ''')

                conn.execute('''
                    CREATE TABLE IF NOT EXISTS game_results_lifecycle (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id TEXT NOT NULL,
                        game_type TEXT,
                        total_actions INTEGER,
                        final_score REAL,
                        game_won BOOLEAN,
                        failure_mode TEXT,
                        action_sequence TEXT,        -- JSON array
                        score_history TEXT,          -- JSON array
                        oscillations_detected TEXT,  -- JSON array
                        strategy_switches TEXT,      -- JSON array
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to initialize lifecycle database: {e}")

    def _load_existing_patterns(self):
        """Load existing patterns from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load game over patterns
                cursor = conn.execute('SELECT * FROM game_lifecycle_patterns')
                for row in cursor.fetchall():
                    game_type = row[1]
                    pattern = GameOverPattern(
                        game_type=game_type,
                        avg_actions_to_failure=row[2] or 0.0,
                        failure_mode=FailureMode(row[4]) if row[4] else FailureMode.UNKNOWN,
                        failure_action_window=json.loads(row[5]) if row[5] else [],
                        score_trajectory=json.loads(row[6]) if row[6] else [],
                        common_oscillation_patterns=json.loads(row[7]) if row[7] else [],
                        success_threshold_actions=row[8],
                        confidence=row[9] or 0.0
                    )
                    self.game_over_patterns[game_type] = pattern

                # Load action effectiveness patterns
                cursor = conn.execute('SELECT * FROM action_effectiveness_patterns')
                for row in cursor.fetchall():
                    key = (row[1], row[2])  # (action_type, game_context_hash)
                    effectiveness = ActionEffectivenessWindow(
                        action_type=row[1],
                        game_context_hash=row[2],
                        early_game_effectiveness=row[3] or 0.0,
                        mid_game_effectiveness=row[4] or 0.0,
                        late_game_effectiveness=row[5] or 0.0,
                        optimal_timing_start=row[6] or 0,
                        optimal_timing_end=row[7] or 100,
                        avoid_after_action_count=row[8]
                    )
                    self.action_effectiveness[key] = effectiveness

                logger.info(f"Loaded {len(self.game_over_patterns)} game over patterns and "
                           f"{len(self.action_effectiveness)} action effectiveness patterns")

        except Exception as e:
            logger.warning(f"Failed to load existing patterns: {e}")

    async def add_game_over_pattern(self, game_data: Dict[str, Any]):
        """Add a new game over pattern to the analysis."""
        try:
            game_id = game_data.get('game_id', 'unknown')
            game_type = game_data.get('game_type', 'default')
            total_actions = game_data.get('actions_to_failure', 0)
            final_score = game_data.get('final_score', 0.0)
            game_won = game_data.get('game_won', False)
            action_sequence = game_data.get('action_sequence_before_end', [])
            score_history = game_data.get('efficiency_trajectory', [])

            # Determine failure mode
            failure_mode = self._classify_failure_mode(game_data)

            # Detect oscillations in the action sequence
            oscillations = self.oscillation_detector.detect_oscillations(action_sequence)

            # Store raw game data
            self._store_game_result(game_id, game_type, total_actions, final_score,
                                  game_won, failure_mode, action_sequence, score_history, oscillations)

            # Update patterns
            await self._update_game_over_patterns(game_type, total_actions, final_score,
                                                game_won, failure_mode, action_sequence[-10:],
                                                score_history, oscillations)

            # Update action effectiveness
            await self._update_action_effectiveness(action_sequence, game_type, total_actions,
                                                  final_score, game_won)

            # Add to recent games for trend analysis
            self.recent_games.append(game_data)

            logger.info(f"Added game over pattern: {game_type}, {total_actions} actions, "
                       f"won={game_won}, failure_mode={failure_mode.value}")

        except Exception as e:
            logger.error(f"Failed to add game over pattern: {e}")

    def _classify_failure_mode(self, game_data: Dict[str, Any]) -> FailureMode:
        """Classify the failure mode based on game data."""
        failure_reason = game_data.get('failure_reason', '').lower()

        if 'timeout' in failure_reason or 'time' in failure_reason:
            return FailureMode.TIMEOUT
        elif 'stuck' in failure_reason or 'loop' in failure_reason:
            return FailureMode.STUCK_LOOP
        elif 'action' in failure_reason and 'limit' in failure_reason:
            return FailureMode.ACTION_LIMIT
        elif 'stagnation' in failure_reason or 'no progress' in failure_reason:
            return FailureMode.SCORE_STAGNATION
        else:
            # Analyze action sequence for oscillations
            action_sequence = game_data.get('action_sequence_before_end', [])
            oscillations = self.oscillation_detector.detect_oscillations(action_sequence)
            if oscillations and len(oscillations) > 0:
                return FailureMode.OSCILLATION

        return FailureMode.UNKNOWN

    def _store_game_result(self, game_id: str, game_type: str, total_actions: int,
                          final_score: float, game_won: bool, failure_mode: FailureMode,
                          action_sequence: List[int], score_history: List[float],
                          oscillations: List[Dict]):
        """Store game result in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO game_results_lifecycle
                    (game_id, game_type, total_actions, final_score, game_won, failure_mode,
                     action_sequence, score_history, oscillations_detected, strategy_switches)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    game_id, game_type, total_actions, final_score, game_won,
                    failure_mode.value, json.dumps(action_sequence), json.dumps(score_history),
                    json.dumps(oscillations), json.dumps([])  # strategy_switches placeholder
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store game result: {e}")

    async def _update_game_over_patterns(self, game_type: str, total_actions: int,
                                       final_score: float, game_won: bool,
                                       failure_mode: FailureMode, last_actions: List[int],
                                       score_trajectory: List[float], oscillations: List[Dict]):
        """Update game over patterns for the game type."""
        try:
            if game_type not in self.game_over_patterns:
                # Create new pattern
                self.game_over_patterns[game_type] = GameOverPattern(
                    game_type=game_type,
                    avg_actions_to_failure=total_actions if not game_won else 0.0,
                    failure_mode=failure_mode,
                    failure_action_window=last_actions,
                    score_trajectory=score_trajectory[-20:] if score_trajectory else [],
                    common_oscillation_patterns=[(osc['actions'][0], osc['actions'][1]) for osc in oscillations],
                    success_threshold_actions=total_actions if game_won else None,
                    confidence=1.0
                )
            else:
                # Update existing pattern
                pattern = self.game_over_patterns[game_type]

                # Update averages with exponential smoothing (alpha=0.2)
                alpha = 0.2
                if not game_won:
                    if pattern.avg_actions_to_failure > 0:
                        pattern.avg_actions_to_failure = (alpha * total_actions +
                                                        (1 - alpha) * pattern.avg_actions_to_failure)
                    else:
                        pattern.avg_actions_to_failure = total_actions

                # Update success threshold if this was a win
                if game_won:
                    if pattern.success_threshold_actions is None:
                        pattern.success_threshold_actions = total_actions
                    else:
                        pattern.success_threshold_actions = (alpha * total_actions +
                                                           (1 - alpha) * pattern.success_threshold_actions)

                # Update failure action window
                pattern.failure_action_window = last_actions
                pattern.score_trajectory = score_trajectory[-20:] if score_trajectory else []

                # Update oscillation patterns
                new_oscillation_pairs = [(osc['actions'][0], osc['actions'][1]) for osc in oscillations]
                pattern.common_oscillation_patterns.extend(new_oscillation_pairs)
                # Keep only unique oscillation patterns, limited to 20
                pattern.common_oscillation_patterns = list(set(pattern.common_oscillation_patterns))[:20]

                # Update confidence
                pattern.confidence = min(pattern.confidence + 0.1, 1.0)

            # Save to database
            await self._save_pattern_to_db(game_type, self.game_over_patterns[game_type])

        except Exception as e:
            logger.error(f"Failed to update game over patterns: {e}")

    async def _update_action_effectiveness(self, action_sequence: List[int], game_type: str,
                                         total_actions: int, final_score: float, game_won: bool):
        """Update action effectiveness based on temporal windows."""
        try:
            if not action_sequence or total_actions == 0:
                return

            game_context_hash = f"{game_type}_{int(final_score/10)*10}"  # Group by score ranges

            # Calculate effectiveness in three temporal windows
            early_end = total_actions // 3
            mid_end = (total_actions * 2) // 3

            early_actions = action_sequence[:early_end]
            mid_actions = action_sequence[early_end:mid_end]
            late_actions = action_sequence[mid_end:]

            # Calculate effectiveness scores for each action type
            unique_actions = set(action_sequence)

            for action_type in unique_actions:
                key = (action_type, game_context_hash)

                early_effectiveness = self._calculate_window_effectiveness(early_actions, action_type, game_won, final_score)
                mid_effectiveness = self._calculate_window_effectiveness(mid_actions, action_type, game_won, final_score)
                late_effectiveness = self._calculate_window_effectiveness(late_actions, action_type, game_won, final_score)

                if key not in self.action_effectiveness:
                    self.action_effectiveness[key] = ActionEffectivenessWindow(
                        action_type=action_type,
                        game_context_hash=game_context_hash,
                        early_game_effectiveness=early_effectiveness,
                        mid_game_effectiveness=mid_effectiveness,
                        late_game_effectiveness=late_effectiveness,
                        optimal_timing_start=0,
                        optimal_timing_end=total_actions
                    )
                else:
                    # Update with exponential smoothing
                    alpha = 0.2
                    effectiveness = self.action_effectiveness[key]
                    effectiveness.early_game_effectiveness = (alpha * early_effectiveness +
                                                            (1 - alpha) * effectiveness.early_game_effectiveness)
                    effectiveness.mid_game_effectiveness = (alpha * mid_effectiveness +
                                                          (1 - alpha) * effectiveness.mid_game_effectiveness)
                    effectiveness.late_game_effectiveness = (alpha * late_effectiveness +
                                                           (1 - alpha) * effectiveness.late_game_effectiveness)

                # Determine optimal timing and avoidance thresholds
                effectiveness = self.action_effectiveness[key]
                max_effectiveness = max(effectiveness.early_game_effectiveness,
                                      effectiveness.mid_game_effectiveness,
                                      effectiveness.late_game_effectiveness)

                if max_effectiveness == effectiveness.early_game_effectiveness:
                    effectiveness.optimal_timing_start = 0
                    effectiveness.optimal_timing_end = early_end
                elif max_effectiveness == effectiveness.mid_game_effectiveness:
                    effectiveness.optimal_timing_start = early_end
                    effectiveness.optimal_timing_end = mid_end
                else:
                    effectiveness.optimal_timing_start = mid_end
                    effectiveness.optimal_timing_end = total_actions

                # Set avoidance threshold if late game effectiveness is very poor
                if effectiveness.late_game_effectiveness < 0.3 and not game_won:
                    effectiveness.avoid_after_action_count = mid_end

                # Save to database
                await self._save_effectiveness_to_db(effectiveness)

        except Exception as e:
            logger.error(f"Failed to update action effectiveness: {e}")

    def _calculate_window_effectiveness(self, actions: List[int], action_type: int,
                                      game_won: bool, final_score: float) -> float:
        """Calculate effectiveness score for an action in a temporal window."""
        if not actions or action_type not in actions:
            return 0.5  # Neutral score

        action_frequency = actions.count(action_type) / len(actions)

        # Base effectiveness score
        if game_won:
            base_score = 0.8 + (final_score / 1000)  # Higher score = better effectiveness
        else:
            base_score = 0.4 + (final_score / 2000)  # Lower effectiveness for failed games

        # Adjust for frequency (too high frequency might indicate oscillation)
        if action_frequency > 0.5:
            base_score *= 0.8  # Penalize over-use
        elif action_frequency < 0.1:
            base_score *= 0.9  # Slightly penalize under-use

        return max(0.0, min(1.0, base_score))

    async def _save_pattern_to_db(self, game_type: str, pattern: GameOverPattern):
        """Save game over pattern to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO game_lifecycle_patterns
                    (game_type, avg_actions_to_failure, avg_actions_to_success, failure_mode,
                     failure_action_window, score_trajectory, oscillation_patterns,
                     success_threshold_actions, confidence, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    game_type,
                    pattern.avg_actions_to_failure,
                    pattern.success_threshold_actions,
                    pattern.failure_mode.value,
                    json.dumps(pattern.failure_action_window),
                    json.dumps(pattern.score_trajectory),
                    json.dumps(pattern.common_oscillation_patterns),
                    pattern.success_threshold_actions,
                    pattern.confidence
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save pattern to database: {e}")

    async def _save_effectiveness_to_db(self, effectiveness: ActionEffectivenessWindow):
        """Save action effectiveness to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO action_effectiveness_patterns
                    (action_type, game_context_hash, early_game_effectiveness, mid_game_effectiveness,
                     late_game_effectiveness, optimal_timing_start, optimal_timing_end,
                     avoid_after_action_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    effectiveness.action_type,
                    effectiveness.game_context_hash,
                    effectiveness.early_game_effectiveness,
                    effectiveness.mid_game_effectiveness,
                    effectiveness.late_game_effectiveness,
                    effectiveness.optimal_timing_start,
                    effectiveness.optimal_timing_end,
                    effectiveness.avoid_after_action_count
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save effectiveness to database: {e}")

    def get_failure_risk(self, game_type: str, current_action_count: int,
                        recent_actions: List[int] = None) -> float:
        """Calculate failure risk based on current game state."""
        try:
            if game_type not in self.game_over_patterns:
                return 0.3  # Default moderate risk for unknown game types

            pattern = self.game_over_patterns[game_type]

            # Risk based on historical failure point
            historical_risk = 0.0
            if pattern.avg_actions_to_failure > 0:
                historical_risk = current_action_count / pattern.avg_actions_to_failure
                historical_risk = min(historical_risk, 1.0)

            # Risk based on oscillation detection
            oscillation_risk = 0.0
            if recent_actions:
                current_oscillations = self.oscillation_detector.detect_oscillations(recent_actions)
                if current_oscillations:
                    # Check if current oscillations match known failure patterns
                    for osc in current_oscillations:
                        osc_pair = tuple(sorted(osc['actions']))
                        if osc_pair in pattern.common_oscillation_patterns:
                            oscillation_risk = max(oscillation_risk, osc['severity'])

            # Combined risk score
            total_risk = (self.historical_failure_weight * historical_risk +
                         self.oscillation_risk_weight * oscillation_risk)

            return min(total_risk, 1.0)

        except Exception as e:
            logger.error(f"Failed to calculate failure risk: {e}")
            return 0.5  # Default moderate risk on error

    def get_alternative_strategy(self, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get alternative strategy recommendation when failure risk is high."""
        try:
            game_type = game_context.get('game_type', 'default')
            current_actions = game_context.get('current_action_count', 0)
            recent_score_change = game_context.get('recent_score_change', 0.0)

            # Analyze current situation
            if recent_score_change > 0:
                # Making progress - suggest conservative approach
                recommended_strategy = StrategyType.CONSERVATIVE
                strategy_params = {
                    'exploration_rate': 0.2,
                    'risk_tolerance': 0.3,
                    'action_diversity': 0.4
                }
            elif recent_score_change == 0:
                # Stagnant - suggest exploration
                recommended_strategy = StrategyType.EXPLORATION
                strategy_params = {
                    'exploration_rate': 0.8,
                    'risk_tolerance': 0.7,
                    'action_diversity': 0.9
                }
            else:
                # Losing score - suggest hybrid approach
                recommended_strategy = StrategyType.HYBRID
                strategy_params = {
                    'exploration_rate': 0.5,
                    'risk_tolerance': 0.5,
                    'action_diversity': 0.6
                }

            return {
                'strategy_type': recommended_strategy.value,
                'parameters': strategy_params,
                'reason': f'Risk mitigation at {current_actions} actions',
                'confidence': 0.8
            }

        except Exception as e:
            logger.error(f"Failed to get alternative strategy: {e}")
            return {
                'strategy_type': StrategyType.EXPLORATION.value,
                'parameters': {'exploration_rate': 0.6},
                'reason': 'Fallback strategy',
                'confidence': 0.5
            }

    def should_avoid_action(self, action_type: int, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if an action should be avoided based on current context."""
        try:
            game_type = game_context.get('game_type', 'default')
            current_actions = game_context.get('current_action_count', 0)
            final_score = game_context.get('current_score', 0)

            game_context_hash = f"{game_type}_{int(final_score/10)*10}"
            key = (action_type, game_context_hash)

            if key not in self.action_effectiveness:
                return {
                    'should_avoid': False,
                    'reason': 'No historical data for this action/context',
                    'confidence': 0.1
                }

            effectiveness = self.action_effectiveness[key]

            # Check if we're past the avoidance threshold
            if (effectiveness.avoid_after_action_count is not None and
                current_actions >= effectiveness.avoid_after_action_count):
                return {
                    'should_avoid': True,
                    'reason': f'Action {action_type} historically ineffective after {effectiveness.avoid_after_action_count} actions',
                    'confidence': 0.8,
                    'alternative_actions': self._suggest_alternative_actions(game_context)
                }

            # Check if we're outside the optimal timing window
            if (current_actions < effectiveness.optimal_timing_start or
                current_actions > effectiveness.optimal_timing_end):
                return {
                    'should_avoid': True,
                    'reason': f'Action {action_type} most effective between actions {effectiveness.optimal_timing_start}-{effectiveness.optimal_timing_end}',
                    'confidence': 0.6,
                    'alternative_actions': self._suggest_alternative_actions(game_context)
                }

            return {
                'should_avoid': False,
                'reason': 'Action is within effective timing window',
                'confidence': 0.7
            }

        except Exception as e:
            logger.error(f"Failed to check action avoidance: {e}")
            return {
                'should_avoid': False,
                'reason': 'Error in analysis',
                'confidence': 0.1
            }

    def _suggest_alternative_actions(self, game_context: Dict[str, Any]) -> List[int]:
        """Suggest alternative actions based on current context."""
        try:
            game_type = game_context.get('game_type', 'default')
            current_actions = game_context.get('current_action_count', 0)
            final_score = game_context.get('current_score', 0)

            game_context_hash = f"{game_type}_{int(final_score/10)*10}"

            # Find actions that are effective in current timing window
            effective_actions = []

            for (action_type, context_hash), effectiveness in self.action_effectiveness.items():
                if context_hash == game_context_hash:
                    # Check if action is in its optimal timing window
                    if (effectiveness.optimal_timing_start <= current_actions <= effectiveness.optimal_timing_end):
                        # Calculate current effectiveness score
                        if current_actions <= effectiveness.optimal_timing_end // 3:
                            score = effectiveness.early_game_effectiveness
                        elif current_actions <= (effectiveness.optimal_timing_end * 2) // 3:
                            score = effectiveness.mid_game_effectiveness
                        else:
                            score = effectiveness.late_game_effectiveness

                        if score > 0.6:  # Only suggest actions with good effectiveness
                            effective_actions.append((action_type, score))

            # Sort by effectiveness and return top 3
            effective_actions.sort(key=lambda x: x[1], reverse=True)
            return [action for action, score in effective_actions[:3]]

        except Exception as e:
            logger.error(f"Failed to suggest alternative actions: {e}")
            return [1, 2, 3]  # Default fallback actions

    def get_lifecycle_insights(self, game_type: str = None) -> Dict[str, Any]:
        """Get comprehensive lifecycle insights for analysis."""
        try:
            insights = {
                'timestamp': datetime.now().isoformat(),
                'total_patterns_tracked': len(self.game_over_patterns),
                'total_effectiveness_patterns': len(self.action_effectiveness),
                'recent_games_analyzed': len(self.recent_games)
            }

            if game_type and game_type in self.game_over_patterns:
                pattern = self.game_over_patterns[game_type]
                insights['game_type_analysis'] = {
                    'avg_actions_to_failure': pattern.avg_actions_to_failure,
                    'success_threshold': pattern.success_threshold_actions,
                    'primary_failure_mode': pattern.failure_mode.value,
                    'common_oscillations': pattern.common_oscillation_patterns,
                    'confidence': pattern.confidence
                }

            # General insights across all game types
            if self.game_over_patterns:
                all_failure_actions = [p.avg_actions_to_failure for p in self.game_over_patterns.values()
                                     if p.avg_actions_to_failure > 0]
                if all_failure_actions:
                    insights['overall_patterns'] = {
                        'avg_failure_actions_across_types': np.mean(all_failure_actions),
                        'failure_action_std': np.std(all_failure_actions),
                        'most_common_failure_modes': self._get_common_failure_modes()
                    }

            return insights

        except Exception as e:
            logger.error(f"Failed to get lifecycle insights: {e}")
            return {'error': str(e)}

    def _get_common_failure_modes(self) -> Dict[str, int]:
        """Get frequency count of different failure modes."""
        failure_mode_counts = defaultdict(int)
        for pattern in self.game_over_patterns.values():
            failure_mode_counts[pattern.failure_mode.value] += 1
        return dict(failure_mode_counts)


# Export the main classes
__all__ = ['GameLifecycleAnalyzer', 'FailureMode', 'StrategyType', 'GameOverPattern', 'ActionEffectivenessWindow']
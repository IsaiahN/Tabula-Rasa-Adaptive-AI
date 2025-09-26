#!/usr/bin/env python3
"""
META-COGNITIVE STRATEGY DETECTOR

This system automatically detects when the current strategy is mismatched
to the game type and triggers appropriate intelligence escalation.

Key Detection Patterns:
1. Action-Response Disconnect: Actions succeed but produce no progress
2. Game Complexity Analysis: Visual pattern density indicates puzzle vs action game
3. Performance Degradation: High effort, zero effectiveness over time
4. Domain Classification: ARC pattern recognition vs mechanical gameplay
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GameType(Enum):
    PUZZLE_SPATIAL = "puzzle_spatial"  # Requires pattern analysis (ARC games)
    ACTION_REFLEX = "action_reflex"    # Requires fast mechanical actions
    STRATEGY_PLANNING = "strategy_planning"  # Requires multi-step planning
    UNKNOWN = "unknown"

class StrategyType(Enum):
    MECHANICAL_SEQUENTIAL = "mechanical_sequential"  # Just cycle through actions
    PATTERN_ANALYTICAL = "pattern_analytical"       # Analyze visual patterns
    EXPLORATION_BASED = "exploration_based"         # Intelligent exploration
    HYBRID_INTELLIGENT = "hybrid_intelligent"       # Full AI capabilities

@dataclass
class StrategyMismatchSignal:
    signal_type: str
    confidence: float
    evidence: Dict[str, Any]
    recommendation: str

class MetaCognitiveStrategyDetector:
    """
    Automatically detects when current strategy is mismatched to game requirements
    and triggers appropriate intelligence escalation.
    """

    def __init__(self):
        self.action_history = []
        self.effectiveness_history = []
        self.game_analysis_cache = {}
        self.strategy_switches = []

        # Thresholds for detection
        self.MIN_ACTIONS_FOR_ANALYSIS = 10
        self.ZERO_PROGRESS_THRESHOLD = 0.05  # Less than 5% effectiveness
        self.PATTERN_COMPLEXITY_THRESHOLD = 0.7

    def analyze_strategy_game_mismatch(
        self,
        action_history: List[Dict],
        game_state: Dict[str, Any],
        current_strategy: StrategyType
    ) -> List[StrategyMismatchSignal]:
        """
        Main analysis function that detects strategy-game mismatches.

        This is what the Governor would call automatically during training.
        """
        signals = []

        # Signal 1: Action-Response Disconnect Analysis
        disconnect_signal = self._detect_action_response_disconnect(action_history)
        if disconnect_signal:
            signals.append(disconnect_signal)

        # Signal 2: Game Complexity vs Strategy Analysis
        complexity_signal = self._analyze_game_complexity_mismatch(game_state, current_strategy)
        if complexity_signal:
            signals.append(complexity_signal)

        # Signal 3: Performance Degradation Pattern
        degradation_signal = self._detect_performance_degradation(action_history)
        if degradation_signal:
            signals.append(degradation_signal)

        # Signal 4: Domain Classification Mismatch
        domain_signal = self._detect_domain_classification_mismatch(game_state, current_strategy)
        if domain_signal:
            signals.append(domain_signal)

        return signals

    def _detect_action_response_disconnect(self, action_history: List[Dict]) -> Optional[StrategyMismatchSignal]:
        """
        Detect when actions execute successfully but produce no meaningful progress.

        Key Pattern: API works fine, actions are valid, but score stays 0
        This indicates "the system works but the approach is wrong"
        """
        if len(action_history) < self.MIN_ACTIONS_FOR_ANALYSIS:
            return None

        recent_actions = action_history[-self.MIN_ACTIONS_FOR_ANALYSIS:]

        # Check for "successful actions with zero progress" pattern
        api_success_rate = sum(1 for a in recent_actions if not a.get('failed', False)) / len(recent_actions)
        score_progress = sum(a.get('score_change', 0) for a in recent_actions)
        effectiveness_rate = sum(1 for a in recent_actions if a.get('success', False)) / len(recent_actions)

        # CRITICAL PATTERN: High API success, zero game progress
        if (api_success_rate > 0.8 and  # Actions execute fine
            abs(score_progress) < 1.0 and  # No score progress
            effectiveness_rate < self.ZERO_PROGRESS_THRESHOLD):  # No effectiveness

            return StrategyMismatchSignal(
                signal_type="action_response_disconnect",
                confidence=0.9,
                evidence={
                    "api_success_rate": api_success_rate,
                    "score_progress": score_progress,
                    "effectiveness_rate": effectiveness_rate,
                    "pattern": "mechanical_actions_ineffective"
                },
                recommendation="ESCALATE: Switch from mechanical to pattern-analytical strategy"
            )

        return None

    def _analyze_game_complexity_mismatch(
        self,
        game_state: Dict[str, Any],
        current_strategy: StrategyType
    ) -> Optional[StrategyMismatchSignal]:
        """
        Analyze if game complexity requires intelligence that current strategy lacks.

        Key Analysis: Visual pattern density, spatial relationships, game ID patterns
        """
        frame_data = game_state.get('frame', [])
        game_id = game_state.get('game_id', '')

        # Analyze visual complexity
        complexity_score = self._calculate_visual_complexity(frame_data)

        # ARC game detection patterns
        is_arc_game = self._detect_arc_game_patterns(game_id, frame_data)

        # Strategy adequacy analysis
        strategy_intelligence_level = self._get_strategy_intelligence_level(current_strategy)
        required_intelligence_level = self._estimate_required_intelligence(complexity_score, is_arc_game)

        intelligence_gap = required_intelligence_level - strategy_intelligence_level

        if intelligence_gap > 0.3:  # Significant intelligence gap
            return StrategyMismatchSignal(
                signal_type="game_complexity_mismatch",
                confidence=min(0.95, 0.6 + intelligence_gap),
                evidence={
                    "visual_complexity": complexity_score,
                    "is_arc_game": is_arc_game,
                    "current_strategy_level": strategy_intelligence_level,
                    "required_level": required_intelligence_level,
                    "intelligence_gap": intelligence_gap
                },
                recommendation=f"ESCALATE: Increase intelligence level by {intelligence_gap:.1f}"
            )

        return None

    def _calculate_visual_complexity(self, frame_data: List) -> float:
        """
        Calculate complexity score of visual frame data.

        High complexity = patterns, spatial relationships, multiple colors
        Low complexity = simple grids, few elements
        """
        try:
            if not frame_data:
                return 0.0

            # Convert to numpy for analysis
            if isinstance(frame_data[0], list):
                arr = np.array(frame_data)
            else:
                return 0.2  # Simple 1D data

            if arr.ndim != 2:
                return 0.1

            # Complexity indicators
            unique_values = len(np.unique(arr))
            height, width = arr.shape
            total_elements = height * width

            # Pattern density
            pattern_score = unique_values / max(1, total_elements) * 10

            # Spatial relationship complexity
            edge_changes = 0
            for i in range(height-1):
                for j in range(width-1):
                    if arr[i,j] != arr[i+1,j] or arr[i,j] != arr[i,j+1]:
                        edge_changes += 1

            spatial_score = edge_changes / max(1, total_elements) * 5

            # Size complexity
            size_score = min(1.0, (height * width) / 100)

            total_complexity = min(1.0, (pattern_score + spatial_score + size_score) / 3)
            return total_complexity

        except Exception as e:
            logger.warning(f"Error calculating visual complexity: {e}")
            return 0.5  # Default moderate complexity

    def _detect_arc_game_patterns(self, game_id: str, frame_data: List) -> bool:
        """
        Detect ARC-specific game patterns that indicate puzzle-solving requirements.
        """
        # ARC game ID patterns
        arc_id_patterns = ['lp', 'vc', 'tr', 'ts']  # Common ARC prefixes
        has_arc_id = any(game_id.startswith(pattern) for pattern in arc_id_patterns)

        # ARC visual patterns (grids with patterns)
        has_grid_structure = (isinstance(frame_data, list) and
                            len(frame_data) > 0 and
                            isinstance(frame_data[0], list))

        return has_arc_id or has_grid_structure

    def _get_strategy_intelligence_level(self, strategy: StrategyType) -> float:
        """Map strategy types to intelligence levels."""
        intelligence_levels = {
            StrategyType.MECHANICAL_SEQUENTIAL: 0.1,
            StrategyType.EXPLORATION_BASED: 0.5,
            StrategyType.PATTERN_ANALYTICAL: 0.8,
            StrategyType.HYBRID_INTELLIGENT: 1.0
        }
        return intelligence_levels.get(strategy, 0.3)

    def _estimate_required_intelligence(self, complexity: float, is_arc: bool) -> float:
        """Estimate intelligence level required for the game."""
        base_requirement = complexity * 0.7

        if is_arc:
            base_requirement += 0.4  # ARC games require high intelligence

        return min(1.0, base_requirement)

    def _detect_performance_degradation(self, action_history: List[Dict]) -> Optional[StrategyMismatchSignal]:
        """
        Detect when performance is consistently degrading despite more actions.

        Pattern: More actions, same (zero) results = wrong approach
        """
        if len(action_history) < 15:
            return None

        # Analyze effectiveness trend over time
        recent_actions = action_history[-15:]
        early_actions = recent_actions[:5]
        late_actions = recent_actions[-5:]

        early_effectiveness = sum(1 for a in early_actions if a.get('success', False)) / len(early_actions)
        late_effectiveness = sum(1 for a in late_actions if a.get('success', False)) / len(late_actions)

        # Check for stagnation (no improvement over time)
        if (early_effectiveness < 0.1 and
            late_effectiveness < 0.1 and
            len(action_history) > 20):

            return StrategyMismatchSignal(
                signal_type="performance_degradation",
                confidence=0.8,
                evidence={
                    "early_effectiveness": early_effectiveness,
                    "late_effectiveness": late_effectiveness,
                    "total_actions": len(action_history),
                    "pattern": "sustained_ineffectiveness"
                },
                recommendation="ESCALATE: Current approach shows no learning - switch strategy"
            )

        return None

    def _detect_domain_classification_mismatch(
        self,
        game_state: Dict[str, Any],
        current_strategy: StrategyType
    ) -> Optional[StrategyMismatchSignal]:
        """
        Detect when game domain (puzzle) doesn't match strategy domain (mechanical).
        """
        game_type = self._classify_game_type(game_state)

        # Check for domain mismatch
        mismatches = {
            (GameType.PUZZLE_SPATIAL, StrategyType.MECHANICAL_SEQUENTIAL): {
                "confidence": 0.95,
                "recommendation": "CRITICAL: Puzzle game requires pattern analysis, not mechanical actions"
            }
        }

        mismatch_key = (game_type, current_strategy)
        if mismatch_key in mismatches:
            mismatch_info = mismatches[mismatch_key]

            return StrategyMismatchSignal(
                signal_type="domain_classification_mismatch",
                confidence=mismatch_info["confidence"],
                evidence={
                    "detected_game_type": game_type.value,
                    "current_strategy": current_strategy.value,
                    "mismatch_severity": "critical"
                },
                recommendation=mismatch_info["recommendation"]
            )

        return None

    def _classify_game_type(self, game_state: Dict[str, Any]) -> GameType:
        """
        Classify the type of game based on available data.
        """
        game_id = game_state.get('game_id', '')
        frame_data = game_state.get('frame', [])

        # ARC games are spatial puzzles
        if self._detect_arc_game_patterns(game_id, frame_data):
            return GameType.PUZZLE_SPATIAL

        # Visual complexity analysis
        complexity = self._calculate_visual_complexity(frame_data)
        if complexity > 0.6:
            return GameType.PUZZLE_SPATIAL

        return GameType.UNKNOWN

    def generate_strategy_recommendations(self, signals: List[StrategyMismatchSignal]) -> Dict[str, Any]:
        """
        Generate concrete recommendations based on detected signals.

        This is what the Governor would use to make decisions.
        """
        if not signals:
            return {"action": "continue", "reason": "No strategy mismatch detected"}

        # Analyze signal strength
        max_confidence = max(s.confidence for s in signals)
        critical_signals = [s for s in signals if s.confidence > 0.8]

        if critical_signals:
            return {
                "action": "escalate_intelligence_immediately",
                "reason": f"Critical mismatch detected: {critical_signals[0].signal_type}",
                "recommended_strategy": StrategyType.PATTERN_ANALYTICAL,
                "evidence": [s.evidence for s in critical_signals],
                "confidence": max_confidence
            }
        elif max_confidence > 0.6:
            return {
                "action": "escalate_intelligence_gradually",
                "reason": f"Moderate mismatch detected",
                "recommended_strategy": StrategyType.EXPLORATION_BASED,
                "evidence": [s.evidence for s in signals],
                "confidence": max_confidence
            }
        else:
            return {
                "action": "monitor",
                "reason": "Weak signals detected, continue monitoring",
                "confidence": max_confidence
            }

# Integration point for Governor/Architect
def create_meta_cognitive_detector() -> MetaCognitiveStrategyDetector:
    """Factory function for creating the detector."""
    return MetaCognitiveStrategyDetector()
"""
Real-Time Learning Engine (Phase 1.1)

Main coordinator for learning during gameplay rather than just after.
Orchestrates mid-game pattern detection, dynamic strategy adjustment,
and immediate action outcome tracking for continuous learning.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

# Import attention coordination (TIER 1)
try:
    from .central_attention_controller import SubsystemDemand
    from .weighted_communication_system import MessagePriority
    ATTENTION_COORDINATION_AVAILABLE = True
except ImportError:
    ATTENTION_COORDINATION_AVAILABLE = False
    SubsystemDemand = None
    MessagePriority = None

logger = logging.getLogger(__name__)

@dataclass
class RealTimeLearningContext:
    """Context for real-time learning during a game session."""
    game_id: str
    session_id: str
    current_action_count: int
    current_score: float
    last_score_change: float
    recent_actions: List[int]
    recent_coordinates: List[Tuple[int, int]]
    active_patterns: List[str]
    active_adjustments: List[str]
    current_focus: Optional[str]
    active_hypotheses: List[str]
    learning_intensity: float = 0.5
    adaptation_mode: str = "normal"  # "normal", "aggressive", "conservative"
    start_time: float = 0.0  # Timestamp when the game context was initialized

class RealTimeLearner:
    """
    Main coordinator for real-time learning during gameplay.

    Integrates pattern detection, strategy adjustment, outcome tracking,
    and hypothesis formation to enable learning DURING gameplay rather
    than only after game completion.
    """

    def __init__(self, db_manager, game_type_classifier=None):
        self.db = db_manager
        self.game_type_classifier = game_type_classifier

        # Core components (will be injected)
        self.pattern_detector = None
        self.strategy_adjuster = None
        self.outcome_tracker = None

        # Active learning contexts by game_id
        self.active_contexts: Dict[str, RealTimeLearningContext] = {}

        # Attention coordination (TIER 1)
        self.attention_controller = None
        self.communication_system = None
        self.attention_coordination_enabled = ATTENTION_COORDINATION_AVAILABLE

        # Real-time learning configuration
        self.config = {
            "pattern_detection_frequency": 3,  # Check for patterns every N actions
            "strategy_adjustment_threshold": 0.7,  # Confidence threshold for adjustments
            "outcome_tracking_window": 5,  # Actions to track for immediate outcomes
            "learning_intensity_base": 0.5,
            "hypothesis_formation_threshold": 0.6,
            "max_concurrent_patterns": 10,
            "max_concurrent_adjustments": 5,
            "max_concurrent_hypotheses": 8
        }

        # Performance metrics
        self.metrics = {
            "patterns_detected": 0,
            "adjustments_made": 0,
            "outcomes_tracked": 0,
            "hypotheses_formed": 0,
            "successful_adaptations": 0,
            "learning_events": 0
        }

    def set_components(self, pattern_detector, strategy_adjuster, outcome_tracker):
        """Inject the core learning components."""
        self.pattern_detector = pattern_detector
        self.strategy_adjuster = strategy_adjuster
        self.outcome_tracker = outcome_tracker
        logger.info("Real-time learning components initialized")

    def set_attention_coordination(self, attention_controller, communication_system):
        """Set attention coordination systems for enhanced coordination."""
        self.attention_controller = attention_controller
        self.communication_system = communication_system
        if attention_controller and communication_system:
            logger.info("Real-time learning enhanced with attention coordination")
        else:
            logger.warning("Attention coordination systems not fully available")

    async def initialize_game_context(self,
                                     game_id: str,
                                     session_id: str,
                                     initial_score: float = 0.0) -> RealTimeLearningContext:
        """Initialize real-time learning context for a new game."""
        try:
            context = RealTimeLearningContext(
                game_id=game_id,
                session_id=session_id,
                current_action_count=0,
                current_score=initial_score,
                last_score_change=0.0,
                recent_actions=[],
                recent_coordinates=[],
                active_patterns=[],
                active_adjustments=[],
                current_focus=None,
                active_hypotheses=[],
                learning_intensity=self.config["learning_intensity_base"],
                start_time=time.time()  # Set start time for processing complexity calculations
            )

            self.active_contexts[game_id] = context

            # Initialize components for this game
            if self.pattern_detector:
                await self.pattern_detector.initialize_game(game_id, session_id)
            if self.strategy_adjuster:
                await self.strategy_adjuster.initialize_game(game_id, session_id)
            if self.outcome_tracker:
                await self.outcome_tracker.initialize_game(game_id, session_id)

            logger.info(f"Initialized real-time learning context for game {game_id}")
            return context

        except Exception as e:
            logger.error(f"Failed to initialize real-time learning context: {e}")
            raise

    async def process_action_taken(self,
                                  game_id: str,
                                  action_number: int,
                                  coordinates: Optional[Tuple[int, int]] = None,
                                  score_before: float = 0.0,
                                  score_after: float = 0.0,
                                  frame_changes: bool = False,
                                  movement_detected: bool = False,
                                  game_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an action taken during gameplay and trigger real-time learning.

        This is the main entry point called after each action during gameplay.
        """
        if game_id not in self.active_contexts:
            logger.warning(f"No active context for game {game_id}")
            return {}

        context = self.active_contexts[game_id]

        try:
            # ENHANCED ATTENTION COORDINATION (TIER 1)
            # Request attention allocation based on processing complexity
            attention_allocation = None
            if (self.attention_coordination_enabled and self.attention_controller and
                ATTENTION_COORDINATION_AVAILABLE):
                try:
                    # Calculate processing complexity based on current situation
                    processing_complexity = self._calculate_processing_complexity(
                        context, score_before, score_after, frame_changes, movement_detected
                    )

                    # Determine urgency level
                    urgency_level = self._determine_urgency_level(context, score_before, score_after)

                    # Create attention demand
                    attention_demand = SubsystemDemand(
                        subsystem_name="real_time_learning",
                        requested_priority=min(0.8, 0.3 + processing_complexity * 0.5),
                        current_load=processing_complexity,
                        processing_complexity=processing_complexity,
                        urgency_level=urgency_level,
                        justification=f"Processing action {action_number} with complexity {processing_complexity:.2f}",
                        context_data={
                            "action_number": action_number,
                            "score_change": score_after - score_before,
                            "frame_changes": frame_changes,
                            "movement_detected": movement_detected,
                            "active_patterns": len(context.active_patterns)
                        }
                    )

                    # Request attention allocation
                    attention_allocation = await self.attention_controller.allocate_attention_resources(
                        game_id, [attention_demand], game_context
                    )

                except Exception as e:
                    logger.debug(f"Attention coordination error: {e}")

            # Update context with new action
            context.current_action_count += 1
            context.last_score_change = score_after - score_before
            context.current_score = score_after
            context.recent_actions.append(action_number)
            if coordinates:
                context.recent_coordinates.append(coordinates)

            # Keep recent history limited
            if len(context.recent_actions) > 20:
                context.recent_actions = context.recent_actions[-20:]
            if len(context.recent_coordinates) > 20:
                context.recent_coordinates = context.recent_coordinates[-20:]

            # Parallel processing of learning components
            learning_results = await asyncio.gather(
                self._track_action_outcome(context, action_number, coordinates,
                                         score_before, score_after, frame_changes,
                                         movement_detected, game_context),
                self._detect_patterns_if_needed(context, game_context),
                self._adjust_strategy_if_needed(context, game_context),
                self._form_hypotheses_if_needed(context, game_context),
                return_exceptions=True
            )

            # Process results and update metrics
            outcome_result, pattern_result, adjustment_result, hypothesis_result = learning_results

            # Compile real-time learning insights
            learning_insights = {
                "action_count": context.current_action_count,
                "immediate_outcome": outcome_result if not isinstance(outcome_result, Exception) else None,
                "patterns_detected": pattern_result if not isinstance(pattern_result, Exception) else [],
                "strategy_adjustments": adjustment_result if not isinstance(adjustment_result, Exception) else [],
                "new_hypotheses": hypothesis_result if not isinstance(hypothesis_result, Exception) else [],
                "learning_intensity": context.learning_intensity,
                "adaptation_mode": context.adaptation_mode
            }

            # Log any exceptions
            for i, result in enumerate(learning_results):
                if isinstance(result, Exception):
                    component_names = ["outcome_tracker", "pattern_detector", "strategy_adjuster", "hypothesis_former"]
                    logger.error(f"Error in {component_names[i]}: {result}")

            return learning_insights

        except Exception as e:
            logger.error(f"Error processing action for real-time learning: {e}")
            return {}

    async def _track_action_outcome(self,
                                   context: RealTimeLearningContext,
                                   action_number: int,
                                   coordinates: Optional[Tuple[int, int]],
                                   score_before: float,
                                   score_after: float,
                                   frame_changes: bool,
                                   movement_detected: bool,
                                   game_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Track immediate outcome of the action."""
        if not self.outcome_tracker:
            return None

        try:
            outcome = await self.outcome_tracker.track_action_outcome(
                context.game_id,
                context.session_id,
                action_number,
                coordinates,
                score_before,
                score_after,
                frame_changes,
                movement_detected,
                context.current_action_count,
                game_context
            )

            if outcome:
                self.metrics["outcomes_tracked"] += 1

                # Trigger learning events based on outcome
                if outcome.get("immediate_classification") == "highly_effective":
                    await self._trigger_learning_event(context, "high_effectiveness_action", outcome)
                elif outcome.get("immediate_classification") == "harmful":
                    await self._trigger_learning_event(context, "harmful_action_detected", outcome)

            return outcome

        except Exception as e:
            logger.error(f"Error tracking action outcome: {e}")
            return None

    async def _detect_patterns_if_needed(self,
                                        context: RealTimeLearningContext,
                                        game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns if conditions are met."""
        if not self.pattern_detector:
            return []

        # Check if we should detect patterns (every N actions)
        if context.current_action_count % self.config["pattern_detection_frequency"] != 0:
            return []

        try:
            patterns = await self.pattern_detector.detect_emerging_patterns(
                context.game_id,
                context.session_id,
                context.recent_actions,
                context.recent_coordinates,
                context.current_score,
                context.current_action_count,
                game_context
            )

            if patterns:
                self.metrics["patterns_detected"] += len(patterns)

                # Update context with new active patterns
                for pattern in patterns:
                    pattern_id = pattern.get("pattern_id")
                    if pattern_id and pattern_id not in context.active_patterns:
                        context.active_patterns.append(pattern_id)

                # Keep active patterns list manageable
                if len(context.active_patterns) > self.config["max_concurrent_patterns"]:
                    context.active_patterns = context.active_patterns[-self.config["max_concurrent_patterns"]:]

            return patterns

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []

    async def _adjust_strategy_if_needed(self,
                                        context: RealTimeLearningContext,
                                        game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adjust strategy if patterns suggest it."""
        if not self.strategy_adjuster:
            return []

        try:
            adjustments = await self.strategy_adjuster.evaluate_strategy_adjustments(
                context.game_id,
                context.session_id,
                context.active_patterns,
                context.last_score_change,
                context.recent_actions,
                context.current_action_count,
                game_context
            )

            if adjustments:
                self.metrics["adjustments_made"] += len(adjustments)

                # Update context with new active adjustments
                for adjustment in adjustments:
                    adjustment_id = adjustment.get("adjustment_id")
                    if adjustment_id and adjustment_id not in context.active_adjustments:
                        context.active_adjustments.append(adjustment_id)

                # Keep active adjustments list manageable
                if len(context.active_adjustments) > self.config["max_concurrent_adjustments"]:
                    context.active_adjustments = context.active_adjustments[-self.config["max_concurrent_adjustments"]:]

                # Adjust learning intensity based on strategy changes
                if any(adj.get("effectiveness_score", 0) > 0.8 for adj in adjustments):
                    context.learning_intensity = min(1.0, context.learning_intensity + 0.1)

            return adjustments

        except Exception as e:
            logger.error(f"Error adjusting strategy: {e}")
            return []

    async def _form_hypotheses_if_needed(self,
                                        context: RealTimeLearningContext,
                                        game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Form new hypotheses based on recent observations."""
        # For now, implement basic hypothesis formation logic
        # This will be expanded in Phase 3.2
        hypotheses = []

        try:
            # Simple hypothesis formation based on recent patterns
            if (context.current_action_count > 10 and
                context.current_action_count % 7 == 0 and  # Every 7 actions
                len(context.recent_actions) >= 5):

                # Form hypothesis about action effectiveness
                recent_effectiveness = []
                for i in range(min(5, len(context.recent_actions))):
                    # This is a simplified hypothesis - will be enhanced in Phase 3.2
                    if context.last_score_change > 0:
                        recent_effectiveness.append(True)
                    else:
                        recent_effectiveness.append(False)

                if len(recent_effectiveness) >= 3:
                    effectiveness_ratio = sum(recent_effectiveness) / len(recent_effectiveness)

                    if effectiveness_ratio > self.config["hypothesis_formation_threshold"]:
                        hypothesis = await self._create_hypothesis(
                            context,
                            "action_effectiveness",
                            f"Recent action sequence is {effectiveness_ratio:.1%} effective",
                            {"actions": context.recent_actions[-5:], "effectiveness_ratio": effectiveness_ratio}
                        )
                        if hypothesis:
                            hypotheses.append(hypothesis)
                            self.metrics["hypotheses_formed"] += 1

            return hypotheses

        except Exception as e:
            logger.error(f"Error forming hypotheses: {e}")
            return []

    async def _create_hypothesis(self,
                                context: RealTimeLearningContext,
                                hypothesis_type: str,
                                statement: str,
                                data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create and store a new hypothesis."""
        try:
            hypothesis_id = f"hyp_{context.game_id}_{int(time.time() * 1000)}"

            hypothesis_data = {
                "hypothesis_id": hypothesis_id,
                "game_id": context.game_id,
                "session_id": context.session_id,
                "hypothesis_type": hypothesis_type,
                "hypothesis_statement": statement,
                "hypothesis_data": json.dumps(data),
                "formation_action": context.current_action_count,
                "confidence": 0.6,  # Default confidence
                "test_actions": json.dumps([]),
                "test_results": json.dumps([]),
                "status": "active"
            }

            # Store in database
            await self.db.execute_query(
                """INSERT INTO real_time_hypotheses
                   (hypothesis_id, game_id, session_id, hypothesis_type, hypothesis_statement,
                    hypothesis_data, formation_action, confidence, test_actions, test_results, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (hypothesis_id, context.game_id, context.session_id, hypothesis_type, statement,
                 hypothesis_data["hypothesis_data"], context.current_action_count, 0.6,
                 hypothesis_data["test_actions"], hypothesis_data["test_results"], "active")
            )

            # Add to context
            if len(context.active_hypotheses) < self.config["max_concurrent_hypotheses"]:
                context.active_hypotheses.append(hypothesis_id)

            return hypothesis_data

        except Exception as e:
            logger.error(f"Error creating hypothesis: {e}")
            return None

    async def _trigger_learning_event(self,
                                     context: RealTimeLearningContext,
                                     event_type: str,
                                     event_data: Dict[str, Any]):
        """Trigger a mid-game learning event."""
        try:
            event_id = f"learn_{context.game_id}_{int(time.time() * 1000)}"

            await self.db.execute_query(
                """INSERT INTO mid_game_learning_events
                   (event_id, game_id, session_id, event_type, event_data, trigger_action, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (event_id, context.game_id, context.session_id, event_type,
                 json.dumps(event_data), context.current_action_count, 0.7)
            )

            self.metrics["learning_events"] += 1
            logger.debug(f"Triggered learning event: {event_type} for game {context.game_id}")

        except Exception as e:
            logger.error(f"Error triggering learning event: {e}")

    def _calculate_processing_complexity(self,
                                       context: RealTimeLearningContext,
                                       score_before: float,
                                       score_after: float,
                                       frame_changes: bool,
                                       movement_detected: bool) -> float:
        """
        Calculate processing complexity for attention allocation.
        
        Returns a value between 0.0 (low complexity) and 1.0 (high complexity).
        """
        complexity = 0.0
        
        # Base complexity from current context state
        if len(context.active_patterns) > 3:
            complexity += 0.2
        if len(context.active_adjustments) > 2:
            complexity += 0.15
        if len(context.active_hypotheses) > 1:
            complexity += 0.1
            
        # Score change impact (larger changes require more processing)
        score_change = abs(score_after - score_before)
        if score_change > 100:
            complexity += 0.3
        elif score_change > 50:
            complexity += 0.2
        elif score_change > 10:
            complexity += 0.1
            
        # Visual/movement changes
        if frame_changes:
            complexity += 0.1
        if movement_detected:
            complexity += 0.05
            
        # Learning intensity affects complexity
        complexity += context.learning_intensity * 0.1
        
        # Action frequency (rapid actions are more complex to process)
        if context.current_action_count > 20:
            action_rate = context.current_action_count / max(1, time.time() - context.start_time)
            if action_rate > 2.0:  # More than 2 actions per second
                complexity += 0.15
                
        return min(1.0, complexity)

    def _determine_urgency_level(self,
                               context: RealTimeLearningContext,
                               score_before: float,
                               score_after: float) -> int:
        """
        Determine urgency level for attention allocation.
        
        Returns urgency level from 1 (low) to 5 (critical).
        """
        urgency = 2  # Default moderate urgency
        
        # Score changes affect urgency
        score_change = score_after - score_before
        if score_change < -50:  # Significant loss
            urgency = 5  # Critical - need immediate attention
        elif score_change < -20:
            urgency = 4  # High urgency
        elif score_change < 0:
            urgency = 3  # Above moderate
        elif score_change > 50:  # Significant gain
            urgency = 4  # High urgency to learn from success
        elif score_change > 20:
            urgency = 3
            
        # Context-based urgency adjustments
        if context.adaptation_mode == "emergency":
            urgency = min(5, urgency + 2)
        elif context.adaptation_mode == "aggressive":
            urgency = min(5, urgency + 1)
            
        # Multiple active patterns suggest complex situation
        if len(context.active_patterns) > 5:
            urgency = min(5, urgency + 1)
            
        # Low learning intensity suggests we need more attention
        if context.learning_intensity < 0.3:
            urgency = min(5, urgency + 1)
            
        return max(1, min(5, urgency))

    async def get_real_time_insights(self, game_id: str) -> Dict[str, Any]:
        """Get current real-time learning insights for a game."""
        if game_id not in self.active_contexts:
            return {}

        context = self.active_contexts[game_id]

        try:
            # Get recent patterns
            recent_patterns = await self.db.fetch_all(
                """SELECT pattern_type, confidence, pattern_strength, detection_timestamp
                   FROM real_time_patterns
                   WHERE game_id = ? AND is_active = 1
                   ORDER BY detection_timestamp DESC LIMIT 5""",
                (game_id,)
            )

            # Get recent adjustments
            recent_adjustments = await self.db.fetch_all(
                """SELECT adjustment_type, effectiveness_score, success
                   FROM real_time_strategy_adjustments
                   WHERE game_id = ?
                   ORDER BY applied_at_action DESC LIMIT 5""",
                (game_id,)
            )

            # Get active hypotheses
            active_hypotheses = await self.db.fetch_all(
                """SELECT hypothesis_type, confidence, status
                   FROM real_time_hypotheses
                   WHERE game_id = ? AND status = 'active'
                   ORDER BY formation_action DESC""",
                (game_id,)
            )

            return {
                "game_id": game_id,
                "action_count": context.current_action_count,
                "current_score": context.current_score,
                "learning_intensity": context.learning_intensity,
                "adaptation_mode": context.adaptation_mode,
                "active_patterns_count": len(context.active_patterns),
                "active_adjustments_count": len(context.active_adjustments),
                "active_hypotheses_count": len(context.active_hypotheses),
                "recent_patterns": [dict(row) for row in recent_patterns] if recent_patterns else [],
                "recent_adjustments": [dict(row) for row in recent_adjustments] if recent_adjustments else [],
                "active_hypotheses": [dict(row) for row in active_hypotheses] if active_hypotheses else [],
                "performance_metrics": self.metrics.copy()
            }

        except Exception as e:
            logger.error(f"Error getting real-time insights: {e}")
            return {"error": str(e)}

    async def finalize_game_context(self, game_id: str) -> Dict[str, Any]:
        """Finalize learning context when game ends."""
        if game_id not in self.active_contexts:
            return {}

        context = self.active_contexts[game_id]

        try:
            # Get final learning summary
            summary = await self.get_real_time_insights(game_id)

            # Mark patterns as inactive
            await self.db.execute_query(
                """UPDATE real_time_patterns
                   SET is_active = 0, last_updated = ?
                   WHERE game_id = ? AND is_active = 1""",
                (time.time(), game_id)
            )

            # Finalize active hypotheses
            await self.db.execute_query(
                """UPDATE real_time_hypotheses
                   SET status = 'inconclusive', updated_at = ?
                   WHERE game_id = ? AND status = 'active'""",
                (time.time(), game_id)
            )

            # Clean up context
            del self.active_contexts[game_id]

            logger.info(f"Finalized real-time learning context for game {game_id}")
            return summary

        except Exception as e:
            logger.error(f"Error finalizing game context: {e}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall real-time learning performance metrics."""
        return {
            "active_games": len(self.active_contexts),
            "total_metrics": self.metrics.copy(),
            "config": self.config.copy()
        }
"""
Continuous Learning Loop - Modular Version

Simplified main orchestrator that uses modular components for all functionality.
This replaces the massive 18,000+ line monolithic file.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from ..analysis.action_selector import create_action_selector
from pathlib import Path
from datetime import datetime

# Import modular components
from ..memory import MemoryManager, ActionMemoryManager, PatternMemoryManager
from ..sessions import TrainingSessionManager, TrainingSessionConfig
from ..api import APIManager
try:
    from src.core.unified_performance_monitor import UnifiedPerformanceMonitor
except ImportError:
    from core.unified_performance_monitor import UnifiedPerformanceMonitor
from ..performance import MetricsCollector
from ..governor import TrainingGovernor, MetaCognitiveController
from ..learning import LearningEngine, PatternLearner, KnowledgeTransfer
from ..utils import LazyImports, ShutdownHandler, CompatibilityShim

# Import losing streak detection components
try:
    from src.core.losing_streak_detector import LosingStreakDetector, FailureType
    from src.core.anti_pattern_learner import AntiPatternLearner
    from src.core.escalated_intervention_system import EscalatedInterventionSystem
except ImportError:
    from core.losing_streak_detector import LosingStreakDetector, FailureType
    from core.anti_pattern_learner import AntiPatternLearner
    from core.escalated_intervention_system import EscalatedInterventionSystem

# Import real-time learning engine components (Phase 1.1)
try:
    from src.core.real_time_learner import RealTimeLearner
    from src.core.mid_game_pattern_detector import MidGamePatternDetector
    from src.core.dynamic_strategy_adjuster import DynamicStrategyAdjuster
    from src.core.action_outcome_tracker import ActionOutcomeTracker
except ImportError:
    from core.real_time_learner import RealTimeLearner
    from core.mid_game_pattern_detector import MidGamePatternDetector
    from core.dynamic_strategy_adjuster import DynamicStrategyAdjuster
    from core.action_outcome_tracker import ActionOutcomeTracker

# Import enhanced attention + communication system components (TIER 1)
try:
    from src.core.central_attention_controller import CentralAttentionController, SubsystemDemand, ResourceUsage
    from src.core.weighted_communication_system import WeightedCommunicationSystem, MessagePriority

    # Import context-dependent fitness evolution system components (TIER 2)
    from src.core.context_dependent_fitness_evolution import ContextDependentFitnessEvolution, LearningPhase, ContextType

    # Import NEAT-based architect system components (TIER 2)
    from src.core.neat_based_architect import NEATBasedArchitect, ModuleType, ModuleCategory

    # Import Bayesian inference engine components (TIER 3)
    from src.core.bayesian_inference_engine import BayesianInferenceEngine, HypothesisType, EvidenceType

    # Import enhanced graph traversal components (TIER 3)
    from src.core.enhanced_graph_traversal import EnhancedGraphTraversal, GraphType, TraversalAlgorithm, NodeType
except ImportError:
    from core.central_attention_controller import CentralAttentionController, SubsystemDemand, ResourceUsage
    from core.weighted_communication_system import WeightedCommunicationSystem, MessagePriority

    # Import context-dependent fitness evolution system components (TIER 2)
    from core.context_dependent_fitness_evolution import ContextDependentFitnessEvolution, LearningPhase, ContextType

    # Import NEAT-based architect system components (TIER 2)
    from core.neat_based_architect import NEATBasedArchitect, ModuleType, ModuleCategory

    # Import Bayesian inference engine components (TIER 3)
    from core.bayesian_inference_engine import BayesianInferenceEngine, HypothesisType, EvidenceType

    # Import enhanced graph traversal components (TIER 3)
    from core.enhanced_graph_traversal import EnhancedGraphTraversal, GraphType, TraversalAlgorithm, NodeType

logger = logging.getLogger(__name__)

class ContinuousLearningLoop:
    """
    Modular Continuous Learning Loop
    
    This simplified version orchestrates all the modular components
    instead of containing all functionality in one massive file.
    """
    
    def __init__(
        self,
        arc_agents_path: str = ".",
        tabula_rasa_path: str = ".",
        api_key: Optional[str] = None,
        save_directory: str = "data"
    ):
        """Initialize the continuous learning loop with modular components."""
        print("[START] Starting Modular ContinuousLearningLoop initialization...")
        
        # Store basic configuration
        self.arc_agents_path = Path(arc_agents_path)
        self.tabula_rasa_path = Path(tabula_rasa_path)
        self.api_key = api_key
        self.save_directory = Path(save_directory)
        
        # Initialize modular components
        self._initialize_components()
        
        # Initialize shutdown handling
        self.shutdown_handler = ShutdownHandler()
        self.shutdown_handler.add_shutdown_callback(self._cleanup)
        
        # Initialize compatibility shim
        self.compatibility_shim = CompatibilityShim()
        
        print("[OK] Modular ContinuousLearningLoop initialized successfully")
    
    async def _ensure_api_initialized(self) -> None:
        """Ensure API manager is initialized (async)."""
        if not self._api_initialized:
            await self.api_manager.initialize()
            self._api_initialized = True
            
            # Initialize action selector
            self.action_selector = create_action_selector(self.api_manager)
    
    def _ensure_initialized(self) -> None:
        """Ensure the system is initialized (synchronous wrapper)."""
        # The system is already initialized in __init__, but we can add validation here
        if not hasattr(self, 'api_manager'):
            raise RuntimeError("ContinuousLearningLoop not properly initialized")
        print("[OK] System initialization verified")
    
    async def get_available_games(self) -> List[Dict[str, Any]]:
        """Get list of available games from the real ARC-AGI-3 API."""
        try:
            await self._ensure_api_initialized()
            # Get real games from ARC-AGI-3 API
            games = await self.api_manager.get_available_games()
            if games:
                print(f" Found {len(games)} real ARC-AGI-3 games available")
                return games
            else:
                print("[WARNING] No real games available from ARC-AGI-3 API")
                return []
        except Exception as e:
            logger.error(f"Error getting available games: {e}")
            return []
    
    async def start_training_with_direct_control(self, game_id: str, max_actions_per_game: int = 5000, 
                                               session_count: int = 1, duration_minutes: int = 15) -> Dict[str, Any]:
        """Start training with direct API control using real ARC-AGI-3 API."""
        try:
            print(f"[TARGET] Starting REAL ARC-AGI-3 training for game {game_id}")
            print(f"   Max actions: {max_actions_per_game}")
            print(f"   Duration: {duration_minutes} minutes")
            
            # Initialize API if needed
            await self._ensure_api_initialized()
            
            # Get real available games from ARC-AGI-3 API
            available_games = await self.api_manager.get_available_games()
            if not available_games:
                raise Exception("No real games available from ARC-AGI-3 API")
            
            # Use the first available real game
            real_game = available_games[0]
            real_game_id = real_game.get('game_id', game_id)
            
            print(f" Using real ARC-AGI-3 game: {real_game_id}")

            # REAL-TIME LEARNING ENGINE: Initialize game context (Phase 1.1)
            if self._real_time_learning_initialized and self.real_time_learner:
                try:
                    session_id = f"session_{session_count}"
                    self._game_real_time_context = await self.real_time_learner.initialize_game_context(
                        real_game_id, session_id, 0.0  # Initial score
                    )
                    print(f" 🧠 Real-time learning context initialized for game {real_game_id}")
                except Exception as e:
                    logger.warning(f"Failed to initialize real-time learning context: {e}")
                    self._game_real_time_context = None
            else:
                self._game_real_time_context = None

            # ENHANCED ATTENTION + COMMUNICATION: Initialize game context (TIER 1)
            if self._attention_communication_initialized and self.attention_controller and self.communication_system:
                try:
                    session_id = f"session_{session_count}"

                    # Initialize attention monitoring
                    self._game_attention_context = await self.attention_controller.initialize_attention_monitoring(
                        real_game_id, session_id
                    )

                    # Initialize communication system
                    self._game_communication_context = await self.communication_system.initialize_communication_system(
                        real_game_id, session_id
                    )

                    print(f" ⚡ Enhanced attention + communication systems initialized for game {real_game_id}")
                except Exception as e:
                    logger.warning(f"Failed to initialize attention + communication systems: {e}")
                    self._game_attention_context = None
                    self._game_communication_context = None
            else:
                self._game_attention_context = None
                self._game_communication_context = None

            # Open scorecard for tracking
            scorecard_id = await self.api_manager.create_scorecard(
                f"Real Training Session {session_count}",
                f"Direct API control training for {real_game_id}"
            )
            
            if not scorecard_id:
                print(" No scorecard created, proceeding without scorecard tracking")
                scorecard_id = None
            
            # Reset the game to get initial state
            # Pass the scorecard_id to the reset_game method
            reset_response = await self.api_manager.reset_game(real_game_id, scorecard_id)
            if not reset_response:
                raise Exception("Failed to reset game")
            
            # reset_response is a GameState object, not a dict
            game_guid = reset_response.guid
            if not game_guid:
                raise Exception("No game GUID received from reset")
            
            print(f" Game start/reset successful. GUID: {game_guid[:8]}...")
            
            # Play the game with real actions
            actions_taken = 0
            total_score = 0
            game_won = False
            current_state = reset_response.state

            # Track action sequences for win pattern learning
            action_sequence = []
            action_coordinates = []  # Track coordinates for Action 6
            score_progression = [0.0]  # Start with initial score

            # Track level-based learning
            level_action_sequence = []
            level_score_progression = [0.0]
            current_level = 1
            last_significant_score = 0.0
            level_completion_threshold = 50.0  # Score increase indicating level completion
            
            print(f"[TARGET] Starting real gameplay with {max_actions_per_game} max actions...")

            # Check for learned winning strategies to apply (both game-level and level-specific)
            if self.action_selector and hasattr(self.action_selector, 'strategy_discovery_system'):
                if self.action_selector.strategy_discovery_system:
                    try:
                        # Check for game-level strategies
                        should_replicate = await self.action_selector.strategy_discovery_system.should_attempt_strategy_replication(real_game_id)
                        if should_replicate:
                            best_strategy = await self.action_selector.strategy_discovery_system.get_best_strategy_for_game(real_game_id)
                            if best_strategy:
                                print(f" APPLYING LEARNED GAME STRATEGY: {best_strategy.strategy_id} - " +
                                      f"efficiency: {best_strategy.efficiency:.2f}, {len(best_strategy.action_sequence)} actions")

                        # Check for level-specific strategies (level 1)
                        level_1_id = f"{real_game_id}_level_1"
                        should_replicate_level = await self.action_selector.strategy_discovery_system.should_attempt_strategy_replication(level_1_id)
                        if should_replicate_level:
                            best_level_strategy = await self.action_selector.strategy_discovery_system.get_best_strategy_for_game(level_1_id)
                            if best_level_strategy:
                                print(f" APPLYING LEARNED LEVEL 1 STRATEGY: {best_level_strategy.strategy_id} - " +
                                      f"efficiency: {best_level_strategy.efficiency:.2f}, {len(best_level_strategy.action_sequence)} actions")

                        # Check for known win conditions
                        try:
                            game_type = self.action_selector.strategy_discovery_system.game_type_classifier.extract_game_type(real_game_id)
                            win_conditions = await self.action_selector.strategy_discovery_system.get_win_conditions_for_game_type(game_type)
                            if win_conditions:
                                print(f" KNOWN WIN CONDITIONS: Found {len(win_conditions)} win conditions for game type {game_type}")
                                high_success_conditions = [c for c in win_conditions if c['success_rate'] > 0.7]
                                if high_success_conditions:
                                    print(f" HIGH SUCCESS CONDITIONS: {len(high_success_conditions)} conditions with >70% success rate")
                        except Exception as e:
                            print(f"[WARNING] Failed to check win conditions: {e}")

                    except Exception as e:
                        print(f"[WARNING] Failed to check for learned strategies: {e}")

            from src.database.persistence_helpers import persist_button_priorities, persist_winning_sequence
            while actions_taken < max_actions_per_game and current_state == 'NOT_FINISHED':
                try:
                    # Check rate limit status and implement dynamic pausing
                    rate_status = self.api_manager.get_rate_limit_status()
                    warning = self.api_manager.rate_limiter.get_usage_warning()
                    should_pause, pause_duration = self.api_manager.rate_limiter.should_pause()
                    if should_pause:
                        print(f"⏸ Rate limit pause: {pause_duration:.1f}s (usage: {rate_status.get('current_usage', 0)}/{rate_status.get('max_requests', 550)})")
                        await asyncio.sleep(pause_duration)
                    elif warning:
                        print(f"[WARNING] {warning}")
                        # Small delay when approaching limits
                        await asyncio.sleep(0.5)  # 0.5 second delay
                    
                    # Get current game state
                    current_response = await self.api_manager.get_game_state(real_game_id, card_id=scorecard_id, guid=game_guid)
                    if not current_response:
                        break
                    
                    current_state = current_response.state
                    # Ensure current_score is a number
                    if isinstance(current_response.score, (int, float)):
                        current_score = float(current_response.score)
                    else:
                        current_score = 0.0
                    available_actions = current_response.available_actions
                    
                    if current_state != 'NOT_FINISHED':
                        break
                    
                    # Choose action based on available actions
                    # Choose and execute an action using intelligent selection
                    if self.action_selector:
                        # Use intelligent action selection with frame analysis
                        game_state = {
                            'frame': current_response.frame,
                            'state': current_response.state,
                            'score': current_response.score,
                            'available_actions': available_actions
                        }
                        action_to_take = await self.action_selector.select_action(game_state, available_actions)
                    else:
                        # Fallback to simple action selection
                        action_to_take = self._choose_smart_action(available_actions, current_response)
                    
                    if not action_to_take or not isinstance(action_to_take, dict):
                        print("[WARNING] No valid actions available, ending game")
                        break
                    
                    # Execute the action
                    action_result = await self.api_manager.take_action(
                        real_game_id, action_to_take, scorecard_id, game_guid
                    )
                    
                    if action_result:
                        actions_taken += 1
                        
                        # Handle both GameState objects and dictionaries
                        if hasattr(action_result, 'score'):
                            # GameState object
                            new_score = action_result.score
                            game_state_str = action_result.state
                            available_actions = action_result.available_actions
                            frame = action_result.frame
                        else:
                            # Dictionary
                            new_score = action_result.get('score', current_score)
                            game_state_str = action_result.get('state', 'NOT_FINISHED')
                            available_actions = action_result.get('available_actions', [])
                            frame = action_result.get('frame', [])
                        
                        
                        # Ensure both scores are numbers
                        if isinstance(new_score, (int, float)):
                            new_score = float(new_score)
                        else:
                            new_score = current_score
                        
                        score_change = new_score - current_score
                        total_score = new_score

                        # Track action and score for win pattern learning
                        if isinstance(action_to_take, dict) and 'id' in action_to_take:
                            action_id = action_to_take['id']
                            action_sequence.append(action_id)

                            # For Action 6, store coordinate information
                            if action_id == 6 and 'x' in action_to_take and 'y' in action_to_take:
                                action_coordinates.append((action_to_take['x'], action_to_take['y']))
                                detailed_action = f"6({action_to_take['x']},{action_to_take['y']})"
                                level_action_sequence.append(detailed_action)
                            else:
                                # For non-Action 6, store None as coordinate placeholder
                                action_coordinates.append(None)
                                level_action_sequence.append(action_id)
                        elif isinstance(action_to_take, int):
                            action_sequence.append(action_to_take)
                            action_coordinates.append(None)  # No coordinates for integer actions
                            level_action_sequence.append(action_to_take)
                        score_progression.append(total_score)
                        level_score_progression.append(total_score)

                        # REAL-TIME LEARNING ENGINE INTEGRATION (Phase 1.1)
                        # Process action outcome through real-time learning system
                        if (self._real_time_learning_initialized and self.real_time_learner and
                            hasattr(self, '_game_real_time_context')):
                            try:
                                # Extract action details
                                action_number = action_to_take.get('id') if isinstance(action_to_take, dict) else action_to_take
                                coordinates = None
                                if isinstance(action_to_take, dict) and action_number == 6:
                                    coordinates = (action_to_take.get('x'), action_to_take.get('y'))

                                # Detect frame changes and movement (simplified detection)
                                frame_changes_detected = (score_change != 0 or
                                                        (frame and len(frame) > 0 and frame != getattr(self, '_last_frame', [])))
                                movement_detected = frame_changes_detected and score_change >= 0

                                # Create game context
                                game_context = {
                                    'game_state': game_state_str,
                                    'available_actions': available_actions,
                                    'actions_taken': actions_taken,
                                    'current_level': current_level,
                                    'frame_data': frame
                                }

                                # Process action through real-time learning
                                learning_insights = await self.real_time_learner.process_action_taken(
                                    real_game_id,
                                    action_number,
                                    coordinates,
                                    current_score,
                                    new_score,
                                    frame_changes_detected,
                                    movement_detected,
                                    game_context
                                )

                                # Log real-time learning insights
                                if learning_insights and learning_insights.get('patterns_detected'):
                                    print(f" 🧠 REAL-TIME LEARNING: {len(learning_insights['patterns_detected'])} patterns detected")
                                if learning_insights and learning_insights.get('strategy_adjustments'):
                                    print(f" ⚡ STRATEGY ADJUSTED: {len(learning_insights['strategy_adjustments'])} real-time adjustments")

                                # Store last frame for comparison
                                self._last_frame = frame

                            except Exception as e:
                                logger.warning(f"Real-time learning processing error: {e}")

                        # ENHANCED ATTENTION + COMMUNICATION INTEGRATION (TIER 1)
                        # Coordinate attention allocation and system communication
                        if (self._attention_communication_initialized and self.attention_controller and
                            self.communication_system and hasattr(self, '_game_attention_context')):
                            try:
                                # Create subsystem demands based on current situation
                                subsystem_demands = []

                                # Real-time learning demand
                                if self._real_time_learning_initialized:
                                    rt_demand = SubsystemDemand(
                                        subsystem_name="real_time_learning",
                                        requested_priority=0.4 if learning_insights and learning_insights.get('patterns_detected') else 0.2,
                                        current_load=0.3,  # Estimate based on processing
                                        processing_complexity=0.5,
                                        urgency_level=3 if score_change < 0 else 2,
                                        justification="Processing action outcomes and detecting patterns",
                                        context_data={"score_change": score_change, "action_count": actions_taken}
                                    )
                                    subsystem_demands.append(rt_demand)

                                # Action selection demand (always high priority)
                                action_demand = SubsystemDemand(
                                    subsystem_name="action_selection",
                                    requested_priority=0.6,
                                    current_load=0.4,
                                    processing_complexity=len(available_actions) / 10.0,
                                    urgency_level=4,  # Always critical for real-time decisions
                                    justification="Critical real-time action selection",
                                    context_data={"available_actions": len(available_actions)}
                                )
                                subsystem_demands.append(action_demand)

                                # Strategy discovery demand (if patterns detected)
                                if learning_insights and learning_insights.get('patterns_detected'):
                                    strategy_demand = SubsystemDemand(
                                        subsystem_name="strategy_discovery",
                                        requested_priority=0.5,
                                        current_load=0.3,
                                        processing_complexity=0.7,
                                        urgency_level=3,
                                        justification="New patterns detected, strategy analysis needed",
                                        context_data={"patterns_count": len(learning_insights.get('patterns_detected', []))}
                                    )
                                    subsystem_demands.append(strategy_demand)

                                # Allocate attention resources
                                attention_allocation = await self.attention_controller.allocate_attention_resources(
                                    real_game_id, subsystem_demands, game_context
                                )

                                # Route communications between systems
                                if learning_insights and learning_insights.get('patterns_detected'):
                                    # Send pattern detection message to strategy discovery
                                    await self.communication_system.route_message(
                                        "real_time_learning", "strategy_discovery", "pattern_detected",
                                        learning_insights, MessagePriority.HIGH, real_game_id
                                    )

                                if learning_insights and learning_insights.get('strategy_adjustments'):
                                    # Send strategy adjustment to action selector
                                    await self.communication_system.route_message(
                                        "strategy_discovery", "action_selection", "strategy_adjustment",
                                        learning_insights.get('strategy_adjustments'), MessagePriority.HIGH, real_game_id
                                    )

                                # Monitor resource usage (simplified)
                                resource_usage = [
                                    ResourceUsage("real_time_learning", 0.3, 50.0, 0.05, 2, 20.0, 0.0, False),
                                    ResourceUsage("action_selection", 0.4, 30.0, 0.02, 1, 50.0, 0.0, False),
                                    ResourceUsage("strategy_discovery", 0.2, 40.0, 0.1, 3, 10.0, 0.0, False)
                                ]

                                monitoring_results = await self.attention_controller.monitor_subsystem_loads(
                                    real_game_id, resource_usage
                                )

                                # Log coordination results
                                if attention_allocation and attention_allocation.allocations:
                                    priority_info = ", ".join([f"{k}:{v:.2f}" for k, v in attention_allocation.allocations.items()])
                                    print(f" ⚡ ATTENTION ALLOCATED: {priority_info}")

                                if monitoring_results.get("bottlenecks_detected"):
                                    print(f" ⚠️  BOTTLENECKS: {len(monitoring_results['bottlenecks_detected'])} detected")

                            except Exception as e:
                                logger.warning(f"Attention + communication coordination error: {e}")

                        # CONTEXT-DEPENDENT FITNESS EVOLUTION INTEGRATION (TIER 2)
                        # Evaluate contextual fitness and evolve success criteria
                        if self._fitness_evolution_initialized and self.fitness_evolution_system:
                            try:
                                # Create context for fitness evaluation
                                fitness_context = {
                                    'game_state': game_state_str,
                                    'action_count': actions_taken,
                                    'current_score': new_score,
                                    'score_change': score_change,
                                    'available_actions': available_actions,
                                    'current_level': current_level,
                                    'coordinates_tried': action_coordinates,
                                    'total_possible_coordinates': 100,  # Estimate
                                    'unique_actions_tried': set(action_sequence),
                                    'context_changes_detected': 1 if frame_changes_detected else 0,
                                    'target_score': 100.0  # Default target, could be dynamic
                                }

                                # Create performance data from current systems
                                performance_data = {
                                    'total_score_improvement': new_score,
                                    'patterns_detected': len(learning_insights.get('patterns_detected', [])) if learning_insights else 0,
                                    'patterns_validated': 0,  # Would need tracking system
                                    'avg_pattern_confidence': 0.7 if learning_insights and learning_insights.get('patterns_detected') else 0.0,
                                    'strategies_discovered': 1 if learning_insights and learning_insights.get('strategy_adjustments') else 0,
                                    'avg_strategy_effectiveness': 0.6 if score_change > 0 else 0.3,
                                    'novel_approaches_tried': min(actions_taken // 10, 5),  # Estimate
                                    'learning_events_triggered': len(learning_insights.get('new_hypotheses', [])) if learning_insights else 0,
                                    'adaptations_made': len(learning_insights.get('strategy_adjustments', [])) if learning_insights else 0,
                                    'avg_adaptation_latency': 5.0,  # Estimate
                                    'adaptation_effectiveness': 0.7 if score_change > 0 else 0.4,
                                    'goals_achieved': 1 if current_state == 'WON' else 0,
                                    'goals_attempted': 1,
                                    'level_completions': current_level - 1,
                                    'performance_consistency': 0.6,  # Could be calculated from score variance
                                    'cross_game_applications': 0,  # Would need cross-game tracking
                                    'generalization_success_rate': 0.5,  # Default
                                    'novel_context_performance': 0.5 if movement_detected else 0.3
                                }

                                # Perform contextual fitness evaluation
                                fitness_evaluation = await self.fitness_evolution_system.evaluate_contextual_fitness(
                                    real_game_id,
                                    f"session_{session_count}",
                                    fitness_context,
                                    performance_data
                                )

                                # Request attention allocation based on fitness priorities if available
                                if fitness_evaluation and fitness_evaluation.individual_scores:
                                    fitness_priorities = fitness_evaluation.individual_scores
                                    attention_request = await self.fitness_evolution_system.request_attention_allocation(
                                        real_game_id, f"session_{session_count}", fitness_priorities
                                    )

                                    if attention_request:
                                        print(f" 🎯 FITNESS EVOLUTION: Composite score {fitness_evaluation.composite_fitness_score:.3f}, " +
                                              f"phase: {fitness_evaluation.learning_phase.value}")

                                        if fitness_evaluation.predicted_improvement_areas:
                                            improvement_areas = ", ".join(fitness_evaluation.predicted_improvement_areas[:3])
                                            print(f" 📈 IMPROVEMENT FOCUS: {improvement_areas}")

                            except Exception as e:
                                logger.warning(f"Fitness evolution integration error: {e}")

                        # NEAT-BASED ARCHITECT SYSTEM INTEGRATION (TIER 2)
                        # Evolve system architecture based on performance patterns
                        if self._neat_architect_initialized and self.neat_architect_system:
                            try:
                                # Determine if architectural evolution should be triggered
                                # Trigger evolution periodically or when performance patterns suggest it
                                should_trigger_evolution = (actions_taken % 50 == 0 or  # Every 50 actions
                                                           score_change < 0 or          # On score decrease
                                                           (learning_insights and       # When patterns detected
                                                            learning_insights.get('patterns_detected')))

                                if should_trigger_evolution:
                                    # Create evolution context
                                    evolution_context = {
                                        'game_state': game_state_str,
                                        'action_count': actions_taken,
                                        'current_score': new_score,
                                        'score_change': score_change,
                                        'current_level': current_level,
                                        'performance_metrics': performance_data if 'performance_data' in locals() else {},
                                        'learning_insights': learning_insights if learning_insights else {},
                                        'available_actions': available_actions,
                                        'patterns_detected': len(learning_insights.get('patterns_detected', [])) if learning_insights else 0,
                                        'recent_effectiveness': 0.7 if score_change > 0 else 0.3
                                    }

                                    # Perform architectural evolution
                                    evolution_results = await self.neat_architect_system.evolve_architecture(
                                        real_game_id, f"session_{session_count}"
                                    )

                                    # Evaluate module effectiveness based on current performance
                                    module_effectiveness = {
                                        'real_time_learning': 0.8 if learning_insights and learning_insights.get('patterns_detected') else 0.4,
                                        'action_selection': 0.9 if score_change > 0 else 0.5,
                                        'strategy_discovery': 0.7 if learning_insights and learning_insights.get('strategy_adjustments') else 0.3,
                                        'attention_control': 0.6,  # Baseline effectiveness
                                        'fitness_evolution': 0.7 if fitness_evaluation and fitness_evaluation.composite_fitness_score > 0.5 else 0.4
                                    }

                                    # Update module effectiveness tracking
                                    await self.neat_architect_system.update_module_effectiveness(
                                        real_game_id, f"session_{session_count}", module_effectiveness
                                    )

                                    # Check for module pruning opportunities
                                    if evolution_results and evolution_results.get('modules_pruned'):
                                        pruned_modules = evolution_results['modules_pruned']
                                        print(f" 🧬 NEAT EVOLUTION: {len(pruned_modules)} ineffective modules pruned")

                                    # Check for new module creations
                                    if evolution_results and evolution_results.get('modules_created'):
                                        new_modules = evolution_results['modules_created']
                                        print(f" 🧬 NEAT EVOLUTION: {len(new_modules)} new modules created")

                                    # Apply architectural changes if beneficial
                                    if evolution_results and evolution_results.get('fitness_improvement'):
                                        fitness_improvement = evolution_results['fitness_improvement']
                                        if fitness_improvement > 0.05:  # 5% improvement threshold
                                            print(f" 🧬 NEAT ARCHITECTURE: {fitness_improvement:.2%} fitness improvement, " +
                                                  f"generation {evolution_results.get('generation', 'unknown')}")

                                    # Request attention allocation for architectural planning if needed
                                    if evolution_results and evolution_results.get('requires_attention'):
                                        architect_demand = SubsystemDemand(
                                            subsystem_name="neat_architect",
                                            requested_priority=0.3,
                                            current_load=0.2,
                                            processing_complexity=0.6,
                                            urgency_level=2,
                                            justification="Architectural evolution planning required",
                                            context_data={"evolution_phase": evolution_results.get('phase', 'unknown')}
                                        )

                                        if hasattr(self, 'attention_controller') and self.attention_controller:
                                            architect_allocation = await self.attention_controller.allocate_attention_resources(
                                                real_game_id, [architect_demand], game_context
                                            )

                            except Exception as e:
                                logger.warning(f"NEAT architect system integration error: {e}")

                        # TIER 3 INTEGRATION: BAYESIAN INFERENCE + ENHANCED GRAPH TRAVERSAL
                        # Advanced probabilistic reasoning and intelligent navigation through pattern spaces
                        if (self._bayesian_inference_initialized and self.bayesian_inference_system and
                            self._graph_traversal_initialized and self.graph_traversal_system):
                            try:
                                # Determine reasoning complexity based on current situation
                                reasoning_complexity = 0.3  # Base complexity
                                if score_change < 0:
                                    reasoning_complexity += 0.2  # Increase for negative outcomes
                                if learning_insights and learning_insights.get('patterns_detected'):
                                    reasoning_complexity += 0.1 * len(learning_insights.get('patterns_detected', []))
                                if len(available_actions) > 5:
                                    reasoning_complexity += 0.1  # Complex decision space

                                reasoning_complexity = min(1.0, reasoning_complexity)  # Cap at 1.0

                                # BAYESIAN INFERENCE: Create and validate hypotheses about game mechanics
                                current_context = {
                                    'game_state': game_state_str,
                                    'action_count': actions_taken,
                                    'current_score': new_score,
                                    'score_change': score_change,
                                    'current_level': current_level,
                                    'available_actions': len(available_actions),
                                    'patterns_detected': len(learning_insights.get('patterns_detected', [])) if learning_insights else 0
                                }

                                # Create hypotheses for current action outcomes
                                if isinstance(action_to_take, dict) and 'id' in action_to_take:
                                    action_id = action_to_take['id']

                                    # Create action-outcome hypothesis
                                    if score_change != 0:  # Only for meaningful outcomes
                                        hypothesis_desc = f"Action {action_id} in context {current_level} leads to score change {score_change}"

                                        action_hypothesis_id = await self.bayesian_inference_system.create_hypothesis(
                                            hypothesis_type=HypothesisType.ACTION_OUTCOME,
                                            description=hypothesis_desc,
                                            prior_probability=0.5,  # Neutral prior
                                            context_conditions={
                                                'action_id': action_id,
                                                'level_range': [current_level-1, current_level+1],
                                                'score_range': [new_score-20, new_score+20]
                                            },
                                            game_id=real_game_id,
                                            session_id=f"session_{session_count}"
                                        )

                                        # Add evidence for this hypothesis
                                        evidence_strength = min(1.0, abs(score_change) / 10.0)  # Stronger evidence for bigger changes
                                        await self.bayesian_inference_system.add_evidence(
                                            hypothesis_id=action_hypothesis_id,
                                            evidence_type=EvidenceType.DIRECT_OBSERVATION,
                                            supports_hypothesis=(score_change > 0),
                                            strength=evidence_strength,
                                            context=current_context,
                                            game_id=real_game_id,
                                            session_id=f"session_{session_count}"
                                        )

                                    # Create coordinate effectiveness hypothesis for Action 6
                                    if action_id == 6 and 'x' in action_to_take and 'y' in action_to_take:
                                        coord_x, coord_y = action_to_take['x'], action_to_take['y']
                                        coord_hypothesis_desc = f"Coordinate ({coord_x},{coord_y}) is effective in level {current_level} contexts"

                                        coord_hypothesis_id = await self.bayesian_inference_system.create_hypothesis(
                                            hypothesis_type=HypothesisType.COORDINATE_EFFECTIVENESS,
                                            description=coord_hypothesis_desc,
                                            prior_probability=0.4,  # Slightly pessimistic prior
                                            context_conditions={
                                                'x': coord_x,
                                                'y': coord_y,
                                                'level': current_level,
                                                'action_type': 6
                                            },
                                            game_id=real_game_id,
                                            session_id=f"session_{session_count}"
                                        )

                                        # Add coordinate evidence
                                        coord_evidence_strength = min(1.0, max(0.1, abs(score_change) / 15.0))
                                        await self.bayesian_inference_system.add_evidence(
                                            hypothesis_id=coord_hypothesis_id,
                                            evidence_type=EvidenceType.DIRECT_OBSERVATION,
                                            supports_hypothesis=(score_change > 0),
                                            strength=coord_evidence_strength,
                                            context=current_context,
                                            game_id=real_game_id,
                                            session_id=f"session_{session_count}"
                                        )

                                # Generate probabilistic predictions for next actions
                                if len(available_actions) > 1:  # Only if there are choices
                                    # Convert available actions to action candidates
                                    action_candidates = []
                                    for act_id in available_actions:
                                        if act_id == 6:
                                            # For Action 6, create several coordinate options
                                            for x in range(10, 55, 15):  # Sample coordinates
                                                for y in range(10, 55, 15):
                                                    action_candidates.append({'id': 6, 'x': x, 'y': y})
                                                    if len(action_candidates) >= 10:  # Limit candidates
                                                        break
                                                if len(action_candidates) >= 10:
                                                    break
                                        else:
                                            action_candidates.append({'id': act_id})

                                    # Get Bayesian prediction
                                    bayesian_prediction = await self.bayesian_inference_system.generate_prediction(
                                        action_candidates=action_candidates,
                                        current_context=current_context,
                                        game_id=real_game_id,
                                        session_id=f"session_{session_count}"
                                    )

                                    if bayesian_prediction:
                                        print(f" 🎯 BAYESIAN PREDICTION: Action {bayesian_prediction.predicted_action.get('id', 'unknown')} " +
                                              f"(probability: {bayesian_prediction.success_probability:.2f}, " +
                                              f"confidence: {bayesian_prediction.confidence_level:.2f})")

                                        if bayesian_prediction.uncertainty_factors:
                                            uncertainty_summary = ", ".join(bayesian_prediction.uncertainty_factors[:2])
                                            print(f" ❓ UNCERTAINTY: {uncertainty_summary}")

                                # ENHANCED GRAPH TRAVERSAL: Model decision trees and pattern spaces
                                traversal_complexity = 0.2  # Base complexity
                                if learning_insights and learning_insights.get('patterns_detected'):
                                    traversal_complexity += 0.1 * len(learning_insights.get('patterns_detected', []))
                                if len(available_actions) > 3:
                                    traversal_complexity += 0.2  # Complex action space

                                traversal_complexity = min(1.0, traversal_complexity)

                                # Create or update game state graph
                                game_state_graph_id = f"game_state_{real_game_id}"

                                # Check if graph exists, if not create it
                                if game_state_graph_id not in self.graph_traversal_system.graphs:
                                    from src.core.enhanced_graph_traversal import GraphNode, GraphEdge

                                    # Create initial game state nodes
                                    initial_nodes = []

                                    # Current state node
                                    current_state_node = GraphNode(
                                        node_id=f"state_{actions_taken}",
                                        node_type=NodeType.STATE_NODE,
                                        properties={
                                            'score': new_score,
                                            'level': current_level,
                                            'actions_taken': actions_taken,
                                            'game_state': game_state_str
                                        },
                                        coordinates=(actions_taken / 100.0, new_score / 100.0)  # Normalize for visualization
                                    )
                                    initial_nodes.append(current_state_node)

                                    # Action nodes for available actions
                                    action_nodes = []
                                    for i, act_id in enumerate(available_actions[:5]):  # Limit to first 5 actions
                                        action_node = GraphNode(
                                            node_id=f"action_{actions_taken}_{act_id}",
                                            node_type=NodeType.ACTION_NODE,
                                            properties={'action_id': act_id, 'available_from_state': f"state_{actions_taken}"},
                                            coordinates=(actions_taken / 100.0 + 0.01, (new_score + i * 5) / 100.0)
                                        )
                                        action_nodes.append(action_node)
                                        initial_nodes.append(action_node)

                                    await self.graph_traversal_system.create_graph(
                                        graph_type=GraphType.GAME_STATE_GRAPH,
                                        initial_nodes=initial_nodes,
                                        initial_edges=[],
                                        game_id=real_game_id
                                    )

                                # Find optimal path through decision space if we have multiple actions
                                if len(available_actions) > 1 and actions_taken > 5:  # Only after some history
                                    try:
                                        # Try to find path from early successful state to current state
                                        early_state_id = f"state_{max(0, actions_taken - 10)}"
                                        current_state_id = f"state_{actions_taken}"

                                        optimal_paths = await self.graph_traversal_system.find_optimal_paths(
                                            graph_id=game_state_graph_id,
                                            start_node=early_state_id,
                                            end_node=current_state_id,
                                            max_alternatives=2
                                        )

                                        if optimal_paths:
                                            best_path = optimal_paths[0]
                                            print(f" 🗺️  GRAPH TRAVERSAL: Found optimal path with {len(best_path.nodes)} states, " +
                                                  f"weight: {best_path.total_weight:.2f}")

                                            if len(optimal_paths) > 1:
                                                alternative_path = optimal_paths[1]
                                                print(f" 🗺️  ALTERNATIVE PATH: {len(alternative_path.nodes)} states, " +
                                                      f"weight: {alternative_path.total_weight:.2f}")

                                    except Exception as e:
                                        logger.debug(f"Graph traversal path finding error: {e}")

                                # Request attention allocation for complex reasoning
                                if reasoning_complexity > 0.6 or traversal_complexity > 0.6:
                                    # Request attention for Bayesian reasoning
                                    if reasoning_complexity > 0.6:
                                        bayesian_attention = await self.bayesian_inference_system.request_attention_allocation(
                                            real_game_id, f"session_{session_count}", reasoning_complexity
                                        )

                                    # Request attention for graph traversal
                                    if traversal_complexity > 0.6:
                                        traversal_attention = await self.graph_traversal_system.request_attention_allocation(
                                            real_game_id, f"session_{session_count}", traversal_complexity
                                        )

                                # Log Tier 3 system performance
                                if reasoning_complexity > 0.5 or traversal_complexity > 0.5:
                                    insights = await self.bayesian_inference_system.get_hypothesis_insights(real_game_id)

                                    if insights:
                                        active_hypotheses = insights.get('total_hypotheses', 0)
                                        high_confidence = len(insights.get('high_confidence_hypotheses', []))
                                        print(f" 🧠 TIER 3 REASONING: {active_hypotheses} hypotheses active, " +
                                              f"{high_confidence} high-confidence, " +
                                              f"reasoning complexity: {reasoning_complexity:.2f}")

                            except Exception as e:
                                logger.warning(f"Tier 3 integration error: {e}")

                        # Check for level completion (significant score increase)
                        score_increase_since_level = total_score - last_significant_score
                        if score_increase_since_level >= level_completion_threshold and len(level_action_sequence) >= 3:
                            # Level completed - learn from level-specific pattern
                            if self.action_selector and hasattr(self.action_selector, 'strategy_discovery_system'):
                                if self.action_selector.strategy_discovery_system:
                                    try:
                                        level_strategy = await self.action_selector.strategy_discovery_system.discover_winning_strategy(
                                            game_id=f"{real_game_id}_level_{current_level}",
                                            action_sequence=level_action_sequence.copy(),
                                            score_progression=level_score_progression.copy()
                                        )
                                        if level_strategy:
                                            print(f" LEVEL {current_level} PATTERN LEARNED: {level_strategy.strategy_id} - "
                                                 f"{len(level_action_sequence)} actions, +{score_increase_since_level:.1f} score")
                                    except Exception as e:
                                        print(f"[WARNING] Failed to learn from level {current_level} strategy: {e}")

                            # Reset for next level
                            current_level += 1
                            last_significant_score = total_score
                            level_action_sequence = []
                            level_score_progression = [total_score]

                        # Check for level failure/stagnation (NEW)
                        elif len(level_action_sequence) >= 15:  # After 15 actions on current level
                            recent_score_change = total_score - (level_score_progression[-10] if len(level_score_progression) > 10 else level_score_progression[0])
                            if recent_score_change <= 5.0:  # Less than 5 points progress in recent actions
                                # Level appears stuck - learn from this failure pattern
                                if self.action_selector and hasattr(self.action_selector, 'strategy_discovery_system'):
                                    if self.action_selector.strategy_discovery_system:
                                        try:
                                            await self._learn_from_level_failure(
                                                self.action_selector.strategy_discovery_system,
                                                real_game_id, current_level, "STAGNATION",
                                                level_action_sequence.copy(), level_score_progression.copy(),
                                                last_significant_score
                                            )
                                            print(f" LEVEL {current_level} STAGNATION DETECTED: Trying new approach")
                                        except Exception as e:
                                            print(f"[WARNING] Failed to learn from level stagnation: {e}")

                                # Reset level tracking to try different approach
                                level_action_sequence = []
                                level_score_progression = [total_score]

                        # Display action with intelligent selection info
                        action_display = f"Action {actions_taken}: {action_to_take}"
                        if isinstance(action_to_take, dict) and 'reason' in action_to_take:
                            action_display += f" ({action_to_take['reason']})"
                        print(f"   {action_display} -> Score: {total_score} (+{score_change})")
                        
                        # Update game state for next iteration
                        game_state = {
                            'game_id': real_game_id,
                            'frame': frame,
                            'state': game_state_str,
                            'score': new_score,
                            'available_actions': available_actions
                        }
                        
                        # Check for win condition
                        if game_state_str == 'WIN':
                            game_won = True
                            print(f" GAME WON! Final score: {total_score}")

                            # Learn from winning strategy
                            if self.action_selector and hasattr(self.action_selector, 'strategy_discovery_system'):
                                if self.action_selector.strategy_discovery_system and len(action_sequence) >= 3:
                                    try:
                                        winning_strategy = await self.action_selector.strategy_discovery_system.discover_winning_strategy(
                                            game_id=real_game_id,
                                            action_sequence=action_sequence,
                                            score_progression=score_progression
                                        )
                                        if winning_strategy:
                                            print(f" WIN PATTERN LEARNED: {winning_strategy.strategy_id} - "
                                                 f"{len(action_sequence)} actions, efficiency: {winning_strategy.efficiency:.2f}")
                                    except Exception as e:
                                        print(f"[WARNING] Failed to learn from winning strategy: {e}")

                            break
                        
                        # Small delay between actions (dynamic pausing handles rate limits)
                        await asyncio.sleep(0.1)  # 100ms delay
                    else:
                        print(f"   Action {actions_taken}: {action_to_take} -> Failed")
                        break
                        
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"   Action {actions_taken}: Error - {e}\n{tb}")
                    logger.exception(f"Exception during action loop: {e}")
                    break

            # Learn from game outcome (success OR failure)
            await self._analyze_game_outcome(
                real_game_id, game_won, current_state, total_score, actions_taken,
                action_sequence, action_coordinates, score_progression, current_level,
                level_action_sequence, level_score_progression, last_significant_score
            )

            # Close scorecard
            try:
                await self.api_manager.close_scorecard(scorecard_id)
            except Exception as e:
                logger.warning(f"Error closing scorecard: {e}")
            
            # Close API manager to prevent resource leaks
            try:
                await self.api_manager.close()
            except Exception as e:
                logger.warning(f"Error closing API manager: {e}")
            
            # Save game result to database
            try:
                from src.database.system_integration import get_system_integration
                integration = get_system_integration()

                await integration.save_game_result(
                    game_id=real_game_id,
                    session_id=f"session_{session_count}",
                    final_score=total_score,
                    total_actions=actions_taken,
                    win_detected=game_won,
                    final_state=current_state,
                    termination_reason="COMPLETED" if game_won else "TIMEOUT"
                )
                print(f" Game result saved to database: {real_game_id}")
            except Exception as e:
                print(f"[WARNING] Failed to save game result to database: {e}")

            # REAL-TIME LEARNING ENGINE: Finalize game context (Phase 1.1)
            if (self._real_time_learning_initialized and self.real_time_learner and
                hasattr(self, '_game_real_time_context') and self._game_real_time_context):
                try:
                    learning_summary = await self.real_time_learner.finalize_game_context(real_game_id)
                    if learning_summary:
                        patterns_count = learning_summary.get('active_patterns_count', 0)
                        adjustments_count = learning_summary.get('active_adjustments_count', 0)
                        learning_events = learning_summary.get('performance_metrics', {}).get('learning_events', 0)
                        print(f" 🧠 Real-time learning summary: {patterns_count} patterns, "
                              f"{adjustments_count} adjustments, {learning_events} learning events")
                except Exception as e:
                    logger.warning(f"Failed to finalize real-time learning context: {e}")

            # ENHANCED ATTENTION + COMMUNICATION: Finalize game context (TIER 1)
            if (self._attention_communication_initialized and self.attention_controller and
                self.communication_system and hasattr(self, '_game_attention_context')):
                try:
                    # Get final attention performance metrics
                    attention_metrics = await self.attention_controller.get_performance_metrics()

                    # Get final communication statistics
                    communication_stats = await self.communication_system.get_communication_stats(real_game_id)

                    # Finalize communication system
                    communication_summary = await self.communication_system.finalize_communication_system(real_game_id)

                    if attention_metrics or communication_stats:
                        allocations_made = attention_metrics.get('total_allocations', 0)
                        messages_routed = communication_stats.total_messages if hasattr(communication_stats, 'total_messages') else 0
                        pathways_count = communication_stats.pathway_count if hasattr(communication_stats, 'pathway_count') else 0
                        print(f" ⚡ Attention + communication summary: {allocations_made} allocations, "
                              f"{messages_routed} messages routed, {pathways_count} pathways")

                except Exception as e:
                    logger.warning(f"Failed to finalize attention + communication systems: {e}")

            # Create real training result
            result = {
                'game_id': real_game_id,
                'session_count': session_count,
                'max_actions': max_actions_per_game,
                'actions_taken': actions_taken,
                'score': total_score,
                'win': game_won,
                'final_state': current_state,
                'training_completed': True,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"[OK] Real training completed: {actions_taken} actions, score {total_score}, win: {game_won}")
            return result
            
        except Exception as e:
            logger.error(f"Error in start_training_with_direct_control: {e}")
            return {
                'game_id': game_id,
                'error': str(e),
                'training_completed': False
            }
    
    def _choose_smart_action(self, available_actions: List[int], game_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Choose a smart action based on available actions and game state."""
        try:
            # Prefer simple actions first (1-5, 7)
            simple_actions = [1, 2, 3, 4, 5, 7]
            for action in simple_actions:
                if action in available_actions:
                    return {'id': action}
            
            # If ACTION6 is available, use it with smart targeting
            if 6 in available_actions:
                # Analyze frame for targeting
                frame_data = game_response.frame
                if frame_data is not None and len(frame_data) > 0:
                    target = self._find_target_coordinates(frame_data[0])
                    if target:
                        return {
                            'id': 6,
                            'x': target[0],
                            'y': target[1]
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error choosing action: {e}")
            return None
    
    def _find_target_coordinates(self, frame: List[List[int]]) -> Optional[Tuple[int, int]]:
        """Find target coordinates for ACTION6 based on frame analysis."""
        try:
            # Simple strategy: find non-zero cells (potential interactive elements)
            for y in range(len(frame)):
                for x in range(len(frame[y])):
                    if frame[y][x] != 0:  # Non-background cell
                        return (x, y)
            
            # Fallback: center of grid
            return (32, 32)
            
        except Exception as e:
            logger.error(f"Error finding target coordinates: {e}")
            return (32, 32)  # Safe fallback
    
    def _initialize_components(self) -> None:
        """Initialize all modular components."""
        try:
            # Memory management
            self.memory_manager = MemoryManager()
            self.action_memory = ActionMemoryManager(self.memory_manager)
            self.pattern_memory = PatternMemoryManager(self.memory_manager)

            # Session management
            session_config = TrainingSessionConfig(
                max_actions=2000,
                timeout_seconds=300,
                enable_position_tracking=True,
                enable_memory_tracking=True,
                enable_performance_tracking=True
            )
            self.session_manager = TrainingSessionManager(session_config)

            # API management (will be initialized when needed)
            self.api_manager = APIManager(self.api_key)
            self._api_initialized = False
            self.action_selector = None

            # Performance monitoring
            self.performance_monitor = UnifiedPerformanceMonitor()
            self.metrics_collector = MetricsCollector()

            # Governor and meta-cognitive systems
            self.governor = TrainingGovernor(persistence_dir=str(self.save_directory))
            self.meta_cognitive = MetaCognitiveController()

            # Learning systems
            self.learning_engine = LearningEngine()
            self.pattern_learner = PatternLearner()
            self.knowledge_transfer = KnowledgeTransfer()

            # Losing streak detection and intervention systems
            # Initialize with database connection when available
            self.losing_streak_detector = None
            self.anti_pattern_learner = None
            self.escalated_intervention_system = None
            self._losing_streak_systems_initialized = False

            # Real-time learning engine components (Phase 1.1)
            # Initialize with database connection when available
            self.real_time_learner = None
            self.pattern_detector = None
            self.strategy_adjuster = None
            self.outcome_tracker = None
            self._real_time_learning_initialized = False

            # Enhanced attention + communication system components (TIER 1)
            # Initialize with database connection when available
            self.attention_controller = None
            self.communication_system = None
            self._attention_communication_initialized = False

            # Context-dependent fitness evolution system components (TIER 2)
            # Initialize with database connection when available
            self.fitness_evolution_system = None
            self._fitness_evolution_initialized = False

            # NEAT-based architect system components (TIER 2)
            # Initialize with database connection when available
            self.neat_architect_system = None
            self._neat_architect_initialized = False

            # Bayesian inference engine components (TIER 3)
            # Initialize with database connection when available
            self.bayesian_inference_system = None
            self._bayesian_inference_initialized = False

            # Enhanced graph traversal system components (TIER 3)
            # Initialize with database connection when available
            self.graph_traversal_system = None
            self._graph_traversal_initialized = False

            # Lazy imports
            self.lazy_imports = LazyImports()

            # API will be initialized when needed (async)

            logger.info("All modular components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    def _initialize_losing_streak_systems(self):
        """Initialize losing streak detection systems with database connection."""
        try:
            if self._losing_streak_systems_initialized:
                return

            # Get database connection from system integration
            from src.database.system_integration import get_system_integration
            integration = get_system_integration()
            db_connection = integration.get_db_connection()

            if db_connection:
                # Initialize anti-pattern learner first (required by intervention system)
                self.anti_pattern_learner = AntiPatternLearner(db_connection)

                # Initialize losing streak detector
                self.losing_streak_detector = LosingStreakDetector(db_connection)

                # Initialize escalated intervention system
                self.escalated_intervention_system = EscalatedInterventionSystem(
                    db_connection, self.anti_pattern_learner
                )

                self._losing_streak_systems_initialized = True
                logger.info("Losing streak detection systems initialized successfully")
            else:
                logger.warning("No database connection available for losing streak systems")

        except Exception as e:
            logger.error(f"Error initializing losing streak systems: {e}")

    def _initialize_real_time_learning_systems(self):
        """Initialize real-time learning engine systems with database connection."""
        try:
            if self._real_time_learning_initialized:
                return

            # Get database connection from system integration
            from src.database.system_integration import get_system_integration
            integration = get_system_integration()
            db_connection = integration.get_db_connection()

            if db_connection:
                # Initialize individual components
                self.pattern_detector = MidGamePatternDetector(db_connection)
                self.strategy_adjuster = DynamicStrategyAdjuster(db_connection)
                self.outcome_tracker = ActionOutcomeTracker(db_connection)

                # Initialize main real-time learner and inject components
                self.real_time_learner = RealTimeLearner(db_connection)
                self.real_time_learner.set_components(
                    self.pattern_detector,
                    self.strategy_adjuster,
                    self.outcome_tracker
                )

                self._real_time_learning_initialized = True
                logger.info("Real-time learning engine initialized successfully")
            else:
                logger.warning("No database connection available for real-time learning engine")
                self._real_time_learning_initialized = False

        except Exception as e:
            logger.error(f"Error initializing real-time learning engine: {e}")
            self._real_time_learning_initialized = False

    def _initialize_attention_communication_systems(self):
        """Initialize enhanced attention + communication systems with database connection."""
        try:
            if self._attention_communication_initialized:
                return

            # Get database connection from system integration
            from src.database.system_integration import get_system_integration
            integration = get_system_integration()
            db_connection = integration.get_db_connection()

            if db_connection:
                # Initialize attention controller
                self.attention_controller = CentralAttentionController(db_connection)

                # Initialize communication system
                self.communication_system = WeightedCommunicationSystem(db_connection)

                # Set communication system on action selector if available
                if self.action_selector and hasattr(self.action_selector, 'set_communication_system'):
                    self.action_selector.set_communication_system(self.communication_system)

                self._attention_communication_initialized = True
                logger.info("Enhanced attention + communication systems initialized successfully")
            else:
                logger.warning("No database connection available for attention + communication systems")
                self._attention_communication_initialized = False

        except Exception as e:
            logger.error(f"Error initializing attention + communication systems: {e}")
            self._attention_communication_initialized = False

    def _initialize_fitness_evolution_system(self):
        """Initialize context-dependent fitness evolution system with database connection."""
        try:
            if self._fitness_evolution_initialized:
                return

            # Get database connection from system integration
            from src.database.system_integration import get_system_integration
            integration = get_system_integration()
            db_connection = integration.get_db_connection()

            if db_connection:
                # Initialize fitness evolution system
                self.fitness_evolution_system = ContextDependentFitnessEvolution(db_connection)

                # Set attention coordination if available
                if (self._attention_communication_initialized and self.attention_controller and
                    self.communication_system):
                    self.fitness_evolution_system.set_attention_coordination(
                        self.attention_controller, self.communication_system
                    )
                    logger.info("Fitness evolution system linked with attention coordination")

                self._fitness_evolution_initialized = True
                logger.info("Context-dependent fitness evolution system initialized successfully")
            else:
                logger.warning("No database connection available for fitness evolution system")
                self._fitness_evolution_initialized = False

        except Exception as e:
            logger.error(f"Error initializing fitness evolution system: {e}")
            self._fitness_evolution_initialized = False

    def _initialize_neat_architect_system(self):
        """Initialize NEAT-based architect system with database connection."""
        try:
            if self._neat_architect_initialized:
                return

            # Get database connection from system integration
            from src.database.system_integration import get_system_integration
            integration = get_system_integration()
            db_connection = integration.get_db_connection()

            if db_connection:
                # Initialize NEAT-based architect system
                self.neat_architect_system = NEATBasedArchitect(db_connection)

                # Set attention coordination if available
                if (self._attention_communication_initialized and self.attention_controller and
                    self.communication_system):
                    self.neat_architect_system.set_attention_coordination(
                        self.attention_controller, self.communication_system
                    )
                    logger.info("NEAT architect system linked with attention coordination")

                # Link with fitness evolution system if available
                if self._fitness_evolution_initialized and self.fitness_evolution_system:
                    self.neat_architect_system.set_fitness_evolution_coordination(
                        self.fitness_evolution_system
                    )
                    logger.info("NEAT architect system linked with fitness evolution")

                self._neat_architect_initialized = True
                logger.info("NEAT-based architect system initialized successfully")
            else:
                logger.warning("No database connection available for NEAT architect system")
                self._neat_architect_initialized = False

        except Exception as e:
            logger.error(f"Error initializing NEAT architect system: {e}")
            self._neat_architect_initialized = False

    def _initialize_bayesian_inference_system(self):
        """Initialize Bayesian inference engine with database connection."""
        try:
            if self._bayesian_inference_initialized:
                return

            # Get database connection from system integration
            from src.database.system_integration import get_system_integration
            integration = get_system_integration()
            db_connection = integration.get_db_connection()

            if db_connection:
                # Initialize Bayesian inference engine
                self.bayesian_inference_system = BayesianInferenceEngine(db_connection)

                # Set attention coordination if available
                if (self._attention_communication_initialized and self.attention_controller and
                    self.communication_system):
                    self.bayesian_inference_system.set_attention_coordination(
                        self.attention_controller, self.communication_system
                    )
                    logger.info("Bayesian inference system linked with attention coordination")

                # Link with fitness evolution system if available
                if self._fitness_evolution_initialized and self.fitness_evolution_system:
                    self.bayesian_inference_system.set_fitness_evolution_coordination(
                        self.fitness_evolution_system
                    )
                    logger.info("Bayesian inference system linked with fitness evolution")

                self._bayesian_inference_initialized = True
                logger.info("Bayesian inference engine initialized successfully")
            else:
                logger.warning("No database connection available for Bayesian inference system")
                self._bayesian_inference_initialized = False

        except Exception as e:
            logger.error(f"Error initializing Bayesian inference system: {e}")
            self._bayesian_inference_initialized = False

    def _initialize_graph_traversal_system(self):
        """Initialize enhanced graph traversal system with database connection."""
        try:
            if self._graph_traversal_initialized:
                return

            # Get database connection from system integration
            from src.database.system_integration import get_system_integration
            integration = get_system_integration()
            db_connection = integration.get_db_connection()

            if db_connection:
                # Initialize enhanced graph traversal system
                self.graph_traversal_system = EnhancedGraphTraversal(db_connection)

                # Set attention coordination if available
                if (self._attention_communication_initialized and self.attention_controller and
                    self.communication_system):
                    self.graph_traversal_system.set_attention_coordination(
                        self.attention_controller, self.communication_system
                    )
                    logger.info("Graph traversal system linked with attention coordination")

                # Link with fitness evolution system if available
                if self._fitness_evolution_initialized and self.fitness_evolution_system:
                    self.graph_traversal_system.set_fitness_evolution_coordination(
                        self.fitness_evolution_system
                    )
                    logger.info("Graph traversal system linked with fitness evolution")

                self._graph_traversal_initialized = True
                logger.info("Enhanced graph traversal system initialized successfully")
            else:
                logger.warning("No database connection available for graph traversal system")
                self._graph_traversal_initialized = False

        except Exception as e:
            logger.error(f"Error initializing graph traversal system: {e}")
            self._graph_traversal_initialized = False

    async def run_continuous_learning(self, max_games: int = None, max_hours: float = 9.0) -> Dict[str, Any]:
        """Run continuous learning with modular components until time limit or game limit reached."""
        try:
            if max_games:
                print(f"[TARGET] Starting continuous learning for {max_games} games")
            else:
                print(f"[TARGET] Starting continuous learning for {max_hours} hours")

            import time
            start_time = time.time()
            max_duration_seconds = max_hours * 3600  # Convert hours to seconds
            
            # Initialize API if not already done
            if not self.api_manager.is_initialized():
                await self.api_manager.initialize()
            
            # Create scorecard for tracking
            scorecard_id = await self.api_manager.create_scorecard(
                f"Continuous Learning {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "Modular continuous learning session"
            )
            # --- FORCE TEST WRITE TO PERSISTENCE HELPERS ---
            from src.database.persistence_helpers import persist_winning_sequence, persist_button_priorities
            # Use obviously non-smoke, non-test data
            await persist_winning_sequence(
                game_id="real_game_test", sequence=[42, 99, 7], frequency=2, avg_score=123.45, success_rate=0.77
            )
            await persist_button_priorities(
                game_type="real_type_test", x=99, y=88, button_type="score_button", confidence=0.55
            )
            # --- END FORCE TEST WRITE ---
            
            results = {
                'games_completed': 0,
                'games_won': 0,
                'total_actions': 0,
                'total_score': 0.0,
                'learning_insights': [],
                'performance_metrics': {}
            }
            
            game_num = 0
            while True:
                # Check time limit first
                elapsed_time = time.time() - start_time
                elapsed_hours = elapsed_time / 3600

                if elapsed_time >= max_duration_seconds:
                    print(f"[TIME LIMIT] Reached {max_hours} hour time limit ({elapsed_hours:.2f} hours elapsed)")
                    break

                # Check game limit if specified
                if max_games and game_num >= max_games:
                    print(f"[GAME LIMIT] Reached {max_games} game limit")
                    break

                if self.shutdown_handler.is_shutdown_requested():
                    print("[STOP] Shutdown requested, stopping continuous learning")
                    break
                
                try:
                    # Create training session
                    session_id = self.session_manager.create_session(f"game_{game_num}")
                    
                    # Run single game
                    game_result = await self._run_single_game(session_id, game_num)
                    
                    # Update results
                    results['games_completed'] += 1
                    if game_result.get('win', False):
                        results['games_won'] += 1
                    results['total_actions'] += game_result.get('actions_taken', 0)
                    results['total_score'] += game_result.get('score', 0.0)
                    
                    # Submit score
                    await self.api_manager.submit_score(
                        f"game_{game_num}",
                        game_result.get('score', 0.0),
                        game_result.get('level', 1),
                        game_result.get('actions_taken', 0),
                        game_result.get('win', False)
                    )
                    
                    # Perform meta-cognitive reflection
                    if self.session_manager.should_reflect(game_result.get('actions_taken', 0)):
                        reflection = self.meta_cognitive.perform_reflection({
                            'game_result': game_result,
                            'session_stats': self.session_manager.get_session_stats(),
                            'performance_metrics': self.performance_monitor.get_performance_report()
                        })
                        results['learning_insights'].extend(reflection.get('insights', []))
                    
                    # Update performance metrics
                    self.performance_monitor.update_metric('total_games_played', 1)
                    self.performance_monitor.update_metric('total_actions_taken', game_result.get('actions_taken', 0))
                    
                    if max_games:
                        print(f"[OK] Game {game_num + 1}/{max_games} completed: {game_result.get('score', 0.0):.2f} score")
                    else:
                        elapsed_hours = (time.time() - start_time) / 3600
                        print(f"[OK] Game {game_num + 1} completed: {game_result.get('score', 0.0):.2f} score ({elapsed_hours:.2f}h elapsed)")
                    
                except Exception as e:
                    logger.error(f"Error in game {game_num}: {e}")
                    continue
                finally:
                    # Increment game counter
                    game_num += 1

                    # Print progress update every 10 games
                    if game_num % 10 == 0:
                        elapsed_hours = (time.time() - start_time) / 3600
                        print(f"[PROGRESS] Completed {game_num} games in {elapsed_hours:.2f} hours ({max_hours - elapsed_hours:.2f} hours remaining)")

            # Final results with timing information
            total_elapsed_time = time.time() - start_time
            total_elapsed_hours = total_elapsed_time / 3600

            results['performance_metrics'] = self.performance_monitor.get_performance_report()
            results['success_rate'] = results['games_won'] / max(results['games_completed'], 1)
            results['total_time_hours'] = total_elapsed_hours
            results['games_per_hour'] = results['games_completed'] / max(total_elapsed_hours, 0.001)

            print(
                f" Continuous learning completed: {results['games_completed']} games, "
                f"{results['success_rate']:.2%} success rate, "
                f"{total_elapsed_hours:.2f} hours, "
                f"{results['games_per_hour']:.1f} games/hour"
            )
            return results
            
        except Exception as e:
            logger.error(f"Error in continuous learning: {e}")
            return {'error': str(e)}
    
    async def _run_single_game(self, session_id: str, game_num: int) -> Dict[str, Any]:
        """Run a single game using modular components."""
        try:
            # Create game
            game_result = await self.api_manager.create_game(f"game_{game_num}")
            if not game_result:
                return {'error': 'Failed to create game', 'win': False, 'score': 0.0, 'actions_taken': 0}
            
            # Reset game
            reset_result = await self.api_manager.reset_game(f"game_{game_num}")
            if not reset_result:
                return {'error': 'Failed to reset game', 'win': False, 'score': 0.0, 'actions_taken': 0}
            
            actions_taken = 0
            max_actions = 1000
            score = 0.0
            win = False
            
            while actions_taken < max_actions and not win:
                if self.shutdown_handler.is_shutdown_requested():
                    break
                
                # Get current game state
                game_state = await self.api_manager.get_game_state(f"game_{game_num}")
                if not game_state:
                    break
                
                # Make decision using governor
                decision = self.governor.make_decision({
                    'game_state': game_state,
                    'actions_taken': actions_taken,
                    'session_id': session_id
                })
                
                # Generate action based on decision
                action = self._generate_action(decision, game_state)
                
                # Submit action
                action_result = await self.api_manager.submit_action(f"game_{game_num}", action)
                if not action_result:
                    break
                
                # Update session
                self.session_manager.update_session_action(session_id, action)
                actions_taken += 1
                
                # Update memory systems
                await self.action_memory.update_action_effectiveness(action, action_result.get('effectiveness', 0.0))
                # Forcefully persist button priorities and winning sequences for every action
                try:
                    from src.database.persistence_helpers import persist_button_priorities, persist_winning_sequence
                    coords = action.get('coordinates', [None, None])
                    await persist_button_priorities(
                        game_type=game_state.get('game_type', 'unknown'),
                        x=coords[0] if coords else None,
                        y=coords[1] if coords else None,
                        button_type=action.get('type', 'unknown'),
                        confidence=action_result.get('effectiveness', 0.0)
                    )
                    # Store every action as a sequence of one for now
                    await persist_winning_sequence(
                        game_id=f"game_{game_num}",
                        sequence=[action.get('id', 0)],
                        frequency=1,
                        avg_score=action_result.get('score', 0.0),
                        success_rate=1.0 if action_result.get('win', False) else 0.0
                    )
                except Exception as e:
                    logger.error(f"Failed to persist gameplay data: {e}")
                
                # Check for win condition
                if action_result.get('win', False):
                    win = True
                    score = action_result.get('score', 0.0)
                    break
                
                # Update performance monitoring
                if self.performance_monitor.should_check_memory(actions_taken):
                    self.performance_monitor.check_memory_usage()
            
            # End session
            self.session_manager.end_session(session_id, 'completed' if win else 'timeout', score, win)
            
            return {
                'win': win,
                'score': score,
                'actions_taken': actions_taken,
                'level': 1
            }
            
        except Exception as e:
            logger.error(f"Error in single game: {e}")
            self.session_manager.end_session(session_id, 'failed', 0.0, False, str(e))
            return {'error': str(e), 'win': False, 'score': 0.0, 'actions_taken': 0}
    
    def _generate_action(self, decision: Dict[str, Any], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an action based on governor decision and game state."""
        # Simple action generation - in a real implementation, this would use
        # the learning engine and pattern learner to generate intelligent actions
        action_type = decision.get('action', 'continue')
        
        if action_type == 'continue':
            # Generate a simple action based on game state
            return {
                'type': 'move',
                'coordinates': [0, 0],
                'direction': 'right'
            }
        elif action_type == 'reduce_memory':
            # Trigger memory cleanup
            self.memory_manager.reset_memory()
            return {'type': 'noop'}
        elif action_type == 'optimize':
            # Trigger optimization
            return {'type': 'optimize'}
        else:
            return {'type': 'noop'}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'memory_status': self.memory_manager.get_memory_status(),
            'session_stats': self.session_manager.get_session_stats(),
            'api_status': self.api_manager.get_status(),
            'performance_report': self.performance_monitor.get_performance_report(),
            'governor_status': self.governor.get_governor_status(),
            'meta_cognitive_status': self.meta_cognitive.get_meta_cognitive_status(),
            'learning_status': self.learning_engine.get_learning_status(),
            'pattern_stats': self.pattern_learner.get_pattern_statistics(),
            'transfer_stats': self.knowledge_transfer.get_transfer_statistics()
        }

    async def _analyze_game_outcome(self, game_id: str, game_won: bool, final_state: str,
                                   final_score: float, actions_taken: int,
                                   action_sequence: List[int], action_coordinates: List[Optional[Tuple[int, int]]],
                                   score_progression: List[float], current_level: int, level_action_sequence: List[Any],
                                   level_score_progression: List[float], last_significant_score: float) -> None:
        """Analyze game outcome and learn from both successes AND failures."""
        try:
            # Initialize losing streak systems if not already done
            self._initialize_losing_streak_systems()

            # Initialize real-time learning systems if not already done
            self._initialize_real_time_learning_systems()

            # Initialize attention + communication systems if not already done
            self._initialize_attention_communication_systems()

            # Initialize fitness evolution system if not already done
            self._initialize_fitness_evolution_system()

            # Initialize NEAT architect system if not already done
            self._initialize_neat_architect_system()

            # Initialize Tier 3 systems if not already done
            self._initialize_bayesian_inference_system()
            self._initialize_graph_traversal_system()

            if not self.action_selector or not hasattr(self.action_selector, 'strategy_discovery_system'):
                return

            strategy_system = self.action_selector.strategy_discovery_system
            if not strategy_system:
                return

            # Extract game type for losing streak tracking
            game_type = strategy_system.game_type_classifier.extract_game_type(game_id)

            # 1. LOSING STREAK DETECTION AND INTERVENTION
            if self._losing_streak_systems_initialized:
                if game_won:
                    # Record success to break any active streak
                    if self.losing_streak_detector:
                        streak_broken = self.losing_streak_detector.record_success(
                            game_type, game_id,
                            break_method=f"winning_strategy_{len(action_sequence)}_actions"
                        )
                        if streak_broken:
                            print(f" LOSING STREAK BROKEN: Success after previous failures")

                    # Record successful patterns to update anti-pattern learning
                    if self.anti_pattern_learner:
                        # Extract real coordinates from action tracking
                        coordinates_used = [coord for coord in action_coordinates if coord is not None]

                        self.anti_pattern_learner.record_pattern_success(
                            game_type, action_sequence, coordinates_used
                        )
                else:
                    # Record failure and check for losing streak
                    failure_type = self._classify_failure_type_for_streak(final_state, actions_taken, final_score)

                    if self.losing_streak_detector:
                        streak_detected, streak_data = self.losing_streak_detector.record_failure(
                            game_type, game_id, failure_type,
                            context_data={
                                "final_score": final_score,
                                "actions_taken": actions_taken,
                                "final_state": final_state,
                                "action_sequence": action_sequence,
                                "score_progression": score_progression
                            }
                        )

                        if streak_detected and streak_data:
                            print(f" LOSING STREAK DETECTED: {streak_data.consecutive_failures} consecutive failures")
                            print(f"   Escalation level: {streak_data.escalation_level.name}")

                            # Apply intervention if needed
                            if self.escalated_intervention_system and streak_data.escalation_level.value > 0:
                                intervention_result = self.escalated_intervention_system.apply_intervention(
                                    streak_data,
                                    {
                                        "game_type": game_type,
                                        "game_id": game_id,
                                        "has_coordinates": 6 in action_sequence,
                                        "recent_actions": action_sequence[-10:] if len(action_sequence) > 10 else action_sequence,
                                        "score_progression": score_progression
                                    }
                                )

                                if intervention_result:
                                    print(f" INTERVENTION APPLIED: {intervention_result.intervention_type.value}")

                    # Analyze failure patterns for anti-pattern learning
                    if self.anti_pattern_learner:
                        # Extract real coordinates from action tracking
                        coordinates_used = [coord for coord in action_coordinates if coord is not None]

                        failure_context = {
                            "final_state": final_state,
                            "final_score": final_score,
                            "actions_taken": actions_taken,
                            "score_progression": score_progression
                        }

                        anti_patterns = self.anti_pattern_learner.analyze_failure(
                            game_type, game_id, action_sequence, coordinates_used, failure_context
                        )

                        if anti_patterns:
                            print(f" ANTI-PATTERNS IDENTIFIED: {len(anti_patterns)} failure patterns learned")

            # 2. GAME-LEVEL ANALYSIS (Original logic)
            if game_won:
                # Success case - analyze win conditions
                print(f" GAME SUCCESS ANALYSIS: {len(action_sequence)} total actions led to win")

                # Analyze win conditions for this successful game
                try:
                    if len(action_sequence) >= 3:  # Need minimum actions for meaningful analysis
                        win_conditions = await strategy_system.analyze_win_conditions(
                            game_id, action_sequence, score_progression
                        )
                        if win_conditions:
                            print(f" WIN CONDITIONS ANALYZED: {len(win_conditions)} conditions identified")

                            # Update win condition frequencies for successful application
                            existing_conditions = await strategy_system.get_win_conditions_for_game_type(game_type)

                            for existing_condition in existing_conditions:
                                # Check if this successful game matches existing conditions
                                await strategy_system.update_win_condition_frequency(
                                    existing_condition['condition_id'], success=True
                                )

                except Exception as e:
                    print(f"[WARNING] Failed to analyze win conditions for successful game: {e}")
            else:
                # FAILURE CASE - NEW LEARNING OPPORTUNITY
                await self._learn_from_game_failure(
                    strategy_system, game_id, final_state, final_score,
                    action_sequence, score_progression, actions_taken
                )

            # 3. LEVEL-SPECIFIC ANALYSIS
            # If we have incomplete level progress when game ends, learn from level failure
            if not game_won and len(level_action_sequence) >= 3:
                # Current level was in progress when game failed
                await self._learn_from_level_failure(
                    strategy_system, game_id, current_level, final_state,
                    level_action_sequence, level_score_progression, last_significant_score
                )

            # 4. PARTIAL SUCCESS ANALYSIS
            # Even in failure, check if any levels were completed during this game
            # (This is already handled in the main loop for level completions)

        except Exception as e:
            print(f"[WARNING] Failed to analyze game outcome: {e}")
            logger.error(f"Error in game outcome analysis: {e}")

    async def _learn_from_game_failure(self, strategy_system, game_id: str, final_state: str,
                                     final_score: float, action_sequence: List[int],
                                     score_progression: List[float], actions_taken: int) -> None:
        """Learn from complete game failures to avoid similar patterns."""
        try:
            # Create failure pattern ID
            failure_id = f"{game_id}_failure"

            # Analyze what went wrong
            failure_type = self._classify_failure_type(final_state, actions_taken, final_score)

            # For now, we'll use the existing discover_winning_strategy method but with negative efficiency
            # to indicate this is a failure pattern. Later, we could extend the system with dedicated failure methods.

            # Calculate "negative efficiency" - how badly this sequence performed
            if len(action_sequence) > 0:
                negative_efficiency = -(final_score / len(action_sequence))  # Negative to indicate failure
            else:
                negative_efficiency = -1.0

            print(f" GAME FAILURE ANALYSIS: {failure_type} after {actions_taken} actions, score {final_score}")
            print(f"   Failure pattern: {action_sequence[-10:] if len(action_sequence) > 10 else action_sequence}")
            print(f"   Avoiding similar sequences in future attempts on {game_id}")

            # Store failure information for future reference
            # This would ideally be stored in a dedicated failure patterns table
            # For now, we'll log it and use it to inform future strategy selection

        except Exception as e:
            logger.error(f"Error learning from game failure: {e}")

    async def _learn_from_level_failure(self, strategy_system, game_id: str, level: int,
                                      final_state: str, level_actions: List[Any],
                                      level_scores: List[float], last_score: float) -> None:
        """Learn from level-specific failures to improve level-specific strategies."""
        try:
            # Create level-specific failure ID
            level_failure_id = f"{game_id}_level_{level}_failure"

            # Analyze level-specific failure
            score_stagnation = len(level_scores) > 1 and (level_scores[-1] - level_scores[0]) < 5.0

            # Format actions with coordinate details for better analysis
            formatted_actions = []
            action_6_coordinates = []

            for action in level_actions:
                if isinstance(action, str) and action.startswith("6("):
                    # Extract coordinates from Action 6
                    formatted_actions.append(action)
                    # Extract x,y from "6(x,y)" format
                    coord_part = action[2:-1]  # Remove "6(" and ")"
                    if ',' in coord_part:
                        try:
                            x, y = map(int, coord_part.split(','))
                            action_6_coordinates.append((x, y))
                        except ValueError:
                            pass
                else:
                    formatted_actions.append(str(action))

            print(f" LEVEL {level} FAILURE ANALYSIS: Failed during level {level}")
            print(f"   Level actions that failed: {formatted_actions}")
            print(f"   Score progress on level: {level_scores[0]:.1f} → {level_scores[-1]:.1f}")

            # Provide more specific hypothesis based on action types
            if score_stagnation:
                if action_6_coordinates:
                    # Analyze coordinate clusters for specific feedback
                    unique_coords = list(set(action_6_coordinates))
                    if len(unique_coords) == 1:
                        print(f"   Hypothesis: Coordinate {unique_coords[0]} appears ineffective for level {level} type puzzles")
                    elif len(unique_coords) <= 3:
                        print(f"   Hypothesis: Coordinate cluster {unique_coords} may not be effective for level {level} type puzzles")
                    else:
                        print(f"   Hypothesis: {len(unique_coords)} different coordinates tried, none effective for level {level} - may need different action types")
                else:
                    print(f"   Hypothesis: Actions {level_actions} may not be effective for level {level} type puzzles")
            else:
                print(f"   Hypothesis: Action sequence was progressing but led to game over - try different approach")

            # Store level-specific failure pattern for future avoidance
            # This helps the system learn "don't do X on level Y of game type Z"

        except Exception as e:
            logger.error(f"Error learning from level failure: {e}")

    def _classify_failure_type(self, final_state: str, actions_taken: int, final_score: float) -> str:
        """Classify the type of failure to better understand what went wrong."""
        if final_state == 'NOT_FINISHED' and actions_taken >= 5000:
            return "TIMEOUT"
        elif final_score <= 0:
            return "ZERO_PROGRESS"
        elif final_score < 50:
            return "LOW_PROGRESS"
        else:
            return "FAILURE_WITH_PROGRESS"

    def _classify_failure_type_for_streak(self, final_state: str, actions_taken: int, final_score: float) -> FailureType:
        """Classify failure type for losing streak detection system."""
        if final_state == 'NOT_FINISHED' and actions_taken >= 5000:
            return FailureType.TIMEOUT
        elif final_score <= 0:
            return FailureType.ZERO_PROGRESS
        elif final_score < 50:
            return FailureType.LOW_PROGRESS
        else:
            return FailureType.GENERAL_FAILURE

    def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            # API cleanup will be handled by the close() method
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def request_shutdown(self) -> None:
        """Request shutdown from external code."""
        self.shutdown_handler.request_shutdown()
        logger.info("External shutdown requested")
    
    async def finish_current_game(self) -> None:
        """Finish the current game and save all data."""
        try:
            if hasattr(self, 'current_game_id') and self.current_game_id:
                from src.database.system_integration import get_system_integration
                integration = get_system_integration()

                # Ensure game is closed in database
                session_id = getattr(self, 'current_session_id', 'unknown')
                await integration.ensure_game_closed(self.current_game_id, session_id)

                logger.info(f"Finished current game: {self.current_game_id}")
        except Exception as e:
            logger.error(f"Error finishing current game: {e}")

    async def finish_current_session(self) -> None:
        """Finish the current session and save all data."""
        try:
            if hasattr(self, 'current_session_id') and self.current_session_id:
                from src.database.system_integration import get_system_integration
                integration = get_system_integration()

                # Ensure session is closed in database
                await integration.ensure_session_closed(self.current_session_id)

                logger.info(f"Finished current session: {self.current_session_id}")
        except Exception as e:
            logger.error(f"Error finishing current session: {e}")

    async def save_scorecard_data(self) -> None:
        """Save scorecard data to database."""
        try:
            if hasattr(self, 'api_manager') and self.api_manager:
                # Get scorecard data from API manager
                scorecard_data = await self.api_manager.get_current_scorecard_data()
                if scorecard_data:
                    # Save to database via system integration
                    from src.database.system_integration import get_system_integration
                    integration = get_system_integration()
                    await integration.save_scorecard_data(scorecard_data)
                    logger.info("Scorecard data saved to database")
                else:
                    logger.warning("No scorecard data to save")
        except Exception as e:
            logger.error(f"Error saving scorecard data: {e}")

    async def close(self) -> None:
        """Close the continuous learning loop."""
        try:
            # Ensure current operations are finished
            await self.finish_current_game()
            await self.finish_current_session()
            await self.save_scorecard_data()

            # Close API manager
            await self.api_manager.close()
            logger.info("Continuous learning loop closed")
        except Exception as e:
            logger.error(f"Error during close: {e}")
            # Still try to close API manager
            try:
                await self.api_manager.close()
            except:
                pass

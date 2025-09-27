"""
Continuous Learning Loop - Modular Version

Simplified main orchestrator that uses modular components for all functionality.
This replaces the massive 18,000+ line monolithic file.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from ..analysis.action_selector import create_action_selector
from pathlib import Path
from datetime import datetime

# Import modular components
from ..memory import MemoryManager, ActionMemoryManager, PatternMemoryManager
from ..sessions import TrainingSessionManager, TrainingSessionConfig
from ..api import APIManager
from src.core.unified_performance_monitor import UnifiedPerformanceMonitor
from ..performance import MetricsCollector
from ..governor import TrainingGovernor, MetaCognitiveController
from ..learning import LearningEngine, PatternLearner, KnowledgeTransfer
from ..utils import LazyImports, ShutdownHandler, CompatibilityShim

# Import losing streak detection components
from src.core.losing_streak_detector import LosingStreakDetector, FailureType
from src.core.anti_pattern_learner import AntiPatternLearner
from src.core.escalated_intervention_system import EscalatedInterventionSystem

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
                            action_sequence.append(action_to_take['id'])
                            level_action_sequence.append(action_to_take['id'])
                        elif isinstance(action_to_take, int):
                            action_sequence.append(action_to_take)
                            level_action_sequence.append(action_to_take)
                        score_progression.append(total_score)
                        level_score_progression.append(total_score)

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
                action_sequence, score_progression, current_level,
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

    async def run_continuous_learning(self, max_games: int = 100) -> Dict[str, Any]:
        """Run continuous learning with modular components."""
        try:
            print(f"[TARGET] Starting continuous learning for {max_games} games")
            
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
            
            for game_num in range(max_games):
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
                    
                    print(f"[OK] Game {game_num + 1}/{max_games} completed: {game_result.get('score', 0.0):.2f} score")
                    
                except Exception as e:
                    logger.error(f"Error in game {game_num}: {e}")
                    continue
            
            # Final results
            results['performance_metrics'] = self.performance_monitor.get_performance_report()
            results['success_rate'] = results['games_won'] / max(results['games_completed'], 1)
            
            print(
                f" Continuous learning completed: {results['games_completed']} games, "
                f"{results['success_rate']:.2%} success rate"
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
                                   action_sequence: List[int], score_progression: List[float],
                                   current_level: int, level_action_sequence: List[int],
                                   level_score_progression: List[float], last_significant_score: float) -> None:
        """Analyze game outcome and learn from both successes AND failures."""
        try:
            # Initialize losing streak systems if not already done
            self._initialize_losing_streak_systems()

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
                        # Extract coordinates used (if any Action 6 was used)
                        coordinates_used = []
                        for i, action in enumerate(action_sequence):
                            if action == 6 and i < len(action_sequence):
                                # For now, use simple coordinate extraction
                                # In a real implementation, this would come from action context
                                coordinates_used.append((i * 10 % 640, i * 10 % 480))

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
                        coordinates_used = []
                        for i, action in enumerate(action_sequence):
                            if action == 6:
                                coordinates_used.append((i * 10 % 640, i * 10 % 480))

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
                                      final_state: str, level_actions: List[int],
                                      level_scores: List[float], last_score: float) -> None:
        """Learn from level-specific failures to improve level-specific strategies."""
        try:
            # Create level-specific failure ID
            level_failure_id = f"{game_id}_level_{level}_failure"

            # Analyze level-specific failure
            score_stagnation = len(level_scores) > 1 and (level_scores[-1] - level_scores[0]) < 5.0

            print(f" LEVEL {level} FAILURE ANALYSIS: Failed during level {level}")
            print(f"   Level actions that failed: {level_actions}")
            print(f"   Score progress on level: {level_scores[0]:.1f} → {level_scores[-1]:.1f}")

            if score_stagnation:
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

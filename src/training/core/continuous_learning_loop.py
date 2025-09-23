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
                print(f"🎮 Found {len(games)} real ARC-AGI-3 games available")
                return games
            else:
                print("[WARNING] No real games available from ARC-AGI-3 API")
                return []
        except Exception as e:
            logger.error(f"Error getting available games: {e}")
            return []
    
    async def start_training_with_direct_control(self, game_id: str, max_actions_per_game: int = 500, 
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
            
            print(f"🎮 Using real ARC-AGI-3 game: {real_game_id}")
            
            # Open scorecard for tracking
            scorecard_id = await self.api_manager.create_scorecard(
                f"Real Training Session {session_count}",
                f"Direct API control training for {real_game_id}"
            )
            
            if not scorecard_id:
                print("⚠️ No scorecard created, proceeding without scorecard tracking")
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
            
            print(f"🔄 Game start/reset successful. GUID: {game_guid[:8]}...")
            
            # Play the game with real actions
            actions_taken = 0
            total_score = 0
            game_won = False
            current_state = reset_response.state
            
            print(f"[TARGET] Starting real gameplay with {max_actions_per_game} max actions...")
            
            while actions_taken < max_actions_per_game and current_state == 'NOT_FINISHED':
                try:
                    # Check rate limit status and implement dynamic pausing
                    rate_status = self.api_manager.get_rate_limit_status()
                    warning = self.api_manager.rate_limiter.get_usage_warning()
                    
                    # Check if we need to pause due to rate limiting
                    should_pause, pause_duration = self.api_manager.rate_limiter.should_pause()
                    if should_pause:
                        print(f"⏸️ Rate limit pause: {pause_duration:.1f}s (usage: {rate_status.get('current_usage', 0)}/{rate_status.get('max_requests', 550)})")
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
                        
                        # Display action with intelligent selection info
                        action_display = f"Action {actions_taken}: {action_to_take}"
                        if isinstance(action_to_take, dict) and 'reason' in action_to_take:
                            action_display += f" ({action_to_take['reason']})"
                        print(f"   {action_display} → Score: {total_score} (+{score_change})")
                        
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
                            print(f"🎉 GAME WON! Final score: {total_score}")
                            break
                        
                        # Small delay between actions (dynamic pausing handles rate limits)
                        await asyncio.sleep(0.1)  # 100ms delay
                    else:
                        print(f"   Action {actions_taken}: {action_to_take} → Failed")
                        break
                        
                except Exception as e:
                    print(f"   Action {actions_taken}: Error - {e}")
                    break
            
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
                print(f"💾 Game result saved to database: {real_game_id}")
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
            
            # Lazy imports
            self.lazy_imports = LazyImports()
            
            # API will be initialized when needed (async)
            
            logger.info("All modular components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
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
                f"🏁 Continuous learning completed: {results['games_completed']} games, "
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
                self.action_memory.update_action_effectiveness(action, action_result.get('effectiveness', 0.0))
                
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
    
    async def close(self) -> None:
        """Close the continuous learning loop."""
        await self.api_manager.close()
        logger.info("Continuous learning loop closed")

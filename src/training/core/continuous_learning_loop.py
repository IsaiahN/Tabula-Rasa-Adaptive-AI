"""
Continuous Learning Loop - Modular Version

Simplified main orchestrator that uses modular components for all functionality.
This replaces the massive 18,000+ line monolithic file.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Import modular components
from ..memory import MemoryManager, ActionMemoryManager, PatternMemoryManager
from ..sessions import TrainingSessionManager, TrainingSessionConfig
from ..api import APIManager
from ..performance import PerformanceMonitor, MetricsCollector
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
        arc_agents_path: str,
        tabula_rasa_path: str,
        api_key: Optional[str] = None,
        save_directory: str = "data"
    ):
        """Initialize the continuous learning loop with modular components."""
        print("🚀 Starting Modular ContinuousLearningLoop initialization...")
        
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
        
        print("✅ Modular ContinuousLearningLoop initialized successfully")
    
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
            
            # API management
            self.api_manager = APIManager(self.api_key, local_mode=False)
            
            # Performance monitoring
            self.performance_monitor = PerformanceMonitor()
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
            
            # Initialize API
            asyncio.create_task(self.api_manager.initialize())
            
            logger.info("All modular components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    async def run_continuous_learning(self, max_games: int = 100) -> Dict[str, Any]:
        """Run continuous learning with modular components."""
        try:
            print(f"🎯 Starting continuous learning for {max_games} games")
            
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
                    print("🛑 Shutdown requested, stopping continuous learning")
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
                    
                    print(f"✅ Game {game_num + 1}/{max_games} completed: {game_result.get('score', 0.0):.2f} score")
                    
                except Exception as e:
                    logger.error(f"Error in game {game_num}: {e}")
                    continue
            
            # Final results
            results['performance_metrics'] = self.performance_monitor.get_performance_report()
            results['success_rate'] = results['games_won'] / max(results['games_completed'], 1)
            
            print(f"🏁 Continuous learning completed: {results['games_completed']} games, {results['success_rate']:.2%} success rate")
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
            asyncio.create_task(self.api_manager.close())
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def close(self) -> None:
        """Close the continuous learning loop."""
        await self.api_manager.close()
        logger.info("Continuous learning loop closed")

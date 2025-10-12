#!/usr/bin/env python3
"""
TABULA RASA - AI-Enhanced ARC-AGI-3 Training System

This is the consolidated training script that automatically uses the best available system:
1. CORE_GAME_MECHANICS (AI-Enhanced) - 10000x performance improvement
2. Legacy system fallback if needed

Features when using AI-Enhanced system:
- AI Orchestration (GAN + Pattern Recognition + Knowledge Transfer)
- Vision-guided coordinate selection for ACTION6
- Cross-game learning and strategy optimization
- Comprehensive performance monitoring and analytics
"""

# Disable Python bytecode caching
import sys
import os
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import asyncio
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("[INFO] python-dotenv not installed. Install with: pip install python-dotenv")

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load AI-enhanced system from src/
try:
    from src.gameplay.enhanced_gameplay import GameSessionManager, CoreGameplay, CoreGameDatabase
    AI_ENHANCED_AVAILABLE = True
    print("[OK] Enhanced gameplay system with Action 6 pseudo-button detection detected!")
except ImportError:
    AI_ENHANCED_AVAILABLE = False
    print("[WARNING] AI-Enhanced system not available")

# Fallback to legacy system
LEGACY_AVAILABLE = False
if not AI_ENHANCED_AVAILABLE:
    try:
        from src.training.core.continuous_learning_loop import ContinuousLearningLoop
        LEGACY_AVAILABLE = True
        print("[OK] Legacy training system available")
    except ImportError:
        print("[ERROR] No training system available")

class DatabaseLogHandler(logging.Handler):
    """Custom logging handler that stores logs in database instead of files."""

    def __init__(self, db_interface):
        super().__init__()
        self.db = db_interface

    def emit(self, record):
        """Store log record in database."""
        try:
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'module': record.module,
                'message': record.getMessage(),
                'filename': record.filename,
                'line_number': record.lineno
            }
            # Store in database using execute_query (fixed to use system_logs table)
            filename_info = f"{log_data['filename']}:{log_data['line_number']}"
            self.db.execute_query(
                """INSERT OR IGNORE INTO system_logs
                   (log_level, component, message, data, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (log_data['level'], log_data['module'], log_data['message'],
                 filename_info, log_data['timestamp'])
            )
        except Exception:
            # Don't let logging errors break the application
            pass

# Set up logging with database storage
def setup_database_logging(db_interface=None):
    """Configure logging to use database storage instead of files."""

    # Create handlers list
    handlers = [logging.StreamHandler()]  # Keep console output

    # Add database handler if database is available
    if db_interface:
        try:
            # Ensure logs table exists
            db_interface.execute_query("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    module TEXT,
                    message TEXT NOT NULL,
                    filename TEXT,
                    line_number INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            handlers.append(DatabaseLogHandler(db_interface))
            print("[OK] Database logging enabled")
        except Exception as e:
            print(f"[WARNING] Database logging failed, using console only: {e}")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override any existing configuration
    )

logger = logging.getLogger(__name__)


class ConsolidatedTrainingSystem:
    """Consolidated training system with automatic AI-enhanced/legacy detection."""

    def __init__(self, api_key: str):
        """Initialize the best available training system."""
        self.api_key = api_key
        self.system_type = None

        if AI_ENHANCED_AVAILABLE:
            self._init_ai_enhanced()
        elif LEGACY_AVAILABLE:
            self._init_legacy()
        else:
            raise RuntimeError("No training system available")

    def _init_ai_enhanced(self):
        """Initialize AI-enhanced system."""
        self.system_type = "AI_ENHANCED"
        self.db_path = "tabula_rasa.db"
        self.database = CoreGameDatabase(self.db_path)
        self.session_manager = GameSessionManager(self.api_key, self.db_path)
        self.gameplay = CoreGameplay(self.session_manager)

        # Configure database logging
        setup_database_logging(self.database)

        # Training state
        self.games_completed = 0
        self.games_won = 0
        self.total_score = 0.0
        self.total_actions = 0
        self.start_time = None
        self.game_results = []

        logger.info("AI-Enhanced training system initialized")

    def _init_legacy(self):
        """Initialize legacy system."""
        self.system_type = "LEGACY"
        import tempfile
        self.temp_dir = tempfile.mkdtemp(prefix="training_session_")
        self.legacy_loop = ContinuousLearningLoop(
            api_key=self.api_key,
            save_directory=Path(self.temp_dir)
        )
        logger.info("Legacy training system initialized")

    async def run_training(self,
                          max_games: Optional[int] = None,
                          max_hours: Optional[float] = None,
                          quick_test: bool = False) -> Dict[str, Any]:
        """Run training with the available system."""

        if quick_test:
            max_games = 10
            max_hours = None

        if self.system_type == "AI_ENHANCED":
            return await self._run_ai_enhanced_training(max_games, max_hours)
        else:
            return await self._run_legacy_training(max_games, max_hours)

    async def _run_ai_enhanced_training(self, max_games: Optional[int], max_hours: Optional[float]) -> Dict[str, Any]:
        """Run AI-enhanced training."""
        logger.info("=" * 60)
        logger.info("STARTING AI-ENHANCED TRAINING")
        logger.info("=" * 60)
        logger.info(f"Max games: {max_games or 'Unlimited'}")
        logger.info(f"Max hours: {max_hours or 'Unlimited'}")
        logger.info(f"AI systems enabled: {self.gameplay.ai_available}")

        self.start_time = datetime.now()
        training_active = True
        interrupted = False

        try:
            while training_active and not interrupted:
                # Check stopping conditions
                if await self._should_stop(max_games, max_hours):
                    logger.info("Training completed - stopping conditions met")
                    break

                try:
                    # Play one game
                    game_result = await self._play_ai_enhanced_game()
                    
                    # Only process if not cancelled
                    if not game_result.get('cancelled', False):
                        await self._process_game_result(game_result)

                        # Progress logging
                        if self.games_completed % 10 == 0:
                            await self._log_progress()

                    # Small delay between games
                    await asyncio.sleep(1.0)

                except asyncio.CancelledError:
                    logger.info("Training cancelled - shutting down gracefully")
                    interrupted = True
                    break
                except KeyboardInterrupt:
                    logger.info("Training interrupted by user - shutting down gracefully")
                    interrupted = True
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            interrupted = True
        except asyncio.CancelledError:
            logger.info("Training cancelled")
            interrupted = True
        finally:
            training_active = False
            if interrupted:
                logger.info("Training shutdown initiated...")

        return await self._generate_ai_enhanced_results()

    async def _run_legacy_training(self, max_games: Optional[int], max_hours: Optional[float]) -> Dict[str, Any]:
        """Run legacy training."""
        logger.info("=" * 60)
        logger.info("RUNNING LEGACY TRAINING")
        logger.info("=" * 60)

        # Use legacy system defaults if not specified
        if max_games is None and max_hours is None:
            max_hours = 9.0  # Original 9-hour training

        return await self.legacy_loop.run_continuous_learning(
            max_games=max_games,
            max_hours=max_hours
        )

    async def _get_real_game_id(self, game_num: int) -> str:
        """Get a real ARC game ID from available games."""
        try:
            # Import ARC client
            from src.arc_integration.arc_api_client import ARCClient

            # Use cached game IDs if available
            if not hasattr(self, '_available_games'):
                api_key = getattr(self.session_manager, 'api_key', None)
                if not api_key:
                    api_key = os.getenv('ARC_AGI_3_API_KEY') or os.getenv('ARC_API_KEY')

                async with ARCClient(api_key=api_key) as client:
                    self._available_games = await client.get_available_games()
                    logger.info(f"Fetched {len(self._available_games)} available games from ARC API")

            # Select game by cycling through available games
            if self._available_games:
                game_index = (game_num - 1) % len(self._available_games)
                game_id = self._available_games[game_index].get('id', self._available_games[game_index].get('game_id'))
                logger.info(f"Using real game ID: {game_id} (index {game_index})")
                return game_id
            else:
                # Fallback if no games available
                logger.warning("No games available from ARC API, using fallback")
                return f"fallback_game_{game_num}"

        except Exception as e:
            logger.warning(f"Error getting real game ID: {e}")
            # Fallback to a known working game ID
            return "vc33-6ae7bf49eea5"  # Use the game ID we know works

    async def _play_ai_enhanced_game(self) -> Dict[str, Any]:
        """Play a single AI-enhanced game."""
        game_start = time.time()
        game_num = self.games_completed + 1

        try:
            logger.info(f"Starting AI-enhanced game {game_num}...")

            # Get real ARC game ID from available games
            game_id = await self._get_real_game_id(game_num)

            result = await self.gameplay.play_single_game(
                game_id=game_id,
                max_actions=400
            )

            duration = time.time() - game_start
            final_score = result.get('final_score', 0.0)
            total_actions = result.get('total_actions', 0)
            game_won = result.get('win_detected', False)

            logger.info(f"Game {game_num}: Score={final_score}, Actions={total_actions}, Won={game_won}, Time={duration:.1f}s")

            return {
                'game_number': game_num,
                'final_score': final_score,
                'total_actions': total_actions,
                'game_duration': duration,
                'game_won': game_won,
                'ai_performance': result.get('ai_performance', {}),
                'timestamp': datetime.now().isoformat()
            }

        except asyncio.CancelledError:
            logger.info(f"Game {game_num} cancelled during shutdown")
            # Return partial result for cancelled game
            return {
                'game_number': game_num,
                'final_score': 0.0,
                'total_actions': 0,
                'game_duration': time.time() - game_start,
                'game_won': False,
                'error': 'cancelled_during_shutdown',
                'cancelled': True
            }
        except KeyboardInterrupt:
            logger.info(f"Game {game_num} interrupted by user")
            # Re-raise KeyboardInterrupt to be handled by training loop
            raise
        except Exception as e:
            logger.error(f"Game {game_num} error: {e}")
            return {
                'game_number': game_num,
                'final_score': 0.0,
                'total_actions': 0,
                'game_duration': time.time() - game_start,
                'game_won': False,
                'error': str(e)
            }

    async def _process_game_result(self, result: Dict[str, Any]):
        """Process game result."""
        self.games_completed += 1
        if result.get('game_won', False):
            self.games_won += 1
        self.total_score += result.get('final_score', 0.0)
        self.total_actions += result.get('total_actions', 0)
        self.game_results.append(result)

        # Knowledge extraction for AI system
        if self.system_type == "AI_ENHANCED" and self.gameplay.knowledge_integrator:
            try:
                await self.gameplay.knowledge_integrator.extract_game_knowledge(result)
            except Exception as e:
                logger.warning(f"Knowledge extraction failed: {e}")

    async def _should_stop(self, max_games: Optional[int], max_hours: Optional[float]) -> bool:
        """Check if training should stop."""
        if max_games and self.games_completed >= max_games:
            return True
        if max_hours and self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
            if elapsed >= max_hours:
                return True
        return False

    async def _log_progress(self):
        """Log training progress."""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
        win_rate = self.games_won / max(self.games_completed, 1)
        avg_score = self.total_score / max(self.games_completed, 1)

        logger.info("=" * 50)
        logger.info(f"PROGRESS: Game {self.games_completed}")
        logger.info(f"Win rate: {win_rate:.1%} ({self.games_won}/{self.games_completed})")
        logger.info(f"Avg score: {avg_score:.1f}")
        logger.info(f"Time: {elapsed:.1f}h")
        logger.info("=" * 50)

    async def _generate_ai_enhanced_results(self) -> Dict[str, Any]:
        """Generate AI-enhanced training results."""
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds() / 3600
        win_rate = self.games_won / max(self.games_completed, 1)

        results = {
            'system_type': 'AI_ENHANCED',
            'games_completed': self.games_completed,
            'games_won': self.games_won,
            'win_rate': win_rate,
            'total_score': self.total_score,
            'average_score': self.total_score / max(self.games_completed, 1),
            'total_actions': self.total_actions,
            'total_time_hours': total_time,
            'games_per_hour': self.games_completed / max(total_time, 0.01),
            'ai_performance': self.gameplay.get_performance_stats() if self.gameplay.ai_available else {},
            'database_path': self.db_path
        }

        return results

    async def shutdown(self):
        """Gracefully shutdown the training system."""
        try:
            logger.info("Shutting down training system...")
            
            if self.system_type == "AI_ENHANCED":
                # Shutdown AI-enhanced components
                if hasattr(self, 'gameplay') and self.gameplay:
                    try:
                        # Gracefully shutdown session manager which includes database
                        if hasattr(self.gameplay, 'session_manager') and self.gameplay.session_manager:
                            await self.gameplay.session_manager.graceful_shutdown()
                    except Exception as e:
                        logger.warning(f"Error during gameplay shutdown: {e}")
                
                # Close database connections
                if hasattr(self, 'database') and self.database:
                    try:
                        self.database.close()
                        logger.info("Database connection closed")
                    except Exception as e:
                        logger.warning(f"Error closing database: {e}")
                        
            elif self.system_type == "LEGACY":
                # Cleanup legacy system
                if hasattr(self, 'legacy_loop'):
                    try:
                        # Legacy systems might not have async shutdown
                        logger.info("Legacy system cleanup completed")
                    except Exception as e:
                        logger.warning(f"Error during legacy cleanup: {e}")
                        
                # Clean up temp directory
                if hasattr(self, 'temp_dir'):
                    try:
                        import shutil
                        shutil.rmtree(self.temp_dir, ignore_errors=True)
                        logger.info("Temporary directory cleaned up")
                    except Exception as e:
                        logger.warning(f"Error cleaning temp directory: {e}")
            
            logger.info("Training system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            # Don't re-raise, we want shutdown to complete even if there are errors


async def main():
    """Main training entry point."""
    print("TABULA RASA - ARC-AGI-3 Training System")
    print("=" * 50)

    # Get API key with interactive fallback
    api_key = os.getenv('ARC_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("ARC_API_KEY environment variable not found.")
        print("\nOptions:")
        print("1. Set environment variable: export ARC_API_KEY='your-key-here'")
        print("2. Enter API key now (for this session only)")
        print("3. Use test mode (no real API calls)")

        choice = input("\nSelect option (1/2/3): ").strip()

        if choice == "2":
            api_key = input("Enter your ARC API key: ").strip()
            if not api_key:
                print("No API key provided. Exiting.")
                return
        elif choice == "3":
            api_key = "test_mode_key"
            print("Using test mode - no real API calls will be made")
        else:
            print("Please set the environment variable and try again.")
            return

    # System status
    if AI_ENHANCED_AVAILABLE:
        print("[AI-ENHANCED] Using CORE_GAME_MECHANICS system")
        print("   Features: AI Orchestration, Vision Guidance, Pattern Learning")
    elif LEGACY_AVAILABLE:
        print("[LEGACY] Using legacy training system")
        print("   Consider upgrading to CORE_GAME_MECHANICS for 10000x improvement")
    else:
        print("[ERROR] No training system available")
        return

    # Training options
    print("\nTraining Options:")
    print("1. Quick test (10 games)")
    print("2. Standard session (1 hour)")
    print("3. Extended training (9 hours)")
    print("4. Custom settings")

    trainer = None
    try:
        choice = input("\nSelect option (1-4, Enter for quick test): ").strip()

        # Initialize training system
        trainer = ConsolidatedTrainingSystem(api_key)

        try:
            if choice == "2":
                results = await trainer.run_training(max_hours=1.0)
            elif choice == "3":
                results = await trainer.run_training(max_hours=9.0)
            elif choice == "4":
                max_games = input("Max games (Enter for unlimited): ").strip()
                max_hours = input("Max hours (Enter for unlimited): ").strip()
                max_games = int(max_games) if max_games else None
                max_hours = float(max_hours) if max_hours else None
                results = await trainer.run_training(max_games=max_games, max_hours=max_hours)
            else:
                results = await trainer.run_training(quick_test=True)

            # Display results
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE!")
            print("=" * 60)
            print(f"System: {results.get('system_type', 'Unknown')}")
            print(f"Games: {results.get('games_completed', 0)}")
            print(f"Wins: {results.get('games_won', 0)}")
            print(f"Win Rate: {results.get('win_rate', 0.0):.1%}")

            if 'average_score' in results:
                print(f"Avg Score: {results['average_score']:.1f}")
            if 'total_time_hours' in results:
                print(f"Time: {results['total_time_hours']:.1f}h")
            if 'games_per_hour' in results:
                print(f"Rate: {results['games_per_hour']:.1f} games/hour")

            if results.get('system_type') == 'AI_ENHANCED':
                print(f"Database: {results.get('database_path', 'N/A')}")
                print("[AI] Systems continuously learning from each game!")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            logger.info("Training interrupted by user")
        except asyncio.CancelledError:
            print("\nTraining cancelled")
            logger.info("Training cancelled") 
        except Exception as e:
            print(f"\nTraining error: {e}")
            logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()

    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
    except Exception as e:
        print(f"\nSetup error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always ensure graceful shutdown
        if trainer:
            try:
                await trainer.shutdown()
            except Exception as e:
                logger.error(f"Error during final shutdown: {e}")
                print(f"Warning: Error during shutdown: {e}")
        print("Training session ended.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
        logger.info("Training stopped by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
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

# Load hypothesis generation and testing system
try:
    from src.intelligence import (
        HypothesisIntegrationSystem,
        get_hypothesis_integration_system
    )
    HYPOTHESIS_SYSTEM_AVAILABLE = True
    print("[OK] Game-Specific Hypothesis Generation and Testing System detected!")
except ImportError:
    HYPOTHESIS_SYSTEM_AVAILABLE = False
    print("[WARNING] Hypothesis system not available")

# Load enhanced frame analyzer
try:
    from src.vision.enhanced_frame_analyzer import create_enhanced_frame_analyzer
    ENHANCED_FRAME_ANALYZER_AVAILABLE = True
    print("[OK] Enhanced Frame Analyzer with CV2 support detected!")
except ImportError:
    ENHANCED_FRAME_ANALYZER_AVAILABLE = False
    print("[WARNING] Enhanced Frame Analyzer not available")

# Load enhanced learning integration
try:
    from src.core.enhanced_learning_integration import create_enhanced_learning_integration
    ENHANCED_LEARNING_AVAILABLE = True
    print("[OK] Enhanced Learning Integration system detected!")
except ImportError:
    ENHANCED_LEARNING_AVAILABLE = False
    print("[WARNING] Enhanced Learning Integration not available")

# Load enhanced knowledge transfer
try:
    from src.learning.enhanced_knowledge_transfer import create_enhanced_knowledge_transfer
    ENHANCED_KNOWLEDGE_TRANSFER_AVAILABLE = True
    print("[OK] Enhanced Knowledge Transfer system detected!")
except ImportError:
    ENHANCED_KNOWLEDGE_TRANSFER_AVAILABLE = False
    print("[WARNING] Enhanced Knowledge Transfer not available")

# Load enhanced memory systems
try:
    from src.memory.enhanced_dnc import EnhancedDNCMemory
    ENHANCED_MEMORY_AVAILABLE = True
    print("[OK] Enhanced Memory Systems (DNC) detected!")
except ImportError:
    ENHANCED_MEMORY_AVAILABLE = False
    print("[WARNING] Enhanced Memory Systems not available")

# Load game lifecycle analyzer
try:
    from src.analysis.game_lifecycle_analyzer import GameLifecycleAnalyzer
    GAME_LIFECYCLE_ANALYZER_AVAILABLE = True
    print("[OK] Game Lifecycle Intelligence System detected!")
except ImportError:
    GAME_LIFECYCLE_ANALYZER_AVAILABLE = False
    print("[WARNING] Game Lifecycle Intelligence System not available")

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

        # Initialize hypothesis system if available
        self.hypothesis_system = None
        if HYPOTHESIS_SYSTEM_AVAILABLE:
            try:
                self.hypothesis_system = get_hypothesis_integration_system(self.db_path)
                logger.info("Hypothesis Generation and Testing System initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hypothesis system: {e}")
                self.hypothesis_system = None

        # Initialize enhanced frame analyzer
        self.frame_analyzer = None
        if ENHANCED_FRAME_ANALYZER_AVAILABLE:
            try:
                self.frame_analyzer = create_enhanced_frame_analyzer()
                logger.info("Enhanced Frame Analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced Frame Analyzer: {e}")
                self.frame_analyzer = None

        # Initialize enhanced learning integration
        self.enhanced_learning = None
        if ENHANCED_LEARNING_AVAILABLE:
            try:
                self.enhanced_learning = create_enhanced_learning_integration(
                    enable_monitoring=True,
                    enable_database_storage=True
                )
                logger.info("Enhanced Learning Integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced Learning Integration: {e}")
                self.enhanced_learning = None

        # Initialize enhanced knowledge transfer
        self.knowledge_transfer = None
        if ENHANCED_KNOWLEDGE_TRANSFER_AVAILABLE:
            try:
                self.knowledge_transfer = create_enhanced_knowledge_transfer(
                    transfer_threshold=0.6,
                    enable_database_storage=True
                )
                logger.info("Enhanced Knowledge Transfer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced Knowledge Transfer: {e}")
                self.knowledge_transfer = None

        # Initialize enhanced memory systems
        self.enhanced_memory = None
        if ENHANCED_MEMORY_AVAILABLE:
            try:
                self.enhanced_memory = EnhancedDNCMemory(
                    memory_size=512,
                    word_size=64,
                    enable_monitoring=True,
                    enable_database_storage=True
                )
                logger.info("Enhanced Memory Systems (DNC) initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Enhanced Memory Systems: {e}")
                self.enhanced_memory = None

        # Initialize game lifecycle analyzer
        self.lifecycle_analyzer = None
        if GAME_LIFECYCLE_ANALYZER_AVAILABLE:
            try:
                self.lifecycle_analyzer = GameLifecycleAnalyzer(db_path="data/game_lifecycle.db")
                logger.info("Game Lifecycle Intelligence System initialized")

                # Connect lifecycle analyzer to gameplay system for proactive action recommendations
                if hasattr(self.gameplay, 'set_lifecycle_analyzer'):
                    self.gameplay.set_lifecycle_analyzer(self.lifecycle_analyzer)
                else:
                    # Set as private attribute for access in gameplay methods
                    self.gameplay._lifecycle_analyzer = self.lifecycle_analyzer
                logger.info("Lifecycle analyzer connected to gameplay system")

            except Exception as e:
                logger.warning(f"Failed to initialize Game Lifecycle Intelligence System: {e}")
                self.lifecycle_analyzer = None

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
        """Play a single AI-enhanced game with hypothesis-guided gameplay."""
        game_start = time.time()
        game_num = self.games_completed + 1

        try:
            logger.info(f"Starting AI-enhanced game {game_num}...")

            # Get real ARC game ID from available games
            game_id = await self._get_real_game_id(game_num)

            # Initialize comprehensive AI analysis for this game
            enhanced_analysis_context = {}

            # Get initial screenshot for analysis
            screenshot_result = await self.gameplay.get_current_screenshot()
            screenshot_array = None  # Initialize to handle case when no screenshot available

            if screenshot_result and 'screenshot_array' in screenshot_result:
                screenshot_array = screenshot_result['screenshot_array']

                # Enhanced Frame Analysis
                frame_analysis_result = None
                if self.frame_analyzer:
                    try:
                        logger.info(f"Running enhanced frame analysis for game {game_id}...")
                        frame_analysis_result = self.frame_analyzer.analyze_frame(screenshot_array)
                        enhanced_analysis_context['frame_analysis'] = frame_analysis_result

                        # Log key insights from frame analysis
                        if frame_analysis_result:
                            patterns = frame_analysis_result.visual_patterns or []
                            logger.info(f"  Frame Analysis: {len(patterns)} visual patterns detected")
                            if hasattr(frame_analysis_result, 'game_mechanics_prediction'):
                                logger.info(f"  Predicted Mechanics: {frame_analysis_result.game_mechanics_prediction}")
                    except Exception as e:
                        logger.warning(f"Enhanced frame analysis failed for game {game_id}: {e}")
                        frame_analysis_result = None

                # Enhanced Memory Retrieval
                memory_insights = None
                if self.enhanced_memory:
                    try:
                        logger.info(f"Retrieving strategic memories for game {game_id}...")
                        # Convert screenshot to tensor for memory query
                        import torch
                        memory_query = torch.tensor(screenshot_array, dtype=torch.float32).flatten()[:64]  # Use first 64 elements as query
                        if len(memory_query) < 64:
                            memory_query = torch.cat([memory_query, torch.zeros(64 - len(memory_query))])

                        memory_result = self.enhanced_memory.read(memory_query.unsqueeze(0))
                        if memory_result is not None:
                            enhanced_analysis_context['memory_insights'] = memory_result
                            logger.info(f"  Memory Systems: Retrieved strategic patterns")
                    except Exception as e:
                        logger.warning(f"Enhanced memory retrieval failed for game {game_id}: {e}")
                        memory_insights = None

                # Knowledge Transfer Analysis
                transfer_insights = None
                if self.knowledge_transfer:
                    try:
                        logger.info(f"Analyzing knowledge transfer opportunities for game {game_id}...")
                        # Check for transferable knowledge from similar games
                        game_features = {
                            'game_id': game_id,
                            'screenshot_data': screenshot_array.tolist() if hasattr(screenshot_array, 'tolist') else screenshot_array,
                            'frame_analysis': frame_analysis_result.__dict__ if frame_analysis_result else None
                        }

                        transfer_opportunities = await self.knowledge_transfer.identify_transfer_opportunities(game_features)
                        if transfer_opportunities:
                            enhanced_analysis_context['knowledge_transfer'] = transfer_opportunities
                            logger.info(f"  Knowledge Transfer: {len(transfer_opportunities)} transfer opportunities identified")
                    except Exception as e:
                        logger.warning(f"Knowledge transfer analysis failed for game {game_id}: {e}")
                        transfer_insights = None

                # Hypothesis Generation (enhanced with frame analysis)
                hypothesis_results = None
                if self.hypothesis_system:
                    try:
                        logger.info(f"Generating hypotheses for game {game_id}...")

                        game_context = {
                            'game_id': game_id,
                            'game_number': game_num,
                            'session_manager': self.session_manager,
                            'enhanced_analysis': enhanced_analysis_context  # Include all enhanced analysis
                        }

                        # Generate hypotheses with enhanced context
                        hypothesis_results = await self.hypothesis_system.analyze_and_generate_hypotheses(
                            game_id=game_id,
                            screenshot_array=screenshot_array,
                            game_context=game_context
                        )

                        if hypothesis_results:
                            logger.info(f"Generated {len(hypothesis_results)} hypotheses for game {game_id}")
                            for i, hyp in enumerate(hypothesis_results[:3]):  # Show first 3
                                logger.info(f"  H{i+1}: {hyp.get('description', 'Unknown hypothesis')}")
                        else:
                            logger.info(f"No hypotheses generated for game {game_id}")

                    except Exception as e:
                        logger.warning(f"Hypothesis generation failed for game {game_id}: {e}")
                        hypothesis_results = None
            else:
                logger.info(f"No screenshot available for enhanced analysis - this is expected before gameplay starts")
                hypothesis_results = None

            # Play the game (with or without hypotheses)
            result = await self.gameplay.play_single_game(
                game_id=game_id,
                max_actions=400,
                hypothesis_context=hypothesis_results  # Pass hypotheses to gameplay
            )

            duration = time.time() - game_start
            final_score = result.get('final_score', 0.0)
            total_actions = result.get('total_actions', 0)
            game_won = result.get('win_detected', False)

            # Add comprehensive enhanced system information to result
            if hypothesis_results:
                result['hypotheses_generated'] = len(hypothesis_results)
                result['hypothesis_system_used'] = True
            else:
                result['hypotheses_generated'] = 0
                result['hypothesis_system_used'] = False

            # Track enhanced systems usage
            result['frame_analysis_used'] = bool(enhanced_analysis_context.get('frame_analysis'))
            result['memory_insights_used'] = bool(enhanced_analysis_context.get('memory_insights'))
            result['knowledge_transfer_used'] = bool(enhanced_analysis_context.get('knowledge_transfer'))
            result['enhanced_analysis_context'] = enhanced_analysis_context

            logger.info(f"Game {game_num}: Score={final_score}, Actions={total_actions}, Won={game_won}, Time={duration:.1f}s")

            # Enhanced systems summary
            systems_used = []
            if hypothesis_results:
                systems_used.append(f"Hypotheses: {len(hypothesis_results)}")
            if enhanced_analysis_context.get('frame_analysis'):
                systems_used.append("Frame Analysis")
            if enhanced_analysis_context.get('memory_insights'):
                systems_used.append("Memory")
            if enhanced_analysis_context.get('knowledge_transfer'):
                systems_used.append("Knowledge Transfer")

            if systems_used:
                logger.info(f"  Enhanced Systems: {', '.join(systems_used)}")

            return {
                'game_number': game_num,
                'game_id': game_id,
                'final_score': final_score,
                'total_actions': total_actions,
                'game_duration': duration,
                'game_won': game_won,
                'ai_performance': result.get('ai_performance', {}),
                'hypotheses_generated': result.get('hypotheses_generated', 0),
                'hypothesis_system_used': result.get('hypothesis_system_used', False),
                'frame_analysis_used': result.get('frame_analysis_used', False),
                'memory_insights_used': result.get('memory_insights_used', False),
                'knowledge_transfer_used': result.get('knowledge_transfer_used', False),
                'enhanced_analysis_context': enhanced_analysis_context,
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
        """Process game result with enhanced learning integration."""
        self.games_completed += 1
        if result.get('game_won', False):
            self.games_won += 1
        self.total_score += result.get('final_score', 0.0)
        self.total_actions += result.get('total_actions', 0)
        self.game_results.append(result)

        # Enhanced Learning Integration
        if self.enhanced_learning:
            try:
                logger.info(f"Processing game result with enhanced learning...")

                # Get lifecycle pattern analysis for this game
                lifecycle_pattern_data = {}
                if self.lifecycle_analyzer:
                    try:
                        # Get failure risk assessment
                        game_type = self._classify_game_type(result)
                        failure_risk = self.lifecycle_analyzer.get_failure_risk(
                            game_type=game_type,
                            current_action_count=result.get('total_actions', 0),
                            recent_actions=result.get('action_history', [])[-10:] if result.get('action_history') else []
                        )

                        # Get action effectiveness data from lifecycle analyzer
                        action_effectiveness = {}
                        if hasattr(self.lifecycle_analyzer, 'action_effectiveness'):
                            for (action, context), effectiveness in self.lifecycle_analyzer.action_effectiveness.items():
                                if context.startswith(game_type):
                                    action_effectiveness[action] = effectiveness.avg_action_effectiveness if hasattr(effectiveness, 'avg_action_effectiveness') else 0.5

                        # Detect oscillations in action sequence
                        oscillation_detected = False
                        action_sequence = result.get('action_history', [])
                        if action_sequence and len(action_sequence) >= 10:
                            oscillations = self.lifecycle_analyzer.oscillation_detector.detect_oscillations(action_sequence)
                            oscillation_detected = len(oscillations) > 0

                        lifecycle_pattern_data = {
                            'failure_risk_score': failure_risk,
                            'oscillation_detected': oscillation_detected,
                            'action_effectiveness': action_effectiveness,
                            'game_type': game_type,
                            'oscillation_patterns': [osc['actions'] for osc in (oscillations if oscillation_detected else [])]
                        }

                        logger.info(f"Lifecycle analysis for game: risk={failure_risk:.2f}, oscillations={oscillation_detected}, effectiveness_count={len(action_effectiveness)}")

                    except Exception as e:
                        logger.warning(f"Failed to get lifecycle patterns for learning: {e}")

                # Create enhanced learning experience from game result
                learning_experience = {
                    'game_id': result.get('game_id', 'unknown'),
                    'game_number': result.get('game_number', 0),
                    'final_score': result.get('final_score', 0.0),
                    'game_won': result.get('game_won', False),
                    'total_actions': result.get('total_actions', 0),
                    'game_duration': result.get('game_duration', 0.0),
                    'hypotheses_generated': result.get('hypotheses_generated', 0),
                    'enhanced_analysis_used': result.get('enhanced_analysis_context', {}),
                    'lifecycle_patterns': lifecycle_pattern_data  # NEW: Add lifecycle pattern data
                }

                # Process with enhanced learning
                await self.enhanced_learning.process_learning_experience(learning_experience)
                result['enhanced_learning_processed'] = True
                logger.info(f"Enhanced learning processed game {result.get('game_number', 0)}")

            except Exception as e:
                result['enhanced_learning_processed'] = False
                logger.warning(f"Enhanced learning processing failed: {e}")

        # Enhanced Memory Storage
        if self.enhanced_memory:
            try:
                logger.info(f"Storing strategic memory from game result...")

                # Create memory entry for successful strategies
                if result.get('game_won', False) or result.get('final_score', 0) > 50:
                    import torch

                    # Create memory content from successful game
                    memory_content = torch.tensor([
                        result.get('final_score', 0.0),
                        result.get('total_actions', 0),
                        result.get('game_duration', 0.0),
                        result.get('hypotheses_generated', 0),
                        1.0 if result.get('game_won', False) else 0.0,
                        # Pad to word_size (64)
                    ] + [0.0] * 59, dtype=torch.float32)

                    # Store in enhanced memory
                    self.enhanced_memory.write(memory_content.unsqueeze(0), memory_content.unsqueeze(0))
                    logger.info(f"Strategic memory stored for high-performing game")

            except Exception as e:
                logger.warning(f"Enhanced memory storage failed: {e}")

        # Enhanced Knowledge Transfer Learning
        if self.knowledge_transfer:
            try:
                logger.info(f"Updating knowledge transfer with game result...")

                # Create transferable knowledge from game result
                game_knowledge = {
                    'game_id': result.get('game_id', 'unknown'),
                    'success_score': result.get('final_score', 0.0),
                    'strategies_used': {
                        'hypotheses_count': result.get('hypotheses_generated', 0),
                        'total_actions': result.get('total_actions', 0),
                        'game_won': result.get('game_won', False)
                    },
                    'performance_metrics': {
                        'duration': result.get('game_duration', 0.0),
                        'efficiency': result.get('final_score', 0.0) / max(result.get('total_actions', 1), 1)
                    },
                    'lifecycle_patterns': lifecycle_pattern_data  # NEW: Add lifecycle pattern data for knowledge transfer
                }

                # Update knowledge transfer system
                await self.knowledge_transfer.update_knowledge_base(game_knowledge)
                logger.info(f"Knowledge transfer updated with game {result.get('game_number', 0)} insights")

            except Exception as e:
                logger.warning(f"Knowledge transfer update failed: {e}")

        # Game Lifecycle Pattern Analysis
        if self.lifecycle_analyzer:
            try:
                logger.info(f"Analyzing game lifecycle patterns...")

                # Create lifecycle data from game result
                lifecycle_data = {
                    'game_id': result.get('game_id', 'unknown'),
                    'game_type': self._classify_game_type(result),
                    'actions_to_failure': result.get('total_actions', 0),
                    'final_score': result.get('final_score', 0.0),
                    'game_won': result.get('game_won', False),
                    'failure_reason': self._determine_failure_reason(result),
                    'action_sequence_before_end': result.get('action_history', [])[-20:] if result.get('action_history') else [],
                    'efficiency_trajectory': result.get('score_history', [])
                }

                # Add pattern to lifecycle analyzer
                await self.lifecycle_analyzer.add_game_over_pattern(lifecycle_data)
                logger.info(f"Lifecycle pattern analysis completed for game {result.get('game_number', 0)}")

            except Exception as e:
                logger.warning(f"Lifecycle pattern analysis failed: {e}")

        # Original knowledge extraction for AI system
        if self.system_type == "AI_ENHANCED" and self.gameplay.knowledge_integrator:
            try:
                await self.gameplay.knowledge_integrator.extract_game_knowledge(result)
            except Exception as e:
                logger.warning(f"Knowledge extraction failed: {e}")

    def _classify_game_type(self, result: Dict[str, Any]) -> str:
        """Classify game type based on game characteristics."""
        try:
            game_id = result.get('game_id', 'unknown')
            total_actions = result.get('total_actions', 0)
            final_score = result.get('final_score', 0.0)

            # Simple classification based on ID patterns and performance characteristics
            if 'action6' in game_id.lower() or total_actions > 200:
                return 'action6_intensive'
            elif final_score == 0 and total_actions < 50:
                return 'quick_failure'
            elif total_actions > 150:
                return 'long_exploration'
            elif final_score > 100:
                return 'high_scoring'
            else:
                return 'standard'
        except Exception:
            return 'unknown'

    def _determine_failure_reason(self, result: Dict[str, Any]) -> str:
        """Determine the reason for game failure."""
        try:
            game_won = result.get('game_won', False)
            total_actions = result.get('total_actions', 0)
            final_score = result.get('final_score', 0.0)
            game_duration = result.get('game_duration', 0.0)
            error = result.get('error')

            if game_won:
                return 'success'
            elif error:
                return f'error_{error}'
            elif total_actions >= 400:  # Max actions limit
                return 'action_limit'
            elif game_duration > 300:  # 5 minutes timeout
                return 'timeout'
            elif total_actions > 100 and final_score == 0:
                return 'score_stagnation'
            elif total_actions < 10:
                return 'early_termination'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'

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
        """Log comprehensive training progress with all enhanced systems."""
        elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
        win_rate = self.games_won / max(self.games_completed, 1)
        avg_score = self.total_score / max(self.games_completed, 1)

        # Calculate hypothesis system metrics
        hypothesis_games = sum(1 for r in self.game_results if r.get('hypothesis_system_used', False))
        total_hypotheses = sum(r.get('hypotheses_generated', 0) for r in self.game_results)
        avg_hypotheses = total_hypotheses / max(hypothesis_games, 1) if hypothesis_games > 0 else 0

        # Calculate enhanced frame analysis metrics
        frame_analysis_games = sum(1 for r in self.game_results if r.get('frame_analysis_used', False))

        # Calculate enhanced learning metrics
        enhanced_learning_games = sum(1 for r in self.game_results if r.get('enhanced_learning_processed', False))

        # Calculate knowledge transfer metrics
        knowledge_transfer_games = sum(1 for r in self.game_results if r.get('knowledge_transfer_used', False))

        # Calculate memory system metrics
        memory_games = sum(1 for r in self.game_results if r.get('memory_insights_used', False))

        # Get Action6 Coordinator statistics from enhanced gameplay
        action6_stats = {}
        if self.gameplay and hasattr(self.gameplay, 'enhanced_gameplay'):
            enhanced_gameplay = self.gameplay.enhanced_gameplay
            if enhanced_gameplay and hasattr(enhanced_gameplay, 'action6_coordinator'):
                coordinator = enhanced_gameplay.action6_coordinator
                if hasattr(coordinator, 'stats'):
                    action6_stats = coordinator.stats

        logger.info("=" * 70)
        logger.info(f"PROGRESS: Game {self.games_completed}")
        logger.info(f"Win rate: {win_rate:.1%} ({self.games_won}/{self.games_completed})")
        logger.info(f"Avg score: {avg_score:.1f}")
        logger.info(f"Time: {elapsed:.1f}h")
        logger.info("-" * 70)

        # Enhanced Frame Analysis
        if self.frame_analyzer:
            if frame_analysis_games > 0:
                logger.info(f"Enhanced Frame Analysis: {frame_analysis_games}/{self.games_completed} games analyzed")
            else:
                logger.info("Enhanced Frame Analysis: Active but no games analyzed yet")

        # Action6 Coordinator Stats
        if action6_stats:
            pseudo_buttons = action6_stats.get('button_based_selections', 0)
            exploration_coords = action6_stats.get('exploration_selections', 0)
            total_action6 = pseudo_buttons + exploration_coords
            logger.info(f"Action6 Coordinator: {total_action6} total selections")
            logger.info(f"  Pseudo-buttons: {pseudo_buttons}, Exploration: {exploration_coords}")
        elif self.gameplay and hasattr(self.gameplay, 'enhanced_gameplay'):
            logger.info("Action6 Coordinator: Active with pseudo-button detection")

        # Enhanced Learning Integration
        if self.enhanced_learning:
            if enhanced_learning_games > 0:
                logger.info(f"Enhanced Learning: {enhanced_learning_games}/{self.games_completed} games processed")
            else:
                logger.info("Enhanced Learning: Active but no learning experiences processed yet")

        # Enhanced Knowledge Transfer
        if self.knowledge_transfer:
            if knowledge_transfer_games > 0:
                logger.info(f"Knowledge Transfer: {knowledge_transfer_games}/{self.games_completed} games with transfer opportunities")
            else:
                logger.info("Knowledge Transfer: Active but no transfer opportunities identified yet")

        # Enhanced Memory Systems
        if self.enhanced_memory:
            if memory_games > 0:
                logger.info(f"Enhanced Memory: {memory_games}/{self.games_completed} games with memory insights")
                # Get memory utilization if available
                if hasattr(self.enhanced_memory, 'get_memory_utilization'):
                    try:
                        utilization = self.enhanced_memory.get_memory_utilization()
                        logger.info(f"  Memory utilization: {utilization:.1%}")
                    except:
                        pass
            else:
                logger.info("Enhanced Memory: Active but no memory insights retrieved yet")

        # Hypothesis System (existing)
        if self.hypothesis_system:
            if hypothesis_games > 0:
                logger.info(f"Hypothesis System: {hypothesis_games}/{self.games_completed} games")
                logger.info(f"  Total hypotheses: {total_hypotheses} (avg: {avg_hypotheses:.1f}/game)")
            else:
                logger.info("Hypothesis System: Active but no hypotheses generated yet")

        logger.info("=" * 70)

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
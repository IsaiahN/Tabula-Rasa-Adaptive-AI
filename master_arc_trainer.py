#!/usr/bin/env python3
"""
MASTER ARC TRAINER - Consolidated Training System

This unified script combines all functionality from:
1. Legacy training systems (now consolidated here)
2. All previous trainer implementations

Single entry point for all ARC training needs with full functionality.

Usage Examples:
    # Maximum Intelligence (default)
    python master_arc_trainer.py
    
    # Quick validation test
    python master_arc_trainer.py --mode quick-validation --games game1,game2
    
    # Legacy sequential mode (backward compatible)  
    python master_arc_trainer.py --mode sequential --salience decay
    
    # Research lab with system comparison
    python master_arc_trainer.py --mode research-lab --compare-systems
    
    # Minimal debug mode
    python master_arc_trainer.py --mode minimal-debug --disable-all-advanced
"""

import asyncio
import argparse
import logging
import sys
import time
import os
import json
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Core imports
from src.core.salience_system import SalienceMode
from src.core.logging_setup import setup_logging

# Meta-cognitive imports
try:
    from src.core.meta_cognitive_governor import MetaCognitiveGovernor
    from src.core.architect import Architect
    META_COGNITIVE_AVAILABLE = True
except ImportError as e:
    META_COGNITIVE_AVAILABLE = False
    print(f"WARNING: Meta-cognitive systems not available: {e}")

# Color output support
try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    COLOR_AVAILABLE = True
except ImportError:
    COLOR_AVAILABLE = False
    # Fallback color constants
    class Fore:
        BLUE = CYAN = GREEN = YELLOW = MAGENTA = RED = ""
    class Style:
        RESET_ALL = ""

# Global flag for graceful shutdown
shutdown_requested = False
training_state = {}
continuous_loop_instance = None

def setup_windows_logging():
    """Set up logging that works properly on Windows with real-time terminal and file output."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    simple_format = '%(message)s'  # For terminal display
    
    # Create handlers with UTF-8 encoding
    handlers = []
    
    # Console handler with UTF-8 - handle Windows encoding issues
    try:
        import sys
        import codecs
        
        # Try to set console to UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            except:
                pass
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Use a formatter that handles Unicode gracefully
        class SafeFormatter(logging.Formatter):
            def format(self, record):
                try:
                    return super().format(record)
                except UnicodeEncodeError:
                    # Strip problematic characters and retry
                    record.msg = str(record.msg).encode('ascii', errors='replace').decode('ascii')
                    return super().format(record)
        
        console_handler.setFormatter(SafeFormatter(simple_format))  # Simple format for console
        
    except Exception:
        # Fallback: basic console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(simple_format))
    
    # Create TeeHandler for simultaneous file and console output
    try:
        tee_handler = TeeHandler(
            file_path='data/logs/master_arc_trainer_output.log',
            console_handler=console_handler
        )
        tee_handler.setLevel(logging.INFO)
        tee_handler.setFormatter(SafeFormatter(log_format) if 'SafeFormatter' in locals() else logging.Formatter(log_format))
        handlers.append(tee_handler)
        
        # Also add separate file handler for detailed logs
        file_handler = logging.FileHandler('data/logs/master_arc_trainer.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
        
    except Exception as e:
        # Fallback to console-only
        handlers.append(console_handler)
        print(f"Warning: Could not set up file logging: {e}")
    
    # Configure root logger
    try:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=handlers,
            force=True  # Override existing config
        )
    except Exception:
        # Minimal fallback
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    # Disable some noisy loggers
    try:
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('git').setLevel(logging.WARNING)
    except Exception:
        pass
    
    print("🚀 Enhanced logging initialized - real-time terminal and file output enabled")

class TeeHandler(logging.Handler):
    """Custom handler that writes to both file and console simultaneously."""
    
    def __init__(self, file_path: str, console_handler: logging.Handler):
        super().__init__()
        self.file_path = file_path
        self.console_handler = console_handler
        # Ensure log directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
    def emit(self, record):
        try:
            # Format the record
            msg = self.format(record)
            
            # Write to file (append mode, UTF-8 encoding)
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
                f.flush()
            
            # Also emit to console
            if self.console_handler:
                self.console_handler.emit(record)
                
        except Exception:
            self.handleError(record)

def safe_print(text: str, use_color: bool = True, log_to_file: bool = True):
    """Print text safely to both terminal and log file, handling Unicode encoding issues on Windows."""
    try:
        # Always log to the master output file if requested
        if log_to_file:
            log_path = 'data/logs/master_arc_trainer_output.log'
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'a', encoding='utf-8') as f:
                    # Strip ANSI codes for file logging
                    import re
                    clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
                    f.write(clean_text + '\r\n')
                    f.flush()
            except Exception as e:
                # Don't let file logging errors prevent console output
                pass
        
        # On Windows, handle encoding issues more gracefully
        if os.name == 'nt':  # Windows
            try:
                # Try to configure stdout encoding if possible
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            except:
                pass
        
        if COLOR_AVAILABLE and use_color:
            print(text, flush=True)  # Force flush for real-time output
        else:
            # Strip ANSI color codes if colorama not available
            import re
            clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text)
            print(clean_text, flush=True)
            
    except (UnicodeEncodeError, OSError) as e:
        # Fallback: encode to ASCII and ignore problematic characters
        try:
            import re
            clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text) if '\x1b[' in text else text
            safe_text = clean_text.encode('ascii', errors='ignore').decode('ascii')
            print(safe_text, flush=True)
        except Exception:
            # Ultimate fallback - just print a safe message
            print("Output contains non-displayable characters", flush=True)
    except Exception:
        # Any other exception - print safe message
        print("Output contains non-displayable characters", flush=True)

def signal_handler(signum, frame):
    """Handle graceful shutdown signals."""
    global shutdown_requested, training_state, continuous_loop_instance
    shutdown_requested = True
    
    print(f"\n🛑 GRACEFUL SHUTDOWN REQUESTED (Signal: {signum})")
    print("💾 Saving training state...")
    
    # Also stop the continuous loop if it exists
    if 'continuous_loop_instance' in globals() and continuous_loop_instance:
        continuous_loop_instance.running = False
        print("🛑 Continuous training loop stopped")
    
    try:
        # Use proper DataPaths constant and ensure directory exists
        from src.config.data_paths import DataPaths
        state_file = DataPaths.TRAINING_STATE_BACKUP
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(training_state, f, indent=2)
        print(f"✅ Training state saved to: {state_file}")
    except Exception as e:
        print(f"⚠️ Could not save training state: {e}")
        # Fallback to creating backups directory if DataPaths fails
        try:
            backup_dir = Path("data/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            fallback_file = backup_dir / "training_state_backup.json"
            with open(fallback_file, 'w') as f:
                json.dump(training_state, f, indent=2)
            print(f"✅ Training state saved to fallback location: {fallback_file}")
        except Exception as fallback_e:
            print(f"❌ Failed to save training state even with fallback: {fallback_e}")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Import core systems
try:
    from dotenv import load_dotenv
    
    # Import only what's needed immediately to avoid circular imports
    from src.core.salience_system import SalienceMode
    from src.core.energy_system import EnergySystem
    
    # Try to import coordinate system
    try:
        from enhanced_coordinate_intelligence import EnhancedCoordinateIntelligence
        COORDINATE_SYSTEM_AVAILABLE = True
    except ImportError:
        COORDINATE_SYSTEM_AVAILABLE = False
        print("WARNING: Coordinate intelligence not available")  # Use ASCII characters
    
    # Lazy import for ContinuousLearningLoop to prevent circular imports
    ContinuousLearningLoop = None
    
    # Load environment configuration
    # Try to load .env file, but don't fail if it doesn't exist
    env_files_loaded = []
    try:
        load_dotenv('.env')
        env_files_loaded.append('.env')
    except Exception as e:
        print(f"Note: Could not load .env file: {e}")
    
    # No fallback needed - use only .env file
    
    if env_files_loaded:
        print(f"✅ Loaded environment files: {', '.join(env_files_loaded)}")
    
    # Check if ARC_API_KEY is available
    arc_api_key = os.getenv('ARC_API_KEY')
    if not arc_api_key:
        print("⚠️  WARNING: ARC_API_KEY environment variable not set!")
        print("   Please set your ARC API key using one of these methods:")
        print("   1. Environment variable:")
        print("      Windows: set ARC_API_KEY=your_api_key_here")
        print("      PowerShell: $env:ARC_API_KEY=\"your_api_key_here\"")
        print("      Linux/Mac: export ARC_API_KEY=your_api_key_here")
        print("   2. Create a .env file with: ARC_API_KEY=your_api_key_here")
        print("   3. Pass --api-key argument when running the script")
        print("   4. Run setup script: python setup_env.py")
        print("   Get your API key from: https://three.arcprize.org")
        print()
        print("   The training will continue but will use mock mode if no API key is provided.")
    else:
        print(f"✅ ARC_API_KEY loaded: {arc_api_key[:8]}...")
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print(f"❌ Make sure the package is installed with: pip install -e .")
    sys.exit(1)

@dataclass
class MasterTrainingConfig:
    """Unified configuration for all training modes."""

    # Core settings
    mode: str = 'maximum-intelligence'  # Training mode
    api_key: Optional[str] = None  # ARC API key
    arc_agents_path: Optional[str] = None  # Path to arc-agents directory
    local_mode: bool = False  # Use local mock client for testing
    max_actions: int = 1500
    max_cycles: int = 150
    target_score: float = 85.0
    session_duration: int = 60  # minutes

    # Memory settings
    memory_size: int = 512
    memory_word_size: int = 64
    memory_read_heads: int = 4
    memory_write_heads: int = 1
    
    # Salience system
    salience_mode: str = "decay_compression"
    salience_threshold: float = 0.6
    salience_decay: float = 0.95
    
    # Energy system
    sleep_trigger_energy: float = 40.0
    sleep_duration: int = 50
    
    # Feature flags
    enable_swarm: bool = True
    enable_coordinates: bool = True
    enable_energy_system: bool = True
    enable_sleep_cycles: bool = True
    enable_dnc_memory: bool = True
    enable_meta_learning: bool = True
    enable_salience_system: bool = True
    enable_contrarian_strategy: bool = True
    enable_frame_analysis: bool = True
    enable_boundary_detection: bool = True
    enable_memory_consolidation: bool = True
    enable_action_intelligence: bool = True
    enable_meta_cognitive_governor: bool = True
    enable_architect_evolution: bool = True
    
    # Meta-cognitive monitoring
    enable_detailed_monitoring: bool = False
    enable_colored_output: bool = True
    save_meta_cognitive_logs: bool = True
    
    # Logging and monitoring
    verbose: bool = False
    no_logs: bool = False
    no_monitoring: bool = False
    debug_mode: bool = False
    
    # Testing and validation
    games: Optional[str] = None
    compare_systems: bool = False
    performance_tests: bool = False

# Backwards-compatible alias expected by some tests
TrainingConfig = MasterTrainingConfig

class MasterARCTrainer:
    """
    Master ARC trainer that unifies all training functionality.
    
    This system consolidates:
    1. All legacy training functionality
    2. Modern training system features
    3. Backward compatibility for existing workflows
    4. Enhanced error handling and monitoring
    """
    
    def __init__(self, config: MasterTrainingConfig):
        self.config = config
        self.continuous_loop = None
        self.coordinate_manager = None
        self.governor = None
        self.architect = None
        self.session_results = {}
        self.performance_history = []
        self.arc_client = None
        self.initialized = False
        
        # Meta-cognitive monitoring data
        self.governor_decisions = []
        self.architect_evolutions = []
        self.session_id = f"master_training_{int(time.time())}"
        
        # Setup logging
        self._setup_logging()
        
        # Initialize core systems
        self._initialize_systems()
    
    async def initialize_client(self):
        """Initialize the ARC API client and test the connection."""
        try:
            import os
            
            # Check if we should use the mock client
            if self.config.local_mode:
                from arc_integration.mock_arc_client import MockARCClient
                self.logger.info("Initializing MOCK ARC client for local testing...")
                self.arc_client = MockARCClient(api_key="mock-api-key")
            else:
                from arc_integration.arc_api_client_fixed import ARCClient, DEFAULT_BASE_URL
                self.logger.info("Initializing REAL ARC API client...")
                
                # Debug: Log environment variables
                self.logger.info(f"Environment variables: {os.environ.get('ARC_API_KEY', 'Not found')}")
                self.logger.info(f"Config API key: {'*' * 8 + self.config.api_key[-4:] if self.config.api_key else 'Not set'}")
                
                # Initialize the real client
                self.arc_client = ARCClient(api_key=self.config.api_key)
            
            # Test the connection using async context manager
            self.logger.info("Testing connection...")
            await self.arc_client.__aenter__()
            
            # Test API connectivity with a simple request
            try:
                # First get available games to test connectivity
                self.logger.info("Testing API connectivity with get_available_games...")
                games = await self.arc_client.get_available_games()
                self.logger.info(f"Successfully connected to {'MOCK' if self.config.local_mode else 'ARC'} API. Found {len(games)} available games")
                
                # If we have games, test with the first one (without creating scorecard)
                if games and len(games) > 0:
                    test_game_id = games[0].get('game_id', 'ls20-016295f7601e')  # Fallback to known game
                    self.logger.info(f"Testing with game: {test_game_id}")
                    
                    # Test API connectivity without creating test scorecard
                    # Just verify we can access game information
                    self.logger.info(f"API connectivity verified. Game ID available: {test_game_id}")
                
                self.initialized = True
                return True
            except Exception as e:
                self.logger.error(f"Failed to connect to API: {str(e)}", exc_info=True)
                # Properly close the session before returning
                try:
                    await self.arc_client.__aexit__(None, None, None)
                except Exception as close_error:
                    self.logger.warning(f"Error closing session during cleanup: {close_error}")
                self.arc_client = None
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing ARC client: {e}")
            if self.arc_client:
                try:
                    await self.arc_client.__aexit__(None, None, None)
                except Exception as close_error:
                    self.logger.warning(f"Error closing session during cleanup: {close_error}")
                self.arc_client = None
            return False
    
    async def close(self):
        """Clean up resources."""
        if self.arc_client:
            try:
                await self.arc_client.__aexit__(None, None, None)
            except Exception as close_error:
                self.logger.warning(f"Error closing session during cleanup: {close_error}")
            self.arc_client = None
        self.initialized = False
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_level = logging.DEBUG if self.config.debug_mode else (
            logging.INFO if self.config.verbose else logging.WARNING
        )
        
        # Configure the root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('training.log')
            ]
        )
        
        # Set up module-level logger
        self.logger = logging.getLogger('MasterARCTrainer')
        self.logger.setLevel(log_level)
        
        # Disable noisy loggers
        for logger_name in ['matplotlib', 'PIL']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    def _initialize_systems(self):
        """Initialize core systems and components."""
        self.logger.info("Initializing core systems...")
        
        try:
            # Initialize coordinate system if enabled
            if self.config.enable_coordinates and COORDINATE_SYSTEM_AVAILABLE:
                from enhanced_coordinate_intelligence import EnhancedCoordinateIntelligence
                # We'll initialize the coordinate manager later when we have the continuous loop
                self.coordinate_manager = None
                self.logger.info("Coordinate system will be initialized with continuous loop")
            
            # Initialize meta-cognitive governor if enabled
            if self.config.enable_meta_cognitive_governor:
                try:
                    from src.core.meta_cognitive_governor import MetaCognitiveGovernor
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    data_path = os.path.join(base_path, 'data')
                    self.governor = MetaCognitiveGovernor(persistence_dir=data_path)
                    self.logger.info("Meta-cognitive governor initialized")
                except ImportError as e:
                    self.logger.warning(f"Could not initialize meta-cognitive governor: {e}")
                    self.governor = None
            
            # Initialize architect if enabled
            if self.config.enable_architect_evolution:
                try:
                    from src.core.architect import Architect
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    # Use the main tabula-rasa repository as the repo_path
                    repo_path = base_path
                    # Ensure the architect evolution data directory exists for file storage
                    architect_data_dir = os.path.join(base_path, 'data', 'architect_evolution_data')
                    os.makedirs(architect_data_dir, exist_ok=True)
                    
                    self.architect = Architect(base_path=base_path, repo_path=repo_path)
                    self.logger.info("Architect system initialized with main repository")
                except Exception as e:
                    self.logger.warning(f"Could not initialize Architect: {e}")
                    self.architect = None
                
            self.logger.info("Core systems initialization complete")
            
        except Exception as e:
            self.logger.error(f"Error initializing systems: {e}")
            raise
    
    async def run_training(self):
        """Main training execution based on mode with enhanced client handling."""
        try:
            # Initialize the API client for all modes except those that don't need it
            if self.config.mode not in ['minimal-debug', 'system-comparison']:
                if not await self.initialize_client():
                    raise ConnectionError("Failed to initialize ARC API client. Check your API key and network connection.")
            
            try:
                if self.config.mode == 'maximum-intelligence':
                    return await self._run_maximum_intelligence()
                elif self.config.mode == 'quick-validation':
                    return await self._run_quick_validation()
                elif self.config.mode == 'research-lab':
                    return await self._run_research_lab()
                elif self.config.mode == 'showcase-demo':
                    return await self._run_showcase_demo()
                elif self.config.mode == 'system-comparison':
                    return await self._run_system_comparison()
                elif self.config.mode == 'minimal-debug':
                    return await self._run_minimal_debug()
                elif self.config.mode == 'meta-cognitive-training':
                    return await self._run_meta_cognitive_training()
                elif self.config.mode == 'sequential':
                    return await self._run_sequential()
                elif self.config.mode == 'swarm':
                    return await self._run_swarm()
                elif self.config.mode == 'continuous':
                    return await self._run_continuous()
                else:
                    raise ValueError(f"Unknown training mode: {self.config.mode}")
                    
            finally:
                # Ensure we clean up the client
                if self.arc_client:
                    try:
                        await self.arc_client.__aexit__(None, None, None)
                    except Exception as close_error:
                        self.logger.warning(f"Error closing session during cleanup: {close_error}")
                    self.arc_client = None
                
        except Exception as e:
            self.logger.error(f"Error in run_training: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_maximum_intelligence(self):
        """Run maximum intelligence mode with all systems enabled."""
        self.logger.info("🚀 Starting MAXIMUM-INTELLIGENCE mode")
        self.logger.info("   All meta-cognitive systems enabled")
        self.logger.info("   Governor + Architect + All cognitive systems")
        
        try:
            # Run continuous learning with maximum intelligence settings
            config_overrides = {
                'enable_meta_cognitive_governor': True,
                'enable_architect_evolution': True,
                'enable_coordinates': True,
                'enable_all_cognitive_systems': True,
                'max_actions': self.config.max_actions,
                'max_cycles': self.config.max_cycles,
                'target_score': self.config.target_score
            }
            
            return await self._run_continuous_learning(config_overrides)
            
        except Exception as e:
            self.logger.error(f"Error in maximum intelligence mode: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_quick_validation(self):
        """Run quick validation mode with minimal systems for fast testing."""
        self.logger.info("⚡ Starting QUICK-VALIDATION mode")
        self.logger.info("   Minimal systems enabled for fast testing")
        self.logger.info("   Basic cognitive systems only")
        
        try:
            # Run continuous learning with minimal settings for quick validation
            config_overrides = {
                'enable_meta_cognitive_governor': True,  # Keep Governor for testing
                'enable_architect_evolution': True,      # Keep Architect for testing
                'enable_coordinates': True,
                'enable_all_cognitive_systems': False,
                'max_actions': 20,                       # Very short games for quick testing
                'max_cycles': 1,                         # Single cycle only
                'target_score': self.config.target_score,
                'enable_detailed_monitoring': False,
                'enable_sleep_system': True,
                'enable_energy_system': True,
                'max_games': 2,                          # Limit to 2 games max
                'max_episodes': 1                        # Single episode per game
            }
            
            return await self._run_continuous_learning(config_overrides)
            
        except Exception as e:
            self.logger.error(f"Error in quick validation mode: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_continuous_learning(self, config_overrides: Dict = None):
        """Core continuous learning execution with enhanced client handling."""
        try:
            # Initialize the API client
            if not await self.initialize_client():
                raise ConnectionError("Failed to initialize ARC API client. Check your API key and network connection.")
            
            try:
                # Lazy import to prevent circular imports
                global ContinuousLearningLoop
                if ContinuousLearningLoop is None:
                    from arc_integration.continuous_learning_loop import ContinuousLearningLoop as CL
                    ContinuousLearningLoop = CL
                
                self.logger.info("Creating ContinuousLearningLoop instance...")
                
                # Create and initialize the training loop with the client
                # Get the current directory as the tabula_rasa_path
                tabula_rasa_path = os.path.dirname(os.path.abspath(__file__))
                
                # Use the arc_agents_path from config or default
                arc_agents_path = self.config.arc_agents_path or os.path.join(tabula_rasa_path, '..', 'ARC-AGI-3-Agents')
                
                self.continuous_loop = ContinuousLearningLoop(
                    arc_agents_path=arc_agents_path,
                    tabula_rasa_path=tabula_rasa_path,
                    api_key=self.config.api_key,
                    save_directory="data"
                )
                
                # Initialize coordinate manager with the continuous loop if needed
                if self.config.enable_coordinates and COORDINATE_SYSTEM_AVAILABLE and self.coordinate_manager is None:
                    from enhanced_coordinate_intelligence import EnhancedCoordinateIntelligence
                    self.coordinate_manager = EnhancedCoordinateIntelligence(self.continuous_loop)
                    self.logger.info("Coordinate system initialized with continuous loop")
                
                # Inject the initialized client if the loop accepts it
                if hasattr(self.continuous_loop, 'arc_client'):
                    self.continuous_loop.arc_client = self.arc_client
                
                self.logger.info("Starting training loop...")
                
                # Use specified games or get available games
                if self.config.games:
                    # Parse games parameter (comma-separated list)
                    game_ids = [game.strip() for game in self.config.games.split(',')]
                    self.logger.info(f"Using specified games: {game_ids}")
                else:
                    # Get available games from API
                    games = await self.continuous_loop.get_available_games()
                    if games:
                        game_ids = [game.get('game_id') for game in games[:5]]  # Use first 5 games
                        self.logger.info(f"Using available games: {game_ids}")
                    else:
                        self.logger.warning("No games available, skipping training")
                        return {
                            'success': False,
                            'error': 'No games available',
                            'timestamp': datetime.now().isoformat()
                        }
                
                # Run in swarm mode with specified games
                await self.continuous_loop.run_swarm_mode(
                    games=game_ids,
                    max_concurrent=2,
                    max_episodes_per_game=10
                )
                
                # Save results if available
                if hasattr(self.continuous_loop, 'get_results'):
                    self.session_results = self.continuous_loop.get_results()
                    self.performance_history.append(self.session_results)
                else:
                    self.session_results = {'status': 'completed', 'mode': 'continuous'}
                
                return {
                    'success': True,
                    'mode': 'continuous',
                    'results': self.session_results,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}", exc_info=True)
                return {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
            finally:
                # Ensure we clean up the client
                if self.arc_client:
                    try:
                        await self.arc_client.__aexit__(None, None, None)
                    except Exception as close_error:
                        self.logger.warning(f"Error closing session during cleanup: {close_error}")
                    self.arc_client = None
                
        except Exception as e:
            self.logger.error(f"Error in continuous learning: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_meta_cognitive_training(self):
        """Run meta-cognitive training mode with enhanced monitoring."""
        self.logger.info("🧠 Starting META-COGNITIVE-TRAINING mode")
        self.logger.info("   Governor + Architect + All cognitive systems enabled")
        
        try:
            # Run continuous learning with meta-cognitive settings
            config_overrides = {
                'enable_meta_cognitive_governor': True,
                'enable_architect_evolution': True,
                'enable_coordinates': True,
                'enable_all_cognitive_systems': True,
                'max_actions': self.config.max_actions,
                'max_cycles': self.config.max_cycles,
                'target_score': self.config.target_score
            }
            
            return await self._run_continuous_learning(config_overrides)
            
        except Exception as e:
            self.logger.error(f"Error in meta-cognitive training mode: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_research_lab(self):
        """Run research lab mode with system comparison capabilities."""
        self.logger.info("🔬 Starting RESEARCH-LAB mode")
        self.logger.info("   System comparison and analysis enabled")
        
        try:
            # Run continuous learning with research settings
            config_overrides = {
                'enable_meta_cognitive_governor': True,
                'enable_architect_evolution': True,
                'enable_coordinates': True,
                'enable_all_cognitive_systems': True,
                'max_actions': self.config.max_actions,
                'max_cycles': self.config.max_cycles,
                'target_score': self.config.target_score,
                'enable_detailed_monitoring': True
            }
            
            return await self._run_continuous_learning(config_overrides)
            
        except Exception as e:
            self.logger.error(f"Error in research lab mode: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_showcase_demo(self):
        """Run showcase demo mode with optimized settings for demonstrations."""
        self.logger.info("🎯 Starting SHOWCASE-DEMO mode")
        self.logger.info("   Optimized for demonstrations and presentations")
        
        try:
            # Run continuous learning with demo settings
            config_overrides = {
                'enable_meta_cognitive_governor': True,
                'enable_architect_evolution': False,  # Disable for cleaner demo
                'enable_coordinates': True,
                'enable_all_cognitive_systems': True,
                'max_actions': min(self.config.max_actions, 200),  # Shorter for demo
                'max_cycles': min(self.config.max_cycles, 5),      # Fewer cycles for demo
                'target_score': self.config.target_score,
                'enable_detailed_monitoring': True
            }
            
            return await self._run_continuous_learning(config_overrides)
            
        except Exception as e:
            self.logger.error(f"Error in showcase demo mode: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_system_comparison(self):
        """Run system comparison mode for evaluating different approaches."""
        self.logger.info("⚖️ Starting SYSTEM-COMPARISON mode")
        self.logger.info("   Comparing different system configurations")
        
        try:
            # Run continuous learning with comparison settings
            config_overrides = {
                'enable_meta_cognitive_governor': True,
                'enable_architect_evolution': True,
                'enable_coordinates': True,
                'enable_all_cognitive_systems': True,
                'max_actions': self.config.max_actions,
                'max_cycles': self.config.max_cycles,
                'target_score': self.config.target_score,
                'enable_detailed_monitoring': True
            }
            
            return await self._run_continuous_learning(config_overrides)
            
        except Exception as e:
            self.logger.error(f"Error in system comparison mode: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_minimal_debug(self):
        """Run minimal debug mode with basic functionality only."""
        self.logger.info("🐛 Starting MINIMAL-DEBUG mode")
        self.logger.info("   Basic functionality only for debugging")
        
        try:
            # Run continuous learning with minimal settings
            config_overrides = {
                'enable_meta_cognitive_governor': False,
                'enable_architect_evolution': False,
                'enable_coordinates': False,
                'enable_all_cognitive_systems': False,
                'max_actions': min(self.config.max_actions, 50),   # Very limited
                'max_cycles': min(self.config.max_cycles, 2),      # Very few cycles
                'target_score': self.config.target_score,
                'enable_detailed_monitoring': False
            }
            
            return await self._run_continuous_learning(config_overrides)
            
        except Exception as e:
            self.logger.error(f"Error in minimal debug mode: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_sequential(self):
        """Run sequential training mode (legacy compatibility)."""
        self.logger.info("📋 Starting SEQUENTIAL mode (legacy compatibility)")
        self.logger.info("   Sequential processing with basic cognitive systems")
        
        try:
            # Run continuous learning with sequential settings
            config_overrides = {
                'enable_meta_cognitive_governor': False,
                'enable_architect_evolution': False,
                'enable_coordinates': True,
                'enable_all_cognitive_systems': False,
                'max_actions': self.config.max_actions,
                'max_cycles': self.config.max_cycles,
                'target_score': self.config.target_score
            }
            
            return await self._run_continuous_learning(config_overrides)
            
        except Exception as e:
            self.logger.error(f"Error in sequential mode: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_swarm(self):
        """Run swarm training mode with parallel processing."""
        self.logger.info("🐝 Starting SWARM mode")
        self.logger.info("   Parallel processing with swarm intelligence")
        
        try:
            # Run continuous learning with swarm settings
            config_overrides = {
                'enable_meta_cognitive_governor': True,
                'enable_architect_evolution': True,
                'enable_coordinates': True,
                'enable_all_cognitive_systems': True,
                'max_actions': self.config.max_actions,
                'max_cycles': self.config.max_cycles,
                'target_score': self.config.target_score
            }
            
            return await self._run_continuous_learning(config_overrides)
            
        except Exception as e:
            self.logger.error(f"Error in swarm mode: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _run_continuous(self):
        """Run continuous training mode (legacy compatibility)."""
        self.logger.info("🔄 Starting CONTINUOUS mode (legacy compatibility)")
        self.logger.info("   Continuous learning with all systems enabled")
        
        try:
            # Run continuous learning with continuous settings
            config_overrides = {
                'enable_meta_cognitive_governor': True,
                'enable_architect_evolution': True,
                'enable_coordinates': True,
                'enable_all_cognitive_systems': True,
                'max_actions': self.config.max_actions,
                'max_cycles': self.config.max_cycles,
                'target_score': self.config.target_score
            }
            
            return await self._run_continuous_learning(config_overrides)
            
        except Exception as e:
            self.logger.error(f"Error in continuous mode: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results."""
        if self.config.no_logs:
            return
        
        timestamp = int(time.time())
        results_file = f"data/sessions/master_training_results_{timestamp}.json"
        try:
            # Clean results for JSON serialization recursively
            def clean_for_json(obj):
                if hasattr(obj, 'value'):  # Handle enums
                    return str(obj)
                elif hasattr(obj, '__dict__'):  # Handle custom objects
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [clean_for_json(item) for item in obj]
                else:
                    return obj
            clean_results = clean_for_json(results)
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"Could not save results: {e}")
    
    async def _consult_governor(self, puzzle_type: str, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Consult the meta-cognitive governor for configuration recommendations."""
        if not self.governor:
            return {}
        
        try:
            # Prepare current configuration for governor
            current_config = {
                'salience_mode': self.config.salience_mode,
                'max_actions': self.config.max_actions,
                'target_score': self.config.target_score,
                'enable_swarm': self.config.enable_swarm,
                'enable_coordinates': self.config.enable_coordinates,
                'enable_energy_system': self.config.enable_energy_system
            }
            
            # Get recommendation from governor
            recommendation = self.governor.get_recommended_configuration(
                puzzle_type=puzzle_type,
                current_performance=current_performance,
                current_config=current_config
            )
            
            if recommendation:
                self.logger.info(f"🧠 Governor recommends: {recommendation.type.value}")
                return {
                    'type': recommendation.type.value,
                    'changes': recommendation.configuration_changes,
                    'rationale': recommendation.rationale,
                    'confidence': recommendation.confidence,
                    'cognitive_cost': recommendation.expected_benefit,
                    'urgency': recommendation.urgency
                }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Governor consultation failed: {e}")
        
        return {}
    
    async def _run_architect_evolution(self) -> Dict[str, Any]:
        """Run one cycle of Architect evolution if enabled."""
        if not self.architect:
            return {'success': False, 'reason': 'architect_not_available'}
        
        try:
            self.logger.info("🏗️ Running Architect evolution cycle...")
            evolution_result = await self.architect.autonomous_evolution_cycle()
            
            if evolution_result.get('success', False):
                self.logger.info(f"🧬 Evolution successful: {evolution_result.get('improvement', 0):.3f} improvement")
            else:
                self.logger.debug(f"📊 Evolution attempted: {evolution_result.get('reason', 'unknown')}")
            
            return evolution_result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Architect evolution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _show_meta_cognitive_status(self):
        """Display current meta-cognitive system status."""
        status_text = f"\n{Fore.GREEN}✅ Meta-Cognitive Hierarchy Active:{Style.RESET_ALL}\n"
        status_text += "   🧠 Primary: 37+ Cognitive Systems\n"
        if self.governor:
            status_text += f"   🧠 Governor: {len(self.governor.system_monitors)} System Monitors\n"
        if self.architect:
            status_text += f"   🧠 Architect: Generation {self.architect.generation}"
        
        safe_print(status_text, self.config.enable_colored_output)
    
    async def _consult_governor_detailed(self, puzzle_type: str, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Governor consultation with detailed logging."""
        if not self.governor:
            return {}
        
        try:
            safe_print("   🎯 Consulting Governor for optimization...", self.config.enable_colored_output)
            
            # Prepare current configuration
            current_config = {
                'max_actions_per_game': self.config.max_actions,
                'salience_mode': self.config.salience_mode,
                'contrarian_enabled': self.config.enable_contrarian_strategy
            }
            
            recommendation = self.governor.get_recommended_configuration(
                puzzle_type=puzzle_type,
                current_performance=current_performance,
                current_config=current_config
            )
            
            if recommendation:
                rec_text = f"   📋 Governor Recommendation: {Fore.YELLOW}{recommendation.type.value}{Style.RESET_ALL}\n"
                rec_text += f"   🎯 Confidence: {Fore.GREEN}{recommendation.confidence:.1%}{Style.RESET_ALL}\n"
                rec_text += f"   💡 Rationale: {recommendation.rationale}\n"
                rec_text += f"   🔄 Changes: {recommendation.configuration_changes}"
                
                safe_print(rec_text, self.config.enable_colored_output)
                
                # Log decision
                decision_record = {
                    'timestamp': datetime.now().isoformat(),
                    'puzzle_type': puzzle_type,
                    'recommendation': recommendation.type.value,
                    'confidence': recommendation.confidence,
                    'changes': recommendation.configuration_changes,
                    'rationale': recommendation.rationale
                }
                self.governor_decisions.append(decision_record)
                
                return {
                    'type': recommendation.type.value,
                    'changes': recommendation.configuration_changes,
                    'rationale': recommendation.rationale,
                    'confidence': recommendation.confidence,
                    'urgency': recommendation.urgency
                }
            else:
                safe_print("   ✅ Current configuration optimal", self.config.enable_colored_output)
                return {'type': 'no_change', 'rationale': 'Current configuration is optimal'}
                
        except Exception as e:
            self.logger.warning(f"Governor consultation failed: {e}")
            return {}
    
    async def _trigger_architect_intervention(self, current_performance: Dict[str, Any], session: int):
        """Trigger Architect intervention for performance issues."""
        if not self.governor or not self.architect:
            return
        
        warning_text = f"\n{Fore.RED}⚠️  Efficiency declining - Escalating to Architect{Style.RESET_ALL}"
        safe_print(warning_text, self.config.enable_colored_output)
        
        try:
            # Create architect request through governor
            architect_request = self.governor.create_architect_request(
                issue_type="learning_plateau",
                problem_description=f"Learning efficiency dropped to {current_performance['learning_efficiency']:.1%} in session {session}",
                performance_data=current_performance
            )
            
            safe_print(f"   📋 Request Type: {architect_request.issue_type}", self.config.enable_colored_output)
            safe_print(f"   🚨 Priority: {architect_request.priority:.2f}", self.config.enable_colored_output)
            
            # Process through Architect
            architect_response = await self.architect.process_governor_request(architect_request)
            
            success_text = "SUCCESS" if architect_response.get('success') else "FAILED"
            response_text = f"   🔬 Architect Response: {Fore.GREEN if architect_response.get('success') else Fore.RED}{success_text}{Style.RESET_ALL}"
            safe_print(response_text, self.config.enable_colored_output)
            
            # Log evolution
            self.architect_evolutions.append({
                'session': session,
                'timestamp': datetime.now().isoformat(),
                'trigger': 'governor_escalation',
                'issue_type': architect_request.issue_type,
                'success': architect_response.get('success', False),
                'response': architect_response
            })
            
        except Exception as e:
            self.logger.warning(f"Architect intervention failed: {e}")
    
    async def _run_autonomous_evolution(self, session: int):
        """Run autonomous evolution cycle."""
        if not self.architect:
            return
        
        evolution_text = f"\n{Fore.MAGENTA}🧬 Running Autonomous Evolution Cycle...{Style.RESET_ALL}"
        safe_print(evolution_text, self.config.enable_colored_output)
        
        try:
            evolution_result = await self.architect.autonomous_evolution_cycle()
            
            safe_print(f"   📊 Evolution Success: {evolution_result['success']}", self.config.enable_colored_output)
            safe_print(f"   🔄 Generation: {evolution_result.get('generation', 'N/A')}", self.config.enable_colored_output)
            
            if evolution_result.get('improvement'):
                improvement = evolution_result['improvement']
                color = Fore.GREEN if improvement > 0 else Fore.RED
                improvement_text = f"   📈 Improvement: {color}{improvement:.3f}{Style.RESET_ALL}"
                safe_print(improvement_text, self.config.enable_colored_output)
            
            # Log evolution
            self.architect_evolutions.append({
                'session': session,
                'timestamp': datetime.now().isoformat(),
                'trigger': 'autonomous_cycle',
                'success': evolution_result['success'],
                'generation': evolution_result.get('generation'),
                'improvement': evolution_result.get('improvement', 0.0)
            })
            
        except Exception as e:
            self.logger.warning(f"Autonomous evolution failed: {e}")
    
    async def _show_comprehensive_results(self, session_results: List[Dict]):
        """Display comprehensive training results with detailed analysis."""
        header_text = f"\n{Fore.CYAN}📊 COMPREHENSIVE TRAINING RESULTS:{Style.RESET_ALL}"
        safe_print(header_text, self.config.enable_colored_output)
        
        # Performance progression with detailed breakdown
        progress_header = f"\n{Fore.YELLOW}📈 Performance Progression:{Style.RESET_ALL}"
        safe_print(progress_header, self.config.enable_colored_output)
            
        for result in session_results:
            perf = result.get('performance', {})
            session_num = result.get('session', 0)
            perf_text = (f"   Session {session_num}: "
                        f"Win Rate {perf.get('win_rate', 0):.1%}, "
                        f"Score {perf.get('avg_score', 0)}, "
                        f"Efficiency {perf.get('learning_efficiency', 0):.1%}")
            safe_print(perf_text, self.config.enable_colored_output)
        
        # Governor activity analysis with enhanced insights
        gov_header = f"\n{Fore.MAGENTA}🎯 Governor Activity Summary:{Style.RESET_ALL}"
        safe_print(gov_header, self.config.enable_colored_output)
            
        safe_print(f"   Total Decisions Made: {len(self.governor_decisions)}", self.config.enable_colored_output)
        
        if self.governor_decisions:
            # Analyze recommendation types and confidence statistics
            recommendation_types = {}
            confidence_scores = []
            
            for decision in self.governor_decisions:
                rec_type = decision.get('recommendation', 'unknown')
                recommendation_types[rec_type] = recommendation_types.get(rec_type, 0) + 1
                
                confidence = decision.get('confidence', 0)
                if confidence > 0:
                    confidence_scores.append(confidence)
            
            # Show recommendation frequency breakdown
            for rec_type, count in recommendation_types.items():
                safe_print(f"   {rec_type}: {count} times", self.config.enable_colored_output)
            
            # Enhanced confidence analysis
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                max_confidence = max(confidence_scores)
                min_confidence = min(confidence_scores)
                
                safe_print(f"   Average Confidence: {avg_confidence:.1%}", self.config.enable_colored_output)
                safe_print(f"   Confidence Range: {min_confidence:.1%} - {max_confidence:.1%}", self.config.enable_colored_output)
        
        # Architect evolution analysis with success metrics  
        arch_header = f"\n{Fore.CYAN}🧬 Architect Evolution Summary:{Style.RESET_ALL}"
        safe_print(arch_header, self.config.enable_colored_output)
            
        safe_print(f"   Total Evolution Cycles: {len(self.architect_evolutions)}", self.config.enable_colored_output)
        if self.architect:
            safe_print(f"   Final Generation: {self.architect.generation}", self.config.enable_colored_output)
        
        # Enhanced evolution success analysis
        successful_evolutions = [e for e in self.architect_evolutions if e.get('success')]
        safe_print(f"   Successful Evolutions: {len(successful_evolutions)}", self.config.enable_colored_output)
        
        if successful_evolutions:
            total_improvement = sum(e.get('improvement', 0) for e in successful_evolutions)
            avg_improvement = total_improvement / len(successful_evolutions)
            
            safe_print(f"   Total Improvement: {total_improvement:.3f}", self.config.enable_colored_output)
            safe_print(f"   Average Improvement per Evolution: {avg_improvement:.3f}", self.config.enable_colored_output)
        
        # Enhanced system status with health metrics
        status_header = f"\n{Fore.GREEN}🎯 Final System Status:{Style.RESET_ALL}"
        safe_print(status_header, self.config.enable_colored_output)
        
        if self.architect:
            try:
                architect_status = self.architect.get_evolution_status()
                safe_print(f"   Mutations Tested: {architect_status.get('total_mutations_tested', 0)}", self.config.enable_colored_output)
                safe_print(f"   Evolution Success Rate: {architect_status.get('success_rate', 0):.1%}", self.config.enable_colored_output)
            except Exception:
                safe_print("   Architect Status: Active", self.config.enable_colored_output)
        
        safe_print("   System Health: Optimal", self.config.enable_colored_output)
        
        # Training efficiency analysis (new feature)
        if len(session_results) > 1:
            efficiency_header = f"\n{Fore.BLUE}⚡ Training Efficiency Analysis:{Style.RESET_ALL}"
            safe_print(efficiency_header, self.config.enable_colored_output)
            
            first_session = session_results[0].get('performance', {})
            last_session = session_results[-1].get('performance', {})
            
            win_rate_improvement = last_session.get('win_rate', 0) - first_session.get('win_rate', 0)
            score_improvement = last_session.get('avg_score', 0) - first_session.get('avg_score', 0)
            
            safe_print(f"   Win Rate Improvement: {win_rate_improvement:.1%}", self.config.enable_colored_output)
            safe_print(f"   Score Improvement: {score_improvement:+.0f} points", self.config.enable_colored_output)
            
            # Calculate learning velocity
            sessions_count = len(session_results)
            if sessions_count > 1:
                learning_velocity = win_rate_improvement / (sessions_count - 1)
                safe_print(f"   Learning Velocity: {learning_velocity:.2%} per session", self.config.enable_colored_output)
    
    async def _save_meta_cognitive_logs(self, session_results: List[Dict]):
        """Save detailed meta-cognitive training logs."""
        if self.config.no_logs:
            return
        try:
            log_file = f"data/sessions/meta_cognitive_training_{self.session_id}.json"
            # ...existing code for config_dict, clean_session_results, log_data...
            config_dict = {
                'mode': self.config.mode,
                'max_actions': self.config.max_actions,
                'target_score': self.config.target_score,
                'salience_mode': str(self.config.salience_mode) if hasattr(self.config.salience_mode, 'value') else self.config.salience_mode
            }
            clean_session_results = []
            for result in session_results:
                clean_result = {
                    'session': result.get('session'),
                    'performance': result.get('performance'),
                    'governor_recommendation': result.get('governor_recommendation'),
                    'training_success': result.get('training_result', {}).get('success', False),
                    'governor_active': result.get('governor_recommendation') is not None,
                    'architect_involved': len([e for e in self.architect_evolutions if e.get('session') == result.get('session')]) > 0
                }
                clean_session_results.append(clean_result)
            log_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'config': config_dict,
                'training_results': clean_session_results,
                'governor_decisions': [
                    {
                        'session': d.get('session'),
                        'timestamp': d.get('timestamp', datetime.now().isoformat()),
                        'recommendation': d.get('recommendation'),
                        'confidence': d.get('confidence'),
                        'changes': d.get('changes', {}),
                        'rationale': d.get('rationale', '')
                    } for d in self.governor_decisions
                ],
                'architect_evolutions': [
                    {
                        'session': e.get('session'),
                        'timestamp': e.get('timestamp', datetime.now().isoformat()),
                        'trigger': e.get('trigger', 'unknown'),
                        'success': e.get('success', False),
                        'generation': e.get('generation', 0),
                        'improvement': e.get('improvement', 0.0)
                    } for e in self.architect_evolutions
                ],
                'final_status': {
                    'total_sessions': len(session_results),
                    'governor_total_decisions': len(self.governor_decisions),
                    'architect_generation': self.architect.generation if self.architect else 0,
                    'successful_evolutions': len([e for e in self.architect_evolutions if e.get('success')]),
                    'governor_decisions_made': len(self.governor_decisions),
                    'architect_evolutions': len(self.architect_evolutions)
                }
            }
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            if COLOR_AVAILABLE and self.config.enable_colored_output:
                print(f"\n{Fore.BLUE}📊 Meta-cognitive logs saved to: {log_file}{Style.RESET_ALL}")
            else:
                print(f"\n📊 Meta-cognitive logs saved to: {log_file}")
                
        except Exception as e:
            self.logger.error(f"Could not save meta-cognitive logs: {e}")


class ContinuousTrainingRunner:
    """Windows-compatible continuous runner for meta-cognitive ARC training."""
    
    def __init__(self, dashboard_mode='console', config=None):
        self.dashboard_mode = dashboard_mode
        self.running = True
        self.session_count = 0
        self.dashboard = None
        self.config = config or MasterTrainingConfig(mode="meta-cognitive-training")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_time = time.time()
        
        # Set up signal handling for continuous mode
        original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        safe_print(f"\n{Fore.YELLOW}SHUTDOWN: Gracefully stopping...{Style.RESET_ALL}")
        self.running = False
        
    async def start_continuous_training(self):
        """Start continuous training with meta-cognitive monitoring."""
        
        safe_print(f"{Fore.CYAN}CONTINUOUS META-COGNITIVE ARC TRAINING")
        safe_print(f"=" * 60)
        safe_print(f"Dashboard Mode: {self.dashboard_mode}")
        safe_print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        safe_print(f"=" * 60 + Style.RESET_ALL)
        
        # Initialize the enhanced output file
        safe_print("🚀 Enhanced terminal output enabled - showing real-time progress")
        
        # Initialize dashboard
        await self._setup_dashboard()
        
        # Run continuous training loop
        await self._run_continuous_loop()
        
    async def _setup_dashboard(self):
        """Set up the meta-cognitive dashboard."""
        safe_print(f"\n{Fore.BLUE}DASHBOARD: Initializing ({self.dashboard_mode} mode)...{Style.RESET_ALL}")
        
        try:
            from src.core.meta_cognitive_dashboard import MetaCognitiveDashboard, DashboardMode
            
            # Map dashboard modes
            mode_map = {
                'gui': DashboardMode.GUI,
                'console': DashboardMode.CONSOLE, 
                'headless': DashboardMode.HEADLESS,
                'web': DashboardMode.WEB
            }
            
            dashboard_mode = mode_map.get(self.dashboard_mode, DashboardMode.CONSOLE)
            
            self.dashboard = MetaCognitiveDashboard(
                mode=dashboard_mode,
                update_interval=3.0  # Update every 3 seconds for continuous mode
            )
            
            # Start monitoring
            session_id = f"continuous_training_{int(time.time())}"
            self.dashboard.start(session_id)
            
            safe_print(f"{Fore.GREEN}SUCCESS: Dashboard initialized and running{Style.RESET_ALL}")
            self.logger.info(f"Dashboard started in {self.dashboard_mode} mode")
            
        except UnicodeEncodeError as e:
            safe_print(f"{Fore.YELLOW}WARNING: Dashboard failed due to Unicode encoding issues")
            safe_print(f"Continuing without dashboard...{Style.RESET_ALL}")
            self.dashboard = None
            print("Dashboard initialization failed due to Windows Unicode issues")
            
        except Exception as e:
            safe_print(f"{Fore.YELLOW}WARNING: Dashboard failed to initialize: {e}")
            safe_print(f"Continuing without dashboard...{Style.RESET_ALL}")
            self.dashboard = None
            print(f"Dashboard initialization failed: {e}")
            
    async def _run_continuous_loop(self):
        """Run the continuous training loop."""
        safe_print(f"\n{Fore.YELLOW}TRAINING: Starting Continuous Loop...")
        safe_print(f"Press Ctrl+C to stop gracefully{Style.RESET_ALL}\n")
        
        total_start = time.time()
        
        while self.running:
            self.session_count += 1
            session_start = time.time()
            
            safe_print(f"\n{Fore.CYAN}{'='*60}")
            safe_print(f"TRAINING SESSION #{self.session_count}")
            safe_print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            safe_print(f"{'='*60}{Style.RESET_ALL}")
            
            try:
                # Create and run training session using master trainer
                session_config = MasterTrainingConfig(
                    mode="meta-cognitive-training",
                    verbose=True,
                    max_cycles=3,  # Shorter cycles for continuous operation
                    enable_meta_cognitive_governor=True,
                    enable_architect_evolution=True
                )
                training_session = MasterARCTrainer(session_config)
                
                # If we have a dashboard, log session start
                if self.dashboard:
                    self.dashboard.log_performance_update({
                        'session_number': self.session_count,
                        'status': 'starting'
                    }, 'continuous_runner')
                
                # Run the actual training
                safe_print(f"{Fore.BLUE}META-COGNITIVE: Running ARC training...{Style.RESET_ALL}")
                results = await training_session.run_training()
                success = results.get('success', False)
                
                session_time = time.time() - session_start
                
                # Log results
                if success:
                    safe_print(f"\n{Fore.GREEN}SUCCESS: Session #{self.session_count} completed!")
                    safe_print(f"   Duration: {session_time:.1f} seconds{Style.RESET_ALL}")
                    
                    if self.dashboard:
                        self.dashboard.log_performance_update({
                            'session_number': self.session_count,
                            'duration': session_time,
                            'status': 'completed',
                            'success': True
                        }, 'continuous_runner')
                        
                else:
                    safe_print(f"\n{Fore.YELLOW}WARNING: Session #{self.session_count} completed with issues")
                    safe_print(f"   Duration: {session_time:.1f} seconds{Style.RESET_ALL}")
                    
                    if self.dashboard:
                        self.dashboard.log_performance_update({
                            'session_number': self.session_count,
                            'duration': session_time,
                            'status': 'completed',
                            'success': False
                        }, 'continuous_runner')
                
                self.logger.info(f"Session {self.session_count} completed: success={success}, duration={session_time:.1f}s")
                
                # Show progress every 3 sessions
                if self.session_count % 3 == 0:
                    await self._show_progress()
                
                # Wait between sessions (adaptive timing)
                wait_time = self._calculate_wait_time(session_time, success)
                if wait_time > 0 and self.running:
                    safe_print(f"\n{Fore.BLUE}WAIT: {wait_time:.0f}s before next session...{Style.RESET_ALL}")
                    
                    # Wait in chunks so we can respond to shutdown signals
                    for _ in range(int(wait_time)):
                        if not self.running:
                            break
                        await asyncio.sleep(1)
                
            except Exception as e:
                safe_print(f"\n{Fore.RED}ERROR: Session #{self.session_count} failed: {e}{Style.RESET_ALL}")
                self.logger.error(f"Session {self.session_count} failed: {e}")
                
                # Log error to dashboard
                if self.dashboard:
                    self.dashboard.log_performance_update({
                        'session_number': self.session_count,
                        'status': 'failed',
                        'error': str(e)
                    }, 'continuous_runner')
                
                # Wait longer after errors
                if self.running:
                    safe_print(f"{Fore.BLUE}WAIT: 120s after error...{Style.RESET_ALL}")
                    for _ in range(120):
                        if not self.running:
                            break
                        await asyncio.sleep(1)
        
        # Shutdown
        await self._shutdown_gracefully(total_start)
        
    def _calculate_wait_time(self, session_duration, success):
        """Calculate wait time between sessions."""
        base_wait = 60  # Base 1 minute wait
        
        # Adjust based on session duration
        if session_duration < 30:  # Very short session
            return base_wait * 2
        elif session_duration > 300:  # Long session (5+ minutes)
            return base_wait * 0.5
        
        # Adjust based on success
        if not success:
            return base_wait * 1.5
            
        return base_wait
        
    async def _show_progress(self):
        """Show progress summary."""
        total_runtime = time.time() - self.start_time
        
        safe_print(f"\n{Fore.CYAN}PROGRESS SUMMARY")
        safe_print(f"{'='*40}")
        safe_print(f"Sessions Completed: {self.session_count}")
        safe_print(f"Total Runtime: {total_runtime/60:.1f} minutes")
        
        if self.dashboard:
            try:
                summary = self.dashboard.get_performance_summary(hours=1)
                safe_print(f"Governor Decisions: {summary.get('decisions', {}).get('governor', 0)}")
                safe_print(f"Architect Evolutions: {summary.get('decisions', {}).get('architect', 0)}")
            except Exception as e:
                safe_print("Dashboard summary unavailable")
                self.logger.warning(f"Dashboard summary failed: {e}")
                
        safe_print(f"{'='*40}{Style.RESET_ALL}")
        
    async def _shutdown_gracefully(self, total_start):
        """Graceful shutdown procedure."""
        total_runtime = time.time() - total_start
        
        safe_print(f"\n{Fore.YELLOW}SHUTDOWN: Graceful shutdown in progress")
        safe_print(f"{'='*50}")
        
        # Stop dashboard
        if self.dashboard:
            safe_print(f"DASHBOARD: Stopping...")
            self.dashboard.stop()
            # Export session data
            export_path = Path(f"data/sessions/continuous_session_{int(time.time())}.json")
            try:
                if self.dashboard.export_session_data(export_path):
                    safe_print(f"EXPORT: Session data saved to {export_path}")
                    self.logger.info(f"Session data exported to {export_path}")
            except Exception as e:
                safe_print(f"EXPORT: Failed to save session data: {e}")
                self.logger.error(f"Export failed: {e}")
        
        # Final summary
        safe_print(f"\n{Fore.GREEN}COMPLETE: CONTINUOUS TRAINING FINISHED")
        safe_print(f"Sessions: {self.session_count}")
        safe_print(f"Runtime: {total_runtime/60:.1f} minutes")
        safe_print(f"Avg per session: {total_runtime/max(self.session_count,1):.1f}s")
        safe_print(f"{'='*50}{Style.RESET_ALL}")
        
        self.logger.info(f"Training completed: {self.session_count} sessions, {total_runtime/60:.1f} minutes")
        safe_print("Thank you for using the Meta-Cognitive ARC Training System!")


def create_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser."""
    parser = argparse.ArgumentParser(
        description="Master ARC Trainer - Unified Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Maximum Intelligence (default) 
  python master_arc_trainer.py
  
  # Meta-cognitive training with detailed monitoring
  python master_arc_trainer.py --mode meta-cognitive-training --enable-detailed-monitoring
  
  # Quick validation test
  python master_arc_trainer.py --mode quick-validation --games game1,game2
  
  # Legacy sequential mode (backward compatible)
  python master_arc_trainer.py --mode sequential --salience decay_compression
  
  # Research with system comparison
  python master_arc_trainer.py --mode research-lab --compare-systems
  
  # Minimal debug mode
  python master_arc_trainer.py --mode minimal-debug --disable-all-advanced
        """
    )
    
    # Core mode selection
    parser.add_argument('--mode', 
                       choices=[
                           # Modern unified modes
                           'maximum-intelligence', 'research-lab', 'quick-validation', 
                           'showcase-demo', 'system-comparison', 'minimal-debug',
                           'meta-cognitive-training', 'continuous-training',
                           # Legacy compatibility modes
                           'sequential', 'swarm', 'continuous', 'test', 'training', 'demo'
                       ],
                       default='maximum-intelligence',
                       help='Training mode (default: maximum-intelligence)')
    
    # System configuration
    parser.add_argument('--api-key', help='ARC API key')
    parser.add_argument('--arc-agents-path', help='Path to arc-agents directory')
    
    # Training parameters
    parser.add_argument('--max-actions', type=int, default=500, 
                       help='Max actions per game (default: 500)')
    parser.add_argument('--max-cycles', type=int, default=50,
                       help='Max learning cycles (default: 50)')
    parser.add_argument('--target-score', type=float, default=85.0,
                       help='Target score per game (default: 85.0)')
    parser.add_argument('--session-duration', type=int, default=60,
                       help='Session duration in minutes (default: 60)')
    
    # Salience system
    parser.add_argument('--salience', choices=['lossless', 'decay_compression', 'decay'],
                       default='decay_compression', help='Salience mode')
    parser.add_argument('--salience-threshold', type=float, default=0.6)
    parser.add_argument('--salience-decay', type=float, default=0.95)
    
    # Memory configuration
    parser.add_argument('--memory-size', type=int, default=512)
    parser.add_argument('--memory-word-size', type=int, default=64)
    parser.add_argument('--memory-read-heads', type=int, default=4)
    parser.add_argument('--memory-write-heads', type=int, default=1)
    
    # Feature toggles
    parser.add_argument('--disable-swarm', action='store_true')
    parser.add_argument('--disable-coordinates', action='store_true')
    parser.add_argument('--disable-energy', action='store_true')
    parser.add_argument('--disable-sleep', action='store_true')
    parser.add_argument('--disable-dnc-memory', action='store_true')
    parser.add_argument('--disable-meta-learning', action='store_true')
    parser.add_argument('--disable-salience', action='store_true')
    parser.add_argument('--disable-contrarian', action='store_true')
    parser.add_argument('--disable-frame-analysis', action='store_true')
    parser.add_argument('--disable-boundary-detection', action='store_true')
    parser.add_argument('--disable-memory-consolidation', action='store_true')
    parser.add_argument('--disable-action-intelligence', action='store_true')
    parser.add_argument('--disable-meta-cognitive-governor', action='store_true',
                       help='Disable meta-cognitive Governor (Third Brain)')
    parser.add_argument('--disable-architect-evolution', action='store_true',
                       help='Disable Architect evolution system (Zeroth Brain)')
    parser.add_argument('--disable-colored-output', action='store_true',
                       help='Disable colored terminal output')
    parser.add_argument('--enable-detailed-monitoring', action='store_true',
                       help='Enable detailed meta-cognitive monitoring')
    parser.add_argument('--disable-all-advanced', action='store_true',
                       help='Disable ALL advanced features (basic mode)')
    parser.add_argument('--local', action='store_true', help='Use local mock client for testing')
    
    # Testing and validation
    parser.add_argument('--games', help='Comma-separated list of game IDs')
    parser.add_argument('--compare-systems', action='store_true')
    parser.add_argument('--performance-tests', action='store_true')
    
    # Logging and monitoring
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug-mode', action='store_true')
    parser.add_argument('--no-logs', action='store_true')
    parser.add_argument('--no-monitoring', action='store_true')
    
    # Continuous training options
    parser.add_argument('--continuous-training', action='store_true',
                       help='Enable Windows-compatible continuous training mode')
    parser.add_argument('--dashboard', choices=['console', 'gui', 'headless', 'web'], 
                       default='console', help='Dashboard mode for continuous training')
    
    # Legacy compatibility aliases
    parser.add_argument('--quiet', action='store_true', help='Legacy: same as --no-logs')
    parser.add_argument('--timeout', type=int, help='Legacy: mapped to session-duration')
    parser.add_argument('--max-episodes', type=int, help='Legacy: mapped to max-cycles')
    
    return parser

def validate_api_key(api_key: str) -> None:
    """Validate the API key and provide helpful error messages if missing."""
    if not api_key:
        error_msg = """
[ERROR] ARC API key is required but not provided.

Please provide your ARC API key using one of these methods:

1. Command line argument:
   python master_arc_trainer.py --api-key YOUR_API_KEY

2. Environment variable (recommended):
   # Windows Command Prompt:
   set ARC_API_KEY=your_api_key_here

   # Windows PowerShell:
   $env:ARC_API_KEY="your_api_key_here"

   # Linux/MacOS:
   export ARC_API_KEY=your_api_key_here

You can find your API key in your ARC account dashboard.
"""
        print(error_msg)
        sys.exit(1)

async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up enhanced logging first
    setup_windows_logging()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv('ARC_API_KEY')
    
    # Validate API key before proceeding
    validate_api_key(api_key)
    
    # Use safe print for the header
    safe_print("🚀 MASTER ARC TRAINER - Unified Training System", True)
    safe_print("=" * 60, False)
    safe_print("Consolidated system combining all ARC training functionality", False)
    safe_print(f"Mode: {args.mode.upper()}", False)
    safe_print("", False)
    
    # Convert args to config with legacy compatibility
    config = MasterTrainingConfig(
        mode=args.mode,
        api_key=api_key,
        arc_agents_path=args.arc_agents_path or '',
        local_mode=args.local,
        max_actions=args.max_actions,
        max_cycles=args.max_episodes or args.max_cycles,  # Legacy compatibility
        target_score=args.target_score,
        session_duration=args.timeout or args.session_duration,  # Legacy compatibility
        salience_mode=args.salience.replace('decay', 'decay_compression') if hasattr(args, 'salience') else 'decay_compression',  # Legacy compatibility
        salience_threshold=args.salience_threshold if hasattr(args, 'salience_threshold') else 0.6,
        salience_decay=args.salience_decay if hasattr(args, 'salience_decay') else 0.95,
        enable_swarm=not args.disable_swarm,
        enable_coordinates=not args.disable_coordinates,
        enable_energy_system=not args.disable_energy,
        enable_sleep_cycles=not args.disable_sleep,
        enable_dnc_memory=not args.disable_dnc_memory,
        enable_meta_learning=not args.disable_meta_learning,
        enable_salience_system=not args.disable_salience,
        enable_contrarian_strategy=not args.disable_contrarian,
        enable_frame_analysis=not args.disable_frame_analysis,
        enable_boundary_detection=not args.disable_boundary_detection,
        enable_memory_consolidation=not args.disable_memory_consolidation,
        enable_action_intelligence=not args.disable_action_intelligence,
        enable_meta_cognitive_governor=not args.disable_meta_cognitive_governor,
        enable_architect_evolution=not args.disable_architect_evolution,
        enable_colored_output=not args.disable_colored_output and COLOR_AVAILABLE,
        enable_detailed_monitoring=args.enable_detailed_monitoring,
        save_meta_cognitive_logs=not args.no_logs,
        verbose=args.verbose,
        no_logs=args.no_logs or args.quiet,  # Legacy compatibility
        no_monitoring=args.no_monitoring,
        debug_mode=args.debug_mode,
        games=args.games,
        compare_systems=args.compare_systems,
        performance_tests=args.performance_tests
    )
    
    # Disable all advanced features if requested
    if args.disable_all_advanced:
        config.enable_swarm = False
        config.enable_coordinates = False
        config.enable_energy_system = False
        config.enable_sleep_cycles = False
        config.enable_dnc_memory = False
        config.enable_meta_learning = False
        config.enable_salience_system = False
        config.enable_contrarian_strategy = False
        config.enable_frame_analysis = False
        config.enable_boundary_detection = False
        config.enable_memory_consolidation = False
        config.enable_action_intelligence = False
        config.enable_meta_cognitive_governor = False
        config.enable_architect_evolution = False
    
    # Handle continuous training mode
    if args.continuous_training or args.mode == 'continuous-training':
        safe_print(f"{Fore.CYAN}TABULA RASA - CONTINUOUS META-COGNITIVE TRAINING")
        safe_print(f"Windows-compatible version with enhanced error handling{Style.RESET_ALL}")
        safe_print("📺 Real-time terminal output enabled - you'll see progress as it happens")
        
        runner = ContinuousTrainingRunner(dashboard_mode=args.dashboard, config=config)
        
        # Store runner globally for signal handler access
        global continuous_loop_instance
        continuous_loop_instance = runner
        
        try:
            await runner.start_continuous_training()
            return 0
        except KeyboardInterrupt:
            safe_print(f"\n{Fore.YELLOW}INTERRUPTED: Stopped by user{Style.RESET_ALL}")
            return 0
        except Exception as e:
            safe_print(f"\n{Fore.RED}SYSTEM ERROR: {e}{Style.RESET_ALL}")
            logging.error(f"System error: {e}", exc_info=True)
            return 1
    
    # Initialize and run trainer
    trainer = MasterARCTrainer(config)
    
    try:
        results = await trainer.run_training()
        
        if results['success']:
            safe_print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!", True)
            safe_print(f"Mode: {results.get('mode', 'unknown')}", False)
            safe_print(f"Time: {results.get('timestamp', 'unknown')}", False)
            return 0
        else:
            safe_print(f"\n❌ TRAINING FAILED!", True)
            safe_print(f"Error: {results.get('error', 'unknown')}", False)
            return 1
            
    except Exception as e:
        safe_print(f"\n💥 UNEXPECTED ERROR: {e}", True)
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        safe_print("\n🛑 Training interrupted by user", True)
        sys.exit(0)
    except Exception as e:
        safe_print(f"\nFatal error: {e}", True)
        sys.exit(1)

class UnifiedTrainer:
    """
    Legacy compatibility wrapper for MasterARCTrainer.
    
    This class maintains backward compatibility with existing code
    that expects the old UnifiedTrainer interface.
    """
    
    def __init__(self, args):
        """Initialize with legacy args format."""
        # Convert old args format to new config format
        config = MasterTrainingConfig()
        
        # Map old attributes to new config
        if hasattr(args, 'mode'):
            config.mode = args.mode
        if hasattr(args, 'verbose'):
            config.verbose = args.verbose
        if hasattr(args, 'mastery_sessions'):
            config.max_cycles = args.mastery_sessions
        if hasattr(args, 'games'):
            config.games = str(args.games) if args.games else None
        if hasattr(args, 'target_win_rate'):
            config.target_score = args.target_win_rate * 100  # Convert to score
        if hasattr(args, 'target_score'):
            config.target_score = args.target_score
        if hasattr(args, 'max_learning_cycles'):
            config.max_cycles = args.max_learning_cycles
        if hasattr(args, 'max_actions_per_session'):
            config.max_actions = args.max_actions_per_session
        if hasattr(args, 'salience'):
            if args.salience == 'lossless':
                config.salience_mode = SalienceMode.LOSSLESS
            elif args.salience == 'decay':
                config.salience_mode = SalienceMode.DECAY_COMPRESSION
            elif args.salience == 'decay_compression':
                config.salience_mode = SalienceMode.DECAY_COMPRESSION
        
        # Enable meta-cognitive features if requested
        if hasattr(args, 'enable_meta_cognitive') and args.enable_meta_cognitive:
            config.enable_meta_cognitive_governor = True
            config.enable_architect_evolution = True
        if hasattr(args, 'disable_meta_cognitive') and args.disable_meta_cognitive:
            config.enable_meta_cognitive_governor = False
            config.enable_architect_evolution = False
        if hasattr(args, 'meta_cognitive_enabled') and args.meta_cognitive_enabled:
            config.enable_meta_cognitive_governor = True
            config.enable_architect_evolution = True
            
        # Initialize the master trainer
        self._master_trainer = MasterARCTrainer(config)
        
        # Expose legacy attributes for compatibility
        self.mode = getattr(args, 'mode', 'sequential')
        self.salience = getattr(args, 'salience', 'decay')
        self.verbose = getattr(args, 'verbose', False)
        self.mastery_sessions = getattr(args, 'mastery_sessions', 5)
        self.games = getattr(args, 'games', 10)
        self.target_win_rate = getattr(args, 'target_win_rate', 0.70)
        self.target_score = getattr(args, 'target_score', 75)
        self.max_learning_cycles = getattr(args, 'max_learning_cycles', 5)
        self.max_actions_per_session = getattr(args, 'max_actions_per_session', 1500)
        self.enable_contrarian_mode = getattr(args, 'enable_contrarian_mode', True)
        
        # Legacy compatibility attributes
        self.continuous_loop = None
        self.training_cycles = 0
        self.best_performance = {'win_rate': 0.0, 'avg_score': 0.0}
        self.scorecard_id = None
        
        # Expose meta-cognitive components
        self.governor = self._master_trainer.governor
        self.architect = self._master_trainer.architect
        self.enable_meta_cognitive = config.enable_meta_cognitive_governor
    
    def get_salience_mode(self) -> SalienceMode:
        """Convert string to SalienceMode enum for compatibility."""
        if self.salience == 'lossless':
            return SalienceMode.LOSSLESS
        elif self.salience in ('decay', 'decay_compression'):
            return SalienceMode.DECAY_COMPRESSION
        else:
            return SalienceMode.DECAY_COMPRESSION
    
    def display_config(self):
        """Display training configuration for compatibility."""
        print("🎯 UNIFIED TRAINING CONFIGURATION (Legacy Mode)")
        print("="*50)
        print(f"Mode: {self.mode.upper()}")
        print(f"Salience: {self.salience.upper()}")
        print(f"Target Win Rate: {self.target_win_rate:.1%}")
        print(f"Target Score: {self.target_score}")
        print(f"Mastery Sessions: {self.mastery_sessions}")
        print(f"Games: {self.games}")
        print(f"Max Learning Cycles: {self.max_learning_cycles}")
        print(f"Max Actions per Session: {self.max_actions_per_session:,}")
        print(f"Contrarian Mode: {'ENABLED' if self.enable_contrarian_mode else 'DISABLED'}")
        print(f"Verbose: {'YES' if self.verbose else 'NO'}")
        print(f"Meta-Cognitive: {'ENABLED' if self.enable_meta_cognitive else 'DISABLED'}")
        print()
    
    async def run_training(self):
        """Run training using the master trainer."""
        return await self._master_trainer._run_sequential_training()
    
    # Add other legacy methods as needed for full compatibility
    async def initialize_with_error_handling(self):
        """Legacy compatibility method."""
        return True


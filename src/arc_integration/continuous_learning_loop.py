"""
Continuous Learning Loop for ARC-AGI-3 Training

This module implements a continuous learning system that runs the Adaptive Learning Agent
against ARC-AGI-3 tasks, collecting insights and improving performance over time.
"""
import asyncio
import aiohttp
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
import numpy as np

# Try to import torch, but don't fail if it's not available
try:
    import torch
except ImportError:
    torch = None

# Action & session trace logger (lightweight, append-only)
try:
    from arc_integration.action_trace_logger import log_action_trace, write_session_trace
except Exception:
    # Leave safe fallbacks if the module isn't available
    def log_action_trace(record):
        try:
            print(" action_trace_logger unavailable")
        except Exception:
            pass
    def write_session_trace(game_id, session_result, raw_output=None):
        try:
            print(" action_trace_logger unavailable")
        except Exception:
            pass

# Set up logger for this module
logger = logging.getLogger(__name__)

# =============================================================================
# ACTION LIMIT CONFIGURATION
# =============================================================================
# Import centralized action limits configuration
try:
    from action_limits_config import ActionLimits
except ImportError:
    # Fallback configuration if the config file is not available
    class ActionLimits:
        """Fallback configuration for action limits - optimized for learning."""
        MAX_ACTIONS_PER_GAME = 2000
        MAX_ACTIONS_PER_SESSION = 5000
        MAX_ACTIONS_PER_SCORECARD = 8000
        MAX_ACTIONS_PER_EPISODE = 1500
        MAX_ACTIONS_SCALING_BASE = 800
        MAX_ACTIONS_SCALING_MAX = 3000
        
        @classmethod
        def get_max_actions_per_game(cls) -> int:
            return cls.MAX_ACTIONS_PER_GAME
        
        @classmethod
        def get_max_actions_per_session(cls) -> int:
            return cls.MAX_ACTIONS_PER_SESSION
        
        @classmethod
        def get_max_actions_per_scorecard(cls) -> int:
            return cls.MAX_ACTIONS_PER_SCORECARD
        
        @classmethod
        def get_scaled_max_actions(cls, efficiency: float) -> int:
            scaled = int(cls.MAX_ACTIONS_SCALING_BASE * (1 + (1 - efficiency)))
            return min(cls.MAX_ACTIONS_SCALING_MAX, scaled)
try:
    # Runtime instrumentation: print the file path when this module is imported so we can
    # confirm which copy of the module the Python process actually loaded at runtime.
    try:
        # Local import to avoid NameError if Path/sys aren't available yet
        from pathlib import Path as _Path
        resolved_path = _Path(__file__).resolve()
    except Exception:
        import sys as _sys
        resolved_path = getattr(_sys.modules.get(__name__), '__file__', 'unknown')
    print(f"MODULE IMPORTED: continuous_learning_loop -> {resolved_path}")
except Exception:
    logger.exception("Failed to emit module import instrumentation")

from collections import deque  # Added for rate limiting
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple  # Add type hints
from arc_integration.arc_meta_learning import ARCMetaLearningSystem
from core.meta_learning import MetaLearningSystem
from core.salience_system import SalienceCalculator, SalienceMode, SalienceWeightedReplayBuffer

# Import Governor and Architect
try:
    from src.core.meta_cognitive_governor import MetaCognitiveGovernor
    from src.core.architect import Architect
    META_COGNITIVE_AVAILABLE = True
except ImportError as e:
    META_COGNITIVE_AVAILABLE = False
    print(f"WARNING: Meta-cognitive systems not available: {e}")
    MetaCognitiveGovernor = None
    Architect = None

# Import Enhanced Simulation Agent
try:
    from src.core.enhanced_simulation_agent import EnhancedSimulationAgent
    from src.core.enhanced_simulation_config import EnhancedSimulationConfig, LearningMode
    from src.core.predictive_core import PredictiveCore
    from src.core.simulation_models import SimulationContext
    SIMULATION_AVAILABLE = True
except ImportError as e:
    SIMULATION_AVAILABLE = False
    print(f"WARNING: Enhanced simulation intelligence not available: {e}")
    EnhancedSimulationAgent = None
    PredictiveCore = None
    EnhancedSimulationConfig = None
# Import FrameAnalyzer
try:
    from vision.frame_analyzer import FrameAnalyzer
except ImportError:
    # Fallback FrameAnalyzer if the main one isn't available
    class FrameAnalyzer:
        def __init__(self):
            pass
        def analyze_frame(self, frame, game_id):
            return {}
        def reset_coordinate_tracking(self):
            pass
        def reset_for_new_game(self, game_id):
            pass

# Import other required systems
try:
    from core.energy_system import EnergySystem
except ImportError:
    # Fallback EnergySystem
    class EnergySystem:
        def __init__(self):
            pass

try:
    from goals.goal_system import GoalInventionSystem, GoalPhase
except ImportError:
    # Fallback GoalInventionSystem
    class GoalInventionSystem:
        def __init__(self):
            pass
    
    # Fallback GoalPhase enum
    class GoalPhase:
        EMERGENT = "emergent"
        EXPLORATION = "exploration"
        EXPLOITATION = "exploitation"

try:
    from core.agent import AdaptiveLearningAgent
except ImportError:
    # Fallback AdaptiveLearningAgent
    class AdaptiveLearningAgent:
        def __init__(self, config):
            pass
# Fallback comparator if salience comparator module unavailable
try:
    from salience_mode_comparison import SalienceModeComparator
except Exception:
    class SalienceModeComparator:
        @staticmethod
        def compare_modes(*args, **kwargs):
            return {'comparison_available': False}

# ARC-3 API Configuration
ARC3_BASE_URL = "https://three.arcprize.org"
ARC3_SCOREBOARD_URL = "https://three.arcprize.org/scorecards"

# Rate Limiting Configuration for ARC-AGI-3 API
ARC3_RATE_LIMIT = {
    'requests_per_minute': 600,
    'requests_per_second': 10,
    'safe_requests_per_second': 8,
    'backoff_base_delay': 1.0,
    'backoff_max_delay': 60.0,
    'request_timeout': 30.0
}
class RateLimiter:
    """
    Rate limiter for ARC-AGI-3 API that respects the 600 RPM limit
    with exponential backoff for 429 responses.
    """
    def __init__(self):
        self.request_times = deque()  # Track request timestamps
        self.requests_per_second = ARC3_RATE_LIMIT['safe_requests_per_second']
        self.backoff_delay = 0.0  # Current backoff delay
        self.consecutive_429s = 0  # Count of consecutive 429 errors
        self.total_requests = 0
        self.total_429s = 0

    async def acquire(self):
        """Acquire permission to make a request, with rate limiting."""
        current_time = time.time()

        # Remove requests older than 1 minute
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()

        # Check if we're at the per-minute limit
        if len(self.request_times) >= ARC3_RATE_LIMIT['requests_per_minute']:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                print(f"⏸ Rate limit: Waiting {wait_time:.1f}s (at {len(self.request_times)}/600 RPM)")
                await asyncio.sleep(wait_time)

        # Check requests per second limit
        recent_requests = sum(1 for t in self.request_times if current_time - t <= 1)
        if recent_requests >= self.requests_per_second:
            wait_time = 1.0 / self.requests_per_second
            await asyncio.sleep(wait_time)

        # Apply backoff delay if we've been getting 429s
        if self.backoff_delay > 0:
            print(f"⏳ Backoff delay: {self.backoff_delay:.1f}s (after {self.consecutive_429s} 429s)")
            await asyncio.sleep(self.backoff_delay)

        # Record this request
        self.request_times.append(current_time)
        self.total_requests += 1

    def handle_429_response(self):
        """Handle a 429 rate limit response with exponential backoff."""
        self.consecutive_429s += 1
        self.total_429s += 1

        # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (max)
        self.backoff_delay = min(
            ARC3_RATE_LIMIT['backoff_base_delay'] * (2 ** (self.consecutive_429s - 1)),
            ARC3_RATE_LIMIT['backoff_max_delay']
        )

        print(f" Rate limit exceeded (429) - backing off {self.backoff_delay:.1f}s")

    def handle_success_response(self):
        """Handle a successful response - reset backoff."""
        if self.consecutive_429s > 0:
            print(f" Request succeeded - resetting backoff (was {self.backoff_delay:.1f}s)")
        self.consecutive_429s = 0
        self.backoff_delay = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            'total_requests': self.total_requests,
            'total_429s': self.total_429s,
            'current_backoff': self.backoff_delay,
            'consecutive_429s': self.consecutive_429s,
            'requests_in_last_minute': len(self.request_times),
            'rate_limit_hit_rate': self.total_429s / max(1, self.total_requests)
        }

# ===== GLOBAL ARC TASK CONFIGURATION =====

# Verified available tasks (will be populated dynamically from API)
ARC_AVAILABLE_TASKS = []

# Task selection limits to prevent overtraining
DEMO_TASK_LIMIT = 3              # Quick demonstration tasks
COMPARISON_TASK_LIMIT = 4        # Tasks for mode comparison
PERSISTENT_TASK_LIMIT = None     # No limit - use all tasks for mastery

def get_demo_tasks(randomize: bool = True) -> List[str]:
    """Get tasks for demo mode - limited and optionally randomized."""
    if randomize:
        return random.sample(ARC_AVAILABLE_TASKS, min(DEMO_TASK_LIMIT, len(ARC_AVAILABLE_TASKS)))
    else:
        return ARC_AVAILABLE_TASKS[:DEMO_TASK_LIMIT]

def get_comparison_tasks(randomize: bool = True) -> List[str]:
    """Get tasks for comparison mode - limited and optionally randomized."""
    if randomize:
        return random.sample(ARC_AVAILABLE_TASKS, min(COMPARISON_TASK_LIMIT, len(ARC_AVAILABLE_TASKS)))
    else:
        return ARC_AVAILABLE_TASKS[:COMPARISON_TASK_LIMIT]

def get_full_training_tasks(randomize: bool = False) -> List[str]:
    """Get tasks for full training mode - all tasks, optionally shuffled."""
    if randomize:
        shuffled = ARC_AVAILABLE_TASKS.copy()
        random.shuffle(shuffled)
        return shuffled
    return ARC_AVAILABLE_TASKS

# ===== END GLOBAL CONFIGURATION =====

@dataclass
class TrainingSession:
    """Represents a learning cycle configuration."""
    session_id: str
    games_to_play: List[str]
    max_mastery_sessions_per_game: int  # Renamed from episodes to mastery_sessions
    learning_rate_schedule: Dict[str, float]
    save_interval: int
    target_performance: Dict[str, float]
    max_actions_per_session: int = ActionLimits.get_max_actions_per_session()  # Default action limit per game session (will be overridden by dynamic calculation)
    enable_contrarian_strategy: bool = False  # New contrarian mode
    salience_mode: SalienceMode = SalienceMode.DECAY_COMPRESSION
    enable_salience_comparison: bool = False
    swarm_enabled: bool = True


class ContinuousLearningLoop:
    """
    Manages continuous learning sessions for the Adaptive Learning Agent on ARC tasks.

    This system:
    1. Uses the official ARC-3 API for game management
    2. Creates scorecards to track performance across sessions
    3. Manages game sessions with proper RESET/ACTION workflows
    4. Collects performance data and learning insights
    5. Adapts training parameters based on performance
    6. Transfers knowledge between different games
    7. Tracks long-term learning progress
    """

    def _initialize_available_actions_memory(self) -> Dict[str, Any]:
        """Initialize the available actions memory with all required keys and proper structure."""
        return {
            'current_game_id': None,
            'current_actions': [],
            'action_history': [],
            'action_effectiveness': {},
            'action_relevance_scores': {},
            'action_sequences': [],
            'winning_action_sequences': [],
            'coordinate_patterns': {},
            'action_transitions': {},
            'action_learning_stats': {
                'total_observations': 0,
                'pattern_confidence_threshold': 0.7,
                'movements_tracked': 0,
                'effects_catalogued': 0,
                'game_contexts_learned': 0
            },
            'action_semantic_mapping': {},
            'sequence_in_progress': [],
            'initial_actions': [],
            'action_stagnation': {},
            'universal_boundary_detection': {
                'boundary_data': {},
                'coordinate_attempts': {},
                'action_coordinate_history': {},
                'stuck_patterns': {},
                'success_zone_mapping': {},
                'danger_zones': {},
                'coordinate_clusters': {},
                'directional_systems': {
                    6: {'current_direction': {}, 'direction_history': {}}
                },
                'current_direction': {},
                'last_coordinates': {},
                'stuck_count': {},
                'coordinate_attempts': {}
            },
            'action6_boundary_detection': {
                'boundary_data': {},
                'coordinate_attempts': {},
                'last_coordinates': {},
                'stuck_count': {},
                'current_direction': {}
            }
        }

    def _ensure_available_actions_memory(self) -> None:
        """Ensure available_actions_memory is properly initialized with all required keys."""
        if not hasattr(self, 'available_actions_memory'):
            self.available_actions_memory = self._initialize_available_actions_memory()
            return
        
        # Ensure all required keys exist
        required_keys = self._initialize_available_actions_memory()
        for key, default_value in required_keys.items():
            if key not in self.available_actions_memory:
                self.available_actions_memory[key] = default_value.copy() if isinstance(default_value, dict) else default_value

    def __init__(
        self,
        arc_agents_path: str,
        tabula_rasa_path: str,
        api_key: Optional[str] = None,
        save_directory: str = "data"
    ):
        """Lightweight constructor - only sets up basic attributes."""
        print("🚀 Starting ContinuousLearningLoop initialization...")
        
        # Store basic configuration
        self.arc_agents_path = Path(arc_agents_path)
        self.tabula_rasa_path = Path(tabula_rasa_path)
        self.api_key = api_key or os.getenv('ARC_API_KEY')
        self.save_directory = Path(save_directory)
        
        # Initialize only the most basic attributes needed immediately
        self._initialized = False
        self._active_sessions = []
        
        # Create save directory
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ Basic initialization complete - complex setup will be done on first use")

    def _ensure_initialized(self):
        """Ensure the object is fully initialized before use."""
        if self._initialized:
            return
            
        print("🔧 Performing complex initialization...")
        
        # Initialize all the complex attributes that were previously in __init__
        self._initialize_complex_attributes()
        
        # Mark as initialized
        self._initialized = True
        print("✅ Complex initialization complete")

    def _initialize_complex_attributes(self):
        """Initialize all the complex attributes that were previously in __init__."""
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
        
        # Initialize energy system
        self.current_energy = 100.0
        self.low_energy_mode = False
        self.ENERGY_CRITICAL = 20.0
        self.ENERGY_RECOVERY = 60.0
        
        # Initialize energy costs and stats
        self.ENERGY_COSTS = {
            'action_base': 0.5,
            'action_effective': 0.3,
            'action_ineffective': 0.8,
            'computation_base': 0.1,
            'exploration_bonus': -0.1,
            'repetitive_penalty': 1.5
        }
        
        self.energy_stats = {
            'total_consumed': 0.0,
            'last_consumed': 0.0,
            'average_consumption': 0.0,
            'peak_consumption': 0.0
        }
        
        # Initialize progress tracker
        self._progress_tracker = {
            'actions_taken': 0,
            'score_progress': 0,
            'level_progress': 0,
            'recent_effectiveness': 0.0,
            'action_pattern_history': [],
            'explored_coordinates': set(),
            'termination_reason': None
        }
        
        # Initialize action cap system
        self._action_cap_system = {
            'enabled': True,
            'base_cap_fraction': 0.3,  # 30% of max_actions_per_game
            'min_cap_fraction': 0.1,   # 10% minimum
            'max_cap_fraction': 0.8,   # 80% maximum
            'multiplier_per_available_action': 0.02,  # 2% of max_actions_per_game per available action
            'early_termination_enabled': False,  # Disable early termination by default
            'progress_extension_enabled': True,
            'extension_threshold': 0.1  # 10% progress threshold
        }
        
        # Initialize global counters
        self.global_counters = {
            'total_memory_operations': 0,
            'total_sleep_cycles': 0,
            'total_memories_deleted': 0,
            'total_memories_combined': 0,
            'total_memories_strengthened': 0,
            'persistent_energy_level': 100.0,
            'total_actions': 0,
            'recent_consecutive_failures': 0
        }
        
        # Initialize available actions memory using the centralized method
        self.available_actions_memory = self._initialize_available_actions_memory()
        
        # Add additional fields specific to this initialization
        self.available_actions_memory.update({
            'action6_strategy': {
                'last_progress_action': 0,
                'last_action6_used': 0,
                'consecutive_failures': 0
            },
            'experiment_mode': False
        })
        
        # Initialize frame analysis and available actions tracking
        self._last_frame_analysis = {}
        self._last_available_actions = {}
        
        # Initialize game session tracking
        self.current_game_sessions = {}
        self.current_scorecard_id = None
        self._created_new_scorecard = False
        
        # Initialize performance tracking
        self.performance_history = []
        self.session_history = []
        
        # Initialize memory protection stats
        self.memory_protection_stats = {
            'total_protected': 0,
            'total_unprotected': 0,
            'protection_rate': 0.0
        }
        
        # Initialize strategy flags
        self.contrarian_strategy_active = False
        
        # Initialize other complex attributes
        self._initialize_remaining_attributes()

    def _initialize_remaining_attributes(self):
        """Initialize the remaining complex attributes."""
        # Initialize basic paths and directories
        self.phase0_experiment_results_dir = self.save_directory / "phase0_experiment_results"
        self.phase0_experiment_results_dir.mkdir(parents=True, exist_ok=True)
        self.lp_validation_results_path = self.phase0_experiment_results_dir / "lp_validation_results.yaml"
        self.survival_test_results_path = self.phase0_experiment_results_dir / "survival_test_results.yaml"
        self.phase0_logs_dir = self.phase0_experiment_results_dir / "logs"
        self.phase0_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize adaptive learning evaluation directory
        self.adaptive_learning_eval_dir = self.save_directory / "adaptive_learning_agi_evaluation_1756519407"
        self.adaptive_learning_eval_dir.mkdir(parents=True, exist_ok=True)
        self.architect_evolution_data_dir = self.save_directory / "architect_evolution_data"
        self.architect_evolution_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize meta-learning system
        try:
            self.arc_meta_learning = ARCMetaLearningSystem(
                base_meta_learning=str(self.save_directory / "base_meta_learning")
            )
        except Exception as e:
            print(f"⚠️  ARCMetaLearningSystem initialization failed: {e}")
            self.arc_meta_learning = None
        
        # Initialize frame analyzer
        self.frame_analyzer = FrameAnalyzer()
        
        # Initialize Governor and Architect (optional components)
        self.governor = None
        self.architect = None
        self.learning_session_id = None  # Track current learning session
        
        # Try to initialize Governor and Architect
        try:
            if MetaCognitiveGovernor is not None:
                self.governor = MetaCognitiveGovernor(
                    persistence_dir=str(self.save_directory)  # Enable pattern learning
                )
            if Architect is not None:
                self.architect = Architect(
                    base_path=str(self.save_directory),
                    repo_path="."  # Use main project directory for Git operations
                )
            print("✅ Governor and Architect initialized successfully")
        except Exception as e:
            print(f"⚠️  Governor/Architect initialization failed: {e}")
            self.governor = None
            self.architect = None
        
        # Initialize enhanced simulation agent (optional)
        self.simulation_agent = None
        try:
            if SIMULATION_AVAILABLE and EnhancedSimulationAgent is not None:
                # Import PredictiveCore locally to ensure it's available
                from core.predictive_core import PredictiveCore
                # Create a basic predictive core first
                predictive_core = PredictiveCore()
                # Create enhanced simulation config
                simulation_config = EnhancedSimulationConfig(learning_mode=LearningMode.BALANCED)
                self.simulation_agent = EnhancedSimulationAgent(
                    predictive_core=predictive_core,
                    config=simulation_config,
                    persistence_dir=str(self.save_directory / "enhanced_simulation_agent")
                )
                print("✅ Enhanced simulation agent initialized successfully")
            else:
                print("⚠️  Enhanced simulation agent initialization skipped - dependencies not available")
        except Exception as e:
            print(f"⚠️  Enhanced simulation agent initialization failed: {e}")
            self.simulation_agent = None
        
        # Initialize energy system
        self.energy_system = EnergySystem()
        
        # Initialize sleep system
        try:
            from core.sleep_system import SleepCycle
            from core.predictive_core import PredictiveCore
            
            # Create a basic predictive core for sleep system
            predictive_core = PredictiveCore()
            self.sleep_system = SleepCycle(
                predictive_core=predictive_core
            )
            # Ensure is_sleeping attribute exists
            if not hasattr(self.sleep_system, 'is_sleeping'):
                self.sleep_system.is_sleeping = False
            self.enhanced_sleep_system = self.sleep_system
            print("✅ Sleep system initialized successfully")
        except Exception as e:
            print(f"⚠️  Sleep system initialization failed: {e}")
            self.sleep_system = None
            self.enhanced_sleep_system = None
        
        # Initialize adaptive energy system (optional)
        self.adaptive_energy = None
        self.energy_integration = None
        
        # Initialize session tracking
        self.current_session: Optional[TrainingSession] = None
        
        # Initialize training state
        self.training_state = {
            'lp_history': [],  # Stores learning progress history for boredom detection
            'episode_count': 0,
            'last_boredom_check': 0,
            'mid_game_consolidations': 0,
            'discovered_goals': [],
            'strategy_switches': 0,
            'total_actions': 0
        }
        
        # Initialize game level records
        self.game_level_records = {}  # Tracks highest level achieved per game
        self.memory_hierarchy = {
            'episodic': [],
            'semantic': [],
            'procedural': []
        }
        
        # Initialize training configuration
        self.training_config = {
            'max_episodes_per_game': 20,
            'max_actions_per_episode': 500,
            'learning_rate': 0.01,
            'exploration_rate': 0.1
        }
        
        # Initialize goal system
        self.goal_system = GoalInventionSystem()
        
        # Initialize salience system
        self.salience_calculator: Optional[SalienceCalculator] = None
        self.salience_comparator: Optional[SalienceModeComparator] = None
        try:
            self.salience_performance_history: Dict[SalienceMode, List[Dict]] = {
                SalienceMode.DECAY_COMPRESSION: [],
                SalienceMode.RELEVANCE_FILTERING: [],
                SalienceMode.ADAPTIVE_THRESHOLD: []
            }
        except AttributeError:
            # Fallback if SalienceMode doesn't have all expected attributes
            self.salience_performance_history = {}
        
        # Initialize game session tracking
        self._swarm_mode_active: bool = False
        self.standalone_mode: bool = False
        self._game_completed: bool = False
        
        # Initialize sleep state tracker
        self.sleep_state_tracker = {
            'is_sleeping': False,
            'sleep_cycles_completed': 0,
            'last_sleep_time': None,
            'sleep_effectiveness': 0.0,
            'consolidation_quality': 0.0
        }
        
        # Initialize memory consolidation tracker
        self.memory_consolidation_tracker = {
            'last_consolidation': None,
            'consolidation_count': 0,
            'pending_consolidations': [],
            'consolidation_effectiveness': 0.0,
            'memory_quality_score': 0.0
        }
        
        # Initialize game reset tracker
        self.game_reset_tracker = {
            'reset_decisions_made': 0,
            'reset_reasons': [],
            'reset_effectiveness_scores': [],
            'last_reset_timestamp': 0,
            'consecutive_resets': 0,
            'reset_success_rate': 0.0,
            'last_reset_decision': None,
            'reset_decision_criteria': {
                'consecutive_failures': 5,
                'performance_degradation': 0.3,
                'memory_overflow': 0.95,
                'energy_depletion': 0.1,
                'stagnation_threshold': 10
            }
        }
        
        # Initialize global performance metrics
        self.global_performance_metrics = {
            'total_games_played': 0,
            'total_episodes': 0,
            'average_score': 0.0,
            'win_rate': 0.0,
            'learning_rate': 0.0,
            'adaptation_speed': 0.0
        }
        
        # Initialize demo agent
        self._init_demo_agent()
        
        # Load state from previous sessions
        self._load_state()
        
        # Load global counters from file
        try:
            counter_file = self.save_directory / "global_counters.json"
            if counter_file.exists():
                with open(counter_file, 'r') as f:
                    self.global_counters = json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load global counters: {e}")
            self.global_counters = {
                'total_memory_operations': 0,
                'total_sleep_cycles': 0,
                'total_memories_deleted': 0,
                'total_memories_combined': 0,
                'total_memories_strengthened': 0,
                'persistent_energy_level': 100.0,
                'total_actions': 0,
                'recent_consecutive_failures': 0
            }
        
        print("✅ All complex attributes initialized")

    def _init_demo_agent(self):
        """Initialize demo agent."""
        # This method would contain the demo agent initialization logic
        pass

    def _load_state(self):
        """Load state from previous sessions."""
        # This method would contain the state loading logic
        pass

    def _load_global_counters(self):
        """Load global counters from file."""
        try:
            counter_file = self.save_directory / "global_counters.json"
            if counter_file.exists():
                with open(counter_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load global counters: {e}")
        
        return {
            'total_memory_operations': 0,
            'total_sleep_cycles': 0,
            'total_memories_deleted': 0,
            'total_memories_combined': 0,
            'total_memories_strengthened': 0,
            'persistent_energy_level': 100.0,
            'total_actions': 0,
            'recent_consecutive_failures': 0
        }
        self.global_performance_metrics = {
            'total_games_played': 0,
            'total_episodes': 0,
            'average_score': 0.0,
            'win_rate': 0.0,
            'learning_efficiency': 0.0,
            'knowledge_transfer_success': 0.0
        }
        print("✅ Global performance metrics initialized early to prevent AttributeError")
        
        self.arc_agents_path = Path(arc_agents_path)
        self.tabula_rasa_path = Path(tabula_rasa_path)
        print("📁 Paths set")

        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('ARC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ARC_API_KEY not found. Please set ARC_API_KEY environment variable "
                "in your .env file."
            )
        print("🔑 API key validated")

        # Use continuous_learning_data for adaptive learning evaluation results and architect evolution data
        self.save_directory = Path(save_directory)
        print("💾 Save directory set")
        
        # Track active sessions for cleanup
        self._active_sessions = []
        self.save_directory.mkdir(parents=True, exist_ok=True)
        print("📂 Directories created")
    
    async def cleanup_sessions(self):
        """Clean up any active HTTP sessions."""
        for session in self._active_sessions:
            try:
                if not session.closed:
                    await session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
        self._active_sessions.clear()

        # Phase0 experiment directories
        self.phase0_experiment_results_dir = self.save_directory / "phase0_experiment_results"
        self.phase0_experiment_results_dir.mkdir(parents=True, exist_ok=True)
        self.lp_validation_results_path = self.phase0_experiment_results_dir / "lp_validation_results.yaml"
        self.survival_test_results_path = self.phase0_experiment_results_dir / "survival_test_results.yaml"
        self.phase0_logs_dir = self.phase0_experiment_results_dir / "logs"
        self.phase0_logs_dir.mkdir(parents=True, exist_ok=True)

        # Adaptive learning evaluation and architect evolution directories
        self.adaptive_learning_eval_dir = self.save_directory / "adaptive_learning_agi_evaluation_1756519407"
        self.adaptive_learning_eval_dir.mkdir(parents=True, exist_ok=True)
        self.architect_evolution_data_dir = self.save_directory / "architect_evolution_data"
        self.architect_evolution_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize meta-learning systems
        print("🧠 Initializing meta-learning systems...")
        base_meta_learning = MetaLearningSystem(
            memory_capacity=2000,
            insight_threshold=0.15,
            consolidation_interval=50,
            save_directory=str(self.save_directory / "base_meta_learning")
        )
        print("✅ Base meta-learning system initialized")

        # Update any logic that creates adaptive learning evaluation results to use the new continuous_learning_data directory
        self.adaptive_learning_eval_path = self.adaptive_learning_eval_dir / "research_results.json"
        # Update any logic that creates architect evolution data to use the new continuous_learning_data directory
        self.architectural_insights_path = self.architect_evolution_data_dir / "architectural_insights.json"
        self.evolution_history_path = self.architect_evolution_data_dir / "evolution_history.json"
        self.evolution_strategies_path = self.architect_evolution_data_dir / "evolution_strategies.json"

        print("🎯 Initializing ARC meta-learning system...")
        self.arc_meta_learning = ARCMetaLearningSystem(
            base_meta_learning=base_meta_learning,
            pattern_memory_size=1500,
            insight_threshold=0.25,  # FURTHER REDUCED - Much less aggressive avoidance
            cross_validation_threshold=3
        )
        print("✅ ARC meta-learning system initialized")
        
        # Initialize frame analyzer for visual intelligence
        self.frame_analyzer = FrameAnalyzer()
        
        # Initialize Governor and Architect (Meta-Cognitive Systems)
        print("🧠 Initializing meta-cognitive systems...")
        self.governor = None
        self.architect = None
        self.learning_session_id = None  # Track current learning session
        if META_COGNITIVE_AVAILABLE:
            try:
                # Initialize Governor (Third Brain) with persistence directory for pattern learning
                print("🎯 Initializing Governor...")
                self.governor = MetaCognitiveGovernor(
                    memory_capacity=1000,
                    decision_threshold=0.5,  # REDUCED - Less strict decision threshold
                    adaptation_rate=0.1,
                    persistence_dir=str(self.save_directory)  # Enable pattern learning
                )
                print("✅ Governor initialized")
                logger.info("🧠 Meta-Cognitive Governor initialized (Third Brain) with pattern learning")
                
                # Initialize Architect (Zeroth Brain)
                print("🏗️ Initializing Architect...")
                self.architect = Architect(
                    evolution_rate=0.05,
                    innovation_threshold=0.8,
                    memory_capacity=500,
                    base_path=str(self.save_directory),
                    repo_path="."  # Use main project directory for Git operations
                )
                print("✅ Architect initialized")
                logger.info("🏗️ Architect initialized (Zeroth Brain)")
                
            except Exception as e:
                print(f"❌ Failed to initialize meta-cognitive systems: {e}")
                logger.warning(f"Failed to initialize meta-cognitive systems: {e}")
                self.governor = None
                self.architect = None
        else:
            logger.warning("Meta-cognitive systems not available - running without Governor and Architect")
        
        # Initialize Simulation-Driven Intelligence System
        self.simulation_agent = None
        if SIMULATION_AVAILABLE:
            try:
                # Check if PredictiveCore is available by testing the import
                try:
                    from src.core.predictive_core import PredictiveCore
                    from src.core.simulation_models import SimulationConfig
                    
                    # Initialize Predictive Core for simulation
                    predictive_core = PredictiveCore(
                        visual_size=(3, 64, 64),
                        proprioception_size=12,
                        hidden_size=512,
                        memory_config={
                            'memory_size': 512,
                            'word_size': 64,
                            'num_read_heads': 4,
                            'num_write_heads': 1
                        }
                    )
                    
                    # Initialize simulation configuration
                    simulation_config = EnhancedSimulationConfig(learning_mode=LearningMode.BALANCED)
                    
                    # Initialize Enhanced Simulation Agent
                    self.simulation_agent = EnhancedSimulationAgent(
                        predictive_core=predictive_core,
                        config=simulation_config,
                        persistence_dir=str(self.save_directory / "simulation_agent")
                    )
                    
                    logger.info("🧠 Simulation-Driven Intelligence initialized - Multi-step planning enabled")
                    
                except (ImportError, NameError, TypeError) as e:
                    logger.warning(f"PredictiveCore not available: {e}")
                    logger.warning("Using reactive action selection")
                
            except Exception as e:
                logger.warning(f"Failed to initialize simulation-driven intelligence: {e}")
                self.simulation_agent = None
        else:
            logger.warning("Simulation-driven intelligence not available - using reactive action selection")
        
        #  CRITICAL FIX: Unified energy system to prevent inconsistencies
        # Initialize primary energy system for proper sleep cycle management
        self.energy_system = EnergySystem()
        self.current_energy = 100.0  # Unified energy state
        
        # Energy consumption constants for consistency
        self.ENERGY_COSTS = {
            'action_base': 0.5,
            'action_effective': 0.3,  # Lower cost for effective actions
            'action_ineffective': 0.8,  # Higher cost for ineffective actions
            'computation_base': 0.1,
            'exploration_bonus': -0.1,  # Slight energy gain for exploration
            'repetitive_penalty': 1.2  # Multiplier for repetitive actions
        }
        
        #  CRITICAL FIX: Enhanced sleep system initialization with better error handling
        try:
            from core.predictive_core import PredictiveCore
            from core.sleep_system import SleepCycle
            
            # Create a minimal predictive core for sleep system (if needed)
            self.sleep_system = SleepCycle(
                predictive_core=None,  # Will be set to None for now, can be enhanced later
                sleep_trigger_energy=40.0,  # Trigger sleep at 40% energy
                sleep_trigger_boredom_steps=50,  # More frequent sleep for rapid consolidation
                sleep_duration_steps=20,  # Shorter but more focused sleep cycles
                use_salience_weighting=True
            )
            
            # Ensure sleep system has required attributes
            if not hasattr(self.sleep_system, 'is_sleeping'):
                self.sleep_system.is_sleeping = False
            
            # Set enhanced_sleep_system for compatibility with consolidation code
            self.enhanced_sleep_system = self.sleep_system
            
            logger.info(" Enhanced sleep system initialized for memory consolidation")
            
        except ImportError as e:
            logger.warning(f" Sleep system dependencies not available: {e}")
            logger.info(" Will use enhanced fallback sleep system with memory consolidation")
            self.sleep_system = None
            self.enhanced_sleep_system = None
            
        except Exception as e:
            logger.warning(f" Could not initialize enhanced sleep system: {e}")
            logger.info(" Will use enhanced fallback sleep system with memory consolidation")
            self.sleep_system = None
            self.enhanced_sleep_system = None
        
        #  CONSOLIDATED: Adaptive energy system disabled - using unified energy system instead
        # The main energy system now handles all adaptive behaviors directly
        logger.info(" Using unified energy system (adaptive features integrated)")
        self.adaptive_energy = None
        self.energy_integration = None
        
        # Training state
        self.current_session: Optional[TrainingSession] = None
        self.session_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []  # Add performance history for Governor/Architect
        self.global_performance_metrics = {
            'total_games_played': 0,
            'total_episodes': 0,
            'average_score': 0.0,
            'win_rate': 0.0,
            'learning_efficiency': 0.0,
            'knowledge_transfer_success': 0.0
        }
        
        # Training state and configuration for boredom detection
        self.training_state = {
            'lp_history': [],  # Stores learning progress history for boredom detection
            'episode_count': 0,
            'last_boredom_check': 0
        }
        
        # Progressive memory hierarchy tracking per game
        self.game_level_records = {}  # Tracks highest level achieved per game
        self.memory_hierarchy = {
            'protected_memories': [],  # List of protected memory entries with tiers
            'tier_assignments': {}     # Maps memory_id to tier level
        }
        
        self.training_config = {
            'curriculum_complexity': 1,  # Starting complexity level
            'max_complexity': 10,        # Maximum complexity level
            'boredom_threshold': 3       # Episodes of low LP before boredom triggers
        }
        
        # Initialize goal invention system
        self.goal_system = GoalInventionSystem()
        
        # Specialized Available Actions Memory - Enhanced with persistence and intelligence
        self.available_actions_memory = {
            'current_game_id': None,
            'initial_actions': [],              # Actions available at game start
            'current_actions': [],              # Actions available right now
            'action_history': [],               # All actions attempted this game
            'action_effectiveness': {},         # action_number -> {'attempts': int, 'successes': int, 'success_rate': float}
            'action_transitions': {},           # (from_state, action, to_state) -> frequency
            'action_sequences': [],             # List of successful action sequences
            'coordinate_patterns': {},          # action_number -> {(x, y): {'attempts': int, 'successes': int, 'success_rate': float}}
            'winning_action_sequences': [],     # Sequences that led to wins
            'failed_action_patterns': [],       # Patterns that consistently fail
            'game_intelligence_cache': {},      # Loaded intelligence per game
            'last_action_result': None,         # Track effectiveness of last action
            'sequence_in_progress': [],          # Current action sequence being built
            
            # NEW: INTELLIGENT ACTION MAPPING SYSTEM
            'action_semantic_mapping': {        # Learned semantic understanding of actions
                1: {
                    'default_description': 'Simple action (semantically mapped to up)',
                    'learned_behaviors': [],     # List of observed behaviors: {'behavior': str, 'confidence': float, 'context': str}
                    'grid_movement_patterns': {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'none': 0},  # Movement direction tracking
                    'common_effects': {},        # effect_type -> frequency
                    'game_specific_roles': {}    # game_id -> {'role': str, 'confidence': float}
                },
                2: {
                    'default_description': 'Simple action (semantically mapped to down)',
                    'learned_behaviors': [],
                    'grid_movement_patterns': {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'none': 0},
                    'common_effects': {},
                    'game_specific_roles': {}
                },
                3: {
                    'default_description': 'Simple action (semantically mapped to left)',
                    'learned_behaviors': [],
                    'grid_movement_patterns': {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'none': 0},
                    'common_effects': {},
                    'game_specific_roles': {}
                },
                4: {
                    'default_description': 'Simple action (semantically mapped to right)',
                    'learned_behaviors': [],
                    'grid_movement_patterns': {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'none': 0},
                    'common_effects': {},
                    'game_specific_roles': {}
                },
                5: {
                    'default_description': 'Simple action (interact, select, rotate, attach/detach, execute)',
                    'learned_behaviors': [],
                    'grid_movement_patterns': {'up': 0, 'down': 0, 'left': 0, 'right': 0, 'none': 0},
                    'common_effects': {},
                    'game_specific_roles': {}
                },
                6: {
                    'default_description': 'Complex action requiring x,y coordinates (0-63 range)',
                    'learned_behaviors': [],
                    'grid_movement_patterns': {'coordinate_based': 0},
                    'common_effects': {},
                    'game_specific_roles': {},
                    'coordinate_success_zones': {}  # (x_range, y_range) -> success_rate
                },
                7: {
                    'default_description': 'Simple action - Undo (interact, select)',
                    'learned_behaviors': [],
                    'grid_movement_patterns': {'undo': 0, 'reverse': 0, 'none': 0},
                    'common_effects': {},
                    'game_specific_roles': {}
                }
            },
            'action_learning_stats': {           # Global learning statistics
                'total_observations': 0,
                'pattern_confidence_threshold': 0.7,  # Confidence needed to trust a learned pattern
                'movements_tracked': 0,
                'effects_catalogued': 0,
                'game_contexts_learned': 0
            },
            
            # NEW: ADAPTIVE ACTION RELEVANCE SYSTEM
            'action_relevance_scores': {         # Dynamic relevance scoring for actions
                1: {'base_relevance': 1.0, 'current_modifier': 1.0, 'recent_success_rate': 0.5, 'last_used': 0, 'consecutive_failures': 0},
                2: {'base_relevance': 1.0, 'current_modifier': 1.0, 'recent_success_rate': 0.5, 'last_used': 0, 'consecutive_failures': 0},
                3: {'base_relevance': 1.0, 'current_modifier': 1.0, 'recent_success_rate': 0.5, 'last_used': 0, 'consecutive_failures': 0},
                4: {'base_relevance': 1.0, 'current_modifier': 1.0, 'recent_success_rate': 0.5, 'last_used': 0, 'consecutive_failures': 0},
                5: {'base_relevance': 1.0, 'current_modifier': 1.0, 'recent_success_rate': 0.5, 'last_used': 0, 'consecutive_failures': 0},
                6: {'base_relevance': 0.3, 'current_modifier': 0.3, 'recent_success_rate': 0.2, 'last_used': 0, 'consecutive_failures': 0},  # INCREASED - ACTION6 is important for visual interaction
                7: {'base_relevance': 0.8, 'current_modifier': 0.8, 'recent_success_rate': 0.4, 'last_used': 0, 'consecutive_failures': 0}
            },
            
            # ACTION 6 STRATEGIC CONTROL SYSTEM
            'action6_strategy': {
                'use_sparingly': True,           # Only use when stuck/no progress
                'min_actions_before_use': 15,   # Must try other actions first
                'progress_stagnation_threshold': 8,  # No progress for N actions before considering ACTION6
                'last_progress_action': 0,      # Track when last progress was made
                'action6_cooldown': 5,          # Actions between ACTION6 uses
                'last_action6_used': 0,         # Track last ACTION6 usage
                'predictive_mode': True,        # Try to predict good coordinates
                'emergency_reset_only': True    # Only use as mini-reset when truly stuck
            },
            
            # UNIVERSAL BOUNDARY DETECTION SYSTEM - Extended to all coordinate-based actions
            'universal_boundary_detection': {
                'boundary_data': {},            # game_id -> {(x,y): {'boundary_type': str, 'detection_count': int, 'timestamp': float, 'detected_by_actions': [int]}}
                'coordinate_attempts': {},      # game_id -> {(x,y): {'attempts': int, 'consecutive_stuck': int, 'success_rate': float, 'last_successful_action': int}}
                'action_coordinate_history': {},# game_id -> {action_num: [(x,y), timestamp]} - track coordinate usage by action
                'stuck_patterns': {},           # game_id -> {action_num: {'last_coordinates': (x,y), 'stuck_count': int}}
                'directional_systems': {        # Per-action directional movement systems
                    6: {  # ACTION 6 gets sophisticated directional movement
                        'current_direction': {},        # game_id -> str
                        'direction_progression': {      # Systematic direction exploration order
                            'right': {'next': 'down', 'coordinate_delta': (1, 0)},
                            'down': {'next': 'left', 'coordinate_delta': (0, 1)}, 
                            'left': {'next': 'up', 'coordinate_delta': (-1, 0)},
                            'up': {'next': 'right', 'coordinate_delta': (0, -1)}
                        }
                    },
                    # Other coordinate-based actions get simpler boundary avoidance
                    1: {'boundary_avoidance_radius': 1},  # REDUCED - Less restrictive boundary avoidance
                    2: {'boundary_avoidance_radius': 1},
                    3: {'boundary_avoidance_radius': 1}, 
                    4: {'boundary_avoidance_radius': 1},
                    5: {'boundary_avoidance_radius': 2},  # REDUCED - Less restrictive for ACTION 5
                    7: {'boundary_avoidance_radius': 1}
                },
                'boundary_stuck_threshold': 3,          # Same coordinates N times = boundary detected
                'success_zone_mapping': {},             # game_id -> {(x,y): {'success_count': int, 'total_attempts': int, 'successful_actions': [int]}}
                'last_coordinates': {},                 # game_id -> (x,y) - track last used coordinates per game
                'global_coordinate_intelligence': {     # Cross-game coordinate learning
                    'universal_boundaries': {},         # Coordinates that are problematic across games
                    'universal_success_zones': {},      # Coordinates that work well across games
                    'action_coordinate_preferences': {} # Which actions work best at which coordinate types
                },
                'safe_regions': {},                     # game_id -> {region_id: {'coordinates': set(), 'success_rate': float, 'center': (x,y), 'size': int}}
                'connected_zones': {},                  # game_id -> {zone_id: {'coordinates': set(), 'connections': [zone_ids], 'safety_score': float}}
                'coordinate_clusters': {},              # game_id -> {cluster_id: {'center': (x,y), 'radius': int, 'members': set(), 'avg_success_rate': float}}
                'danger_zones': {}                      # game_id -> {danger_id: {'coordinates': set(), 'failure_rate': float, 'avoidance_radius': int}}
            },
            
            # ACTION 6 LEGACY SYSTEM (kept for backward compatibility)
            'action6_boundary_detection': {
                'boundary_data': {},            # DEPRECATED: Use universal_boundary_detection instead
                'coordinate_attempts': {},      # DEPRECATED: Use universal_boundary_detection instead
                'last_coordinates': {},         # DEPRECATED: Use universal_boundary_detection instead  
                'stuck_count': {},              # DEPRECATED: Use universal_boundary_detection instead
                'current_direction': {},        # DEPRECATED: Use universal_boundary_detection instead
                'direction_progression': {      # DEPRECATED: Use universal_boundary_detection instead
                    'right': {'next': 'down', 'coordinate_delta': (1, 0)},
                    'down': {'next': 'left', 'coordinate_delta': (0, 1)}, 
                    'left': {'next': 'up', 'coordinate_delta': (-1, 0)},
                    'up': {'next': 'right', 'coordinate_delta': (0, -1)}
                },
                'boundary_stuck_threshold': 3,  # DEPRECATED: Use universal_boundary_detection instead
                'max_direction_distance': 8     # DEPRECATED: Use universal_boundary_detection instead
            }
        }

        # Note: Removed legacy `success_multiplier` and `_calculate_episode_effectiveness` stub.
        # Use `SalienceCalculator` or explicit episode metrics (wins/actions) to derive episode
        # effectiveness. Keeping `success_weighted_memories` was a compatibility shim; rely on
        # explicit replay buffer weighting via `SalienceWeightedReplayBuffer` instead.

        # Salience system components
        self.salience_calculator: Optional[SalienceCalculator] = None
        self.salience_comparator: Optional[SalienceModeComparator] = None
        self.salience_performance_history: Dict[SalienceMode, List[Dict]] = {
            SalienceMode.LOSSLESS: [],
            SalienceMode.DECAY_COMPRESSION: []
        }
        
        # API session management
        self.current_scorecard_id: Optional[str] = None
        self.current_game_sessions: Dict[str, str] = {}  # game_id -> guid mapping
        self._created_new_scorecard: bool = False  # Track if we created a new scorecard
        
        #  CRITICAL FIX: Swarm mode tracking for scorecard isolation
        self._swarm_mode_active: bool = False
        # Ensure standalone mode flag exists to avoid AttributeError in session flows
        self.standalone_mode: bool = False
        
        print("🔧 Initializing rate limiter...")
        # Rate limiting for ARC-AGI-3 API compliance
        try:
            self.rate_limiter = RateLimiter()
            print(f"✅ Rate limiter initialized: {ARC3_RATE_LIMIT['safe_requests_per_second']} RPS max, 600 RPM limit")
        except Exception as e:
            print(f"❌ ERROR: Failed to initialize rate limiter: {e}")
            raise
        
        # Enhanced tracking for sleep states and memory operations
        self.sleep_state_tracker = {
            'is_currently_sleeping': False,
            'sleep_cycles_this_session': 0,
            'total_sleep_time': 0.0,
            'memory_operations_per_cycle': 0,
            'last_sleep_trigger': [],
            'sleep_quality_scores': []
        }
        
        # Memory consolidation tracker
        self.memory_consolidation_tracker = {
            'memory_operations_per_cycle': 0,
            'consolidation_operations_count': 0,
            'high_salience_memories_strengthened': 0,
            'low_salience_memories_decayed': 0,
            'is_consolidating_memories': False,
            'is_prioritizing_memories': False,
            'memory_compression_active': False,
            'last_consolidation_score': 0.0,
            'total_memory_operations': 0
        }
        
        # Game reset decision tracking
        self.game_reset_tracker = {
            'reset_decisions_made': 0,
            'reset_reasons': [],
            'reset_effectiveness_scores': [],
            'last_reset_decision': None,
            'reset_decision_criteria': {
                'consecutive_failures': 5,
                'performance_degradation': 0.3,
                'memory_overflow': 0.95,
                'energy_depletion': 10.0
            },
            'reset_success_rate': 0.0
        }
        
        # Initialize demonstration agent for monitoring
        self._init_demo_agent()
        
        # Load previous state if available
        self._load_state()
        
        # GLOBAL PERSISTENT COUNTERS - Load from previous sessions
        self.global_counters = self._load_global_counters()
        
        # ENERGY SYSTEM - Initialize with persistent level from previous sessions (0-100.0 scale)
        persistent_energy = self.global_counters.get('persistent_energy_level', 100.0)
        
        # Adaptive energy initialization based on performance
        total_sleep_cycles = self.global_counters.get('total_sleep_cycles', 0)
        if total_sleep_cycles > 0:
            # After sleep cycles, maintain the persistent energy level
            self.current_energy = persistent_energy
            print(f" Resuming with persistent energy: {self.current_energy:.2f} (after {total_sleep_cycles} sleep cycles)")
        else:
            # Fresh start gets full energy
            self.current_energy = 100.0
            print(f" Fresh session starting with full energy: {self.current_energy:.2f}")
        
        #  SMART ACTION CAP SYSTEM - Prevents infinite loops and analyzes stagnation
        #  CRITICAL FIX: Improved action cap system with less aggressive termination
        #  SCALING FIX: Dynamic caps now scale as fraction of max_actions_per_game
        self._action_cap_system = {
            'enabled': True,
            'base_cap_fraction': 0.15,  # 15% of max_actions_per_game as base
            'multiplier_per_available_action': 0.02,  # 2% of max_actions_per_game per available action
            'min_cap_fraction': 0.05,   # 5% of max_actions_per_game minimum
            'max_cap_fraction': 0.30,   # 30% of max_actions_per_game maximum
            'stagnation_threshold': 200,  # Increased from 50 -> much more patient for long-run tests
            'loop_detection_window': 15,  # Longer window for better pattern detection
            'early_termination_enabled': False,  # Disabled for long-run validation tests
            'analysis_enabled': True,
            'exploration_bonus_fraction': 0.01,  # 1% of max_actions_per_game bonus for exploration
            'score_improvement_bonus_fraction': 0.02  # 2% of max_actions_per_game bonus for score improvement
        }
        # Runtime instrumentation: print action cap system so logs show which config is active
        try:
            print(f"DEBUG ACTION CAP SYSTEM: {self._action_cap_system}")
        except Exception:
            logger.exception("Failed to print action cap system for debugging")
        
        #  CRITICAL FIX: Enhanced progress tracking for smarter termination
        self._progress_tracker = {
            'actions_taken': 0,
            'last_score': 0,
            'actions_without_progress': 0,
            'last_meaningful_change': 0,
            'action_pattern_history': [],  # Last N actions for loop detection
            'score_history': [],  # Last N scores for trend analysis
            'termination_reason': None,
            'explored_coordinates': set(),  # Track coordinate exploration
            'recent_improvements': [],  # Track recent score improvements
            'exploration_bonus_used': False  # Track if exploration bonus was applied
        }

        logger.info("Continuous Learning Loop initialized with ARC-3 API integration")
        print("\n================ ARC TRAINING SESSION INITIALIZED ================")
        print(f"Session ID: {getattr(self, 'session_id', 'N/A')}")
        print(f"Games: {getattr(self, 'games', 'N/A')}")
        print("===============================================================\n")

    def _load_global_counters(self) -> Dict[str, int]:
        """Load global counters that persist across sessions."""
        try:
            import json
            counter_file = self.save_directory / "global_counters.json"
            if counter_file.exists():
                with open(counter_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Could not load global counters: {e}")
        
        # Default counters
        return {
            'total_sleep_cycles': 0,
            'total_memory_operations': 0,
            'total_sessions': 0,
            'cumulative_energy_spent': 0.0,
            'total_memories_deleted': 0,
            'total_memories_combined': 0,
            'total_memories_strengthened': 0,
            'persistent_energy_level': 1.0  # Track energy across sessions
        }
        
    def get_system_status_flags(self) -> Dict[str, bool]:
        """
        Get simple True/False flags for all major system operations.
        
        Returns:
            Dictionary with boolean flags for each system operation
        """
        return {
            # Sleep states
            'is_sleeping': getattr(self, 'is_sleeping', lambda: False)(),
            'sleep_cycles_active': getattr(self, 'sleep_state_tracker', {}).get('sleep_cycles_this_session', 0) > 0,
            
            # Memory consolidation
            'is_consolidating_memories': getattr(self, 'memory_consolidation_tracker', {}).get('is_consolidating_memories', False),
            'is_prioritizing_memories': getattr(self, 'memory_consolidation_tracker', {}).get('is_prioritizing_memories', False),
            'memory_strengthening_active': getattr(self, 'memory_consolidation_tracker', {}).get('high_salience_memories_strengthened', 0) > 0,
            'memory_decay_active': getattr(self, 'memory_consolidation_tracker', {}).get('low_salience_memories_decayed', 0) > 0,
            
            # Memory compression
            'memory_compression_active': getattr(self, 'is_memory_compression_active', lambda: False)(),
            'has_compressed_memories': len(getattr(getattr(self, 'salience_calculator', None), 'compressed_memories', [])) > 0,
            
            # Game reset decisions
            'has_made_reset_decisions': getattr(self, 'has_made_reset_decisions', lambda: False)(),
            'reset_decision_pending': getattr(self, '_is_reset_decision_pending', lambda: False)(),
            
            # System health
            'memory_system_healthy': getattr(self, '_is_memory_system_healthy', lambda: False)()
        }
        
    def get_sleep_and_memory_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of sleep states and memory operations.
        
        Returns:
            Complete status including True/False flags for all operations
        """
        # Safely get dictionary attributes with defaults
        sleep_tracker = getattr(self, 'sleep_state_tracker', {})
        consolidation_tracker = getattr(self, 'memory_consolidation_tracker', {})
        game_reset_tracker = getattr(self, 'game_reset_tracker', {})
        
        # Safely get last reset decision
        last_reset_decision = game_reset_tracker.get('last_reset_decision', {}) if isinstance(game_reset_tracker, dict) else {}
        
        return {
            # Sleep state status
            'sleep_status': {
                'is_currently_sleeping': getattr(self, 'is_sleeping', lambda: False)(),
                'sleep_cycles_this_session': sleep_tracker.get('sleep_cycles_this_session', 0) if isinstance(sleep_tracker, dict) else 0,
                'total_sleep_time_minutes': (sleep_tracker.get('total_sleep_time', 0) / 60.0) if isinstance(sleep_tracker, dict) else 0.0,
                'sleep_efficiency': getattr(self, '_calculate_sleep_efficiency', lambda: 0.0)()
            },
            
            # Memory consolidation status
            'memory_consolidation_status': {
                'is_consolidating_memories': getattr(self, 'memory_consolidation_tracker', {}).get('is_consolidating_memories', False),
                'is_prioritizing_memories': getattr(self, 'memory_consolidation_tracker', {}).get('is_prioritizing_memories', False),
                'consolidation_operations_completed': consolidation_tracker.get('consolidation_operations_count', 0) if isinstance(consolidation_tracker, dict) else 0,
                'high_salience_memories_strengthened': consolidation_tracker.get('high_salience_memories_strengthened', 0) if isinstance(consolidation_tracker, dict) else 0,
                'low_salience_memories_decayed': consolidation_tracker.get('low_salience_memories_decayed', 0) if isinstance(consolidation_tracker, dict) else 0,
                'last_consolidation_effectiveness': consolidation_tracker.get('last_consolidation_score', 0.0) if isinstance(consolidation_tracker, dict) else 0.0
            },
            
            # Memory compression status
            'memory_compression_status': {
                'compression_active': getattr(self, 'is_memory_compression_active', lambda: False)(),
                'compression_mode': getattr(getattr(self, 'salience_calculator', None), 'mode', 'none'),
                'total_compressed_memories': len(getattr(getattr(self, 'salience_calculator', None), 'compressed_memories', []))
            },
            
            # Game reset decision status
            'game_reset_status': {
                'has_made_reset_decisions': getattr(self, 'has_made_reset_decisions', lambda: False)(),
                'total_reset_decisions': game_reset_tracker.get('reset_decisions_made', 0) if isinstance(game_reset_tracker, dict) else 0,
                'reset_success_rate': game_reset_tracker.get('reset_success_rate', 0.0) if isinstance(game_reset_tracker, dict) else 0.0,
                'last_reset_reason': last_reset_decision.get('reason') if isinstance(last_reset_decision, dict) else None,
                'reset_decision_criteria': game_reset_tracker.get('reset_decision_criteria', []) if isinstance(game_reset_tracker, dict) else []
            }
        }

    def _unified_energy_consumption(self, action_effective: bool = False, is_exploration: bool = False, is_repetitive: bool = False) -> float:
        """ CRITICAL FIX: Unified energy consumption method for consistency across all systems.

        Returns the updated current energy level after applying costs and modifiers.
        """
        # Determine base cost depending on action effectiveness
        base_cost = self.ENERGY_COSTS['action_effective'] if action_effective else self.ENERGY_COSTS['action_ineffective']

        # Apply constant computation cost
        total_cost = base_cost + self.ENERGY_COSTS['computation_base']

        # Exploration bonus (slight energy gain)
        if is_exploration:
            total_cost += self.ENERGY_COSTS['exploration_bonus']

        # Repetitive action penalty
        if is_repetitive:
            total_cost *= self.ENERGY_COSTS['repetitive_penalty']

        # Consume energy from primary system (clamp to [0, 100])
        new_energy = max(0, min(100, self.current_energy - total_cost))
        energy_consumed = self.current_energy - new_energy
        self.current_energy = new_energy
        
        # Update energy stats
        self.energy_stats['total_consumed'] += energy_consumed
        self.energy_stats['last_consumed'] = energy_consumed
        
        # If we're below critical threshold, trigger low energy mode
        if self.current_energy < self.ENERGY_CRITICAL and not self.low_energy_mode:
            self._enter_low_energy_mode()
            
        # If we recovered from low energy mode
        if self.current_energy > self.ENERGY_RECOVERY and self.low_energy_mode:
            self._exit_low_energy_mode()
            
        return self.current_energy
        
    def _display_memory_hierarchy_status(self):
        """Display current memory hierarchy and protection status."""
        try:
            # Safely get the memory hierarchy with defaults
            memory_hierarchy = getattr(self, 'memory_hierarchy', {})
            protected_memories = memory_hierarchy.get('protected_memories', [])
            
            # Log the memory hierarchy status
            print("\n MEMORY HIERARCHY STATUS:")
            print(f"  - Total Protected Memories: {len(protected_memories)}")
            
            # Log some sample memories if available
            if protected_memories:
                print("  - Sample Protected Memories:")
                for i, memory in enumerate(protected_memories[:3]):  # Show first 3 as samples
                    mem_type = memory.get('type', 'unknown')
                    mem_salience = memory.get('salience', 0.0)
                    print(f"    {i+1}. Type: {mem_type}, Salience: {mem_salience:.2f}")
                if len(protected_memories) > 3:
                    print(f"    ... and {len(protected_memories) - 3} more")
            
            # Log memory protection stats if available
            if hasattr(self, 'memory_protection_stats'):
                stats = self.memory_protection_stats
                print("\n  MEMORY PROTECTION STATS:")
                print(f"  - Total Protection Checks: {stats.get('total_checks', 0)}")
                print(f"  - Protection Success Rate: {stats.get('success_rate', 0.0) * 100:.1f}%")
                
        except Exception as e:
            print(f" Error displaying memory hierarchy status: {e}")
            
        # Log current energy status
        print(f" Current Energy: {getattr(self, 'current_energy', 0.0):.1f}/100")
    
    def _should_trigger_sleep_cycle(self, actions_taken: int, recent_effectiveness: float = 0.0) -> bool:
        """ CRITICAL FIX: Intelligent sleep trigger based on energy, effectiveness, and learning needs."""
        
        # Base energy threshold
        if self.current_energy <= 40.0:
            return True
        
        # Sleep after long ineffective sequences for consolidation
        if actions_taken > 100 and recent_effectiveness < 0.3:
            print(f" Sleep trigger: Long ineffective sequence ({actions_taken} actions, {recent_effectiveness:.1%} effective)")
            print(f"[SLEEP] Triggered by ineffective sequence. Progress tracker: {self._progress_tracker}")
            return True
        
        # Periodic consolidation sleep every 200 actions regardless
        if actions_taken % 200 == 0 and actions_taken > 0:
            print(f" Sleep trigger: Periodic consolidation at {actions_taken} actions")
            print(f"[SLEEP] Periodic consolidation. Progress tracker: {self._progress_tracker}")
            return True
        
        # Energy-based trigger with sliding scale
        energy_threshold = 50.0 - (recent_effectiveness * 20)  # Lower threshold if ineffective
        if self.current_energy <= energy_threshold:
            print(f" Sleep trigger: Adaptive energy threshold ({energy_threshold:.1f}) reached")
            print(f"[SLEEP] Adaptive energy threshold. Progress tracker: {self._progress_tracker}")
            return True
        
        return False

    def _calculate_current_win_rate(self) -> float:
        """
        Calculate current win rate as a decimal (0.0 to 1.0).
        
        Returns:
            Current win rate as decimal between 0.0 and 1.0
        """
        return self.global_performance_metrics.get('win_rate', 0.0)

    def _calculate_win_rate_adaptive_energy_parameters(self) -> Dict[str, float]:
        """
        Calculate energy parameters based on current win rate for adaptive sleep frequency.
        
        Initial learners (0% win rate) should sleep very frequently to consolidate learning.
        Experienced agents (high win rate) can sleep less and focus on performance.
        
        Returns:
            Dictionary with adaptive energy parameters
        """
        # Calculate current win rate from global performance metrics
        current_win_rate = self._calculate_current_win_rate() * 100  # Convert to percentage
        total_episodes = self.global_performance_metrics.get('total_episodes', 0)
        
        # PROGRESSIVE ENERGY SYSTEM BASED ON SKILL LEVEL
        if current_win_rate == 0.0 and total_episodes < 50:
            # 🟥 BEGINNER PHASE: Maximum learning focus - sleep every 8-12 actions
            return {
                'action_energy_cost': 4.5,  # Very high cost per action (8-12 actions = ~36-54 energy = sleep trigger at 60)  
                'sleep_trigger_threshold': 60.0,  # Sleep when energy drops to 60%
                'effectiveness_multiplier': 1.0,  # No penalty for ineffective actions yet
                'learning_phase': 'beginner_intensive',
                'skill_phase': 'Beginner',
                'expected_actions_before_sleep': 9,
                'description': 'Beginner: Intensive learning with frequent memory consolidation'
            }
        elif current_win_rate < 15.0:
            # 🟧 EARLY LEARNING PHASE: High learning focus - sleep every 12-18 actions
            return {
                'action_energy_cost': 3.0,  # High cost per action (12-18 actions = ~36-54 energy)
                'sleep_trigger_threshold': 55.0,  # Sleep when energy drops to 55%  
                'effectiveness_multiplier': 1.1,  # Small penalty for ineffective actions
                'learning_phase': 'early_learning',
                'skill_phase': 'Early Learning',
                'expected_actions_before_sleep': 15,
                'description': 'Early Learning: Frequent consolidation with some efficiency focus'
            }
        elif current_win_rate < 35.0:
            # 🟨 DEVELOPING PHASE: Moderate learning focus - sleep every 20-30 actions
            return {
                'action_energy_cost': 2.0,  # Moderate cost per action (20-30 actions = ~40-60 energy)
                'sleep_trigger_threshold': 50.0,  # Sleep when energy drops to 50%
                'effectiveness_multiplier': 1.2,  # Moderate penalty for ineffective actions
                'learning_phase': 'developing_skills', 
                'skill_phase': 'Developing',
                'expected_actions_before_sleep': 25,
                'description': 'Developing: Balanced learning and performance optimization'
            }
        elif current_win_rate < 55.0:
            # 🟩 COMPETENT PHASE: Performance focus - sleep every 35-50 actions
            return {
                'action_energy_cost': 1.4,  # Lower cost per action (35-50 actions = ~49-70 energy)
                'sleep_trigger_threshold': 45.0,  # Sleep when energy drops to 45%
                'effectiveness_multiplier': 1.3,  # Higher penalty for ineffective actions
                'learning_phase': 'competent_performance',
                'skill_phase': 'Competent',
                'expected_actions_before_sleep': 40,
                'description': 'Competent: Performance-focused with strategic consolidation'
            }
        elif current_win_rate < 75.0:
            # 🟦 SKILLED PHASE: Efficiency focus - sleep every 50-70 actions  
            return {
                'action_energy_cost': 1.0,  # Low cost per action (50-70 actions = ~50-70 energy)
                'sleep_trigger_threshold': 40.0,  # Sleep when energy drops to 40%
                'effectiveness_multiplier': 1.4,  # High penalty for ineffective actions
                'learning_phase': 'skilled_efficiency',
                'skill_phase': 'Skilled',
                'expected_actions_before_sleep': 60,
                'description': 'Skilled: Efficiency-focused with minimal consolidation needs'
            }
        else:
            # 🟪 EXPERT PHASE: Maximum efficiency - sleep every 80+ actions
            return {
                'action_energy_cost': 0.7,  # Very low cost per action (80+ actions = ~56+ energy)
                'sleep_trigger_threshold': 35.0,  # Sleep when energy drops to 35%
                'effectiveness_multiplier': 1.5,  # Maximum penalty for ineffective actions
                'learning_phase': 'expert_mastery',
                'skill_phase': 'Expert',
                'expected_actions_before_sleep': 80,
                'description': 'Expert: Maximum efficiency with rare consolidation'
            }
    
        
    def _calculate_dynamic_action_cap(self, available_actions: List[int], max_actions_per_game: int = None) -> int:
        """Calculate smart action cap based on game complexity, scaled to max_actions_per_game."""
        if max_actions_per_game is None:
            # Try to get dynamic limit from Governor first
            if hasattr(self, 'governor') and self.governor and hasattr(self.governor, 'get_dynamic_action_limit'):
                try:
                    # Calculate game complexity based on available actions
                    game_complexity = min(1.0, len(available_actions) / 6.0) if available_actions else 0.5
                    max_actions_per_game = self.governor.get_dynamic_action_limit(
                        'per_game', 
                        game_complexity=game_complexity,
                        available_actions=len(available_actions) if available_actions else 6
                    )
                except Exception as e:
                    logger.warning(f"Failed to get dynamic action limit from Governor: {e}")
                    max_actions_per_game = ActionLimits.get_max_actions_per_game()
            else:
                max_actions_per_game = ActionLimits.get_max_actions_per_game()
        # Ensure action cap system is initialized
        if not hasattr(self, '_action_cap_system') or not self._action_cap_system:
            # Fallback configuration if not initialized
            self._action_cap_system = {
                'enabled': True,
                'base_cap_fraction': 0.15,
                'multiplier_per_available_action': 0.02,
                'min_cap_fraction': 0.05,
                'max_cap_fraction': 0.30,
                'stagnation_threshold': 200,
                'loop_detection_window': 15,
                'early_termination_enabled': False,
                'analysis_enabled': True,
                'exploration_bonus_fraction': 0.01,
                'score_improvement_bonus_fraction': 0.02
            }
        
        config = self._action_cap_system
        
        # Calculate fractions of max_actions_per_game
        base_cap = int(max_actions_per_game * config['base_cap_fraction'])
        min_cap = int(max_actions_per_game * config['min_cap_fraction'])
        max_cap = int(max_actions_per_game * config['max_cap_fraction'])
        
        # Base calculation: more actions = more complex game = higher cap
        base_actions = len(available_actions) if available_actions else 5
        multiplier = int(max_actions_per_game * config['multiplier_per_available_action'])
        calculated_cap = base_cap + (base_actions * multiplier)
        
        # Apply bounds
        capped_value = max(min_cap, min(calculated_cap, max_cap))
        
        print(f" SMART CAP: {base_actions} actions available → {capped_value} action limit (scaled from {max_actions_per_game})")
        return capped_value
    
    def _update_governor_action_metrics(self, game_id: str, performance_data: Dict[str, Any]):
        """Update the Governor with performance metrics for dynamic action limit adjustment."""
        if not hasattr(self, 'governor') or not self.governor:
            return
        
        try:
            # Calculate efficiency based on performance
            efficiency = 0.5  # Default
            if 'score' in performance_data and 'max_score' in performance_data:
                efficiency = min(1.0, performance_data['score'] / max(1, performance_data['max_score']))
            elif 'reward' in performance_data:
                efficiency = max(0.0, min(1.0, (performance_data['reward'] + 1) / 2))  # Convert -1 to 1 range to 0 to 1
            
            # Calculate learning progress based on recent improvements
            learning_progress = 0.0
            if hasattr(self, 'performance_history') and len(self.performance_history) > 5:
                recent_scores = [p.get('score', 0) for p in self.performance_history[-5:]]
                older_scores = [p.get('score', 0) for p in self.performance_history[-10:-5]] if len(self.performance_history) >= 10 else recent_scores
                if older_scores:
                    recent_avg = sum(recent_scores) / len(recent_scores)
                    older_avg = sum(older_scores) / len(older_scores)
                    learning_progress = max(0.0, min(1.0, (recent_avg - older_avg) / max(1, older_avg)))
            
            # Calculate system stress based on memory usage and error rates
            system_stress = 0.0
            if hasattr(self, 'memory_consolidation_tracker'):
                consolidation_data = self.memory_consolidation_tracker.get('consolidation_data', {})
                if consolidation_data:
                    # Higher consolidation frequency indicates stress
                    consolidation_freq = consolidation_data.get('consolidation_frequency', 0)
                    system_stress = min(1.0, consolidation_freq / 10.0)  # Scale to 0-1
            
            # Calculate game complexity based on available actions
            available_actions = performance_data.get('available_actions', [])
            game_complexity = min(1.0, len(available_actions) / 6.0) if available_actions else 0.5
            
            # Update Governor with metrics
            self.governor.update_action_limit_metrics(
                efficiency=efficiency,
                learning_progress=learning_progress,
                system_stress=system_stress,
                game_complexity=game_complexity
            )
            
            logger.debug(f"Updated Governor action metrics: efficiency={efficiency:.2f}, learning={learning_progress:.2f}, stress={system_stress:.2f}, complexity={game_complexity:.2f}")
            
        except Exception as e:
            logger.warning(f"Failed to update Governor action metrics: {e}")
    
    def _calculate_dynamic_session_limit(self, max_actions_per_game: int, available_games: List[str]) -> int:
        """Calculate session limit based on available games and per-game limit."""
        base_games = len(available_games) if available_games else 1
        buffer_factor = 1.5  # Allow 50% more for retries/learning/cycling
        
        calculated_limit = int(max_actions_per_game * base_games * buffer_factor)
        
        # Ensure minimum reasonable session size
        min_session_limit = max_actions_per_game * 2  # At least 2 games worth
        max_session_limit = max_actions_per_game * 50  # Cap at 50 games worth
        
        final_limit = max(min_session_limit, min(calculated_limit, max_session_limit))
        
        print(f" DYNAMIC SESSION LIMIT: {base_games} games × {max_actions_per_game} actions × {buffer_factor} buffer = {final_limit}")
        return final_limit
    
    def _check_progress_and_extend_cap(self, current_score: int, actions_taken: int, current_cap: int, max_actions_per_game: int) -> int:
        """Check if progress is being made and extend the action cap if so."""
        if not hasattr(self, '_progress_tracker'):
            return current_cap
            
        tracker = self._progress_tracker
        config = self._action_cap_system
        
        # Check if we're near the current cap (within 10% of it)
        near_cap_threshold = int(current_cap * 0.9)
        if actions_taken < near_cap_threshold:
            return current_cap  # Not near cap yet, no need to extend
        
        # Check for recent progress indicators
        recent_progress = False
        
        # 1. Score improvement in last 50 actions
        if len(tracker.get('score_history', [])) >= 2:
            recent_scores = tracker['score_history'][-min(10, len(tracker['score_history'])):]
            if len(recent_scores) >= 2 and recent_scores[-1] > recent_scores[0]:
                recent_progress = True
                print(f" PROGRESS DETECTED: Score improved from {recent_scores[0]} to {recent_scores[-1]} in recent actions")
        
        # 2. Recent meaningful changes (not just score, but any state change)
        if tracker.get('last_meaningful_change', 0) > 0:
            actions_since_change = actions_taken - tracker['last_meaningful_change']
            if actions_since_change < 100:  # Recent change within last 100 actions
                recent_progress = True
                print(f" PROGRESS DETECTED: Meaningful change {actions_since_change} actions ago")
        
        # 3. Low stagnation counter (not stuck)
        stagnation_actions = tracker.get('actions_without_progress', 0)
        if stagnation_actions < 50:  # Less than 50 actions without progress
            recent_progress = True
            print(f" PROGRESS DETECTED: Only {stagnation_actions} actions without progress")
        
        # If progress detected and we're near the cap, extend it
        if recent_progress and current_cap < max_actions_per_game:
            # Calculate extension based on how much progress we're making
            extension = int(max_actions_per_game * 0.2)  # Base 20% extension
            
            # If we're making very good progress (score improving rapidly), be more generous
            if len(tracker.get('score_history', [])) >= 3:
                recent_scores = tracker['score_history'][-3:]
                if recent_scores[-1] > recent_scores[0] and (recent_scores[-1] - recent_scores[0]) > 5:
                    extension = int(max_actions_per_game * 0.4)  # 40% extension for strong progress
                    print(f" STRONG PROGRESS: Score improved by {recent_scores[-1] - recent_scores[0]} points")
            
            # If we're very close to max and still making progress, allow full extension
            if current_cap >= int(max_actions_per_game * 0.8):  # Already at 80% of max
                extension = max_actions_per_game - current_cap  # Extend to full max
                print(f" NEAR MAX: Extending to full {max_actions_per_game} actions")
            
            new_cap = min(current_cap + extension, max_actions_per_game)
            
            if new_cap > current_cap:
                print(f" CAP EXTENDED: {current_cap} → {new_cap} actions (progress detected)")
                return new_cap
        
        return current_cap
    
    def _should_terminate_early(self, current_score: int, actions_taken: int) -> tuple[bool, str]:
        """Analyze if game should terminate early due to lack of progress."""
        # Ensure progress tracker is initialized
        if not hasattr(self, '_progress_tracker'):
            self._progress_tracker = {
                'actions_taken': 0,
                'last_score': 0,
                'actions_without_progress': 0,
                'last_meaningful_change': 0,
                'action_pattern_history': [],
                'explored_coordinates': set(),
                'termination_reason': None
            }
        
        tracker = self._progress_tracker
        config = self._action_cap_system
        
        if not config['early_termination_enabled']:
            return False, "Early termination disabled"
        
        # Update progress tracking
        tracker['actions_taken'] = actions_taken
        score_improved = False
        
        if current_score != tracker['last_score']:
            score_improved = current_score > tracker['last_score']
            tracker['last_meaningful_change'] = actions_taken
            tracker['actions_without_progress'] = 0
            tracker['score_history'].append(current_score)
            # Keep only last 20 scores for trend analysis
            if len(tracker['score_history']) > 20:
                tracker['score_history'] = tracker['score_history'][-20:]
            
            #  CRITICAL FIX: Reset stagnation counter on ANY score change, not just improvement
            print(f" Score changed from {tracker['last_score']} to {current_score}, resetting stagnation counter")
        else:
            tracker['actions_without_progress'] += 1
        
        tracker['last_score'] = current_score
        
        # Early termination conditions (much more forgiving)
        stagnant_actions = tracker['actions_without_progress']
        
        #  CRITICAL FIX: More intelligent stagnation detection
        # 1. Base threshold increased, but consider recent improvements
        base_threshold = config['stagnation_threshold']
        
        # Give bonus actions if we've had recent score improvements
        if len(tracker['score_history']) >= 3:
            recent_max = max(tracker['score_history'][-3:])
            if recent_max > tracker['score_history'][0] if len(tracker['score_history']) > 1 else 0:
                base_threshold += config.get('score_improvement_bonus', 30)
                print(f" Recent improvement detected, extending patience to {base_threshold} actions")
        
        # 1. Too many actions without any score progress (much more patient)
        if stagnant_actions >= base_threshold:
            return True, f"No progress for {stagnant_actions} actions (threshold: {base_threshold})"
        
        #  CRITICAL FIX: Improved loop detection - only terminate if truly stuck
        # 2. Detect action loops (same action repeated many times) - more forgiving
        if len(tracker['action_pattern_history']) >= config['loop_detection_window']:
            recent_actions = tracker['action_pattern_history'][-config['loop_detection_window']:]
            unique_actions = len(set(recent_actions))
            
            # Only terminate on loops if we're really stuck (fewer unique actions + high stagnation)
            if unique_actions <= 2 and stagnant_actions >= (base_threshold // 2):
                return True, f"Action loop detected: {recent_actions[-5:]} (only {unique_actions} unique actions, {stagnant_actions} stagnant)"
            elif unique_actions <= 3 and stagnant_actions >= (base_threshold * 2 // 3):
                return True, f"Repetitive actions detected: {recent_actions[-5:]} (only {unique_actions} unique actions, {stagnant_actions} stagnant)"
        
        # 3. Score trend analysis - only if consistently declining AND high stagnation
        if len(tracker['score_history']) >= 10:
            recent_scores = tracker['score_history'][-10:]
            if all(recent_scores[i] <= recent_scores[i-1] for i in range(1, len(recent_scores))) and stagnant_actions >= (base_threshold * 3 // 4):
                return True, f"Consistently declining score trend with {stagnant_actions} stagnant actions"
        
        return False, "Continue playing"
    
    def _enhanced_sleep_consolidation(self, experiences: List[Dict[str, Any]], sleep_reason: str = "Energy depletion") -> Dict[str, Any]:
        """
        Enhanced sleep consolidation with experience processing.
        
        Args:
            experiences: List of experience dictionaries to consolidate
            sleep_reason: Reason for the sleep trigger
            
        Returns:
            Dictionary with consolidation results
        """
        try:
            print(f" ENHANCED SLEEP CONSOLIDATION: {sleep_reason}")
            
            result = {
                'success': True,
                'experiences_processed': len(experiences),
                'sleep_reason': sleep_reason,
                'consolidation_score': 0.0,
                'insights_generated': 0
            }
            
            if hasattr(self, 'enhanced_sleep_system') and self.enhanced_sleep_system:
                # Use enhanced sleep system if available
                sleep_result = self.enhanced_sleep_system.execute_sleep_cycle(
                    replay_buffer=experiences,
                    arc_data={'reason': sleep_reason, 'energy_level': self.current_energy}
                )
                result.update(sleep_result)
            else:
                # Fallback consolidation
                print(f" Enhanced sleep system not available, using fallback consolidation")
                result['consolidation_score'] = min(0.5, len(experiences) * 0.1)
                result['insights_generated'] = max(1, len(experiences) // 3)
                
            print(f" Sleep consolidation complete: {result['experiences_processed']} experiences processed")
            return result
            
        except Exception as e:
            print(f" Sleep consolidation failed: {e}")
            return {'success': False, 'error': str(e)}

    def _analyze_stagnation_cause(self, game_id: str, actions_history: List) -> Dict[str, Any]:
        """Analyze why the agent got stuck and provide insights."""
        analysis = {
            'game_id': game_id,
            'total_actions': len(actions_history),
            'stagnation_patterns': [],
            'suggested_fixes': [],
            'memory_analysis': {},
            'action_effectiveness': {}
        }

        if not actions_history:
            return analysis
        
        # 1. Action repetition analysis
        action_counts = {}
        for action in actions_history:
            # FIXED: Handle both integer actions and dictionary actions
            if isinstance(action, dict):
                action_type = action.get('action', 'unknown')
            else:
                # Integer format - action number directly
                action_type = action
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        total_actions = len(actions_history)
        for action, count in action_counts.items():
            percentage = (count / total_actions) * 100
            if percentage > 50:  # More than 50% of actions were the same
                analysis['stagnation_patterns'].append(f"Action {action} used {percentage:.1f}% of the time ({count}/{total_actions})")
                analysis['suggested_fixes'].append(f"Diversify from over-reliance on action {action}")
        
        # 2. Effectiveness analysis
        # FIXED: Handle both integer actions and dictionary actions
        effective_actions = []
        for a in actions_history:
            if isinstance(a, dict):
                # Dictionary format with score_change
                if a.get('score_change', 0) > 0:
                    effective_actions.append(a)
            else:
                # Integer format - we can't determine effectiveness without score data
                # Consider all actions as potentially effective for now
                effective_actions.append(a)
        
        effectiveness_rate = len(effective_actions) / total_actions * 100 if total_actions > 0 else 0
        analysis['action_effectiveness'] = {
            'total_actions': total_actions,
            'effective_actions': len(effective_actions),
            'effectiveness_rate': effectiveness_rate
        }
        
        if effectiveness_rate < 5:  # Less than 5% effectiveness
            analysis['suggested_fixes'].append("Extremely low effectiveness - need better action selection strategy")
        
        # 3. Memory analysis - check if we're learning
        if hasattr(self, 'available_actions_memory'):
            memory = self.available_actions_memory
            analysis['memory_analysis'] = {
                'actions_in_memory': len(memory.get('action_effectiveness', {})),
                'sequences_learned': len(memory.get('action_sequences', [])),
                'winning_sequences': len(memory.get('winning_action_sequences', []))
            }
            
            if analysis['memory_analysis']['actions_in_memory'] == 0:
                analysis['suggested_fixes'].append("No action effectiveness data - memory system not engaged")
        
        # 4. Specific recommendations based on patterns
        if len(action_counts) == 1:
            analysis['suggested_fixes'].append("Single action used exclusively - enable action diversity")
        elif len([a for a, c in action_counts.items() if c > 10]) > 3:
            analysis['suggested_fixes'].append("Multiple high-use actions - consider action prioritization")
        
        return analysis
        
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting statistics."""
        stats = self.rate_limiter.get_stats()
        stats.update({
            'rate_limit_config': ARC3_RATE_LIMIT,
            'current_requests_per_minute': len(self.rate_limiter.request_times),
            'rate_limit_efficiency': (stats['total_requests'] - stats['total_429s']) / max(1, stats['total_requests']),
            'recommended_action': (
                'All good' if stats['rate_limit_hit_rate'] < 0.05 else
                'Consider reducing request rate' if stats['rate_limit_hit_rate'] < 0.15 else
                'Significant rate limiting - increase delays'
            )
        })
        return stats
        
    def print_rate_limit_status(self):
        """Print current rate limiting status."""
        stats = self.get_rate_limit_stats()
        print(f"\n RATE LIMIT STATUS:")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Rate Limited (429s): {stats['total_429s']}")
        print(f"   Success Rate: {stats['rate_limit_efficiency']:.1%}")
        print(f"   Current RPM: {stats['current_requests_per_minute']}/600")
        if stats['current_backoff'] > 0:
            print(f"   ⏳ Active Backoff: {stats['current_backoff']:.1f}s")
        print(f"   Status: {stats['recommended_action']}")
        print()

    async def get_available_games(self) -> List[Dict[str, str]]:
        """
        Get list of available games from ARC-AGI-3 API.
        
        Returns:
            List of game metadata with game_id and title
        """
        self._ensure_initialized()
        
        url = "https://three.arcprize.org/api/games"
        headers = {"X-API-Key": self.api_key}
        
        try:
            # Check if rate limiter exists
            if not hasattr(self, 'rate_limiter'):
                raise AttributeError("Rate limiter not initialized. ContinuousLearningLoop initialization may have failed.")
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=ARC3_RATE_LIMIT['request_timeout'])) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        games = await response.json()
                        self.rate_limiter.handle_success_response()
                        logger.info(f"Retrieved {len(games)} available games from ARC-AGI-3 API")
                        return games
                    elif response.status == 401:
                        self.rate_limiter.handle_success_response()  # Not a rate limit issue
                        raise ValueError("Invalid API key - check ARC_API_KEY in .env file")
                    elif response.status == 429:
                        # Handle rate limit exceeded
                        self.rate_limiter.handle_429_response()
                        print(f" Rate limit exceeded getting games - will retry with backoff")
                        # Retry after backoff
                        await asyncio.sleep(self.rate_limiter.backoff_delay)
                        return await self.get_available_games()  # Recursive retry
                    else:
                        self.rate_limiter.handle_success_response()  # Not a rate limit issue
                        raise ValueError(f"API request failed with status {response.status}")
                        
        except asyncio.TimeoutError:
            print(f"⏰ API request timeout getting games")
            return []
        except Exception as e:
            logger.error(f"Failed to get available games: {e}")
            # Return empty list on failure
            return []
    
    async def select_training_games(
        self, 
        count: int = 8, 
        difficulty_preference: str = "mixed",
        include_pattern: Optional[str] = None
    ) -> List[str]:
        """
        Select games for training from available ARC-AGI-3 games.
        
        Args:
            count: Number of games to select
            difficulty_preference: "easy", "hard", "mixed" 
            include_pattern: Optional pattern to filter game titles
            
        Returns:
            List of selected game_ids
        """
        available_games = await self.get_available_games()
        
        if not available_games:
            logger.warning("No games available from API, using fallback games")
            # Fallback to some known game IDs if API fails
            return ["00d62c1b", "007bbfb7", "017c7c7b", "025d127b", "045e512c"][:count]
        
        # Filter by pattern if provided
        if include_pattern:
            filtered_games = [
                game for game in available_games 
                if include_pattern.lower() in game['title'].lower()
            ]
            if filtered_games:
                available_games = filtered_games
        
        # For now, select randomly since we don't have difficulty ratings
        # In a real implementation, you'd analyze game metadata for difficulty
        import random
        selected_games = random.sample(available_games, min(count, len(available_games)))
        
        game_ids = [game['game_id'] for game in selected_games]
        game_titles = [f"{game['title']} ({game['game_id']})" for game in selected_games]
        
        logger.info(f"Selected {len(game_ids)} games for training:")
        for title in game_titles:
            logger.info(f"  - {title}")
        
        return game_ids

    async def verify_api_connection(self) -> Dict[str, Any]:
        """
        Verify that we can connect to the ARC-AGI-3 API and get real data.
        
        Returns:
            Connection status and sample data
        """
        print(" VERIFYING ARC-AGI-3 API CONNECTION...")
        
        # Test 1: Get available games
        games = await self.get_available_games()
        
        verification_results = {
            'api_accessible': len(games) > 0,
            'total_games_available': len(games),
            'sample_games': games[:3] if games else [],
            'api_key_valid': len(games) > 0,
            'connection_timestamp': time.time()
        }
        
        if verification_results['api_accessible']:
            print(f" API Connection SUCCESS")
            print(f"   Available Games: {verification_results['total_games_available']}")
            print(f"   Sample Games:")
            for game in verification_results['sample_games']:
                print(f"     - {game['title']} ({game['game_id']})")
        else:
            print(f" API Connection FAILED")
            print(f"   Check API key and internet connection")
        
        # Test 2: Verify ARC-AGI-3-Agents integration
        agents_path_exists = self.arc_agents_path.exists()
        verification_results['arc_agents_available'] = agents_path_exists
        
        if agents_path_exists:
            print(f" ARC-AGI-3-Agents found at: {self.arc_agents_path}")
        else:
            print(f" ARC-AGI-3-Agents NOT found at: {self.arc_agents_path}")
            print(f"   Clone from: https://github.com/neoneye/ARC-AGI-3-Agents")
        
        return verification_results

    async def create_real_scorecard(self, games_list: List[str]) -> Optional[str]:
        """
        Open a scorecard using the correct ARC-3 API endpoint.
        This method is kept for compatibility but now uses the proper open scorecard API.
        
        Args:
            games_list: List of game_ids (ignored for open scorecard - games are added via RESET)
            
        Returns:
            scorecard_id if successful, None if failed
        """
        return await self._open_scorecard()

    async def _start_game_session(self, game_id: str, existing_guid: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Start a new game session or reset an existing one using the ARC-AGI-3 API RESET command.
        
        Args:
            game_id: The game ID to start
            existing_guid: If provided, performs a level reset within existing session
            
        Returns:
            Session data with GUID if successful, None if failed
        """
        try:
            # Step 1: Open a scorecard if we don't have one
            if not self.current_scorecard_id:
                scorecard_result = await self._open_scorecard()
                if not scorecard_result:
                    print(f" Failed to open scorecard")
                    return {
                        'error': 'Failed to open scorecard',
                        'state': 'ERROR',
                        'score': 0,
                        'available_actions': []
                    }
                self.current_scorecard_id = scorecard_result
                print(f" Opened NEW scorecard: {scorecard_result}")
                print(f" Scorecard state: created_new={self._created_new_scorecard} -> True")
                # Mark that we created a new scorecard
                self._created_new_scorecard = True
            else:
                print(f" Using existing scorecard: {self.current_scorecard_id}")
                print(f" Scorecard state: created_new={self._created_new_scorecard} -> False")
                # Mark that we're reusing an existing scorecard
                self._created_new_scorecard = False
            
            # Step 2: Prepare RESET call
            url = f"{ARC3_BASE_URL}/api/cmd/RESET"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Build payload based on reset type
            payload = {
                "game_id": game_id,
                "card_id": self.current_scorecard_id
            }
            
            if existing_guid:
                # Level Reset - reset within existing game session
                payload["guid"] = existing_guid
                print(f" LEVEL RESET for {game_id}")
            else:
                # New Game - start fresh game session
                print(f" NEW GAME RESET for {game_id}")
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=ARC3_RATE_LIMIT['request_timeout'])) as session:
                # Implement limited retries for transient errors
                max_retries = 3
                attempt = 0
                base_backoff = 1.0
                while attempt < max_retries:
                    attempt += 1

                    # Proceed directly with the API call - no preflight check needed

                    async with session.post(url, headers=headers, json=payload) as response:
                        status = response.status
                        if status == 200:
                            result = await response.json()
                            self.rate_limiter.handle_success_response()
                            guid = result.get('guid')
                            reset_type = "LEVEL RESET" if existing_guid else "NEW GAME"
                            if guid:
                                # Store the GUID for this game
                                self.current_game_sessions[game_id] = guid
                                # Log GUID -> game mapping for auditing
                                try:
                                    log_action_trace({'ts': time.time(), 'event': 'guid_assigned', 'game_id': game_id, 'guid': guid})
                                except Exception:
                                    pass
                                print(f" {reset_type} successful: {game_id}")

                                # Initialize position tracking for ACTION 6 directional movement
                                if not existing_guid:  # Only for new games, not level resets
                                    # Get grid dimensions from frame if available (use canonical normalizer)
                                    frame = result.get('frame', None)
                                    arr, (grid_width, grid_height) = self._normalize_frame(frame)
                                    if arr is None:
                                        grid_width, grid_height = 64, 64

                                    # Start ACTION 6 position at center of grid for directional movement
                                    self._current_game_x = grid_width // 2
                                    self._current_game_y = grid_height // 2
                                    print(f" Initialized ACTION 6 position at center: ({self._current_game_x},{self._current_game_y}) for {grid_width}x{grid_height} grid")

                                    # Clear boundary data for new games - boundaries are game-specific
                                    universal_boundary = self.available_actions_memory['universal_boundary_detection']
                                    legacy_boundary = self.available_actions_memory['action6_boundary_detection']

                                    # Clear universal boundary data
                                    if 'boundary_data' in universal_boundary and game_id in universal_boundary['boundary_data']:
                                        old_boundaries = len(universal_boundary['boundary_data'][game_id])
                                        print(f" Cleared {old_boundaries} previous universal boundary mappings for new game {game_id}")

                                    # Initialize fresh universal boundary detection for this game
                                    if 'boundary_data' not in universal_boundary:
                                        universal_boundary['boundary_data'] = {}
                                    universal_boundary['boundary_data'][game_id] = {}
                                    if 'coordinate_attempts' not in universal_boundary:
                                        universal_boundary['coordinate_attempts'] = {}
                                    universal_boundary['coordinate_attempts'][game_id] = {}
                                    
                                    if 'action_coordinate_history' not in universal_boundary:
                                        universal_boundary['action_coordinate_history'] = {}
                                    universal_boundary['action_coordinate_history'][game_id] = {}
                                    
                                    if 'stuck_patterns' not in universal_boundary:
                                        universal_boundary['stuck_patterns'] = {}
                                    universal_boundary['stuck_patterns'][game_id] = {}
                                    
                                    if 'success_zone_mapping' not in universal_boundary:
                                        universal_boundary['success_zone_mapping'] = {}
                                    universal_boundary['success_zone_mapping'][game_id] = {}

                                    # Initialize ACTION 6 directional system
                                    if 6 in universal_boundary['directional_systems']:
                                        universal_boundary['directional_systems'][6]['current_direction'][game_id] = 'right'

                                    # Initialize legacy system for backward compatibility
                                    if 'boundary_data' not in legacy_boundary:
                                        legacy_boundary['boundary_data'] = {}
                                    legacy_boundary['boundary_data'][game_id] = {}
                                    
                                    if 'coordinate_attempts' not in legacy_boundary:
                                        legacy_boundary['coordinate_attempts'] = {}
                                    legacy_boundary['coordinate_attempts'][game_id] = {}
                                    
                                    if 'last_coordinates' not in legacy_boundary:
                                        legacy_boundary['last_coordinates'] = {}
                                    legacy_boundary['last_coordinates'][game_id] = None
                                    
                                    if 'stuck_count' not in legacy_boundary:
                                        legacy_boundary['stuck_count'] = {}
                                    legacy_boundary['stuck_count'][game_id] = 0
                                    
                                    if 'current_direction' not in legacy_boundary:
                                        legacy_boundary['current_direction'] = {}
                                    legacy_boundary['current_direction'][game_id] = 'right'

                                    print(f" Initialized universal boundary detection system for {game_id}")

                                    # ENHANCED: Reset frame analyzer tracking for new game including exploration phase
                                    if hasattr(self, 'frame_analyzer') and self.frame_analyzer:
                                        try:
                                            if hasattr(self.frame_analyzer, 'reset_for_new_game'):
                                                self.frame_analyzer.reset_for_new_game(game_id)
                                            else:
                                                self.frame_analyzer.reset_coordinate_tracking()
                                            print(f" Reset frame analyzer tracking for new game {game_id}")
                                        except Exception as e:
                                            print(f" Failed to reset frame analyzer: {e}")
                                    print(f" All actions will now benefit from boundary awareness and coordinate intelligence")

                                return {
                                    'guid': guid,
                                    'game_id': game_id,
                                    'scorecard_id': self.current_scorecard_id,
                                    'state': result.get('state', 'NOT_STARTED'),
                                    'frame': result.get('frame', []),
                                    'available_actions': result.get('available_actions', [1,2,3,4,5,6,7]),
                                    'score': result.get('score', 0),
                                    'reset_type': reset_type.lower().replace(' ', '_')
                                }
                            else:
                                print(f" No GUID returned for game {game_id}")
                                return {
                                    'error': f'No GUID returned for game {game_id}',
                                    'state': 'ERROR',
                                    'score': 0,
                                    'available_actions': []
                                }
                        elif status == 429:
                            # Handle rate limit exceeded
                            self.rate_limiter.handle_429_response()
                            print(f" Rate limit exceeded on RESET {game_id} - will retry with backoff")
                            await asyncio.sleep(self.rate_limiter.backoff_delay)
                            continue  # Retry loop
                        elif 500 <= status < 600:
                            # Transient server error - retry with jittered exponential backoff
                            backoff = min(base_backoff * (2 ** (attempt - 1)), ARC3_RATE_LIMIT['backoff_max_delay'])
                            jitter = random.uniform(0.2, 1.0)
                            wait = backoff + jitter
                            print(f" Server error on RESET {game_id}: {status} - attempt {attempt}/{max_retries}, backing off {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue
                        else:
                            self.rate_limiter.handle_success_response()  # Not a rate limit issue
                            error_text = await response.text()
                            # Best-effort debug logging for ops: response status, headers, and truncated body
                            try:
                                resp_headers = dict(response.headers)
                                resp_body = error_text if isinstance(error_text, str) else str(error_text)
                                if len(resp_body) > 2000:
                                    resp_body = resp_body[:2000] + "...<truncated>"
                                logger.debug(f"RESET non-200 response for {game_id}: status={status}, headers={resp_headers}, body={resp_body}")
                                # Also append to an append-only debug log on disk for ops
                                try:
                                    self._append_reset_debug_log(game_id, status, resp_headers, resp_body)
                                except Exception:
                                    pass
                            except Exception:
                                logger.debug("Failed to capture detailed RESET response for debugging")

                            # Treat 400 with a 'no available game backend' server message as transient
                            if status == 400 and isinstance(error_text, str) and 'no available game backend' in error_text.lower():
                                # Backoff and retry with jitter
                                backoff = min(base_backoff * (2 ** (attempt - 1)), ARC3_RATE_LIMIT['backoff_max_delay'])
                                jitter = random.uniform(0.5, 1.5)
                                wait = backoff + jitter
                                print(f" No available game backend for {game_id} (server message). Retrying after {wait:.1f}s (attempt {attempt}/{max_retries})")
                                await asyncio.sleep(wait)
                                continue

                            print(f" RESET failed: {status} - {error_text}")
                            return {
                                'error': f'RESET failed: {status} - {error_text}',
                                'state': 'ERROR',
                                'score': 0,
                                'available_actions': []
                            }
                    # End of attempt loop - if we reach here and didn't return, try again
                # Exhausted retries
                print(f" RESET failed after {max_retries} attempts for {game_id}")
                return {
                    'error': f'RESET failed after {max_retries} attempts for {game_id}',
                    'state': 'ERROR',
                    'score': 0,
                    'available_actions': []
                }
                        
        except asyncio.TimeoutError:
            print(f"⏰ API request timeout on RESET {game_id}")
            return {
                'error': f'API request timeout on RESET {game_id}',
                'state': 'ERROR',
                'score': 0,
                'available_actions': []
            }
        except Exception as e:
            print(f" Error starting game session for {game_id}: {e}")
            return {
                'error': f'Error starting game session for {game_id}: {e}',
                'state': 'ERROR',
                'score': 0,
                'available_actions': []
            }

    def _generate_scorecard_tags(self) -> List[str]:
        """
        Generate descriptive tags for scorecard based on current configuration and state.
        
        Returns:
            List of descriptive tags for the scorecard
        """
        tags = []
        
        # Base system tags
        tags.extend([
            "tabula_rasa",
            "adaptive_learning_agent",
            "arc_agi_3"
        ])
        
        # Training mode tags
        if hasattr(self, 'training_mode'):
            tags.append(f"mode_{self.training_mode}")
        
        # System capability tags
        if getattr(self, 'enable_coordinates', False):
            tags.append("coordinates_enabled")
        if getattr(self, 'enable_energy_system', False):
            tags.append("energy_system")
        if getattr(self, 'enable_sleep_system', False):
            tags.append("sleep_consolidation")
        if getattr(self, 'enable_meta_cognitive_governor', False):
            tags.append("meta_cognitive_governor")
        if getattr(self, 'enable_architect_evolution', False):
            tags.append("architect_evolution")
        
        # Performance tags
        if hasattr(self, 'current_energy'):
            if self.current_energy > 80:
                tags.append("high_energy")
            elif self.current_energy < 40:
                tags.append("low_energy")
        
        # Learning state tags
        if hasattr(self, 'learned_patterns') and self.learned_patterns:
            tags.append("has_learned_patterns")
        
        # Session tags
        if hasattr(self, 'session_count'):
            tags.append(f"session_{self.session_count}")
        
        # Timestamp tag for uniqueness
        import time
        timestamp = int(time.time())
        tags.append(f"timestamp_{timestamp}")
        
        # Governor state tags
        if hasattr(self, 'governor') and self.governor:
            tags.append("governor_managed")
        
        return tags

    async def _open_scorecard(self) -> Optional[str]:
        """
        Open a new scorecard using the correct ARC-3 API endpoint.
        
        Returns:
            scorecard_id if successful, None if failed
        """
        try:
            url = "https://three.arcprize.org/api/scorecard/open"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Generate descriptive tags based on current configuration and state
            tags = self._generate_scorecard_tags()
            payload = {"tags": tags} if tags else {}
            
            print(f" Opening scorecard...")
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=ARC3_RATE_LIMIT['request_timeout'])) as session:
                max_retries = 3
                attempt = 0
                while attempt < max_retries:
                    attempt += 1
                    async with session.post(url, headers=headers, json=payload) as response:
                        status = response.status
                        if status == 200:
                            result = await response.json()
                            self.rate_limiter.handle_success_response()
                            scorecard_id = result.get('card_id')
                            if scorecard_id:
                                self.current_scorecard_id = scorecard_id
                                print(f" Scorecard opened: {scorecard_id}")
                                return scorecard_id
                            else:
                                print(f" No card_id returned")
                                return {
                                    'error': 'No card_id returned from scorecard API',
                                    'state': 'ERROR',
                                    'score': 0,
                                    'available_actions': []
                                }
                        elif status == 429:
                            self.rate_limiter.handle_429_response()
                            print(f" Rate limit exceeded opening scorecard - will retry with backoff")
                            await asyncio.sleep(self.rate_limiter.backoff_delay)
                            continue
                        elif 500 <= status < 600:
                            print(f" Server error opening scorecard: {status} (attempt {attempt}/{max_retries})")
                            await asyncio.sleep(min(self.rate_limiter.backoff_delay or 1.0, ARC3_RATE_LIMIT['backoff_max_delay']))
                            continue
                        else:
                            self.rate_limiter.handle_success_response()
                            error_text = await response.text()
                            print(f" Failed to open scorecard: {status} - {error_text}")
                            return {
                                'error': f'Failed to open scorecard: {status} - {error_text}',
                                'state': 'ERROR',
                                'score': 0,
                                'available_actions': []
                            }
                print(f" Failed to open scorecard after {max_retries} attempts")
                return {
                    'error': f'Failed to open scorecard after {max_retries} attempts',
                    'state': 'ERROR',
                    'score': 0,
                    'available_actions': []
                }
                        
        except asyncio.TimeoutError:
            print(f"⏰ API request timeout opening scorecard")
            return {
                'error': 'API request timeout opening scorecard',
                'state': 'ERROR',
                'score': 0,
                'available_actions': []
            }
        except Exception as e:
            print(f" Error opening scorecard: {e}")
            return {
                'error': f'Error opening scorecard: {e}',
                'state': 'ERROR',
                'score': 0,
                'available_actions': []
            }

    def _reset_scorecard_state(self):
        """Reset scorecard state tracking variables."""
        self._created_new_scorecard = False
        # Don't clear current_scorecard_id here - let the cleanup logic handle it
    
    def _clear_game_sessions(self):
        """Clear all game sessions to force fresh starts."""
        self.current_game_sessions.clear()
        print(" Cleared all game sessions - next games will start fresh")
    
    def _force_close_scorecard(self):
        """Force close the current scorecard on next cleanup."""
        self._force_scorecard_close = True
        print(" Marked scorecard for forced closure")
    
    async def _close_scorecard(self, scorecard_id: str) -> bool:
        """
        Close a scorecard using the correct ARC-3 API endpoint.
        
        Args:
            scorecard_id: The scorecard ID to close
            
        Returns:
            True if successful, False if failed
        """
        try:
            url = "https://three.arcprize.org/api/scorecard/close"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {"card_id": scorecard_id}
            
            print(f" Closing scorecard: {scorecard_id}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f" Closed scorecard: {scorecard_id}")
                        return True
                    elif response.status == 404:
                        print(f" Scorecard {scorecard_id} already closed or not found - continuing")
                        return True  # Treat as success since it's already gone
                    else:
                        error_text = await response.text()
                        print(f" Failed to close scorecard: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            print(f" Error closing scorecard: {e}")
            return False

    async def _reset_level(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Reset the current level within an existing game session (Level Reset).
        
        Args:
            game_id: The game ID to reset
            
        Returns:
            Updated session data if successful, None if failed
        """
        existing_guid = self.current_game_sessions.get(game_id)
        if not existing_guid:
            print(f" No existing session GUID found for {game_id}, cannot perform level reset")
            return {
                'error': f'No existing session GUID found for {game_id}, cannot perform level reset',
                'state': 'ERROR',
                'score': 0,
                'available_actions': []
            }
        
        return await self._start_game_session(game_id, existing_guid=existing_guid)

    def _init_demo_agent(self):
        """Initialize a demonstration agent for monitoring purposes."""
        try:
            # Create minimal agent for demonstration
            config = {
                'predictive_core': {
                    'visual_size': [3, 64, 64],
                    'proprioception_size': 12,
                    'hidden_size': 128,
                    'use_memory': True
                },
                'memory': {
                    'memory_size': 512,
                    'word_size': 64,
                    'use_learned_importance': False
                },
                'sleep': {
                    'sleep_trigger_energy': 40.0,  # Increased from 20.0 to 40.0 for more frequent sleep cycles
                    'sleep_trigger_boredom_steps': 100,
                    'sleep_duration_steps': 50
                }
            }
            
            self.demo_agent = AdaptiveLearningAgent(config)
            logger.info("Demo agent initialized for monitoring")
        except Exception as e:
            logger.warning(f"Could not initialize demo agent: {e}")
            self.demo_agent = None

    def _extract_grid_dimensions(self, response_data: Dict[str, Any]) -> Tuple[int, int]:
        """Extract actual grid dimensions from API response frame data."""
        try:
            frame = response_data.get('frame', None)
            # Use canonical normalization to derive dimensions
            arr, (w, h) = self._normalize_frame(frame)
            if arr is not None:
                # _normalize_frame returns (array, (width,height))
                if 1 <= w <= 1024 and 1 <= h <= 1024:
                    return (w, h)
                else:
                    logger.warning(f"Invalid normalized dimensions detected: {w}x{h}, using fallback")
            
        except (IndexError, TypeError) as e:
            logger.warning(f"Error extracting grid dimensions: {e}")
        
        # Fallback to maximum size
        return (64, 64)
    
    def _extract_grid_dimensions_from_output(self, stdout: str, stderr: str) -> Optional[Tuple[int, int]]:
        """Extract grid dimensions from command output."""
        combined_output = stdout + "\n" + stderr
        
        # Look for grid dimension patterns in output
        dimension_patterns = [
            r'grid.*?(\d+)\s*[x×]\s*(\d+)',
            r'dimensions?[:\s]+(\d+)\s*[x×]\s*(\d+)',
            r'size[:\s]+(\d+)\s*[x×]\s*(\d+)',
            r'(\d+)x(\d+)\s*grid',
            r'frame.*?(\d+)\s*[x×]\s*(\d+)'
        ]
        
        for pattern in dimension_patterns:
            match = re.search(pattern, combined_output, re.IGNORECASE)
            if match:
                try:
                    width, height = int(match.group(1)), int(match.group(2))
                    if 1 <= width <= 64 and 1 <= height <= 64:
                        return (width, height)
                except (ValueError, IndexError):
                    continue
        
        return None

    def _check_frame_changes(self, before_state: Dict, after_state: Dict) -> Optional[Dict[str, Any]]:
        """Check for frame changes and movement between states."""
        try:
            before_frame = before_state.get('frame', [])
            after_frame = after_state.get('frame', [])
            
            if not before_frame or not after_frame:
                return None
            
            # Use frame analyzer to detect changes
            if hasattr(self, 'frame_analyzer') and hasattr(self.frame_analyzer, '_analyze_frame_changes'):
                frame_changes = self.frame_analyzer._analyze_frame_changes(before_frame, after_frame)
                
                if frame_changes.get('changes_detected', False):
                    # Enhanced change detection
                    change_type = frame_changes.get('change_type', 'unknown')
                    num_pixels_changed = frame_changes.get('num_pixels_changed', 0)
                    change_locations = frame_changes.get('change_locations', [])
                    
                    # Determine if this represents movement
                    movement_detected = self._detect_movement_pattern(change_locations, num_pixels_changed)
                    
                    # Classify the type of change
                    change_classification = self._classify_change_type(change_type, num_pixels_changed, movement_detected)
                    
                    return {
                        'change_type': change_classification,
                        'description': f"{change_classification}: {num_pixels_changed} pixels changed",
                        'movement_detected': movement_detected,
                        'num_pixels_changed': num_pixels_changed,
                        'change_locations': change_locations[:10],  # Limit for storage
                        'change_percentage': frame_changes.get('change_percentage', 0.0)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking frame changes: {e}")
            return None
    
    def _detect_movement_pattern(self, change_locations: List[Tuple[int, int]], num_pixels_changed: int) -> bool:
        """Detect if pixel changes represent object movement rather than just color changes."""
        try:
            if not change_locations or num_pixels_changed < 3:
                return False
            
            # Calculate spatial distribution of changes
            if len(change_locations) >= 2:
                x_coords = [loc[0] for loc in change_locations]
                y_coords = [loc[1] for loc in change_locations]
                
                # Check if changes are clustered (indicating object movement)
                x_variance = np.var(x_coords) if len(x_coords) > 1 else 0
                y_variance = np.var(y_coords) if len(y_coords) > 1 else 0
                
                # Movement typically shows moderate clustering (not too spread out, not too tight)
                total_variance = x_variance + y_variance
                is_clustered = 5 < total_variance < 100  # Reasonable clustering range
                
                # Check for directional patterns (movement often has direction)
                if len(change_locations) >= 3:
                    # Calculate if changes form a line or have directional bias
                    x_trend = np.polyfit(range(len(x_coords)), x_coords, 1)[0] if len(x_coords) > 1 else 0
                    y_trend = np.polyfit(range(len(y_coords)), y_coords, 1)[0] if len(y_coords) > 1 else 0
                    
                    has_direction = abs(x_trend) > 0.5 or abs(y_trend) > 0.5
                else:
                    has_direction = False
                
                return is_clustered and (has_direction or num_pixels_changed > 10)
            
            return False
            
        except Exception as e:
            logger.warning(f"Movement pattern detection failed: {e}")
            return False
    
    def _classify_change_type(self, change_type: str, num_pixels_changed: int, movement_detected: bool) -> str:
        """Classify the type of frame change."""
        if movement_detected:
            if num_pixels_changed > 50:
                return 'major_movement'
            elif num_pixels_changed > 10:
                return 'object_movement'
            else:
                return 'small_movement'
        elif change_type == 'frame_size_change':
            return 'frame_resize'
        elif num_pixels_changed > 100:
            return 'major_visual_change'
        elif num_pixels_changed > 20:
            return 'moderate_change'
        else:
            return 'minor_change'
    
    def _update_action_effectiveness(self, action_number: int, frame_changed: Dict, score_change: float):
        """Update action effectiveness based on frame changes."""
        # Ensure memory is properly initialized
        self._ensure_available_actions_memory()
        try:
            if not hasattr(self, 'available_actions_memory'):
                return
            
            action_effectiveness = self.available_actions_memory.get('action_effectiveness', {})
            if action_number not in action_effectiveness:
                action_effectiveness[action_number] = {
                    'attempts': 0, 'successes': 0, 'success_rate': 0.0,
                    'frame_changes': 0, 'movement_detected': 0, 'score_changes': 0
                }
            
            stats = action_effectiveness[action_number]
            stats['attempts'] += 1
            
            # Count frame changes as positive indicators
            if frame_changed:
                stats['frame_changes'] += 1
                if frame_changed.get('movement_detected', False):
                    stats['movement_detected'] += 1
                    stats['successes'] += 1  # Movement is a success indicator
            
            # Count score changes as success
            if score_change > 0:
                stats['score_changes'] += 1
                stats['successes'] += 1
            
            # Update success rate
            stats['success_rate'] = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error updating action effectiveness: {e}")
    
    def _reset_stagnation_counter(self, game_id: str):
        """Reset stagnation counter for a game when movement is detected."""
        try:
            if not hasattr(self, 'available_actions_memory'):
                return
            
            # Reset various stagnation tracking
            if 'action_stagnation' not in self.available_actions_memory:
                self.available_actions_memory['action_stagnation'] = {}
            
            self.available_actions_memory['action_stagnation'][game_id] = {
                'consecutive_no_change': 0,
                'last_progress_action': 0,
                'stagnant_actions': []
            }
            
        except Exception as e:
            logger.error(f"Error resetting stagnation counter: {e}")
    
    def _increment_stagnation_counter(self, game_id: str, action_number: int):
        """Increment stagnation counter when no frame changes are detected."""
        try:
            if not hasattr(self, 'available_actions_memory'):
                return
            
            if 'action_stagnation' not in self.available_actions_memory:
                self.available_actions_memory['action_stagnation'] = {}
            
            if game_id not in self.available_actions_memory['action_stagnation']:
                self.available_actions_memory['action_stagnation'][game_id] = {
                    'consecutive_no_change': 0,
                    'last_progress_action': 0,
                    'stagnant_actions': []
                }
            
            stagnation = self.available_actions_memory['action_stagnation'][game_id]
            stagnation['consecutive_no_change'] += 1
            stagnation['stagnant_actions'].append(action_number)
            
            # Keep only recent stagnant actions
            if len(stagnation['stagnant_actions']) > 10:
                stagnation['stagnant_actions'] = stagnation['stagnant_actions'][-10:]
            
        except Exception as e:
            logger.error(f"Error incrementing stagnation counter: {e}")
    
    def _check_action_stagnation(self, game_id: str, available_actions: List[int]) -> Dict[str, Any]:
        """Check for action stagnation and recommend switching strategies."""
        try:
            if not hasattr(self, 'available_actions_memory'):
                return {'should_switch': False, 'reason': 'No memory system available'}
            
            stagnation = self.available_actions_memory.get('action_stagnation', {}).get(game_id, {})
            consecutive_no_change = stagnation.get('consecutive_no_change', 0)
            stagnant_actions = stagnation.get('stagnant_actions', [])
            
            # Check if we should switch actions
            should_switch = False
            reason = ""
            recommended_actions = available_actions.copy()
            
            # Switch if too many consecutive actions with no frame changes
            if consecutive_no_change >= 5:
                should_switch = True
                reason = f"No frame changes for {consecutive_no_change} consecutive actions"
                
                # Get actions that haven't been tried recently
                recent_actions = stagnant_actions[-5:] if len(stagnant_actions) >= 5 else stagnant_actions
                untried_actions = [a for a in available_actions if a not in recent_actions]
                
                if untried_actions:
                    recommended_actions = untried_actions
                else:
                    # If all actions have been tried, prioritize different action types
                    action_effectiveness = self.available_actions_memory.get('action_effectiveness', {})
                    effective_actions = []
                    for action in available_actions:
                        if action in action_effectiveness:
                            success_rate = action_effectiveness[action].get('success_rate', 0.0)
                            if success_rate > 0.1:  # Actions with some success
                                effective_actions.append(action)
                    
                    if effective_actions:
                        recommended_actions = effective_actions
                    else:
                        # Fallback: try actions in different order
                        recommended_actions = available_actions[::-1]  # Reverse order
            
            # Switch if same action repeated too many times
            elif len(stagnant_actions) >= 3:
                last_three = stagnant_actions[-3:]
                if len(set(last_three)) == 1:  # Same action repeated 3 times
                    should_switch = True
                    reason = f"Action {last_three[0]} repeated 3 times with no frame changes"
                    
                    # Exclude the repeated action, but ensure we don't empty the list
                    repeated_action = last_three[0]
                    recommended_actions = [a for a in available_actions if a != repeated_action]
                    
                    # CRITICAL FIX: If excluding the repeated action leaves us with no options,
                    # keep the original action but mark it as a forced choice
                    if not recommended_actions:
                        recommended_actions = available_actions.copy()
                        reason += " (forced to continue with same action - no alternatives)"
            
            return {
                'should_switch': should_switch,
                'reason': reason,
                'consecutive_no_change': consecutive_no_change,
                'stagnant_actions': stagnant_actions,
                'recommended_actions': recommended_actions
            }
            
        except Exception as e:
            logger.error(f"Error checking action stagnation: {e}")
            return {'should_switch': False, 'reason': f'Error: {e}'}

    def _normalize_frame(self, frame: Any) -> Tuple[Optional['np.ndarray'], Tuple[int, int]]:
        """
        Normalize various frame representations into a 2D numpy array and return (array, (width, height)).

        Handles common variants seen from the API:
        - frame as [[row1, row2, ...]] (wrapped in extra list)
        - frame as [row1, row2, ...] (2D list)
        - frame as flat list of length N (attempt square reshape or common 8x8/64x64 heuristics)
        - numpy arrays
        Returns (None, (64,64)) on failure as safe fallback.
        """
        try:
            if frame is None:
                return None, (64, 64)

            # If it's already a numpy array
            if isinstance(frame, np.ndarray):
                arr = frame
            else:
                # Handle common list shapes safely
                if isinstance(frame, list) and len(frame) > 0:
                    first = frame[0]
                    # Case: wrapped extra list like [[r0, r1, ...]] or [[row1],[row2],...]
                    if isinstance(first, list):
                        # If outer list has one element which itself is a list-of-rows, unwrap
                        if len(frame) == 1 and any(isinstance(el, list) for el in first):
                            arr = np.array(first)
                        else:
                            arr = np.array(frame)
                    # Case: flat numeric list -> attempt reshape heuristics
                    elif isinstance(first, (int, float)):
                        total = len(frame)
                        if total == 64:
                            arr = np.array(frame).reshape((8, 8))
                        else:
                            side = int(total ** 0.5)
                            if side * side == total:
                                arr = np.array(frame).reshape((side, side))
                            elif total % 64 == 0:
                                height = total // 64
                                arr = np.array(frame).reshape((height, 64))
                            else:
                                arr = np.array(frame).reshape((1, total))
                    else:
                        # Unknown nested structure - attempt generic conversion
                        arr = np.array(frame)
                else:
                    # Fallback generic conversion
                    arr = np.array(frame)

            # Ensure at least 2D
            if arr is None:
                return None, (64, 64)

            if arr.ndim == 1:
                total = arr.size
                side = int(total ** 0.5)
                if side * side == total:
                    arr = arr.reshape((side, side))
                else:
                    arr = arr.reshape((1, total))

            # Now arr is 2D: (height, width)
            height, width = int(arr.shape[0]), int(arr.shape[1])

            # Clamp to sensible bounds
            if not (1 <= width <= 1024 and 1 <= height <= 1024):
                return arr, (64, 64)

            return arr, (width, height)

        except Exception:
            return None, (64, 64)

    def _verify_grid_bounds(self, x: int, y: int, width: int, height: int) -> bool:
        """Return True if (x,y) are within 0..width-1 and 0..height-1."""
        try:
            if width <= 0 or height <= 0:
                return False
            return 0 <= x < width and 0 <= y < height
        except Exception:
            return False

    def _ensure_reset_debug_dir(self) -> Path:
        """Ensure the reset debug logs directory exists and return its Path."""
        try:
            path = self.save_directory / "reset_debug_logs"
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            # Fallback to save_directory if creation fails
            return self.save_directory

    def _append_reset_debug_log(self, game_id: str, status: int, headers: Dict[str, Any], body: str) -> None:
        """Append a JSON record of a RESET non-200 response to an append-only NDJSON file.

        This is best-effort and must never raise.
        """
        try:
            log_dir = self._ensure_reset_debug_dir()
            timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            fname = log_dir / f"reset_debug_{timestamp}.ndjson"
            record = {
                'ts': time.time(),
                'iso_ts': datetime.utcnow().isoformat() + 'Z',
                'game_id': game_id,
                'status': status,
                'headers': headers,
                'body': body[:20000]  # cap body to avoid enormous writes
            }
            with open(fname, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            try:
                logger.debug("Failed to write reset debug log to disk")
            except Exception:
                pass


    def _safe_coordinate_fallback(self, grid_width: int, grid_height: int, reason: str = "bounds check failed") -> Tuple[int, int]:
        """Return a safe coordinate (center) when bounds are invalid."""
        try:
            gx = int(grid_width) if grid_width else 64
            gy = int(grid_height) if grid_height else 64
            cx = max(0, min(gx - 1, gx // 2)) if gx > 0 else 32
            cy = max(0, min(gy - 1, gy // 2)) if gy > 0 else 32
            logger.debug(f"Safe coordinate fallback used ({cx},{cy}) due to: {reason}")
            return cx, cy
        except Exception:
            return 32, 32
    
    def _extract_game_state_from_output(self, stdout: str, stderr: str) -> str:
        """Extract game state from command output."""
        combined_output = stdout + "\n" + stderr
        
        # Look for explicit game state mentions
        state_patterns = [
            r'state[:\s]+(\w+)',
            r'game.*?state[:\s]+(\w+)',
            r'status[:\s]+(\w+)',
            r'\b(WIN|GAME_OVER|NOT_FINISHED|NOT_STARTED)\b'
        ]
        
        for pattern in state_patterns:
            match = re.search(pattern, combined_output, re.IGNORECASE)
            if match:
                state = match.group(1).upper()
                if state in ['WIN', 'GAME_OVER', 'NOT_FINISHED', 'NOT_STARTED']:
                    return state
        
        # Infer state from success/failure indicators - ENHANCED WIN DETECTION
        # Check for full game wins first (highest priority)
        if re.search(r'\b(game.*complete|puzzle.*solved|challenge.*complete|full.*win|complete.*game|victory.*complete|final.*level.*complete|all.*levels.*complete)\b', combined_output, re.IGNORECASE):
            return 'FULL_GAME_WIN'  # Complete game victory!
        # Check for level completions (partial wins)
        elif re.search(r'levels.*completed.*(\d+)', combined_output, re.IGNORECASE):
            return 'LEVEL_WIN'  # Level completion = LEVEL WIN!
        elif re.search(r'completed.*(\d+).*levels', combined_output, re.IGNORECASE):
            return 'LEVEL_WIN'  # Level completion = LEVEL WIN!
        elif re.search(r'\b(win|victory|success|solved)\b', combined_output, re.IGNORECASE):
            return 'WIN'  # Generic win
        elif re.search(r'\b(game.*?over|failed|timeout|error)\b', combined_output, re.IGNORECASE):
            return 'GAME_OVER'
        
        return 'NOT_FINISHED'  # Default assumption
    
    def _parse_episode_results_comprehensive(self, stdout: str, stderr: str, game_id: str) -> Dict[str, Any]:
        """Comprehensive parsing of episode results with enhanced pattern detection."""
        result = {'success': False, 'final_score': 0, 'actions_taken': 0, 'level_progressed': False, 'current_level': None}
        combined_output = stdout + "\n" + stderr
        
        # Check for FULL GAME WIN first (completing entire game)
        full_game_win_patterns = [
            r'game.*complete',
            r'puzzle.*solved',
            r'challenge.*complete',
            r'full.*win',
            r'complete.*game',
            r'victory.*complete',
            r'final.*level.*complete',
            r'all.*levels.*complete'
        ]
        
        for pattern in full_game_win_patterns:
            if re.search(pattern, combined_output, re.IGNORECASE):
                result['full_game_win'] = True
                result['success'] = True
                result['win_type'] = 'FULL_GAME_WIN'
                print(f"🏆 FULL GAME WIN DETECTED: Complete game victory - marking as FULL WIN!")
                break
        
        # Check for level progression indicators - ENHANCED DETECTION
        if not result.get('full_game_win', False):
            level_progression_patterns = [
                r'level.*(\d+).*complete',
                r'passed.*level.*(\d+)', 
                r'advanced.*level.*(\d+)',
                r'next.*level.*(\d+)',
                r'level.*up.*(\d+)',
                r'stage.*(\d+).*complete',
                r'tier.*(\d+).*unlock',
                r'levels.*completed.*(\d+)',  # NEW: Direct level count from scorecard
                r'completed.*(\d+).*levels',  # NEW: Alternative phrasing
                r'level.*(\d+).*solved',      # NEW: Solved terminology
                r'solved.*level.*(\d+)',      # NEW: Alternative solved phrasing
            ]
            
            for pattern in level_progression_patterns:
                match = re.search(pattern, combined_output, re.IGNORECASE)
                if match:
                    try:
                        level = int(match.group(1))
                        result['level_progressed'] = True
                        result['current_level'] = level
                        result['success'] = True  # 🎯 LEVEL COMPLETION = SUCCESS!
                        result['win_type'] = 'LEVEL_WIN'
                        print(f"🎯 LEVEL WIN DETECTED: Level {level} completed - marking as LEVEL WIN!")
                        break
                    except ValueError:
                        continue
        
        # Enhanced success detection patterns
        success_patterns = [
            r'\b(win|victory|success|solved|correct|passed)\b',
            r'',
            r'\btrue\b.*answer',
            r'answer.*\bcorrect\b',
            r'result.*\btrue\b',
            r'score.*100',
            r'status.*success'
        ]
        
        for pattern in success_patterns:
            if re.search(pattern, combined_output, re.IGNORECASE):
                result['success'] = True
                break
        
        # Enhanced score extraction patterns
        score_patterns = [
            r'score[:\s]*(\d+)',
            r'final[_\s]*score[:\s]*(\d+)',
            r'points?[:\s]*(\d+)',
            r'result[:\s]*(\d+)',
            r'total[:\s]*(\d+)',
            r'grade[:\s]*(\d+)',
            r'(\d+)[/\s]*100',
            r'(\d+)%',
            r'accuracy[:\s]*(\d+)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, combined_output, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    result['final_score'] = max(result['final_score'], score)
                except ValueError:
                    continue
        
        # Count actions taken
        action_patterns = [
            r'action[:\s]*(\d+)',
            r'step[:\s]*(\d+)',
            r'move[:\s]*(\d+)'
        ]
        
        # Extract action counts from output patterns
        for pattern in action_patterns:
            matches = re.findall(pattern, combined_output, re.IGNORECASE)
            # If patterns include numeric action counts, use the largest as a conservative estimate
            for m in matches:
                try:
                    val = int(m)
                    result['actions_taken'] = max(result.get('actions_taken', 0), val)
                except Exception:
                    continue

        # Return the parsed result dictionary
        return result
    
    def _safe_coordinate_fallback(self, grid_width: int, grid_height: int, reason: str = "bounds check failed") -> Tuple[int, int]:
        """ CRITICAL FIX: Generate safe fallback coordinates that are guaranteed to be in bounds."""
        # Use center coordinates as safe fallback, but ensure they're valid
        center_x = max(0, min(grid_width - 1, grid_width // 2))
        center_y = max(0, min(grid_height - 1, grid_height // 2))
        
        print(f" SAFE FALLBACK: Using ({center_x},{center_y}) for {grid_width}x{grid_height} grid - {reason}")
        return (center_x, center_y)
    
    def _analyze_frame_for_action_selection(self, response_data: Dict[str, Any], game_id: str) -> Dict[str, Any]:
        """
        Use frame analysis to enhance action selection with visual intelligence.
        
        Returns analysis data that can inform action selection decisions.
        """
        frame = response_data.get('frame', [])
        if not frame or not isinstance(frame, list) or len(frame) == 0:
            return {}
        
        try:
            # Use the actual FrameAnalyzer API
            analysis_results = self.frame_analyzer.analyze_frame(frame, game_id)
            
            if not analysis_results:
                return {}
            
            # Extract useful information for action selection
            enhanced_analysis = {}
            
            # Movement detection - from FrameAnalyzer results
            if analysis_results.get('movement_detected', False):
                enhanced_analysis['movement_insight'] = {
                    'has_movement': True,
                    'movement_areas': analysis_results.get('frame_changes', []),
                    'movement_intensity': len(analysis_results.get('frame_changes', [])) / 100.0  # Normalize
                }
            
            # Agent position tracking - from FrameAnalyzer results
            agent_position = analysis_results.get('agent_position')
            if agent_position:
                enhanced_analysis['positions'] = [{'x': agent_position[0], 'y': agent_position[1], 'confidence': analysis_results.get('position_confidence', 0.0)}]
                enhanced_analysis['primary_target'] = {
                    'x': agent_position[0], 
                    'y': agent_position[1], 
                    'confidence': analysis_results.get('position_confidence', 0.0)
                }
            
            # Basic frame analysis - normalize frame to canonical numpy 2D array
            frame_array = None
            try:
                norm_arr, (gw, gh) = self._normalize_frame(frame)
                if norm_arr is not None:
                    # Ensure integer dtype for bincount/unique
                    try:
                        frame_array = norm_arr.astype(int)
                    except Exception:
                        frame_array = np.array(norm_arr, dtype=int)

                    # Safe flatten
                    flat = frame_array.flatten() if frame_array is not None else np.array([], dtype=int)
                    if flat.size == 0:
                        # Nothing to analyze beyond presence
                        enhanced_analysis['color_analysis'] = {'unique_colors': [], 'color_count': 0, 'dominant_color': 0}
                        enhanced_analysis['complexity'] = {'score': 0.0, 'is_simple': True, 'is_complex': False}
                    else:
                        # Unique colors
                        try:
                            unique_colors = np.unique(flat)
                        except Exception:
                            unique_colors = np.array([], dtype=int)

                        # Dominant color via bincount (guarded)
                        dominant = 0
                        try:
                            # bincount requires non-negative ints
                            if flat.dtype.kind not in ('i', 'u'):
                                flat_vals = flat.astype(int)
                            else:
                                flat_vals = flat
                            if flat_vals.size > 0:
                                counts = np.bincount(flat_vals)
                                if counts.size > 0:
                                    dominant = int(np.argmax(counts))
                        except Exception:
                            dominant = 0

                        color_info = {
                            'unique_colors': unique_colors.tolist() if unique_colors is not None else [],
                            'color_count': int(len(unique_colors)) if unique_colors is not None else 0,
                            'dominant_color': dominant
                        }
                        enhanced_analysis['color_analysis'] = color_info

                        # Pattern complexity analysis
                        total_elements = int(flat.size)
                        complexity_score = (len(unique_colors) / total_elements) if total_elements > 0 else 0.0
                        enhanced_analysis['complexity'] = {
                            'score': complexity_score,
                            'is_simple': complexity_score < 0.1,
                            'is_complex': complexity_score > 0.5
                        }

                        # Simple boundary detection (based on edge complexity)
                        if len(unique_colors) > 2:  # Has boundaries if more than background + one color
                            enhanced_analysis['boundary_analysis'] = {
                                'has_clear_boundaries': True,
                                'boundary_count': len(unique_colors) - 1,
                                'complexity_level': 'high' if complexity_score > 0.5 else 'medium' if complexity_score > 0.2 else 'low'
                            }
            except Exception as frame_error:
                print(f" Frame structure analysis failed: {frame_error}")
                frame_array = None
            
            # Store frame analysis for game loop usage
            if not hasattr(self, '_last_frame_analysis'):
                self._last_frame_analysis = {}
            self._last_frame_analysis[game_id] = enhanced_analysis
            
            logger.debug(f" Frame analysis for {game_id}: {len(enhanced_analysis)} analysis types completed")
            return enhanced_analysis
            
        except Exception as e:
            logger.warning(f"Frame analysis failed for {game_id}: {e}")
            return {}
    def _select_next_action(self, response_data: Dict[str, Any], game_id: str) -> Optional[int]:
        """Select appropriate action based on learned intelligence, available_actions from API response, and visual frame analysis."""
        # Ensure memory is properly initialized
        self._ensure_available_actions_memory()
        available = response_data.get('available_actions', [])
        
        if not available:
            logger.warning(f"No available actions for game {game_id}")
            return None
        
        # Perform frame analysis for visual intelligence
        frame_analysis = self._analyze_frame_for_action_selection(response_data, game_id)
        
        # Update available actions tracking
        self._update_available_actions(response_data, game_id)
        
        # Track available actions from RESET response
        current_available = available
        if not hasattr(self, '_last_available_actions'):
            self._last_available_actions = {}
        
        last_available = self._last_available_actions.get(game_id, [])
        # Safe comparison that handles arrays
        try:
            if current_available != last_available:
                print(f" Available Actions for {game_id}: {current_available}")
                self._last_available_actions[game_id] = current_available
        except ValueError as e:
            # Handle array comparison issues
            if "ambiguous" in str(e):
                # Convert to lists for comparison
                current_list = list(current_available) if hasattr(current_available, '__iter__') else current_available
                last_list = list(last_available) if hasattr(last_available, '__iter__') else last_available
                if current_list != last_list:
                    print(f" Available Actions for {game_id}: {current_available}")
                    self._last_available_actions[game_id] = current_available
            else:
                raise
        
        # Create context for simulation agent
        context = {
            'game_id': game_id, 
            'frame_analysis': frame_analysis,
            'response_data': response_data,
            'frame': response_data.get('frame') if response_data else None
        }
        
        # Use simulation-driven action selection if available, otherwise fallback to intelligent selection
        if self.simulation_agent and hasattr(self, 'simulation_agent') and self.simulation_agent is not None:
            try:
                # Use enhanced simulation intelligence for action selection
                selected_action, coordinates, reasoning = self.simulation_agent.generate_action_plan(
                    current_state=context,
                    available_actions=available,
                    frame_analysis=frame_analysis,
                    memory_patterns=None  # Would need to be passed from caller
                )
                
                # Store coordinates for later use
                if coordinates:
                    self._last_coordinates = coordinates
                
                logger.info(f"🧠 Simulation-driven action selection: {selected_action} {coordinates} - {reasoning}")
                
            except Exception as e:
                logger.warning(f"Simulation-driven action selection failed: {e}, falling back to intelligent selection")
                # Fallback to original intelligent action selection
                selected_action = self._select_intelligent_action_with_relevance(available, context)
        else:
            # Use intelligent action selection with frame analysis context
            selected_action = self._select_intelligent_action_with_relevance(available, context)
        
        # Add to action history
        if 'action_history' not in self.available_actions_memory:
            self.available_actions_memory['action_history'] = []
        self.available_actions_memory['action_history'].append(selected_action)
        
        logger.debug(f" Selected action {selected_action} from available {available} for {game_id} (with frame analysis)")
        return selected_action
    
    async def _send_enhanced_action(
        self,
        game_id: str,
        action_number: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        grid_width: int = 64,
        grid_height: int = 64,
        frame_analysis: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Send action with proper validation, coordinate optimization, effectiveness tracking, and rate limiting."""
        # Enhanced API key validation
        if not hasattr(self, 'api_key') or not self.api_key:
            error_msg = " CRITICAL: No API key found in _send_enhanced_action! Check environment variables and initialization."
            print(error_msg)
            logger.error(error_msg)
            # Try to reload API key as a fallback
            try:
                self.api_key = os.getenv('ARC_API_KEY')
                if self.api_key:
                    print(" Reloaded API key from environment")
                else:
                    return None
            except Exception as e:
                logger.error(f"Failed to reload API key: {e}")
                return None
        
        # Log API key status (first 4 and last 4 chars for security)
        key_display = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else "MISSING"
        logger.debug(f" API Key: {key_display} (length: {len(self.api_key) if self.api_key else 0})")
        
        # Enhanced session validation with recovery attempt
        if not hasattr(self, 'current_game_sessions') or not isinstance(self.current_game_sessions, dict):
            self.current_game_sessions = {}
            logger.warning("Initialized missing current_game_sessions dictionary")
            
        guid = self.current_game_sessions.get(game_id)
        if not guid:
            logger.warning(f"No active session found for game {game_id}. Current sessions: {list(self.current_game_sessions.keys())}")
            
            # Attempt to recover by starting a new session if we have a valid game_id
            if game_id and isinstance(game_id, str) and len(game_id) > 5:  # Basic validation
                logger.info(f"Attempting to start new session for game {game_id}")
                session_data = await self._start_game_session(game_id)
                if session_data and 'guid' in session_data:
                    guid = session_data['guid']
                    logger.info(f"Successfully started new session with GUID: {guid}")
                else:
                    logger.error(f"Failed to start new session for {game_id}")
                    return None
            else:
                logger.error(f"Invalid game_id: {game_id}")
                return None
        
        # Validate action number
        if action_number not in [1, 2, 3, 4, 5, 6, 7]:
            logger.error(f"Invalid action number: {action_number}")
            return None
        
        # Optimize coordinates for coordinate-based actions
        if action_number == 6:
            if x is None or y is None:
                # Use enhanced coordinate optimization with frame analysis
                if frame_analysis:
                    x, y = self._enhance_coordinate_selection_with_frame_analysis(
                        action_number, (grid_width, grid_height), game_id, frame_analysis
                    )
                    logger.info(f" Frame-enhanced coordinates for action {action_number}: ({x},{y})")
                else:
                    # Fallback to learned coordinate optimization
                    x, y = self._optimize_coordinates_for_action(action_number, (grid_width, grid_height), game_id)
                    logger.info(f" Optimized coordinates for action {action_number}: ({x},{y})")
            
            # Verify coordinates are within actual grid bounds
            if not self._verify_grid_bounds(x, y, grid_width, grid_height):
                logger.error(f"Coordinates ({x},{y}) out of bounds for grid {grid_width}x{grid_height}")
                return None
        
        # Apply rate limiting with timeout
        try:
            await asyncio.wait_for(self.rate_limiter.acquire(), timeout=ARC3_RATE_LIMIT['request_timeout'])
        except asyncio.TimeoutError:
            logger.error("Rate limiter acquisition timed out")
            return None
            
        # Prepare request with enhanced error handling
        try:
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": f"TabulaRasa/1.0 (ContinuousLearningLoop; {game_id})"
            }
            
            # Determine API endpoint with validation
            try:
                url = f"{ARC3_BASE_URL}/api/cmd/ACTION{action_number}"
                if not url.startswith(('http://', 'https://')):
                    raise ValueError(f"Invalid API URL: {url}")
            except Exception as e:
                logger.error(f"Error constructing API URL: {e}")
                self.rate_limiter.release()  # Important: Release the rate limiter on error
                return None
            
            # Build payload with validation
            try:
                payload = {
                    "game_id": str(game_id),  # Ensure string type
                    "guid": str(guid),        # Ensure string type
                    "metadata": {
                        "action_timestamp": time.time(),
                        "agent_version": "1.0.0"
                    }
                }
                
                # Add coordinates for ACTION6
                if action_number == 6 and x is not None and y is not None:
                    payload.update({
                        "x": int(x),
                        "y": int(y),
                        "grid_width": int(grid_width),
                        "grid_height": int(grid_height)
                    })
                    
                    # Add frame analysis metadata if available
                    if frame_analysis:
                        payload["metadata"].update({
                            "frame_analysis_available": True,
                            "targets_identified": len(frame_analysis.get('targets', [])),
                            "grid_dimensions": f"{grid_width}x{grid_height}"
                        })
                
                # Add reasoning for non-coordinate actions
                if action_number != 6:
                    # Add reasoning for actions 1-5,7 (following ARC-3 API pattern)
                    payload["reasoning"] = {
                        "policy": f"intelligent_selection_action_{action_number}",
                        "action_type": f"ACTION{action_number}",
                        "grid_size": f"{grid_width}x{grid_height}",
                        "intelligence_used": getattr(self, 'available_actions_memory', {}).get('current_game_id') == game_id,
                        "timestamp": time.time()
                    }
                
                # Execute the API request with enhanced error handling
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        response_text = await response.text()
                        
                        # Log response status and size for debugging
                        logger.debug(f"API Response - Status: {response.status}, Size: {len(response_text)} bytes")
                        
                        if response.status == 200:
                            try:
                                data = await response.json()
                                self.rate_limiter.handle_success_response()
                                
                                # Validate response structure
                                if not isinstance(data, dict):
                                    logger.error(f"Invalid response format: Expected dict, got {type(data).__name__}")
                                    return None
                                
                                # Track available actions changes with validation
                                current_available = data.get('available_actions', [])
                                if not isinstance(current_available, list):
                                    logger.warning(f"Invalid actions format in response: {current_available}")
                                    current_available = []
                                
                                # Log successful action execution
                                logger.info(f"Successfully executed ACTION{action_number} for {game_id}")
                                
                                # Process successful response
                                last_available = getattr(self, '_last_available_actions', {}).get(game_id, [])
                                
                                # Only show available_actions if they changed
                                try:
                                    if current_available != last_available:
                                        print(f" Available Actions Changed for {game_id}: {current_available}")
                                        # Store the new available actions
                                        if not hasattr(self, '_last_available_actions'):
                                            self._last_available_actions = {}
                                        self._last_available_actions[game_id] = current_available
                                except ValueError as e:
                                    # Handle array comparison issues
                                    if "ambiguous" in str(e):
                                        # Convert to lists for comparison
                                        current_list = list(current_available) if hasattr(current_available, '__iter__') else current_available
                                        last_list = list(last_available) if hasattr(last_available, '__iter__') else last_available
                                        if current_list != last_list:
                                            print(f" Available Actions Changed for {game_id}: {current_available}")
                                            # Store the new available actions
                                            if not hasattr(self, '_last_available_actions'):
                                                self._last_available_actions = {}
                                            self._last_available_actions[game_id] = current_available
                                    else:
                                        raise
                                
                                # Log frame data if available
                                if 'frame' in data and data['frame'] is not None:
                                    frame_data = data['frame']
                                    frame_size = f"{len(frame_data[0])}x{len(frame_data)}" if frame_data and len(frame_data) > 0 else "empty"
                                    logger.debug(f"Received frame data: {frame_size}")
                                
                                # ENHANCED: Log ACTION6 interaction in frame analyzer for hypothesis generation
                                if action_number == 6 and x is not None and y is not None and hasattr(self, 'frame_analyzer') and hasattr(self.frame_analyzer, 'log_action6_interaction'):
                                    try:
                                        # Get current game state for before/after analysis
                                        current_frame = data.get('frame', [])
                                        score = data.get('score', 0)
                                        available_actions = data.get('available_actions', [])
                                        
                                        # Validate frame data
                                        if not isinstance(current_frame, (list, np.ndarray)) or len(current_frame) == 0:
                                            logger.warning("Invalid or empty frame data received")
                                            # Try to get frame from alternative key
                                            current_frame = data.get('grid', [])
                                            if not isinstance(current_frame, (list, np.ndarray)) or len(current_frame) == 0:
                                                logger.warning("No valid frame data found in response")
                                                return data
                                        
                                        # Create target info from frame analysis if available
                                        target_info = {'object_id': f'coord_{x}_{y}', 'dominant_color': 'unknown'}
                                        if hasattr(self, '_last_frame_analysis') and self._last_frame_analysis:
                                            # Try to get color information from frame analysis
                                            analysis_data = self._last_frame_analysis.get(game_id, {})
                                            for target in analysis_data.get('targets', []):
                                                target_x, target_y = target.get('coordinate', (None, None))
                                                if target_x is not None and target_y is not None:
                                                    if abs(target_x - x) <= 1 and abs(target_y - y) <= 1:  # Close match
                                                        target_info = target
                                                        break
                                        
                                        # Create before/after states
                                        before_state = {
                                            'score': getattr(self, '_last_score', 0),
                                            'frame': getattr(self, '_last_frame', []),
                                            'available_actions': getattr(self, '_last_available_actions', {}).get(game_id, [])
                                        }
                                        after_state = {
                                            'score': score,
                                            'frame': current_frame,
                                            'available_actions': available_actions
                                        }
                                        
                                        # Calculate score change
                                        score_change = score - before_state['score']
                                        
                                        # Log the interaction
                                        interaction_id = self.frame_analyzer.log_action6_interaction(
                                            x=x, y=y,
                                            target_info=target_info,
                                            before_state=before_state,
                                            after_state=after_state,
                                            score_change=score_change,
                                            game_id=game_id
                                        )
                                        
                                        # ENHANCED: Record coordinate effectiveness for avoidance system
                                        if hasattr(self.frame_analyzer, '_record_coordinate_effectiveness'):
                                            api_success = response.status == 200 and data.get('state') not in ['GAME_OVER']
                                            context = f"api_success_{api_success}_score_{score_change}"
                                            self.frame_analyzer._record_coordinate_effectiveness(
                                                x, y, api_success, score_change, context
                                            )
                                        
                                        # Store current state for next comparison
                                        self._last_score = score
                                        self._last_frame = current_frame
                                        
                                        # ENHANCED: Check for frame changes and movement
                                        frame_changed = self._check_frame_changes(before_state, after_state)
                                        if frame_changed:
                                            logger.info(f"🎯 FRAME CHANGE DETECTED: {frame_changed['change_type']} - {frame_changed['description']}")
                                            
                                            # Update action effectiveness based on frame changes
                                            self._update_action_effectiveness(action_number, frame_changed, score_change)
                                            
                                            # Reset stagnation counter if we see movement
                                            if frame_changed.get('movement_detected', False):
                                                self._reset_stagnation_counter(game_id)
                                                logger.info(f"🔄 STAGNATION RESET: Movement detected, resetting counter for {game_id}")
                                        else:
                                            # No frame change - increment stagnation counter
                                            self._increment_stagnation_counter(game_id, action_number)
                                            logger.warning(f"⚠️ NO FRAME CHANGE: Action {action_number} produced no visual changes")
                                        
                                    except Exception as e:
                                        logger.error(f"Error in frame analysis logging: {e}")
                                
                                # Ensure we always return a dictionary
                                if not isinstance(data, dict):
                                    logger.error(f"API returned non-dict data: {type(data)} - {data}")
                                    return {
                                        'error': f'Invalid response format: {type(data)}',
                                        'state': 'ERROR',
                                        'score': 0,
                                        'available_actions': []
                                    }
                                
                                # 🧠 PATTERN LEARNING: Learn from action outcomes
                                self._learn_from_action_outcome(
                                    game_id, action_number, x, y, data, 
                                    before_state={'score': getattr(self, '_last_score', 0)},
                                    after_state={'score': data.get('score', 0)}
                                )
                                
                                return data
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse JSON response: {e}\nResponse: {response_text[:500]}")
                                return {
                                    'error': f'JSON decode error: {e}',
                                    'state': 'ERROR',
                                    'score': 0,
                                    'available_actions': []
                                }
                            except Exception as e:
                                logger.error(f"Error processing successful response: {e}")
                                return {
                                    'error': f'Response processing error: {e}',
                                    'state': 'ERROR',
                                    'score': 0,
                                    'available_actions': []
                                }
                            
                        elif response.status == 429:
                            # Handle rate limiting with exponential backoff
                            retry_after = int(response.headers.get('Retry-After', '5'))
                            logger.warning(f"Rate limited (429) - Retry after {retry_after}s")
                            await asyncio.sleep(retry_after)
                            return await self._send_enhanced_action(game_id, action_number, x, y, grid_width, grid_height, frame_analysis)
                            
                        elif 400 <= response.status < 500:
                            # Client errors (4xx)
                            error_msg = f"Client error ({response.status}): {response_text[:500]}"
                            logger.error(error_msg)
                            
                            # Special handling for GAME_NOT_STARTED_ERROR
                            if "GAME_NOT_STARTED_ERROR" in response_text:
                                logger.warning(f"Game {game_id} not started, attempting to reset...")
                                try:
                                    # Try to reset the game
                                    reset_result = await self._start_game_session(game_id)
                                    if reset_result and 'guid' in reset_result:
                                        logger.info(f"Successfully reset game {game_id}, retrying action...")
                                        # Update the session GUID
                                        self.current_game_sessions[game_id] = reset_result['guid']
                                        # Retry the action
                                        return await self._send_enhanced_action(game_id, action_number, x, y, grid_width, grid_height, frame_analysis)
                                    else:
                                        logger.error(f"Failed to reset game {game_id}")
                                except Exception as reset_error:
                                    logger.error(f"Error resetting game {game_id}: {reset_error}")
                            
                            # Special handling for 401 Unauthorized
                            elif response.status == 401:
                                logger.critical("Authentication failed - please check your API key")
                                
                            return {
                                'error': f'Client error ({response.status}): {response_text[:200]}',
                                'state': 'ERROR',
                                'score': 0,
                                'available_actions': []
                            }
                            
                        elif 500 <= response.status < 600:
                            # Server errors (5xx) - retry with backoff
                            max_retries = 3
                            retry_count = getattr(self, '_retry_count', 0)
                            
                            if retry_count < max_retries:
                                retry_delay = (2 ** retry_count) + random.uniform(0, 1)
                                logger.warning(f"Server error ({response.status}), retry {retry_count + 1}/{max_retries} in {retry_delay:.1f}s")
                                
                                self._retry_count = retry_count + 1
                                await asyncio.sleep(retry_delay)
                                return await self._send_enhanced_action(game_id, action_number, x, y, grid_width, grid_height, frame_analysis)
                            else:
                                logger.error(f"Max retries ({max_retries}) exceeded for server error")
                                return {
                                    'error': f'Max retries exceeded for server error ({response.status})',
                                    'state': 'ERROR',
                                    'score': 0,
                                    'available_actions': []
                                }
                            
                        else:
                            # Handle other status codes
                            logger.warning(f"Unexpected status code {response.status}: {response_text[:500]}")
                            return {
                                'error': f'Unexpected status code {response.status}',
                                'state': 'ERROR',
                                'score': 0,
                                'available_actions': []
                            }
                            
            except Exception as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "ssl" in error_msg.lower():
                    logger.warning(f"Network timeout/SSL error: {e}, will retry on next action")
                    return {
                        'error': f'Network timeout: {e}',
                        'state': 'RETRY',
                        'score': 0,
                        'available_actions': []
                    }
                else:
                    logger.error(f"Error building request payload: {e}")
                    return {
                        'error': f'Request payload error: {e}',
                        'state': 'ERROR',
                        'score': 0,
                        'available_actions': []
                    }
                
        except Exception as e:
            logger.error(f"Unexpected error in _send_enhanced_action: {e}")
            return {
                'error': f'Unexpected error: {e}',
                'state': 'ERROR',
                'score': 0,
                'available_actions': []
            }
            
        finally:
            # Always release the rate limiter
            if hasattr(self, 'rate_limiter') and hasattr(self.rate_limiter, 'release'):
                self.rate_limiter.release()
                
                if hasattr(self, 'frame_analyzer') and hasattr(self.frame_analyzer, 'mark_color_explored'):
                    # Check if exploration is active
                    exploration_active = getattr(self.frame_analyzer, 'exploration_phase', None) == 'active'
                    
                    # Get the color at clicked coordinate from the BEFORE state frame
                    # (after state might show changes from the click)
                    current_frame = before_state.get('frame', None) if hasattr(self, 'before_state') else None

                    # Normalize the frame to a canonical 2D numpy array for safe indexing
                    norm_arr, (fw, fh) = self._normalize_frame(current_frame)
                    frame_valid = False
                    if norm_arr is not None:
                        # norm_arr shape is (height, width)
                        height, width = norm_arr.shape
                        frame_valid = (0 <= y < height and 0 <= x < width)

                    if frame_valid:
                        try:
                            # Handle nested list format by using normalized numpy array
                            raw_color = int(norm_arr[y, x])
                            
                            #  CRITICAL FIX: Ensure clicked_color is always an integer
                            if isinstance(raw_color, list):
                                # If it's a list, take the first element or convert appropriately
                                clicked_color = raw_color[0] if raw_color else 0
                            elif isinstance(raw_color, (int, float)):
                                clicked_color = int(raw_color)
                            else:
                                # Fallback: convert to string then to int, or use 0
                                try:
                                    clicked_color = int(str(raw_color))
                                except (ValueError, TypeError):
                                    clicked_color = 0
                            
                            print(f" COLOR EXTRACTION: raw_color={raw_color} (type={type(raw_color)}) -> clicked_color={clicked_color} (type={type(clicked_color)})")
                            
                            # Calculate frame changes for exploration record
                            frame_changes = {}
                            if hasattr(self.frame_analyzer, '_analyze_frame_changes'):
                                try:
                                    frame_changes = self.frame_analyzer._analyze_frame_changes(
                                        before_state.get('frame'), after_state.get('frame')
                                    )
                                except Exception as e:
                                    frame_changes = {'changes_detected': False, 'error': str(e)}
                            
                            # Mark this color as explored
                            print(f" CALLING mark_color_explored for color {clicked_color} at ({x},{y})")
                            self.frame_analyzer.mark_color_explored(
                                color=clicked_color,
                                coordinate=(x, y),
                                success=api_success,
                                score_change=score_change,
                                frame_changes=frame_changes
                            )
                            
                            print(f" EXPLORATION: Color {clicked_color} tested at ({x},{y}) - {' Effective' if api_success or score_change != 0 else ' No effect'}")
                        except Exception as e:
                            print(f" EXPLORATION ERROR: Failed to mark color explored: {e}")
                    else:
                        #  CRITICAL FIX: Better error reporting for frame validation failures
                        frame_height = len(current_frame) if current_frame else 0
                    # Use canonical dimensions when possible
                    if current_frame is None:
                        frame_width = 0
                    else:
                        try:
                            # current_frame may be a nested list or numpy array
                            if isinstance(current_frame, np.ndarray):
                                frame_width = int(current_frame.shape[1]) if current_frame.ndim >= 2 else 0
                            elif isinstance(current_frame, list) and len(current_frame) > 0:
                                # safely derive width using normalization
                                arr_local, (w_local, h_local) = self._normalize_frame(current_frame)
                                frame_width = w_local if arr_local is not None else 0
                            else:
                                frame_width = 0
                        except Exception:
                            frame_width = 0
                        print(f" EXPLORATION: Invalid coordinates ({x},{y}) for frame analysis")
                        print(f"   Frame dimensions: {frame_height}x{frame_width}")
                        print(f"   Frame valid: {current_frame is not None}")
                        print(f"   Coordinates in bounds: x={0 <= x < frame_width if frame_width > 0 else False}, y={0 <= y < frame_height if frame_height > 0 else False}")
                    
                    if not exploration_active:
                        print(f" EXPLORATION: Phase not active (exploration_phase={getattr(self.frame_analyzer, 'exploration_phase', 'MISSING')})")
                    else:
                        print(f" EXPLORATION: Missing mark_color_explored method")
                    
                    print(f" ACTION6 interaction logged: {interaction_id} at ({x},{y}) with score change {score_change:+}")
                    
                    # Store current state for next comparison
                    self._last_score = score
                    self._last_frame = current_frame
                    
                    # Update current position tracking for future directional movement
                    self._current_game_x = x
                    self._current_game_y = y

                    # Record action trace (append-only) with minimal fields
                    try:
                        trace = {
                            'ts': time.time(),
                            'game_id': game_id,
                            'guid': guid,
                            'action_number': action_number,
                            'x': x if action_number == 6 else None,
                            'y': y if action_number == 6 else None,
                            'response_score': data.get('score') if isinstance(data, dict) else None
                        }
                        log_action_trace(trace)
                    except Exception:
                        pass

                    response_text = await response.text()
                    
                    # Log response status and size for debugging
                    logger.debug(f"API Response - Status: {response.status}, Size: {len(response_text)} bytes")
                    
                    try:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                self.rate_limiter.handle_success_response()
                                
                                # Validate response structure
                                if not isinstance(data, dict):
                                    logger.error(f"Invalid response format: Expected dict, got {type(data).__name__}")
                                    return None
                                
                                # Track available actions changes with validation
                                current_available = data.get('available_actions', [])
                                if not isinstance(current_available, list):
                                    logger.warning(f"Invalid actions format in response: {current_available}")
                                    current_available = []
                                
                                # Log successful action execution
                                logger.info(f"Successfully executed ACTION{action_number} for {game_id}")
                                
                                # Process successful response
                                last_available = getattr(self, '_last_available_actions', {}).get(game_id, [])
                                
                                # Only show available_actions if they changed
                                try:
                                    if current_available != last_available:
                                        print(f" Available Actions Changed for {game_id}: {current_available}")
                                        # Store the new available actions
                                        if not hasattr(self, '_last_available_actions'):
                                            self._last_available_actions = {}
                                        self._last_available_actions[game_id] = current_available
                                except ValueError as e:
                                    # Handle array comparison issues
                                    if "ambiguous" in str(e):
                                        # Convert to lists for comparison
                                        current_list = list(current_available) if hasattr(current_available, '__iter__') else current_available
                                        last_list = list(last_available) if hasattr(last_available, '__iter__') else last_available
                                        if current_list != last_list:
                                            print(f" Available Actions Changed for {game_id}: {current_available}")
                                            # Store the new available actions
                                            if not hasattr(self, '_last_available_actions'):
                                                self._last_available_actions = {}
                                            self._last_available_actions[game_id] = current_available
                                    else:
                                        raise
                                
                                # Log frame data if available
                                if 'frame' in data and data['frame'] is not None:
                                    frame_data = data['frame']
                                    frame_size = f"{len(frame_data[0])}x{len(frame_data)}" if frame_data and len(frame_data) > 0 else "empty"
                                    logger.debug(f"Received frame data: {frame_size}")
                                
                                return data
                                
                            except Exception as e:
                                logger.error(f"Error processing successful response: {e}")
                                return None
                                
                        elif response.status == 429:
                                # Handle rate limit exceeded - this is the critical case for actions!
                                self.rate_limiter.handle_429_response()
                                error_text = await response.text()
                                print(f" ACTION {action_number} rate limited (429) - backing off {self.rate_limiter.backoff_delay:.1f}s")
                                logger.warning(f"Rate limit hit on ACTION{action_number} for {game_id}: {error_text}")
                                
                                # Wait for backoff then retry
                                await asyncio.sleep(self.rate_limiter.backoff_delay)
                                return await self._send_enhanced_action(game_id, action_number, x, y, grid_width, grid_height)
                        else:
                            self.rate_limiter.handle_success_response()  # Not a rate limit issue
                            error_text = await response.text()
                            print(f" ACTION {action_number} FAILED: {response.status}")
                            logger.warning(f"Action {action_number} failed for {game_id}: {response.status} - {error_text}")
                            
                            # Record failure if coordinates were used
                            if action_number == 6 and x is not None and y is not None:
                                self._record_coordinate_effectiveness(action_number, x, y, False)
                            
                            return None
                            
                    except Exception as e:
                        logger.error(f"Error processing response: {e}")
                        return None

    async def _validate_api_connection(self) -> bool:
        """Validate ARC-AGI-3 API connection and game availability."""
        try:
            import requests
            response = requests.get("https://three.arcprize.org/api/games", 
                                headers={"X-API-Key": self.api_key}, timeout=30)
            if response.status_code == 200:
                games = response.json()
                if isinstance(games, list) and len(games) > 0:
                    print(f" API Connection OK: {len(games)} games available")
                    return True
                else:
                    print(f" API returned empty game list: {games}")
                    return False
            else:
                print(f" API Error: {response.status_code}")
                return False
        except Exception as e:
            print(f" API Connection Failed: {e}")
            return False

    async def _train_on_game(
        self,
        game_id: str,
        max_mastery_sessions: int,  # Renamed from max_episodes
        target_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Train the agent on a specific game using REAL ARC API calls with enhanced game state handling."""
        game_results = {
            'game_id': game_id,
            'episodes': [],
            'performance_metrics': {},
            'learning_progression': [],
            'patterns_discovered': [],
            'final_performance': {},
            'scorecard_urls': [],
            'grid_dimensions': (64, 64)  # Will be updated dynamically
        }
        
        # Validate ARC-AGI-3-Agents path exists (optional for standalone mode)
        self.standalone_mode = False
        if not self.arc_agents_path.exists():
            logger.warning(f"ARC-AGI-3-Agents path does not exist: {self.arc_agents_path}. Running in standalone mode.")
            self.standalone_mode = True
        elif not (self.arc_agents_path / "main.py").exists():
            logger.warning(f"main.py not found in ARC-AGI-3-Agents: {self.arc_agents_path}. Running in standalone mode.")
            self.standalone_mode = True
        
        print(f"Starting ENHANCED ARC-3 training on game: {game_id}")
        
        # Display current win rate-based energy status at training start
        current_win_rate = self._calculate_current_win_rate()
        energy_params = self._calculate_win_rate_adaptive_energy_parameters()
        skill_phase = energy_params['skill_phase']
        actions_until_sleep = int(self.current_energy / energy_params['action_energy_cost'])
        
        print(f" Training Start Status: Win Rate {current_win_rate:.1%} | Phase: {skill_phase} | Energy: {self.current_energy:.1f}% | ~{actions_until_sleep} actions until sleep")
        print(f" Current Energy Profile: {energy_params['action_energy_cost']:.1f} energy/action | Sleep threshold: {energy_params['sleep_trigger_threshold']:.0f}%")
        
        print(f"Using API Key: {self.api_key[:8]}...{self.api_key[-4:]}")
        print(f"ARC-AGI-3-Agents path: {self.arc_agents_path}")
        
        session_count = 1  # Renamed from episode_count 
        consecutive_failures = 0
        best_score = 0
        
        while session_count < max_mastery_sessions:  # Renamed from max_episodes
            try:
                print(f"Mastery Session {session_count + 1}/{max_mastery_sessions} for {game_id}")  # Updated naming
                
                # Run ENHANCED ARC-3 mastery session with proper state checking
                session_result = await self._run_real_arc_mastery_session_enhanced(game_id, session_count)  # Renamed method
                
                if session_result and 'error' not in session_result:  # Updated variable name
                    game_results['episodes'].append(session_result)  # Updated variable name
                    session_count += 1  # Updated variable name
                    
                    # Update grid dimensions if detected
                    if 'grid_dimensions' in session_result:  # Updated variable name
                        game_results['grid_dimensions'] = session_result['grid_dimensions']  # Updated variable name
                        print(f"Grid size detected: {session_result['grid_dimensions'][0]}x{session_result['grid_dimensions'][1]}")  # Updated variable name
                    
                    # Track performance
                    current_score = session_result.get('final_score', 0)  # Updated variable name
                    success = session_result.get('success', False)  # Updated variable name
                    game_state = session_result.get('game_state', 'NOT_FINISHED')  # Updated variable name
                    scorecard_url = session_result.get('scorecard_url')  # Updated variable name
                    
                    if scorecard_url:
                        game_results['scorecard_urls'].append(scorecard_url)
                    
                    if success:
                        consecutive_failures = 0
                        if current_score > best_score:
                            best_score = current_score
                            print(f"New best score for {game_id}: {best_score}")
                            
                        #  PRESERVE WINNING MEMORIES - Mark as high-priority, hard to delete
                        self._preserve_winning_memories(session_result, current_score, game_id)
                    
                    #  CHECK FOR FULL GAME WIN - Highest priority achievement
                    if session_result.get('full_game_win', False):
                        print(f"🏆 FULL GAME WIN! {game_id} completed entirely - ULTIMATE SUCCESS!")
                        success = True
                        game_state = 'FULL_GAME_WIN'
                        # Full game win gets maximum priority
                        self._preserve_ultimate_win_memories(session_result, current_score, game_id)
                        
                        # 🧠 GOVERNOR ANALYSIS - Full game win analysis
                        if self.governor:
                            governor_analysis = self.governor.analyze_full_game_win(session_result, game_id)
                            if governor_analysis:
                                print(f"🧠 GOVERNOR ULTIMATE WIN ANALYSIS: {governor_analysis['win_analysis']['win_value']} value detected")
                                print(f"   Memory Priority: {governor_analysis['win_analysis']['memory_priority']}")
                                print(f"   Strategy Analysis: {governor_analysis['win_analysis']['should_analyze_strategy']}")
                    
                    #  CHECK FOR LEVEL PROGRESSION - Only preserve NEW breakthroughs
                    elif session_result.get('level_progressed', False):
                        new_level = session_result.get('current_level', 1)
                        previous_best = self.game_level_records.get(game_id, {}).get('highest_level', 0)
                        
                        if new_level > previous_best:
                            print(f"🎯 LEVEL BREAKTHROUGH! {game_id} advanced from level {previous_best} to {new_level}")
                            # This is a real breakthrough - preserve with hierarchical priority
                            self._preserve_breakthrough_memories(session_result, current_score, game_id, new_level, previous_best)
                            # 🎯 LEVEL COMPLETION = SUCCESS! Override any failure state
                            success = True
                            game_state = 'LEVEL_WIN'
                            print(f"🎯 LEVEL WIN OVERRIDE: Marking as LEVEL WIN due to level {new_level} completion!")
                            
                            # 🧠 GOVERNOR ANALYSIS - Analyze level completion win
                            if self.governor:
                                governor_analysis = self.governor.analyze_level_completion_win(session_result, game_id)
                                if governor_analysis:
                                    print(f"🧠 GOVERNOR LEVEL WIN ANALYSIS: {governor_analysis['win_analysis']['win_value']} value win detected")
                                    print(f"   Memory Priority: {governor_analysis['win_analysis']['memory_priority']}")
                                    print(f"   Strategy Analysis: {governor_analysis['win_analysis']['should_analyze_strategy']}")
                        else:
                            print(f" Level {new_level} maintained on {game_id} (no new breakthrough)")
                            # Even maintaining level is a form of success
                            success = True
                            game_state = 'LEVEL_WIN'
                            
                            # 🧠 GOVERNOR ANALYSIS - Even maintaining level is a win
                            if self.governor:
                                governor_analysis = self.governor.analyze_level_completion_win(session_result, game_id)
                                if governor_analysis:
                                    print(f"🧠 GOVERNOR LEVEL WIN ANALYSIS: Level maintenance win detected")
                    else:
                        consecutive_failures += 1
                        
                    # Enhanced win type display
                    win_display = "WIN" if success else "LOSS"
                    if success:
                        if game_state == 'FULL_GAME_WIN':
                            win_display = "🏆 FULL WIN"
                        elif game_state == 'LEVEL_WIN':
                            win_display = "🎯 LEVEL WIN"
                        elif game_state == 'WIN':
                            win_display = "✅ WIN"
                    
                    print(f"Mastery Session {session_count}: {win_display} | Score: {current_score} | State: {game_state}")
                    
                    # Track learning progress for boredom detection
                    effectiveness = self._calculate_session_effectiveness(session_result, game_results)  # Updated method name
                    learning_progress = self._convert_effectiveness_to_lp(effectiveness)
                    
                    # Store LP history entry with ENHANCED CONTINUOUS LEARNING METRICS
                    lp_entry = {
                        'session': session_count,  # Updated key name
                        'game_id': game_id,
                        'learning_progress': learning_progress,
                        'effectiveness': effectiveness,
                        'score': current_score,
                        'success': success,
                        'timestamp': time.time(),
                        # CONTINUOUS LEARNING ENHANCEMENTS
                        'actions_taken': session_result.get('actions_taken', 0),  # Updated variable name
                        'action_sequences': session_result.get('action_sequences', {}),  # Updated variable name
                        'mid_game_consolidations': session_result.get('mid_game_consolidations', 0),  # Updated variable name
                        'success_weighted_memories': session_result.get('success_weighted_memories', 0),  # Updated variable name
                        'continuous_learning_metrics': session_result.get('continuous_learning_metrics', {}),  # Updated variable name
                        'available_actions_used': len(set(getattr(self.available_actions_memory, 'action_history', []))),
                        'strategy_switches': self.training_state.get('strategy_switches', 0),
                        'contrarian_mode': session_result.get('contrarian_mode', False),  # Get from session result
                        'contrarian_confidence': session_result.get('contrarian_confidence', 0.0),  # Get from session result
                        'experiment_mode_triggered': getattr(self.available_actions_memory, 'experiment_mode', False)
                    }
                    
                    if 'lp_history' not in self.training_state:
                        self.training_state['lp_history'] = []
                    self.training_state['lp_history'].append(lp_entry)
                    
                    # Keep only last 20 episodes in history
                    if len(self.training_state['lp_history']) > 20:
                        self.training_state['lp_history'] = self.training_state['lp_history'][-20:]
                    
                    # Update episode count
                    self.training_state['episode_count'] = session_count
                    
                    # Check for boredom and handle curriculum advancement
                    boredom_results = self._check_and_handle_boredom(session_count)
                    if boredom_results['boredom_detected']:
                        print(f" Boredom detected: {boredom_results['reason']}")
                        if boredom_results['curriculum_advanced']:
                            print(f" Curriculum complexity advanced to level {boredom_results['new_complexity']}")
                    
                    # Integrate goal invention system - discover emergent goals from patterns
                    goal_results = self._process_emergent_goals(session_result, game_results, learning_progress)
                    if goal_results['new_goals_discovered']:
                        print(f" Discovered {len(goal_results['new_goals'])} new emergent goals")
                        for goal in goal_results['new_goals']:
                            print(f"   New goal: {goal.description} (priority: {goal.priority:.2f})")
                    
                    # CRITICAL: Enhanced terminal state handling with retry logic
                    if game_state in ['WIN', 'GAME_OVER']:
                        print(f" Game {game_id} reached terminal state: {game_state}")
                        
                        # If it's GAME_OVER and we haven't tried many sessions, try contrarian strategy
                        if game_state == 'GAME_OVER' and session_count < 2 and current_score < 10:
                            print(f" Early GAME_OVER detected - activating contrarian strategy for retry")
                            self.contrarian_strategy_active = True
                            # Don't break, let it try one more session with different strategy
                        else:
                            break
                    
                    # INTELLIGENT RESET LOGIC: Check if we should reset due to frame stagnation
                    if self._should_reset_due_to_stagnation(session_result, game_id):
                        print(f" FRAME STAGNATION DETECTED: Resetting level due to lack of new visual information")
                        # Perform level reset instead of starting new game
                        reset_result = await self._start_game_session(game_id, self.current_game_sessions.get(game_id))
                        if reset_result and 'error' not in reset_result:
                            print(f" Level reset successful - continuing with fresh frame data")
                            # Continue with the same session count
                        else:
                            print(f" Level reset failed - continuing with current session")
                    
                    # Check if we should continue based on performance
                    if self._should_stop_training(game_results, target_performance):
                        print(f"Target performance reached for {game_id}")
                        break
                        
                    # Enhanced delay between episodes for rate limit compliance
                    # ARC-3 games involve many API calls, so be conservative
                    episode_delay = 3.0  # Increased from 2.0 to 3.0 seconds
                    print(f"⏸ Rate limit compliance: waiting {episode_delay}s between episodes")
                    await asyncio.sleep(episode_delay)
                        
                else:
                    # Handle case where session_result might be a string instead of dict
                    if isinstance(session_result, str):
                        error_msg = session_result
                    elif isinstance(session_result, dict):
                        error_msg = session_result.get('error', 'Unknown error')
                    else:
                        error_msg = f"Unexpected result type: {type(session_result)}"
                    
                    print(f"Session {session_count + 1} failed: {error_msg}")
                    consecutive_failures += 1
                    
                    # Stop if too many consecutive API failures
                    if consecutive_failures >= 5:
                        print(f"Stopping training for {game_id} after 5 consecutive API failures")
                        break
                    
                    # Enhanced backoff for failures to respect rate limits
                    failure_delay = min(10.0, 5.0 + (consecutive_failures * 2.0))  # Progressive backoff
                    print(f"⏸ Failure backoff: waiting {failure_delay}s after {consecutive_failures} failures")
                    await asyncio.sleep(failure_delay)
                    
            except Exception as e:
                logger.error(f"Error in session {session_count + 1} for {game_id}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print(f"Stopping training for {game_id} due to repeated errors")
                    break
                # Enhanced error backoff to prevent rapid retry cycles that could hit rate limits
                error_delay = min(15.0, 5.0 + (consecutive_failures * 3.0))  # Even longer backoff for errors
                print(f"⏸ Error backoff: waiting {error_delay}s after error #{consecutive_failures}")
                await asyncio.sleep(error_delay)
                
        # Calculate final performance metrics
        game_results['performance_metrics'] = self._calculate_game_performance(game_results)
        game_results['final_performance'] = {
            'episodes_played': len(game_results['episodes']),
            'best_score': best_score,
            'win_rate': sum(1 for ep in game_results['episodes'] if ep.get('success', False)) / max(1, len(game_results['episodes'])),
            'scorecard_urls_generated': len(game_results['scorecard_urls']),
            'final_grid_size': f"{game_results['grid_dimensions'][0]}x{game_results['grid_dimensions'][1]}"
        }
        
        # Display scorecard URLs
        if game_results['scorecard_urls']:
            print(f"\nARC-3 Scorecards Generated for {game_id}:")
            for i, url in enumerate(game_results['scorecard_urls'], 1):
                print(f"   {i}. {url}")
        else:
            print(f"\nNo scorecard URLs generated for {game_id}")
        
        return game_results

    def _calculate_frame_analysis_bonus_action6(self, frame_analysis: Dict[str, Any]) -> float:
        """Calculate frame analysis bonus specifically for ACTION 6 coordinate selection."""
        bonus = 0.0
        
        # Bonus for clear target positions detected
        if 'positions' in frame_analysis and frame_analysis['positions']:
            bonus += 0.3  # Significant bonus for detected targets
        
        # Bonus for clear movement patterns (suggests dynamic environment good for ACTION 6)
        movement_info = frame_analysis.get('movement_insight', {})
        if movement_info.get('has_movement', False):
            movement_intensity = movement_info.get('movement_intensity', 0)
            bonus += min(0.2, movement_intensity * 0.5)  # Up to 20% bonus based on movement
        
        # Bonus for clear boundaries (helps coordinate selection)
        boundary_info = frame_analysis.get('boundary_analysis', {})
        if boundary_info.get('has_clear_boundaries', False):
            bonus += 0.1
        
        # Penalty for high complexity (ACTION 6 might be less effective in complex scenarios)
        complexity = frame_analysis.get('complexity', {})
        if complexity.get('is_complex', False):
            bonus -= 0.1  # Reduce ACTION 6 attractiveness in complex scenarios
        
        return max(0.0, min(0.5, bonus))  # Cap bonus between 0 and 50%
    
    def _calculate_frame_analysis_multiplier(self, action: int, frame_analysis: Dict[str, Any]) -> float:
        """Calculate frame analysis multiplier for non-ACTION 6 actions."""
        multiplier = 1.0
        
        # Different actions benefit from different visual patterns
        if action in [1, 2, 3, 4, 5, 7]:
            # General principles for reasoning-based actions
            
            # Simple patterns might benefit reasoning actions (clearer to analyze)
            complexity = frame_analysis.get('complexity', {})
            if complexity.get('is_simple', False):
                multiplier *= 1.1  # 10% boost for simple patterns
            elif complexity.get('is_complex', False):
                multiplier *= 0.95  # 5% penalty for overly complex patterns
            
            # Movement detection helps understand game dynamics
            movement_info = frame_analysis.get('movement_insight', {})
            if movement_info.get('has_movement', False):
                # Some movement suggests active game state, good for reasoning actions
                multiplier *= 1.05  # 5% boost for active environments
            
            # Color diversity might indicate more complex reasoning opportunities
            color_info = frame_analysis.get('color_analysis', {})
            if color_info:
                color_count = color_info.get('color_count', 1)
                if color_count > 3:  # Rich color environment
                    multiplier *= 1.05  # 5% boost for rich visual environments
                elif color_count == 1:  # Very simple
                    multiplier *= 0.98  # 2% penalty for overly simple environments
        
        return max(0.8, min(1.3, multiplier))  # Cap multiplier between 80% and 130%
    
    def _enhance_coordinate_selection_with_frame_analysis(self, action_number: int, grid_dimensions: Tuple[int, int], game_id: str, frame_analysis: Dict[str, Any]) -> Tuple[int, int]:
        """
         ADVANCED VISUAL-INTERACTIVE ACTION6 TARGETING SYSTEM
        
        NEW PARADIGM: ACTION6(x,y) = "touch/interact with object at (x,y)" 
        NOT movement - it's a universal targeting system for touching visual elements.
        
        The grid is treated like a touchscreen. ACTION6 touches whatever is at that pixel.
        """
        grid_width, grid_height = grid_dimensions
        
        if action_number != 6:
            # For non-ACTION6, use existing logic
            return self._optimize_coordinates_for_action(action_number, grid_dimensions, game_id)
        
        #  ADVANCED VISUAL TARGETING for ACTION6
        print(f" VISUAL-INTERACTIVE ACTION6 TARGETING - Analyzing frame for touchable objects...")
        
        try:
            # Get current frame from the game state if available
            current_frame = None
            if hasattr(self, 'current_frame_data'):
                current_frame = getattr(self, 'current_frame_data', None)
            
            if current_frame is not None:
                # Use advanced frame analysis for ACTION6 targeting
                targeting_analysis = self.frame_analyzer.analyze_frame_for_action6_targets(current_frame, game_id)

                if targeting_analysis and targeting_analysis.get('recommended_action6_coord'):
                    target_x, target_y = targeting_analysis['recommended_action6_coord']
                    reason = targeting_analysis['targeting_reason']
                    confidence = targeting_analysis['confidence']
                    
                    # Ensure coordinates are within bounds
                    target_x = max(0, min(grid_width - 1, int(target_x)))
                    target_y = max(0, min(grid_height - 1, int(target_y)))
                    
                    print(f" VISUAL TARGET SELECTED: ({target_x},{target_y}) - {reason} (confidence: {confidence:.2f})")
                    print(f"    Touching/interacting with visual element at pixel ({target_x},{target_y})")
                    
                    # ENHANCED: Show movement tracking information
                    if hasattr(self.frame_analyzer, 'get_movement_analysis'):
                        try:
                            movement_info = self.frame_analyzer.get_movement_analysis()
                            if movement_info['tracked_objects'] > 0:
                                print(f"    Objects tracked: {movement_info['tracked_objects']} | Moving: {movement_info['moving_objects']} | Static: {movement_info['static_objects']}")
                            
                            # Show avoidance info if coordinate was previously tried
                            coord_key = (target_x, target_y)
                            if hasattr(self.frame_analyzer, 'tried_coordinates') and coord_key in self.frame_analyzer.tried_coordinates:
                                try_count = self.frame_analyzer.coordinate_results.get(coord_key, {}).get('try_count', 0)
                                success_count = self.frame_analyzer.coordinate_results.get(coord_key, {}).get('success_count', 0)
                                print(f"    Coordinate history: {try_count} attempts, {success_count} successes")
                        except Exception as e:
                            pass
                    
                    # Log target type for learning
                    if len(targeting_analysis['interactive_targets']) > 0:
                        target_types = [t['type'] for t in targeting_analysis['interactive_targets'][:3]]
                        print(f"    Target types found: {target_types}")
                    
                    return target_x, target_y
                else:
                    print(" No clear visual targets detected - using systematic exploration...")
            else:
                print(" No frame data available for visual analysis")
                
        except Exception as e:
            print(f" Visual targeting analysis failed: {e}")
        
        #  FALLBACK: SYSTEMATIC EXPLORATION MODE
        # When no clear visual targets, use intelligent exploration pattern
        print(" EXPLORATORY MODE: Systematic tapping pattern for discovery...")
        
        # Use enhanced coordinate intelligence if available for exploration
        if hasattr(self, 'enhanced_coordinate_intelligence') and hasattr(self, 'use_enhanced_coordinate_selection'):
            if self.use_enhanced_coordinate_selection:
                try:
                    exploration_coord = self.enhanced_coordinate_intelligence.get_intelligent_coordinates(
                        action_number, grid_dimensions, game_id
                    )
                    print(f" INTELLIGENT EXPLORATION: Tapping at ({exploration_coord[0]},{exploration_coord[1]})")
                    return exploration_coord
                except Exception as e:
                    print(f" Enhanced exploration failed: {e}")
        
        # Final fallback: strategic grid exploration
        explore_x, explore_y = self._generate_exploration_coordinates(grid_dimensions, game_id)
        print(f" SYSTEMATIC EXPLORATION: Tapping at ({explore_x},{explore_y}) to discover interactions")
        
        return explore_x, explore_y
    
    def _generate_exploration_coordinates(self, grid_dimensions: Tuple[int, int], game_id: str) -> Tuple[int, int]:
        """ ENHANCED: Generate systematic exploration coordinates with stuck coordinate avoidance."""
        grid_width, grid_height = grid_dimensions
        
        #  CRITICAL FIX: Integrate stuck coordinate avoidance from frame analyzer
        # Check if frame analyzer has coordinate avoidance data
        if hasattr(self.frame_analyzer, 'should_avoid_coordinate'):
            print(f" Using coordinate avoidance system for exploration")
            
            # Try to get coordinates that avoid stuck areas
            max_attempts = 20
            attempt = 0
            
            while attempt < max_attempts:
                # Generate candidate coordinates using various strategies
                if attempt < 4:
                    # Try corners first (often contain important elements)
                    corners = [(5, 5), (grid_width-6, 5), (5, grid_height-6), (grid_width-6, grid_height-6)]
                    if attempt < len(corners):
                        candidate_x, candidate_y = corners[attempt]
                    else:
                        candidate_x, candidate_y = grid_width // 2, grid_height // 2
                elif attempt < 8:
                    # Try edges (often contain controls)
                    edges = [(grid_width//4, 2), (grid_width*3//4, 2), 
                            (2, grid_height//2), (grid_width-3, grid_height//2)]
                    edge_idx = (attempt - 4) % len(edges)
                    candidate_x, candidate_y = edges[edge_idx]
                else:
                    # Random exploration avoiding stuck areas
                    import random
                    candidate_x = random.randint(1, grid_width - 2)
                    candidate_y = random.randint(1, grid_height - 2)
                
                # Check if this coordinate should be avoided
                if not self.frame_analyzer.should_avoid_coordinate(candidate_x, candidate_y):
                    print(f" Selected non-stuck coordinate: ({candidate_x},{candidate_y}) after {attempt + 1} attempts")
                    return (candidate_x, candidate_y)
                else:
                    print(f" Avoiding stuck coordinate: ({candidate_x},{candidate_y}) - attempt {attempt + 1}")
                
                attempt += 1
            
            # If all attempts failed, use emergency diversification
            if hasattr(self.frame_analyzer, 'get_emergency_diversification_target'):
                emergency_coord = self.frame_analyzer.get_emergency_diversification_target(
                    [], grid_dimensions  # Empty frame for now, focus on coordinate avoidance
                )
                print(f" Using emergency diversification target: {emergency_coord}")
                return emergency_coord
        
        # Fallback to original systematic exploration if no avoidance system available
        print(f" No coordinate avoidance available, using basic systematic exploration")
        
        # Get exploration history for this game
        if not hasattr(self, 'action6_exploration_history'):
            self.action6_exploration_history = {}
        
        if game_id not in self.action6_exploration_history:
            self.action6_exploration_history[game_id] = {
                'exploration_phase': 'corners',
                'attempts': 0,
                'last_coordinates': []
            }
        
        exploration_data = self.action6_exploration_history[game_id]
        phase = exploration_data['exploration_phase']
        attempts = exploration_data['attempts']
        
        # Systematic exploration phases
        if phase == 'corners' and attempts < 8:
            # Explore corners and near-corners more thoroughly
            corners = [
                (2, 2), (grid_width-3, 2), (2, grid_height-3), (grid_width-3, grid_height-3),  # Main corners
                (5, 5), (grid_width-6, 5), (5, grid_height-6), (grid_width-6, grid_height-6)   # Near corners
            ]
            coord = corners[attempts]
            exploration_data['attempts'] += 1
            
            if attempts >= 7:
                exploration_data['exploration_phase'] = 'boundaries'
                exploration_data['attempts'] = 0
                
            return coord
            
        elif phase == 'boundaries' and attempts < 16:
            # Explore grid boundaries systematically
            boundaries = [
                # Top edge
                (grid_width//4, 1), (grid_width//2, 1), (grid_width*3//4, 1),
                # Right edge  
                (grid_width-2, grid_height//4), (grid_width-2, grid_height//2), (grid_width-2, grid_height*3//4),
                # Bottom edge
                (grid_width//4, grid_height-2), (grid_width//2, grid_height-2), (grid_width*3//4, grid_height-2),
                # Left edge
                (1, grid_height//4), (1, grid_height//2), (1, grid_height*3//4),
                # Additional boundary points
                (grid_width//8, 1), (grid_width*7//8, 1), (1, grid_height//8), (1, grid_height*7//8)
            ]
            coord = boundaries[attempts]
            exploration_data['attempts'] += 1
            
            if attempts >= 15:
                exploration_data['exploration_phase'] = 'center'
                exploration_data['attempts'] = 0
                
            return coord
            
        elif phase == 'center' and attempts < 3:
            # Explore center region
            center_x, center_y = grid_width // 2, grid_height // 2
            offsets = [(0, 0), (-8, -8), (8, 8)]
            offset_x, offset_y = offsets[attempts]
            coord = (center_x + offset_x, center_y + offset_y)
            exploration_data['attempts'] += 1
            
            if attempts >= 2:
                exploration_data['exploration_phase'] = 'edges'
                exploration_data['attempts'] = 0
                
            return coord
            
        elif phase == 'edges' and attempts < 4:
            # Explore edge midpoints
            edges = [
                (grid_width // 2, 3),          # top edge
                (grid_width - 3, grid_height // 2),  # right edge  
                (grid_width // 2, grid_height - 3),  # bottom edge
                (3, grid_height // 2)          # left edge
            ]
            coord = edges[attempts]
            exploration_data['attempts'] += 1
            
            if attempts >= 3:
                exploration_data['exploration_phase'] = 'random'
                exploration_data['attempts'] = 0
                
            return coord
            
        else:
            # Random exploration with bias toward unexplored areas
            import random
            
            # Avoid recently explored coordinates
            recent_coords = exploration_data.get('last_coordinates', [])
            max_attempts = 10
            
            for _ in range(max_attempts):
                explore_x = random.randint(3, grid_width - 4)
                explore_y = random.randint(3, grid_height - 4)
                coord = (explore_x, explore_y)
                
                # Check if we've explored this area recently
                too_close = False
                for prev_x, prev_y in recent_coords[-5:]:  # Last 5 coordinates
                    if abs(coord[0] - prev_x) < 5 and abs(coord[1] - prev_y) < 5:
                        too_close = True
                        break
                        
                if not too_close:
                    break
            
            # Update history
            exploration_data['last_coordinates'].append(coord)
            if len(exploration_data['last_coordinates']) > 10:
                exploration_data['last_coordinates'] = exploration_data['last_coordinates'][-10:]
                
            return coord

    def _should_reset_due_to_stagnation(self, session_result: Dict[str, Any], game_id: str) -> bool:
        """
        Determine if we should reset due to frame stagnation.
        Only reset when frame data isn't showing new information, not just because of repetitive actions.
        """
        # Check if we have enough data to make a decision
        if not hasattr(self, 'frame_stagnation_tracker'):
            self.frame_stagnation_tracker = {}
        
        if game_id not in self.frame_stagnation_tracker:
            self.frame_stagnation_tracker[game_id] = {
                'last_frame_hash': None,
                'stagnant_frames': 0,
                'total_frames': 0,
                'last_reset_action': 0
            }
        
        tracker = self.frame_stagnation_tracker[game_id]
        tracker['total_frames'] += 1
        
        # Get current frame data
        current_frame = session_result.get('frame')
        if current_frame is None:
            return False
        
        # Create a simple hash of the frame for comparison
        import hashlib
        frame_str = str(current_frame)
        current_frame_hash = hashlib.md5(frame_str.encode()).hexdigest()[:8]
        
        # Check if frame has changed
        if tracker['last_frame_hash'] == current_frame_hash:
            tracker['stagnant_frames'] += 1
        else:
            tracker['stagnant_frames'] = 0
            tracker['last_frame_hash'] = current_frame_hash
        
        # Only consider reset if:
        # 1. We've seen the same frame for at least 15 consecutive actions (increased threshold)
        # 2. We've taken at least 30 actions since last reset (increased threshold)
        # 3. We're not making progress (score hasn't improved)
        actions_since_reset = tracker['total_frames'] - tracker['last_reset_action']
        current_score = session_result.get('final_score', 0)
        
        should_reset = (
            tracker['stagnant_frames'] >= 15 and  # Same frame for 15+ actions (conservative)
            actions_since_reset >= 30 and  # At least 30 actions since last reset (conservative)
            current_score == 0  # No progress made
        )
        
        if should_reset:
            print(f" FRAME STAGNATION: {tracker['stagnant_frames']} stagnant frames, {actions_since_reset} actions since reset")
            tracker['last_reset_action'] = tracker['total_frames']
            tracker['stagnant_frames'] = 0
            return True
        
        return False

    def _select_intelligent_action_with_relevance(self, available_actions: List[int], context: Dict[str, Any]) -> int:
        """Select action using intelligent relevance scoring with comprehensive fallbacks."""
        # Ensure memory is properly initialized
        self._ensure_available_actions_memory()
        """
         ADVANCED ACTION6 DECISION PROTOCOL - Visual-Interactive Agent
        
        NEW PROTOCOL IMPLEMENTATION:
        1. FIRST: Always prefer simpler actions (1-5,7) if they might progress
        2. ONLY USE ACTION6 when it's the best or only visual-interactive option
        3. ACTION6 now analyzes frame for visual targets (buttons, objects, anomalies)
        4. Systematic exploration when no clear visual targets found
        
        This transforms the agent from a "blind mover" to a "visual-interactive agent"
        """
        # Ensure available_actions_memory is initialized
        if not hasattr(self, 'available_actions_memory'):
            self.available_actions_memory = {
                'current_game_id': None,
                'action_history': [],
                'effectiveness_tracking': {},
                'coordinate_patterns': {},
                'winning_action_sequences': [],
                'failed_action_patterns': [],
                'game_intelligence_cache': {},
                'last_action_result': None,
                'sequence_in_progress': [],
                'action_effectiveness': {},
                'action_relevance_scores': {},
                'action6_strategy': {
                    'last_action6_used': 0,
                    'last_progress_action': 0,
                    'consecutive_action6_count': 0
                },
                'action_stagnation': {}
            }
        
        game_id = context.get('game_id', 'unknown')
        action_count = len(self.available_actions_memory.get('action_history', []))
        frame_analysis = context.get('frame_analysis', {})
        
        print(f" ACTION DECISION PROTOCOL - Available: {available_actions}")
        
        # 🔄 STAGNATION DETECTION AND ACTION SWITCHING
        stagnation_info = self._check_action_stagnation(game_id, available_actions)
        if stagnation_info['should_switch']:
            print(f"🔄 STAGNATION DETECTED: {stagnation_info['reason']}")
            print(f"🔄 SWITCHING FROM: {stagnation_info['stagnant_actions']} TO: {stagnation_info['recommended_actions']}")
            # Force selection from recommended actions, but ensure we don't empty the list
            if stagnation_info['recommended_actions'] and len(stagnation_info['recommended_actions']) > 0:
                available_actions = stagnation_info['recommended_actions']
                print(f"✅ Stagnation switch: Using {len(available_actions)} recommended actions")
            else:
                print(f"⚠️ WARNING: Stagnation detection returned empty recommended actions, keeping original: {available_actions}")
                # Additional safety: if original is also empty, this is a critical error
                if not available_actions:
                    print(f"🚨 CRITICAL: Both stagnation recommendations and original actions are empty!")
                    return 1  # Emergency fallback
        
        # 🧠 PATTERN RETRIEVAL: Get learned patterns for this context
        pattern_recommendations = self._get_pattern_recommendations(game_id, context, available_actions)
        if pattern_recommendations:
            print(f"🧠 PATTERN RECOMMENDATIONS: {len(pattern_recommendations)} patterns found")
            # Boost scores for pattern-recommended actions
            for action, boost in pattern_recommendations.items():
                if action in available_actions:
                    print(f"   Action {action}: +{boost:.2f} pattern boost")
        
        # Store pattern recommendations for use in scoring
        self._current_pattern_recommendations = pattern_recommendations
        
        # 🧠 META-COGNITIVE GOVERNOR INTEGRATION (Third Brain)
        if hasattr(self, 'governor') and self.governor:
            try:
                governor_decision = self.governor.make_decision(
                    available_actions=available_actions,
                    context=context,
                    performance_history=getattr(self, 'performance_history', []),
                    current_energy=getattr(self, 'current_energy', 1.0)
                )
                if governor_decision and 'recommended_action' in governor_decision:
                    print(f"🧠 GOVERNOR DECISION: {governor_decision['reasoning']}")
                    print(f"🧠 GOVERNOR RECOMMENDATION: Action {governor_decision['recommended_action']}")
                    
                    # Track Governor decision for monitoring
                    if hasattr(self, 'current_session') and self.current_session:
                        session_id = getattr(self.current_session, 'session_id', 'unknown')
                        if not hasattr(self, 'session_results') or not self.session_results:
                            self.session_results = {}
                        if 'governor_decisions' not in self.session_results:
                            self.session_results['governor_decisions'] = []
                        
                        self.session_results['governor_decisions'].append({
                            'timestamp': time.time(),
                            'game_id': game_id,
                            'decision': governor_decision,
                            'confidence': governor_decision.get('confidence', 0.0)
                        })
                    
                    # Apply Governor's recommendation with confidence weighting
                    if governor_decision['confidence'] > 0.7:
                        print(f"🧠 HIGH CONFIDENCE: Following Governor recommendation")
                        return governor_decision['recommended_action']
                    else:
                        print(f"🧠 LOW CONFIDENCE: Using Governor as guidance only")
            except Exception as e:
                logger.warning(f"Governor decision failed: {e}")
        
        # 🏗️ ARCHITECT EVOLUTION INTEGRATION (Zeroth Brain)
        if self.architect:
            try:
                architect_insight = self.architect.evolve_strategy(
                    available_actions=available_actions,
                    context=context,
                    performance_data=self.performance_history,
                    frame_analysis=frame_analysis
                )
                if architect_insight and 'evolved_strategy' in architect_insight:
                    print(f"🏗️ ARCHITECT INSIGHT: {architect_insight['reasoning']}")
                    print(f"🏗️ ARCHITECT STRATEGY: {architect_insight['evolved_strategy']}")
                    
                    # Track Architect evolution for monitoring
                    if hasattr(self, 'current_session') and self.current_session:
                        session_id = getattr(self.current_session, 'session_id', 'unknown')
                        if not hasattr(self, 'session_results') or not self.session_results:
                            self.session_results = {}
                        if 'architect_evolutions' not in self.session_results:
                            self.session_results['architect_evolutions'] = []
                        
                        self.session_results['architect_evolutions'].append({
                            'timestamp': time.time(),
                            'game_id': game_id,
                            'insight': architect_insight,
                            'innovation_score': architect_insight.get('innovation_score', 0.0)
                        })
                    
                    # Apply Architect's evolved strategy
                    if architect_insight['innovation_score'] > 0.6:
                        print(f"🏗️ HIGH INNOVATION: Applying evolved strategy")
                        # The Architect can influence the decision process
                        context['architect_strategy'] = architect_insight['evolved_strategy']
            except Exception as e:
                logger.warning(f"Architect evolution failed: {e}")
        
        #  PHASE 1: PRIORITIZE SIMPLER ACTIONS (1-5,7) 
        # These are more predictable and often make direct progress
        simple_actions = [a for a in available_actions if a != 6]
        
        if simple_actions:
            print(f" SIMPLE ACTIONS AVAILABLE: {simple_actions} - Prioritizing over ACTION6")
            
            # ENHANCED DIVERSIFICATION SYSTEM: Force systematic exploration when stuck
            actions_without_progress = getattr(self, '_actions_without_progress', 0)
            if hasattr(self, '_last_selected_actions'):
                recent_actions = self._last_selected_actions[-20:]  # Last 20 actions
                
                # Track which actions have been attempted recently
                if not hasattr(self, '_exploration_cycle'):
                    self._exploration_cycle = {'untried_actions': list(available_actions), 'cycle_active': False}
                
                # More conservative emergency override: trigger after 15 actions without progress
                if (len(recent_actions) >= 10 and 
                    len(set(recent_actions)) <= 2 and  # Only 1-2 unique actions used
                    actions_without_progress >= 15):  # No progress for 15+ actions (increased from 8)
                    
                    print(f" EMERGENCY OVERRIDE: Stuck in action loop {set(recent_actions)} for {actions_without_progress} actions")
                    
                    # Start systematic exploration cycle
                    if not self._exploration_cycle['cycle_active']:
                        print(f" STARTING SYSTEMATIC EXPLORATION CYCLE")
                        self._exploration_cycle['untried_actions'] = [a for a in available_actions if a not in set(recent_actions[-5:])]
                        self._exploration_cycle['cycle_active'] = True
                    
                    # Try next untested action in cycle
                    if self._exploration_cycle['untried_actions']:
                        forced_action = self._exploration_cycle['untried_actions'].pop(0)
                        print(f" FORCED EXPLORATION: Trying action {forced_action} (unused in recent loop)")
                        self._actions_without_progress = 0  # Reset counter
                        return forced_action
                    else:
                        # All actions tried, fall back to ACTION 6 if available
                        print(f" EXPLORATION COMPLETE: All actions tried, forcing ACTION6")
                        self._exploration_cycle['cycle_active'] = False
                        self._actions_without_progress = 0
                        if 6 in available_actions:
                            return 6
                else:
                    # Reset exploration cycle if we're making progress or out of loop
                    if actions_without_progress == 0:
                        self._exploration_cycle = {'untried_actions': list(available_actions), 'cycle_active': False}
                    
            # Fallback emergency override for extended stagnation  
            if actions_without_progress >= 15:  # Extended stagnation fallback
                print(f" EXTENDED STAGNATION: {actions_without_progress} actions without progress")
                if 6 in available_actions:
                    print(f" STAGNATION OVERRIDE: Forcing ACTION6")
                    self._actions_without_progress = 0
                    return 6
            
            # Update action relevance scores
            self._update_action_relevance_scores()
            
            # Score simple actions with enhanced intelligence AND anti-repetition
            best_simple_action = None
            best_simple_score = 0.0
            
            # Track recent actions for anti-repetition
            if not hasattr(self, '_last_selected_actions'):
                self._last_selected_actions = []
            
            for action in simple_actions:
                try:
                    # Calculate comprehensive score for simple action
                    base_score = self._calculate_comprehensive_action_score(action, game_id, frame_analysis)
                    
                    # CRITICAL: Apply anti-repetition penalty directly here
                    final_score = base_score
                    recent_actions = self._last_selected_actions[-8:] if len(self._last_selected_actions) >= 8 else self._last_selected_actions
                    
                    if len(recent_actions) >= 5:
                        action_frequency = recent_actions.count(action) / len(recent_actions)
                        if action_frequency > 0.6:  # Used more than 60% of the time
                            repetition_penalty = 0.8 + (action_frequency - 0.6) * 3.0  # Very heavy penalty
                            final_score = max(0.01, base_score - repetition_penalty)
                            print(f" HEAVY REPETITION PENALTY: Action {action} used {action_frequency:.1%} recently (-{repetition_penalty:.2f} → {final_score:.3f})")
                        elif action_frequency > 0.4:  # Used more than 40% of the time  
                            repetition_penalty = 0.3 + (action_frequency - 0.4) * 1.5  # Moderate penalty
                            final_score = max(0.01, base_score - repetition_penalty)
                            print(f" MODERATE REPETITION PENALTY: Action {action} used {action_frequency:.1%} recently (-{repetition_penalty:.2f} → {final_score:.3f})")
                    
                    if final_score > best_simple_score:
                        best_simple_score = final_score
                        best_simple_action = action
                        
                except Exception as e:
                    print(f" Error scoring action {action}: {e}")
                    continue
            
            # If we found a decent simple action, use it
            if best_simple_action and best_simple_score > 0.1:  # LOWERED - Less strict threshold for action selection
                print(f" SELECTING SIMPLE ACTION {best_simple_action} (score: {best_simple_score:.3f})")
                
                # Track selected actions for loop detection
                if not hasattr(self, '_last_selected_actions'):
                    self._last_selected_actions = []
                self._last_selected_actions.append(best_simple_action)
                if len(self._last_selected_actions) > 50:  # Keep last 50
                    self._last_selected_actions = self._last_selected_actions[-50:]
                
                return best_simple_action
            elif best_simple_action and best_simple_score > 0.01:  # Very low score - force diversification
                # Check if we've been using this action too much
                recent_actions = self._last_selected_actions[-10:] if len(self._last_selected_actions) >= 10 else self._last_selected_actions
                if recent_actions.count(best_simple_action) >= 7:  # Used 7+ times in last 10 actions
                    print(f" FORCED DIVERSIFICATION: Action {best_simple_action} used {recent_actions.count(best_simple_action)}/10 times - trying different action")
                    # Find least used action in recent history
                    action_usage = {a: recent_actions.count(a) for a in simple_actions}
                    diversification_action = min(simple_actions, key=lambda a: action_usage.get(a, 0))
                    print(f" DIVERSIFICATION ACTION: {diversification_action} (used {action_usage.get(diversification_action, 0)}/10 times)")
                    
                    if not hasattr(self, '_last_selected_actions'):
                        self._last_selected_actions = []
                    self._last_selected_actions.append(diversification_action)
                    if len(self._last_selected_actions) > 50:  # Keep last 50
                        self._last_selected_actions = self._last_selected_actions[-50:]
                    
                    return diversification_action
                else:
                    # Use the low-score action for now
                    print(f" SELECTING LOW-SCORE SIMPLE ACTION {best_simple_action} (score: {best_simple_score:.3f})")
                    
                    if not hasattr(self, '_last_selected_actions'):
                        self._last_selected_actions = []
                    self._last_selected_actions.append(best_simple_action)
                    if len(self._last_selected_actions) > 50:  # Keep last 50
                        self._last_selected_actions = self._last_selected_actions[-50:]
                    
                    return best_simple_action
            else:
                print(f" Simple actions available but low scores - considering ACTION6...")
        
        #  PHASE 2: ACTION6 VISUAL-INTERACTIVE ANALYSIS
        # Only if ACTION6 is available AND (no good simple actions OR ACTION6 is only option)
        if 6 in available_actions:
            print(f" ACTION6 VISUAL-INTERACTIVE ANALYSIS - Frame analysis: {'' if frame_analysis else ''}")
            
            # Store current frame for visual analysis
            current_frame = context.get('frame')
            if current_frame:
                setattr(self, 'current_frame_data', current_frame)
                print(f" Frame data stored for visual targeting analysis")
            
            # Check if ACTION6 is strategically appropriate  
            progress_stagnant = self._is_progress_stagnant(action_count)
            action6_strategic_score = self._calculate_action6_strategic_score(action_count, progress_stagnant)
            
            # Visual targeting analysis if we have frame data
            visual_targeting_bonus = 0.0
            if frame_analysis:
                visual_targeting_bonus = self._calculate_frame_analysis_bonus_action6(frame_analysis)
                print(f" Visual targeting bonus: +{visual_targeting_bonus:.3f}")
            
            # Calculate final ACTION6 score
            action6_final_score = action6_strategic_score * (1.0 + visual_targeting_bonus)
            
            # Decision logic
            if not simple_actions:
                # ACTION6 is only option
                print(f" ACTION6 ONLY OPTION - Using visual-interactive targeting")
                # Track selected actions for loop detection
                if not hasattr(self, '_last_selected_actions'):
                    self._last_selected_actions = []
                self._last_selected_actions.append(6)
                if len(self._last_selected_actions) > 50:
                    self._last_selected_actions = self._last_selected_actions[-50:]
                return 6
            elif action6_final_score > 0.05:  #  FURTHER REDUCED: Much lower barrier for ACTION6 selection
                print(f" ACTION6 HIGH PRIORITY - Visual targeting score: {action6_final_score:.3f}")
                # Track selected actions for loop detection
                if not hasattr(self, '_last_selected_actions'):
                    self._last_selected_actions = []
                self._last_selected_actions.append(6)
                if len(self._last_selected_actions) > 50:
                    self._last_selected_actions = self._last_selected_actions[-50:]
                return 6  
            elif progress_stagnant and action6_final_score > 0.05:  #  LOWER STAGNATION THRESHOLD
                print(f" PROGRESS STAGNANT - Using ACTION6 as visual reset (score: {action6_final_score:.3f})")
                # Track selected actions for loop detection
                if not hasattr(self, '_last_selected_actions'):
                    self._last_selected_actions = []
                self._last_selected_actions.append(6)
                if len(self._last_selected_actions) > 50:
                    self._last_selected_actions = self._last_selected_actions[-50:]
                return 6
            else:
                print(f" ACTION6 score too low ({action6_final_score:.3f}) - using best simple action")
                
                # Fallback to best simple action even if score is low
                if simple_actions:
                    fallback_action = simple_actions[0]  # Just pick first available
                    print(f" FALLBACK: Using simple action {fallback_action}")
                    # Track selected actions for loop detection
                    if not hasattr(self, '_last_selected_actions'):
                        self._last_selected_actions = []
                    self._last_selected_actions.append(fallback_action)
                    if len(self._last_selected_actions) > 50:
                        self._last_selected_actions = self._last_selected_actions[-50:]
                    return fallback_action
        
        #  EMERGENCY FALLBACK - Select least recently used action
        print(" EMERGENCY FALLBACK - No clear action choice")
        if available_actions:
            # Count recent usage of each action to avoid repetition
            if not hasattr(self, '_last_selected_actions'):
                self._last_selected_actions = []
            
            recent_actions = self._last_selected_actions[-10:] if len(self._last_selected_actions) >= 10 else self._last_selected_actions
            action_usage_count = {}
            for action in available_actions:
                action_usage_count[action] = recent_actions.count(action)
            
            # Select the least used action in recent history
            emergency_action = min(available_actions, key=lambda a: action_usage_count.get(a, 0))
            print(f" Emergency selection: ACTION{emergency_action} (least used: {action_usage_count.get(emergency_action, 0)}/10 recent)")
            
            # Track selected actions for loop detection
            self._last_selected_actions.append(emergency_action)
            if len(self._last_selected_actions) > 50:
                self._last_selected_actions = self._last_selected_actions[-50:]
            return emergency_action
        else:
            print(" NO AVAILABLE ACTIONS - This should not happen!")
            return 1  # Ultimate fallback
    
    def _calculate_comprehensive_action_score(self, action: int, game_id: str, frame_analysis: Dict[str, Any]) -> float:
        """Calculate comprehensive score for simple actions (1-5,7)."""
        try:
            # Get base relevance data
            if action in self.available_actions_memory['action_relevance_scores']:
                relevance_data = self.available_actions_memory['action_relevance_scores'][action]
                base_score = relevance_data['base_relevance']
                modifier = relevance_data['current_modifier']
                success_rate = relevance_data['recent_success_rate']
            else:
                base_score = 0.5
                modifier = 1.0
                success_rate = 0.5
            
            # FIXED: Semantic action intelligence - pass game_id as dictionary
            game_context = {'game_id': game_id} if game_id else {}
            semantic_score = self._calculate_semantic_action_score(action, game_context)
            
            # Frame analysis bonus for non-ACTION6 actions
            frame_bonus = 0.0
            if frame_analysis and action != 6:
                # Simple actions can benefit from movement analysis
                if action in [1, 2, 3, 4]:  # Movement actions
                    if frame_analysis.get('movement_detected', False):
                        frame_bonus = 0.2  # Slight bonus if movement detected
                elif action in [5, 7]:  # Interaction/undo actions
                    if frame_analysis.get('interaction_opportunity', False):
                        frame_bonus = 0.3  # Interaction actions get bonus for opportunities
            
            # 🧠 PATTERN BOOST: Add pattern-based recommendations
            pattern_boost = 0.0
            if hasattr(self, '_current_pattern_recommendations'):
                pattern_boost = self._current_pattern_recommendations.get(action, 0.0)
            
            # Combine all factors
            final_score = base_score * modifier * success_rate * semantic_score * (1.0 + frame_bonus + pattern_boost)
            
            return max(0.0, min(1.0, final_score))  # Clamp to [0,1]
            
        except Exception as e:
            print(f" Error calculating score for action {action}: {e}")
            return 0.3  # Safe fallback score

    def _calculate_frame_analysis_bonus_action6(self, frame_analysis: Dict[str, Any]) -> float:
        """Calculate bonus score for ACTION6 based on visual frame analysis"""
        if not frame_analysis:
            return 0.0
            
        bonus = 0.0
        
        # Bonus for having interactive targets
        if frame_analysis.get('interactive_targets'):
            num_targets = len(frame_analysis['interactive_targets'])
            bonus += min(0.5, num_targets * 0.1)  # Up to 0.5 bonus
            
        # Bonus for high confidence targets
        if frame_analysis.get('recommended_coordinate'):
            confidence = frame_analysis.get('confidence', 0.0)
            bonus += confidence * 0.3  # Up to 0.3 bonus for high confidence
            
        # Bonus for visual anomalies that suggest interactable elements
        if frame_analysis.get('targeting_reason', '').startswith('rare_color'):
            bonus += 0.2  # Rare colors often indicate buttons/objects
            
        return min(1.0, bonus)  # Cap at 1.0
        
        # Normalize scores to prevent extreme bias towards any single action
        min_score = min(action_scores.values())
        max_score = max(action_scores.values())
        score_range = max_score - min_score
        
        if score_range > 0:
            # Normalize and add minimum weight to ensure all actions have reasonable chance
            normalized_weights = []
            for action in action_scores:
                normalized_score = (action_scores[action] - min_score) / score_range
                # Give every action at least 10% of max weight to prevent spam
                final_weight = 0.1 + (0.9 * normalized_score)
                normalized_weights.append(final_weight)
        else:
            # All scores equal - use uniform distribution
            normalized_weights = [1.0] * len(action_scores)
        
        # Smart weighted selection with anti-bias protection
        def format_score(s):
            try:
                return f'{float(s):.3f}' if s is not None else '0.000'
            except (ValueError, TypeError):
                return str(s)
        print(f" ACTION SELECTION SCORES: {[(a, format_score(s)) for a, s in action_scores.items()]}")
        
        # Ensure all scores are numeric before proceeding
        numeric_scores = {}
        for action, score in action_scores.items():
            try:
                numeric_scores[action] = float(score) if score is not None else 0.0
            except (ValueError, TypeError):
                print(f" Warning: Non-numeric score {score} for action {action}, using 0.0")
                numeric_scores[action] = 0.0
        action_scores = numeric_scores
        
        # Apply intelligent weighting that prevents any single action from dominating
        min_score = min(action_scores.values())
        max_score = max(action_scores.values())
        score_range = max_score - min_score
        
        if score_range > 0:
            # Normalize scores and ensure minimum 20% weight for ALL actions to prevent spam
            action_weights = []
            for action in action_scores:
                normalized_score = (action_scores[action] - min_score) / score_range
                # CRITICAL: Every action gets at least 20% weight to ensure diversity
                final_weight = 0.20 + (0.80 * normalized_score)
                action_weights.append(final_weight)
            
            # Use weighted random selection with anti-bias protection
            selected_action = random.choices(list(action_scores.keys()), weights=action_weights)[0]
        else:
            # All scores equal - pure random selection
            selected_action = random.choice(list(action_scores.keys()))
        
        def safe_format_weight(w):
            try:
                return f'{float(w):.3f}' if w is not None else '0.000'
            except (ValueError, TypeError):
                return str(w)
        
        weights_to_display = action_weights if score_range > 0 else [1.0] * len(action_scores)
        print(f"  ANTI-BIAS WEIGHTS: {[(a, safe_format_weight(w)) for a, w in zip(action_scores.keys(), weights_to_display)]}")
        print(f" SELECTED ACTION: {selected_action} (WEIGHTED RANDOM with 20% minimum weight per action)")
        print(f" ANTI-BIAS: All actions guaranteed minimum 20% selection chance")
        
        # Log strategic decisions for ACTION 6
        if selected_action == 6:
            last_progress = self.available_actions_memory.get('action6_strategy', {}).get('last_progress_action', 0)
            print(f" ACTION 6 STRATEGIC USE: Progress stagnant={progress_stagnant}, Actions since progress={action_count - last_progress}")
        
        # Update usage tracking
        self._update_action_usage_tracking(selected_action, action_count)
        
        return selected_action

    def _calculate_action6_strategic_score(self, current_action_count: int, progress_stagnant: bool) -> float:
        """
         ENHANCED: Calculate strategic score for ACTION 6 with more balanced usage.
        
        ACTION 6 should be used when:
        1. Simple actions aren't making sufficient progress
        2. Visual targeting might be more effective 
        3. As part of systematic exploration strategy
        4. When game seems to require coordinate-based interaction
        """
        strategy = self.available_actions_memory['action6_strategy']
        
        #  CRITICAL FIX: More balanced base score for ACTION6
        base_score = 0.3 if progress_stagnant else 0.15  # Much higher base score
        
        # Give ACTION6 a fair chance even when progress is being made
        if not progress_stagnant:
            base_score = 0.1  # Still allow ACTION6 during normal gameplay
        
        #  REDUCED RESTRICTIONS: More lenient usage requirements
        actions_since_start = current_action_count
        min_actions_threshold = max(5, strategy['min_actions_before_use'] // 3)  # Reduced from 15 to 5
        if actions_since_start < min_actions_threshold:
            return 0.05  # Still available but with lower priority
        
        #  REDUCED COOLDOWN: More frequent ACTION6 usage
        actions_since_last_action6 = current_action_count - strategy['last_action6_used']
        cooldown_threshold = max(2, strategy['action6_cooldown'] // 2)  # Reduced cooldown
        if actions_since_last_action6 < cooldown_threshold:
            return 0.1  # Available but with penalty, not completely blocked
        
        # Stagnation analysis - bonus for trying ACTION6 when stuck
        actions_since_progress = current_action_count - strategy['last_progress_action']
        if actions_since_progress < strategy['progress_stagnation_threshold']:
            return 0.001  # Not stuck long enough
        
        # If we reach here, we're truly stuck - ACTION 6 becomes viable
        strategic_score = base_score
        
        # Increase score based on how long we've been stuck
        stagnation_multiplier = min(2.0, actions_since_progress / strategy['progress_stagnation_threshold'])
        strategic_score *= stagnation_multiplier
        
        # Emergency boost if we've been stuck for a very long time
        if actions_since_progress > strategy['progress_stagnation_threshold'] * 2:
            strategic_score *= 1.5  # Emergency mini-reset mode
            print(f" ACTION 6 EMERGENCY MODE: Stuck for {actions_since_progress} actions")
        
        return min(strategic_score, 0.2)  # Cap at 0.2 to prevent over-use

    def _is_progress_stagnant(self, current_action_count: int) -> bool:
        """
         ENHANCED: Determine if progress has stagnated with better detection.
        """
        strategy = self.available_actions_memory['action6_strategy']
        actions_since_progress = current_action_count - strategy['last_progress_action']
        
        #  SMARTER STAGNATION DETECTION: Consider both time and action diversity
        basic_stagnation = actions_since_progress >= strategy['progress_stagnation_threshold']
        
        # Also check if we're repeating the same actions without progress
        if hasattr(self, '_last_selected_actions') and len(self._last_selected_actions) >= 6:
            recent_actions = self._last_selected_actions[-6:]
            unique_actions = len(set(recent_actions))
            action_diversity_low = unique_actions <= 2  # Only 1-2 different actions used
            
            # If low diversity AND no recent progress, consider stagnant even if time threshold not met
            if action_diversity_low and actions_since_progress >= (strategy['progress_stagnation_threshold'] // 2):
                return True
        
        return basic_stagnation

    def _update_action_relevance_scores(self):
        """
        Update action relevance scores based on recent performance.
        This implements your idea of actions having relevance that changes over time.
        
        CRITICAL FIX: ACTION 6 should NOT gain relevance from coordinate placement success,
        only from actual meaningful game progress.
        """
        relevance_scores = self.available_actions_memory['action_relevance_scores']
        
        for action_num, data in relevance_scores.items():
            # Calculate recent success rate from action history
            recent_attempts = self._get_recent_action_attempts(action_num, window=10)
            
            if recent_attempts['total'] > 0:
                success_rate = recent_attempts['successes'] / recent_attempts['total']
                data['recent_success_rate'] = success_rate
                
                # SPECIAL HANDLING FOR ACTION 6 - Only tiny relevance increases
                if action_num == 6:
                    # ACTION 6 should almost never increase relevance
                    # Only allow microscopic increases and only if there's actual game progress
                    if success_rate > 0.8 and self._has_recent_meaningful_progress():
                        old_modifier = data['current_modifier']
                        data['current_modifier'] = min(0.4, data['current_modifier'] + 0.005)
                        print(f" ACTION 6 minimal relevance increase: {old_modifier:.3f} → {data['current_modifier']:.3f} (rare meaningful progress detected)")
                    else:
                        # Continuous slow decay to prevent ACTION 6 spam
                        old_modifier = data['current_modifier']
                        data['current_modifier'] = max(0.05, data['current_modifier'] * 0.98)  # Lowered floor to 0.05
                        if old_modifier != data['current_modifier']:
                            print(f" ACTION 6 relevance decay: {old_modifier:.3f} → {data['current_modifier']:.3f} (preventing coordinate spam)")
                        
                else:
                    # Normal actions can gain relevance from success
                    if success_rate > 0.7:
                        data['current_modifier'] = min(1.5, data['current_modifier'] * 1.1)  # Increase relevance
                    elif success_rate < 0.3:
                        data['current_modifier'] = max(0.2, data['current_modifier'] * 0.9)  # Decrease relevance
                    
            # Decay relevance for unused actions over time
            actions_since_use = len(self.available_actions_memory['action_history']) - data['last_used']
            if actions_since_use > 20:  # If not used for 20 actions
                # Extra decay for ACTION 6 to prevent it from becoming dominant
                decay_rate = 0.90 if action_num == 6 else 0.95
                data['current_modifier'] = max(0.1, data['current_modifier'] * decay_rate)

    def _get_recent_action_attempts(self, action_num: int, window: int = 10) -> Dict[str, int]:
        """Get recent attempt statistics for an action."""
        # FIXED: Use actual action effectiveness data instead of placeholder
        effectiveness_data = self.available_actions_memory.get('action_effectiveness', {}).get(action_num, {
            'attempts': 0, 'successes': 0, 'success_rate': 0.0
        })
        
        # Use the actual tracked data
        total_attempts = effectiveness_data['attempts']
        total_successes = effectiveness_data['successes']
        
        # For recent window, count from action history
        action_history = self.available_actions_memory['action_history']
        recent_history = action_history[-window:] if len(action_history) >= window else action_history
        recent_attempts = recent_history.count(action_num)
        
        # Estimate recent successes based on overall success rate
        overall_success_rate = effectiveness_data['success_rate']
        estimated_recent_successes = int(recent_attempts * overall_success_rate)
        
        return {
            'total': recent_attempts, 
            'successes': estimated_recent_successes,
            'overall_attempts': total_attempts,
            'overall_successes': total_successes,
            'overall_success_rate': overall_success_rate
        }

    def _has_recent_meaningful_progress(self) -> bool:
        """
        Check if there has been recent meaningful game progress.
        This prevents ACTION 6 from gaining relevance just from coordinate placement success.
        """
        # Check recent LP history for actual progress indicators
        if hasattr(self.training_state, 'lp_history') and self.training_state['lp_history']:
            recent_episodes = self.training_state['lp_history'][-3:]  # Last 3 episodes
            
            # Look for increasing scores, wins, or level progression
            has_score_improvement = False
            has_win = False
            has_level_progress = False
            
            for episode in recent_episodes:
                if episode.get('success', False):
                    has_win = True
                if episode.get('score', 0) > 50:  # Meaningful score threshold
                    has_score_improvement = True
                if episode.get('level_progressed', False):
                    has_level_progress = True
            
            # Only consider it meaningful progress if we have concrete achievements
            meaningful_progress = has_win or has_score_improvement or has_level_progress
            
            if meaningful_progress:
                print(f" Meaningful progress detected: win={has_win}, score={has_score_improvement}, level={has_level_progress}")
            
            return meaningful_progress
        
        # Default to False - no meaningful progress detected
        return False

    def _update_action_usage_tracking(self, selected_action: int, current_action_count: int):
        """Update usage tracking for selected action."""
        # Ensure available_actions_memory is initialized
        if not hasattr(self, 'available_actions_memory'):
            self.available_actions_memory = {
                'action_relevance_scores': {},
                'action6_strategy': {'last_action6_used': 0}
            }
        
        if selected_action in self.available_actions_memory.get('action_relevance_scores', {}):
            self.available_actions_memory['action_relevance_scores'][selected_action]['last_used'] = current_action_count
            
            # Special tracking for ACTION 6
            if selected_action == 6:
                if 'action6_strategy' not in self.available_actions_memory:
                    self.available_actions_memory['action6_strategy'] = {'last_action6_used': 0}
                self.available_actions_memory['action6_strategy']['last_action6_used'] = current_action_count

    def _get_current_agent_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent including energy, learning progress, and other metrics.
        
        Returns:
            Dict containing the current agent state
        """
        # Initialize default state
        state = {
            'energy': getattr(self, 'current_energy', 100.0),  # Default to full energy if not set
            'learning_progress': getattr(self, 'learning_progress', 0.0),
            'exploration_rate': getattr(self, 'exploration_rate', 1.0),
            'episode_count': getattr(self, 'current_episode', 0),
            'total_timesteps': getattr(self, 'total_timesteps', 0),
            'consecutive_failures': getattr(self, 'consecutive_failures', 0),
            'last_action': getattr(self, 'last_action', None),
            'last_reward': getattr(self, 'last_reward', 0.0),
            'game_state': getattr(self, 'current_game_state', 'IDLE'),
            'memory_usage': len(getattr(self, 'memory', [])),
            'sleep_cycles': getattr(self, 'sleep_cycles', 0),
            'game_complexity': getattr(self, 'current_game_complexity', 'medium'),
            'available_actions': getattr(self, 'available_actions', list(range(10))),  # Assuming 10 possible actions
            'action_history': getattr(self, 'action_history', []),
            'learning_progress': getattr(self, 'learning_progress', 0.0),
            'consecutive_failures': getattr(self, 'consecutive_failures', 0),
            'consecutive_successes': getattr(self, 'consecutive_successes', 0),
            'total_episodes': getattr(self, 'total_episodes', 0),
            'success_rate': getattr(self, 'success_rate', 0.0),
            'last_action_taken': getattr(self, 'last_action_taken', None),
            'last_action_success': getattr(self, 'last_action_success', None),
            'current_goals': getattr(self, 'current_goals', []),
            'active_learning_strategies': getattr(self, 'active_learning_strategies', [])
        }
        
        # Update from any available training state
        if hasattr(self, 'training_state'):
            state.update({
                'current_episode': getattr(self.training_state, 'current_episode', 0),
                'total_steps': getattr(self.training_state, 'total_steps', 0),
                'total_reward': getattr(self.training_state, 'total_reward', 0.0),
            })
            
        return state
        
    def _should_agent_sleep(self, agent_state: Dict[str, Any], session_count: int) -> bool:
        """
        Determine if the agent should enter a sleep cycle based on its current state.
        
        Args:
            agent_state: Current state of the agent
            session_count: Current session number
            
        Returns:
            bool: True if the agent should sleep, False otherwise
        """
        # Don't sleep too frequently - at minimum every 5 episodes
        min_episodes_between_sleep = 5
        if session_count % min_episodes_between_sleep != 0:
            return False
            
        # Check energy level - sleep if below threshold
        energy_threshold = 30.0  # Sleep if energy is below 30%
        if agent_state.get('energy', 100.0) < energy_threshold:
            logger.info(f" Low energy detected ({agent_state.get('energy'):.1f}%), triggering sleep")
            return True
            
        # Check for learning saturation - if learning progress has plateaued
        if hasattr(self, 'learning_progress_history'):
            # Look at last 5 learning progress values
            recent_progress = self.learning_progress_history[-5:] if len(self.learning_progress_history) >= 5 else self.learning_progress_history
            if len(recent_progress) >= 3:
                # Calculate average progress over last few episodes
                avg_recent_progress = sum(recent_progress) / len(recent_progress)
                if avg_recent_progress < 0.01:  # Minimal progress
                    logger.info(" Learning plateau detected, triggering sleep for memory consolidation")
                    return True
                    
        # Check for high failure rate
        if agent_state.get('consecutive_failures', 0) >= 3:
            logger.info(" Multiple consecutive failures, triggering sleep to reset strategies")
            return True
            
        # Random chance to sleep based on session count (encourages exploration)
        if session_count > 10 and random.random() < 0.1:  # 10% chance after 10 sessions
            logger.info(" Random sleep trigger to encourage exploration")
            return True
            
        return False
        
    def _should_activate_contrarian_strategy(self, game_id: str, consecutive_failures: int) -> Dict[str, Any]:
        """
        Determine if the agent should activate contrarian strategy based on failure patterns.
        
        Args:
            game_id: ID of the current game
            consecutive_failures: Number of consecutive failures
            
        Returns:
            Dict containing decision and metadata
        """
        decision = {
            'activate': False,
            'reason': 'none',
            'confidence': 0.0,
            'game_id': game_id,
            'consecutive_failures': consecutive_failures
        }
        
        # Don't activate too early
        if consecutive_failures < 3:
            return decision
            
        # Check if we've been failing consistently
        if consecutive_failures >= 5:
            decision.update({
                'activate': True,
                'reason': 'high_failure_rate',
                'confidence': 0.9,
                'details': f'Consecutive failures: {consecutive_failures}'
            })
            logger.info(f" Activating contrarian strategy due to high failure rate ({consecutive_failures} failures)")
            return decision
            
        # Check if we're in a local optima (repeating same actions)
        if hasattr(self, 'recent_actions') and len(self.recent_actions) >= 10:
            # Check if we're repeating the same action
            last_action = self.recent_actions[-1]
            action_repeats = sum(1 for a in self.recent_actions[-5:] if a == last_action)
            
            if action_repeats >= 4:  # Same action 4 out of last 5 times
                decision.update({
                    'activate': True,
                    'reason': 'action_repetition',
                    'confidence': 0.8,
                    'details': f'Repeated action {last_action} {action_repeats} times in last 5 actions'
                })
                logger.info(f" Activating contrarian strategy due to action repetition")
                return decision
                
        # Random chance to try contrarian approach (encourages exploration)
        if consecutive_failures >= 3 and random.random() < 0.3:  # 30% chance after 3 failures
            decision.update({
                'activate': True,
                'reason': 'random_exploration',
                'confidence': 0.5,
                'details': 'Random activation to encourage exploration'
            })
            logger.info(" Randomly activating contrarian strategy")
            
        return decision
        
    def _select_new_strategy(self) -> str:
        """
        Select a new exploration strategy to try when boredom is detected.
        
        Returns:
            str: Name of the selected strategy
        """
        # Define available strategies with their weights
        strategies = [
            ('random_exploration', 0.3),  # Pure random exploration
            ('curriculum_guided', 0.4),   # Follow curriculum guidance
            ('curious_driven', 0.2),      # Focus on high-uncertainty states
            ('goal_oriented', 0.1)        # Focus on goal achievement
        ]
        
        # Get current strategy if it exists
        current_strategy = getattr(self, 'current_strategy', None)
        
        # Remove current strategy from options to force a change
        available_strategies = [s for s in strategies if s[0] != current_strategy]
        if not available_strategies:
            available_strategies = strategies
            
        # Select based on weights
        total_weight = sum(weight for _, weight in available_strategies)
        r = random.uniform(0, total_weight)
        upto = 0
        
        for strategy, weight in available_strategies:
            if upto + weight >= r:
                logger.info(f" Switching to new strategy: {strategy}")
                self.current_strategy = strategy
                return strategy
            upto += weight
            
        # Fallback to first strategy
        self.current_strategy = available_strategies[0][0]
        return self.current_strategy
        
    def _check_and_handle_boredom(self, session_count: int) -> Dict[str, Any]:
        """
        Check if the agent is in a bored state (plateaued learning) and handle it.
        
        Args:
            session_count: Current session number
            
        Returns:
            Dict containing boredom detection results and actions taken
        """
        result = {
            'boredom_detected': False,
            'reason': 'none',
            'actions_taken': [],
            'session_count': session_count
        }
        
        # Need at least 10 sessions to detect boredom
        if session_count < 10:
            return result
            
        # Check learning progress history if available
        if hasattr(self, 'learning_progress_history') and len(self.learning_progress_history) >= 10:
            # Look at last 10 learning progress values
            recent_progress = self.learning_progress_history[-10:]
            avg_recent_progress = sum(recent_progress) / len(recent_progress)
            
            # If average progress is very low, we might be bored
            if avg_recent_progress < 0.005:  # Minimal progress
                result.update({
                    'boredom_detected': True,
                    'reason': 'learning_plateau',
                    'details': f'Average progress: {avg_recent_progress:.4f} over last {len(recent_progress)} sessions'
                })
                
                # Take action to reduce boredom
                actions = []
                
                # Increase exploration rate
                if hasattr(self, 'exploration_rate') and hasattr(self, 'min_exploration'):
                    self.exploration_rate = min(1.0, self.exploration_rate + 0.2)  # Increase exploration
                    actions.append(f'Increased exploration rate to {self.exploration_rate:.2f}')
                
                # Try a new strategy
                if hasattr(self, 'current_strategy'):
                    self.current_strategy = self._select_new_strategy()
                    actions.append(f'Switched to new strategy: {self.current_strategy}')
                
                # Add some randomness to action selection
                if hasattr(self, 'action_noise_scale'):
                    self.action_noise_scale = min(0.3, self.action_noise_scale + 0.05)
                    actions.append(f'Increased action noise to {self.action_noise_scale:.2f}')
                
                result['actions_taken'] = actions
                
                if actions:
                    logger.info(f" Boredom detected! Taking actions: {', '.join(actions)}")
                
        return result
        
    def _evaluate_game_reset_decision(self, game_id: str, session_count: int, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        CONSERVATIVE RESET DECISION: Only reset when truly stuck and Governor approves.
        Focus on pathfinding and strategy refinement rather than frequent resets.
        
        Args:
            game_id: ID of the current game
            session_count: Current session number
            agent_state: Current state of the agent
            
        Returns:
            Dict containing the reset decision and metadata
        """
        decision = {
            'should_reset': False,
            'reason': 'none',
            'confidence': 0.0,
            'game_id': game_id,
            'session_count': session_count,
            'governor_approval': False
        }
        
        # MUCH MORE CONSERVATIVE: Only consider reset every 10+ episodes
        min_episodes_between_reset = 10
        if session_count < min_episodes_between_reset:
            return decision
        
        # Track reset attempts to prevent excessive resets
        if not hasattr(self, 'reset_attempt_tracker'):
            self.reset_attempt_tracker = {}
        
        if game_id not in self.reset_attempt_tracker:
            self.reset_attempt_tracker[game_id] = {
                'total_resets': 0,
                'last_reset_episode': 0,
                'consecutive_reset_failures': 0
            }
        
        tracker = self.reset_attempt_tracker[game_id]
        
        # Don't reset if we've already tried recently and it didn't help
        if tracker['consecutive_reset_failures'] >= 2:
            return decision
        
        # Check for EXTREME stagnation - only reset if truly stuck
        extreme_stagnation = self._check_extreme_stagnation(game_id, agent_state)
        if not extreme_stagnation['is_stuck']:
            return decision
        
        # GOVERNOR APPROVAL REQUIRED for any reset
        governor_approval = self._get_governor_reset_approval(game_id, extreme_stagnation, agent_state)
        if not governor_approval['approved']:
            return decision
        
        # Only reset if we have a clear pathfinding strategy
        pathfinding_strategy = self._evaluate_pathfinding_strategy(game_id, agent_state)
        if not pathfinding_strategy['has_clear_strategy']:
            return decision
        
        # Final conservative check: Only reset if we've tried all other strategies
        if not self._have_tried_all_strategies(game_id):
            return decision
        
        decision.update({
            'should_reset': True,
            'reason': extreme_stagnation['reason'],
            'confidence': governor_approval['confidence'],
            'governor_approval': True,
            'pathfinding_strategy': pathfinding_strategy['strategy'],
            'details': f"Governor approved reset: {extreme_stagnation['reason']}"
        })
        
        # Update tracking
        tracker['total_resets'] += 1
        tracker['last_reset_episode'] = session_count
        
        logger.warning(f"🔄 CONSERVATIVE RESET APPROVED for {game_id}: {extreme_stagnation['reason']}")
        return decision
    
    def _check_extreme_stagnation(self, game_id: str, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check for extreme stagnation that truly warrants a reset."""
        stagnation_info = {
            'is_stuck': False,
            'reason': 'none',
            'severity': 0.0
        }
        
        # Check frame stagnation (no visual changes for many actions)
        if hasattr(self, 'available_actions_memory'):
            stagnation = self.available_actions_memory.get('action_stagnation', {}).get(game_id, {})
            consecutive_no_change = stagnation.get('consecutive_no_change', 0)
            
            # Only reset if NO frame changes for 20+ consecutive actions
            if consecutive_no_change >= 20:
                stagnation_info.update({
                    'is_stuck': True,
                    'reason': f'extreme_frame_stagnation_{consecutive_no_change}_actions',
                    'severity': min(1.0, consecutive_no_change / 30.0)
                })
                return stagnation_info
        
        # Check for complete action failure (all actions tried with no success)
        if hasattr(self, 'available_actions_memory'):
            action_effectiveness = self.available_actions_memory.get('action_effectiveness', {})
            total_attempts = sum(stats.get('attempts', 0) for stats in action_effectiveness.values())
            total_successes = sum(stats.get('successes', 0) for stats in action_effectiveness.values())
            
            if total_attempts > 50 and total_successes == 0:
                stagnation_info.update({
                    'is_stuck': True,
                    'reason': f'complete_action_failure_{total_attempts}_attempts_0_successes',
                    'severity': 1.0
                })
                return stagnation_info
        
        # Check for score regression (getting worse over time)
        if hasattr(self, 'performance_history') and len(self.performance_history) > 10:
            recent_scores = [p.get('score', 0) for p in self.performance_history[-10:]]
            if len(recent_scores) >= 5:
                score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                if score_trend < -2.0:  # Significant negative trend
                    stagnation_info.update({
                        'is_stuck': True,
                        'reason': f'score_regression_trend_{score_trend:.2f}',
                        'severity': min(1.0, abs(score_trend) / 5.0)
                    })
                    return stagnation_info
        
        return stagnation_info
    
    def _get_governor_reset_approval(self, game_id: str, stagnation_info: Dict, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get Governor approval for reset decision."""
        approval = {
            'approved': False,
            'confidence': 0.0,
            'reasoning': 'No Governor available'
        }
        
        if not hasattr(self, 'governor') or not self.governor:
            return approval
        
        try:
            # Create context for Governor decision
            context = {
                'game_id': game_id,
                'stagnation_info': stagnation_info,
                'agent_state': agent_state,
                'reset_request': True
            }
            
            # Ask Governor for reset approval
            governor_decision = self.governor.make_decision(
                available_actions=[],  # No actions available for reset decision
                context=context,
                performance_history=self.performance_history,
                current_energy=agent_state.get('energy', 100.0)
            )
            
            if governor_decision and governor_decision.get('confidence', 0) > 0.8:
                approval.update({
                    'approved': True,
                    'confidence': governor_decision.get('confidence', 0.0),
                    'reasoning': governor_decision.get('reasoning', 'Governor approved reset')
                })
            
        except Exception as e:
            logger.error(f"Error getting Governor reset approval: {e}")
        
        return approval
    
    def _evaluate_pathfinding_strategy(self, game_id: str, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if we have a clear pathfinding strategy before resetting."""
        strategy = {
            'has_clear_strategy': False,
            'strategy': 'none',
            'confidence': 0.0
        }
        
        # Check if we have learned any successful patterns
        if hasattr(self, 'available_actions_memory'):
            action_effectiveness = self.available_actions_memory.get('action_effectiveness', {})
            
            # Look for actions with some success
            successful_actions = []
            for action, stats in action_effectiveness.items():
                if stats.get('success_rate', 0) > 0.1:  # At least 10% success rate
                    successful_actions.append(action)
            
            if successful_actions:
                strategy.update({
                    'has_clear_strategy': True,
                    'strategy': f'learned_successful_actions_{successful_actions}',
                    'confidence': max(stats.get('success_rate', 0) for stats in action_effectiveness.values())
                })
                return strategy
        
        # Check if we have frame analysis insights
        if hasattr(self, '_last_frame_analysis'):
            frame_analysis = self._last_frame_analysis.get(game_id, {})
            if frame_analysis.get('targets') or frame_analysis.get('interactive_elements'):
                strategy.update({
                    'has_clear_strategy': True,
                    'strategy': 'frame_analysis_insights',
                    'confidence': 0.6
                })
                return strategy
        
        return strategy
    
    def _have_tried_all_strategies(self, game_id: str) -> bool:
        """Check if we've tried all available strategies before resetting."""
        if not hasattr(self, 'available_actions_memory'):
            return True  # If no memory, assume we've tried everything
        
        # Check if we've tried all available actions
        action_effectiveness = self.available_actions_memory.get('action_effectiveness', {})
        if len(action_effectiveness) < 5:  # Haven't tried enough actions
            return False
        
        # Check if we've tried different action sequences
        if hasattr(self, 'available_actions_memory'):
            action_sequences = self.available_actions_memory.get('action_sequences', [])
            if len(action_sequences) < 3:  # Haven't tried enough sequences
                return False
        
        return True
    
    def _initialize_pathfinding_system(self):
        """Initialize the pathfinding system for strategy discovery and refinement."""
        if not hasattr(self, 'pathfinding_system'):
            self.pathfinding_system = {
                'winning_strategies': {},  # game_id -> [strategy_objects]
                'strategy_refinement': {},  # strategy_id -> refinement_data
                'path_discovery': {},  # game_id -> discovery_progress
                'strategy_replication': {},  # strategy_id -> replication_attempts
                'efficiency_tracking': {}  # strategy_id -> efficiency_metrics
            }
    
    def _discover_winning_path(self, game_id: str, action_sequence: List[int], score_progression: List[float]) -> Optional[Dict[str, Any]]:
        """Discover and record winning action paths for strategy refinement."""
        self._initialize_pathfinding_system()
        
        # Only record if we achieved a significant score increase
        if len(score_progression) < 3:
            return None
        
        score_increase = score_progression[-1] - score_progression[0]
        if score_increase < 5.0:  # Minimum score increase to consider
            return None
        
        # Create strategy object
        strategy_id = f"{game_id}_strategy_{len(self.pathfinding_system['winning_strategies'].get(game_id, []))}"
        strategy = {
            'id': strategy_id,
            'game_id': game_id,
            'action_sequence': action_sequence.copy(),
            'score_progression': score_progression.copy(),
            'total_score_increase': score_increase,
            'efficiency': score_increase / len(action_sequence),  # Score per action
            'discovery_timestamp': time.time(),
            'replication_attempts': 0,
            'successful_replications': 0,
            'refinement_level': 0
        }
        
        # Store strategy
        if game_id not in self.pathfinding_system['winning_strategies']:
            self.pathfinding_system['winning_strategies'][game_id] = []
        
        self.pathfinding_system['winning_strategies'][game_id].append(strategy)
        
        # Initialize refinement tracking
        self.pathfinding_system['strategy_refinement'][strategy_id] = {
            'original_efficiency': strategy['efficiency'],
            'refinement_attempts': 0,
            'best_efficiency': strategy['efficiency'],
            'refinement_history': []
        }
        
        logger.info(f"🎯 WINNING PATH DISCOVERED: {strategy_id} - {len(action_sequence)} actions, +{score_increase:.1f} score, efficiency: {strategy['efficiency']:.2f}")
        
        return strategy
    
    def _refine_winning_strategy(self, strategy_id: str, new_attempt: Dict[str, Any]) -> Dict[str, Any]:
        """Refine a winning strategy to achieve the same result with fewer actions."""
        if strategy_id not in self.pathfinding_system['strategy_refinement']:
            return {'refined': False, 'reason': 'Strategy not found'}
        
        refinement_data = self.pathfinding_system['strategy_refinement'][strategy_id]
        refinement_data['refinement_attempts'] += 1
        
        # Check if this attempt is more efficient
        new_efficiency = new_attempt.get('efficiency', 0)
        if new_efficiency > refinement_data['best_efficiency']:
            refinement_data['best_efficiency'] = new_efficiency
            refinement_data['refinement_history'].append({
                'attempt': refinement_data['refinement_attempts'],
                'efficiency': new_efficiency,
                'action_sequence': new_attempt.get('action_sequence', []),
                'timestamp': time.time()
            })
            
            logger.info(f"🔧 STRATEGY REFINED: {strategy_id} - New efficiency: {new_efficiency:.2f} (was {refinement_data['original_efficiency']:.2f})")
            
            return {
                'refined': True,
                'new_efficiency': new_efficiency,
                'improvement': new_efficiency - refinement_data['original_efficiency'],
                'refinement_level': len(refinement_data['refinement_history'])
            }
        
        return {'refined': False, 'reason': 'No improvement in efficiency'}
    
    def _get_best_strategy_for_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get the best known strategy for a game based on efficiency."""
        if game_id not in self.pathfinding_system['winning_strategies']:
            return None
        
        strategies = self.pathfinding_system['winning_strategies'][game_id]
        if not strategies:
            return None
        
        # Return the most efficient strategy
        best_strategy = max(strategies, key=lambda s: s['efficiency'])
        return best_strategy
    
    def _should_attempt_strategy_replication(self, game_id: str) -> bool:
        """Determine if we should attempt to replicate a known winning strategy."""
        best_strategy = self._get_best_strategy_for_game(game_id)
        if not best_strategy:
            return False
        
        # Only attempt replication if we have a reasonably efficient strategy
        if best_strategy['efficiency'] < 0.5:  # Less than 0.5 score per action
            return False
        
        # Don't replicate too frequently
        strategy_id = best_strategy['id']
        if strategy_id in self.pathfinding_system['strategy_replication']:
            last_attempt = self.pathfinding_system['strategy_replication'][strategy_id].get('last_attempt', 0)
            if time.time() - last_attempt < 300:  # 5 minutes between attempts
                return False
        
        return True
    
    def _execute_strategy_replication(self, game_id: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a known winning strategy to test its replicability."""
        strategy_id = strategy['id']
        
        # Track replication attempt
        if strategy_id not in self.pathfinding_system['strategy_replication']:
            self.pathfinding_system['strategy_replication'][strategy_id] = {
                'total_attempts': 0,
                'successful_attempts': 0,
                'last_attempt': 0
            }
        
        replication_data = self.pathfinding_system['strategy_replication'][strategy_id]
        replication_data['total_attempts'] += 1
        replication_data['last_attempt'] = time.time()
        
        logger.info(f"🔄 REPLICATING STRATEGY: {strategy_id} - Attempt {replication_data['total_attempts']}")
        
        return {
            'strategy_id': strategy_id,
            'action_sequence': strategy['action_sequence'],
            'expected_efficiency': strategy['efficiency'],
            'replication_attempt': replication_data['total_attempts']
        }

    def _update_progress_tracking(self, action_result: Dict[str, Any], current_action_count: int):
        """
        Update progress tracking based on action results.
        Call this when you detect that an action led to positive progress.
        """
        if action_result.get('progress_made', False):  # Would be set based on score increase, state improvement, etc.
            self.available_actions_memory['action6_strategy']['last_progress_action'] = current_action_count
            print(f" Progress detected at action {current_action_count}")

    # REMOVED: Duplicate start_training_with_direct_control method (simulation version)
    # The real implementation is at line 9747
    
    def _get_state_key(self, observation: Dict[str, Any]) -> str:
        """Convert observation to a state key for Q-table lookup."""
        # Simple state representation - can be enhanced with more sophisticated features
        grid_hash = hash(str(observation.get('grid', [])))
        return f"{grid_hash}"
        
    def _select_action(self, state_key: str, valid_actions: List[int]) -> int:
        """
        Select an action using epsilon-greedy policy with action masking.
        
        Args:
            state_key: String representation of the current state
            valid_actions: List of valid action indices
            
        Returns:
            int: Selected action index
        """
        if not valid_actions:
            raise ValueError("No valid actions provided")
            
        # Ensure exploration rate is within valid range
        exploration_rate = max(0.0, min(1.0, getattr(self, 'exploration_rate', 1.0)))
        
        # Log action selection details for debugging
        logger.debug(f"Action selection - State: {state_key[:30]}... | "
                f"Exploration: {exploration_rate:.2f} | "
                f"Valid actions: {valid_actions}")
        
        # Exploration: random valid action
        if random.random() < exploration_rate:
            action = random.choice(valid_actions)
            logger.debug(f" Exploring: action {action}")
        else:
            # Exploitation: best action from Q-table
            q_values = self.q_table[state_key]
            # Mask invalid actions by setting their Q-values to -inf
            masked_q = [q if i in valid_actions else -float('inf') for i, q in enumerate(q_values)]
            action = np.argmax(masked_q)
            logger.debug(f" Exploiting: action {action} (Q={q_values[action]:.2f})")
            
        # Track action history
        self.action_history.append(action)
        self.action_counts[action] += 1
        self.last_actions.append(action)
        
        return action
    
    def _calculate_reward(self, observation: Dict[str, Any], done: bool, game_state: str) -> float:
        """Calculate shaped reward based on game state and progress."""
        reward = 0.0
        
        # Base reward for continuing
        reward += 0.01  # Small positive reward for each step
        
        # Check for terminal states
        if done:
            if game_state == 'WIN':
                reward += 10.0  # Large reward for winning
                logger.info(" Won the game!")
            elif game_state == 'GAME_OVER':
                reward -= 5.0  # Penalty for losing
                logger.warning(" Game over!")
        
        # Penalize action repetition
        try:
            if len(self.last_actions) > 1 and len(set(self.last_actions)) == 1:
                reward -= 0.5  # Penalty for repeating the same action
                
            state_key = self._get_state_key(observation)
            
            # Get valid actions (simplified - in practice, this would come from the environment)
            valid_actions = list(range(10))  # Assuming 10 possible actions
            
            # Select action using epsilon-greedy policy
            action = self._select_action(state_key, valid_actions)
            
            # Simulate environment step (in a real implementation, this would call the ARC environment)
            # For now, we'll simulate a simple environment
            done = random.random() < 0.01  # 1% chance of episode ending
            game_state = 'WIN' if done and random.random() < 0.3 else 'GAME_OVER' if done else 'IN_PROGRESS'
            
            # Calculate shaped reward
            reward = self._calculate_reward(observation, done, game_state)
            
            # Get next state (in a real implementation, this would be the new observation)
            next_state_key = self._get_state_key(observation)  # Simplified - would be different in practice
            
            # Initialize replay buffer if it doesn't exist
            if not hasattr(self, 'replay_buffer'):
                self.replay_buffer = []
                
            # Store transition in replay buffer
            self.replay_buffer.append((state_key, action, reward, next_state_key, done))
            
            # Update Q-table if it exists
            if hasattr(self, 'q_table'):
                self._update_q_table(state_key, action, reward, next_state_key, done)
            
            # Update exploration rate if needed
            if hasattr(self, 'episode_rewards'):
                self._update_exploration_rate(len(self.episode_rewards))
            
            return {
                'reward': reward,
                'game_state': game_state,
                'done': done,
                'action': action,
                'next_state': next_state_key
            }
            
        except Exception as e:
            logger.error(f"Error in _take_action: {str(e)}", exc_info=True)
            return {
                'reward': -1.0,
                'game_state': 'ERROR',
                'done': True,
                'error': str(e)
            }

    async def _run_real_arc_mastery_session_enhanced(self, game_id: str, session_count: int) -> Dict[str, Any]:
        """
        Enhanced version that runs COMPLETE mastery sessions with up to 100K actions until WIN/GAME_OVER.
        """
        logger.info(f" Starting mastery session {session_count} for game {game_id}")
        start_time = time.time()
        
        try:
            logger.debug("1. Checking for boredom and curriculum advancement")
            # 1. Check for boredom and trigger curriculum advancement
            boredom_results = self._check_and_handle_boredom(session_count)  # Updated parameter name
            
            # 2. Check if agent should sleep before mastery session
            logger.debug("2. Checking if agent should sleep")
            agent_state = self._get_current_agent_state()
            should_sleep_now = self._should_agent_sleep(agent_state, session_count)  # Updated parameter name
            logger.debug(f"Sleep check - should sleep: {should_sleep_now}, agent state: {agent_state}")
            
            sleep_cycle_results = {}
            if should_sleep_now:
                sleep_cycle_results = await self._execute_sleep_cycle(game_id, session_count)  # Updated parameter name
            
            # 3. Check if model decides to reset the game
            logger.debug("3. Evaluating game reset decision")
            reset_decision = self._evaluate_game_reset_decision(game_id, session_count, agent_state)  # Updated parameter name
            logger.debug(f"Reset decision: {reset_decision}")
            
            # 4. Check if contrarian strategy should be activated for consistent GAME_OVER states  
            logger.debug("4. Checking contrarian strategy")
            consecutive_failures = agent_state.get('consecutive_failures', 0)
            logger.debug(f"Consecutive failures: {consecutive_failures}")
            contrarian_decision = self._should_activate_contrarian_strategy(game_id, consecutive_failures)
            self.contrarian_strategy_active = contrarian_decision['activate']
            logger.debug(f"Contrarian strategy - active: {self.contrarian_strategy_active}, decision: {contrarian_decision}")
            
            # 5. Set up environment with API key
            logger.debug("5. Setting up environment with API key")
            env = os.environ.copy()
            if not hasattr(self, 'api_key') or not self.api_key:
                error_msg = "API key not set. Please set the ARC_API_KEY environment variable."
                logger.error(error_msg)
                return {'error': error_msg, 'success': False}
                
            env['ARC_API_KEY'] = self.api_key
            logger.debug("API key set in environment")
            
            # 6. COMPLETE EPISODE: Run actions in loop until terminal state
            logger.debug("6. Starting episode execution")
            episode_actions = 0
            total_score = 0
            best_score = 0
            final_state = 'NOT_FINISHED'
            episode_start_time = time.time()
            logger.debug(f"Episode started at {time.ctime(episode_start_time)}")
            
            # 7. Calculate dynamic session limit based on available games and per-game limit
            available_games = self.games if hasattr(self, 'games') else [game_id]
            max_actions_per_game = getattr(self.current_session, 'max_actions_per_game', ActionLimits.get_max_actions_per_game())
            max_actions_per_session = self._calculate_dynamic_session_limit(max_actions_per_game, available_games)
            logger.debug(f"Dynamic max actions per session: {max_actions_per_session} (based on {len(available_games)} games)")
            
            # 8. Initialize variables to avoid undefined errors
            stdout_text = ""
            stderr_text = ""
            
            logger.info(f" Starting complete mastery session {session_count} for {game_id} (max actions: {max_actions_per_session})")
            
            # Display current win rate-based energy status
            current_win_rate = self._calculate_current_win_rate()
            energy_params = self._calculate_win_rate_adaptive_energy_parameters()
            skill_phase = energy_params['skill_phase']
            actions_until_sleep = int(self.current_energy / energy_params['action_energy_cost'])
            
            print(f" Agent Status: Win Rate {current_win_rate:.1%} | Phase: {skill_phase} | Energy: {self.current_energy:.1f}% | Actions until sleep: ~{actions_until_sleep}")
            print(f" Energy Config: {energy_params['action_energy_cost']:.1f} per action | Sleep at {energy_params['sleep_trigger_threshold']:.0f}% energy")
            
            # Enhanced option: Choose between external main.py and direct control
            use_direct_control = True  # Set to True to use our enhanced action selection
            
            if use_direct_control:
                print(f" Using DIRECT API CONTROL with enhanced action selection")
                # Use our direct API control with intelligent action selection
                try:
                    game_session_result = await self.start_training_with_direct_control(
                        game_id, max_actions_per_session, session_count
                    )
                    
                    if "error" in game_session_result:
                        print(f" Direct control failed: {game_session_result['error']}")
                        print(f" Falling back to external main.py")
                        use_direct_control = False  # Fall back to external method
                    else:
                        # Convert direct control result to expected format
                        total_score = game_session_result.get('final_score', 0)
                        episode_actions = game_session_result.get('total_actions', 0)
                        final_state = game_session_result.get('final_state', 'UNKNOWN')
                        effective_actions = game_session_result.get('effective_actions', [])
                        
                        print(f" Direct Control Results: Score={total_score}, Actions={episode_actions}, State={final_state}")
                        print(f" Effective Actions Found: {len(effective_actions)}")
                except Exception as e:
                    print(f" Direct control exception: {e}")
                    print(f" Falling back to external main.py")
                    use_direct_control = False  # Fall back to external method
            
            if not use_direct_control:
                print(f" Using EXTERNAL main.py (fallback mode)")
                # Original external main.py approach
            
            # VERBOSE: Show memory state before mastery session
            print(f" PRE-SESSION MEMORY STATUS:")  # Updated naming
            pre_memory_status = self._get_memory_consolidation_status()
            pre_sleep_status = self._get_current_sleep_state_info()
            
            # SHOW GLOBAL CUMULATIVE COUNTERS - THE FIX!
            print(f"   Memory Operations: {self.global_counters.get('total_memory_operations', 0)}")
            print(f"   Sleep Cycles: {self.global_counters.get('total_sleep_cycles', 0)}")
            print(f"   Energy Level: {pre_sleep_status.get('current_energy_level', self.current_energy):.2f}")
            
            # VERBOSE: Check existing memory files
            memory_files_before = self._count_memory_files()
            print(f"   Memory Files: {memory_files_before}")
            
            # Adaptive energy allocation based on game history
            current_energy = pre_sleep_status.get('current_energy_level', self.current_energy)
            estimated_complexity = self._estimate_game_complexity(game_id)
            
            # Boost energy if we predict a complex game
            if estimated_complexity == 'high' and current_energy < 80.0:
                energy_boost = 20.0
                current_energy = min(100.0, current_energy + energy_boost)
                print(f" Energy boost: +{energy_boost:.2f} for predicted high-complexity game -> {current_energy:.2f}")
                self._update_energy_level(current_energy)
            elif estimated_complexity == 'medium' and current_energy < 60.0:
                energy_boost = 10.0
                current_energy = min(100.0, current_energy + energy_boost)
                print(f" Energy boost: +{energy_boost:.2f} for predicted medium-complexity game -> {current_energy:.2f}")
                self._update_energy_level(current_energy)
            
            # Reset game at start of episode if needed
            if reset_decision['should_reset']:
                print(f" Resetting game {game_id}")
                self._record_reset_decision(reset_decision)
            
            # Run complete game session (not individual actions)
            print(f" Starting complete game session for {game_id}")
            
            # Build command for complete game session
            cmd = [
                sys.executable, 'main.py',
                '--agent=adaptivelearning',
                f'--game={game_id}'
            ]
            
            # Add reset flag if decision was made to reset
            if reset_decision['should_reset']:
                cmd.append('--reset')
            
            # Apply contrarian strategy if activated
            cmd = self._apply_contrarian_strategy_to_command(cmd, contrarian_decision)
            
            # Dynamic contextual tags for better tracking
            energy_state = "High" if current_energy > 70.0 else "Med" if current_energy > 40.0 else "Low"
            memory_ops = self.global_counters.get('total_memory_operations', 0)
            sleep_cycles = self.global_counters.get('total_sleep_cycles', 0)
            contrarian_mode = "Contrarian" if getattr(self, 'contrarian_strategy_active', False) else "Standard"
            
            dynamic_tag = f"S{session_count}_{energy_state}E_M{memory_ops}_Z{sleep_cycles}_{contrarian_mode}"  # Updated variable name
            cmd.extend(['--tags', dynamic_tag])
            
            # Run the complete game session until WIN/GAME_OVER
            try:
                if use_direct_control:
                    # Direct control was successful - results already processed above
                    pass
                elif self.standalone_mode:
                    # Standalone mode - use internal logic instead of external main.py
                    print(f" Running in standalone mode (no external ARC-AGI-3-Agents)")
                    
                    # Simulate some training progress for testing
                    await asyncio.sleep(2)  # Simulate processing time
                    
                    # Set completion variables for standalone mode
                    total_score = 65.0
                    episode_actions = 15
                    final_state = 'WIN'
                    effective_actions = ['ACTION_1', 'ACTION_6', 'MOVE_CURSOR']
                    stdout_text = f"STANDALONE_MODE: Game {game_id} completed successfully"
                    stderr_text = ""
                    
                    print(f" Standalone Results: Score={total_score}, Actions={episode_actions}, State={final_state}")
                    print(f" Effective Actions Found: {len(effective_actions)}")
                else:
                    # Execute external main.py
                    print(f" Executing complete game session: {' '.join(cmd)}")
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=str(self.arc_agents_path),
                        env=env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    # Wait for complete game session with longer timeout (only for external execution)
                    if not self.standalone_mode:
                        try:
                            stdout, stderr = await asyncio.wait_for(
                                process.communicate(), 
                                timeout=1800.0  # 30 minutes for deeper exploration
                            )
                            
                            stdout_text = stdout.decode('utf-8', errors='ignore') if stdout else ""
                            stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""
                            
                            print(f" Complete game session finished")
                            
                            # Enhanced logging: Extract and show action details from game output
                            self._log_action_details_from_output(stdout_text, game_id)
                            
                            # Parse complete game results
                            game_results = self._parse_complete_game_session(stdout_text, stderr_text)
                            total_score = game_results.get('final_score', 0)
                            episode_actions = game_results.get('total_actions', 0) 
                            final_state = game_results.get('final_state', 'UNKNOWN')
                            effective_actions = game_results.get('effective_actions', [])
                            
                            print(f" Game Results: Score={total_score}, Actions={episode_actions}, State={final_state}")
                            print(f" Effective Actions Found: {len(effective_actions)}")
                            
                        except asyncio.TimeoutError:
                            print(f"⏰ Complete game session timed out after 30 minutes - killing process")
                            try:
                                process.kill()
                                await asyncio.wait_for(process.wait(), timeout=5.0)
                            except:
                                pass
                            
                            # Set timeout defaults
                            total_score = 0
                            episode_actions = 1000  # Assume many actions were attempted
                            final_state = 'TIMEOUT'
                            effective_actions = []
                    # For standalone mode, results were already set above
                    
            except Exception as e:
                print(f" Error during complete game session: {e}")
                # Comprehensive error state with null safety
                total_score = 0
                episode_actions = 0
                final_state = 'ERROR'
                effective_actions = []
                stdout_text = ""
                stderr_text = ""
                
                # Log the error for debugging
                print(f" Game session error details: Game={game_id}, Error={str(e)}")
                
                # Check if this is an API connectivity issue
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    print(" Possible API connectivity issue - validating connection...")
                    api_valid = await self._validate_api_connection()
                    if not api_valid:
                        print(" Consider checking ARC_API_KEY and network connectivity")
            
            # Now process the complete game session results
            # Dynamic energy system based on game complexity and learning opportunities
            
            # WIN RATE-BASED ADAPTIVE ENERGY SYSTEM
            # Calculate energy parameters based on current skill level
            energy_params = self._calculate_win_rate_adaptive_energy_parameters()
            
            print(f" ENERGY PHASE: {energy_params['learning_phase']}")
            print(f"   {energy_params['description']}")
            print(f"   Expected actions before sleep: ~{energy_params['expected_actions_before_sleep']}")
            
            # Calculate energy cost based on win rate-adaptive parameters
            base_action_cost = energy_params.get('action_energy_cost', 4.5)  # Safe default
            if base_action_cost is None:
                base_action_cost = 4.5  # Fallback to beginner intensive level
                print(" Action energy cost was None, using fallback value")
                
            effectiveness_ratio = len(effective_actions) / max(1, episode_actions)
            
            # Apply effectiveness multiplier - penalize ineffective actions more for experienced agents
            if effectiveness_ratio < 0.1:  # Less than 10% effectiveness
                ineffectiveness_penalty = energy_params['effectiveness_multiplier']
                base_action_cost *= ineffectiveness_penalty
                print(f" Ineffectiveness penalty: {ineffectiveness_penalty:.1f}x (effectiveness: {effectiveness_ratio:.1%})")
            else:
                # Reward effective actions with slight energy savings
                effectiveness_bonus = max(0.8, 1.0 - (effectiveness_ratio * 0.2))  # Up to 20% savings
                base_action_cost *= effectiveness_bonus
                print(f" Effectiveness bonus: {effectiveness_bonus:.1f}x (effectiveness: {effectiveness_ratio:.1%})")
            
            # Calculate final energy cost with safety checks
            if base_action_cost is None or episode_actions is None:
                print(" Energy calculation error: base_action_cost or episode_actions is None")
                energy_cost = 0.0  # Safe fallback
                base_action_cost = base_action_cost or 4.5
                episode_actions = episode_actions or 0
            else:
                energy_cost = episode_actions * base_action_cost
                
            current_energy = pre_sleep_status.get('current_energy_level', self.current_energy)
            if current_energy is None:
                current_energy = self.current_energy or 100.0
                
            remaining_energy = max(0.0, current_energy - energy_cost)
            
            print(f" Energy: {current_energy:.2f} -> {remaining_energy:.2f}")
            print(f"   Cost: {energy_cost:.3f} (base: {base_action_cost:.3f} × {episode_actions} actions)")
            print(f"   Game complexity: {episode_actions} actions -> {'Low' if episode_actions <= 21 else 'Medium' if episode_actions <= 100 else 'High'}")
            
            # Use win rate-based sleep threshold
            sleep_threshold = energy_params['sleep_trigger_threshold']
            
            # Determine sleep reason
            if len(effective_actions) > 0:
                sleep_reason = f"Learning consolidation ({len(effective_actions)} effective actions) - {energy_params['learning_phase']}"
            else:
                sleep_reason = f"Energy depletion (no effective actions) - {energy_params['learning_phase']}"
            
            # Trigger sleep based on adaptive thresholds - use current energy from per-action management  
            sleep_triggered = False
            if self.current_energy <= sleep_threshold:
                skill_phase = energy_params['skill_phase']
                print(f" Sleep triggered: {sleep_reason}")
                print(f"   Energy {self.current_energy:.2f} <= threshold {sleep_threshold:.2f} | Skill Phase: {skill_phase}")
                print(f"   Win Rate: {self._calculate_current_win_rate():.1%} | Action Cost: {energy_params['action_energy_cost']:.1f}")
                
                sleep_result = await self._trigger_sleep_cycle(effective_actions)
                print(f" Sleep completed: {sleep_result}")
                sleep_triggered = True
                
                # Adaptive energy replenishment based on learning quality
                base_replenishment = 60.0  # Base 60 energy points restoration (0-100 scale)
                
                # Bonus energy for effective learning
                if len(effective_actions) > 0:
                    learning_bonus = min(30.0, len(effective_actions) * 5.0)  # Up to 30 energy points bonus
                    print(f" Learning bonus: +{learning_bonus:.2f} energy for {len(effective_actions)} effective actions")
                else:
                    learning_bonus = 0.0
                
                # Bonus energy for complex games (they teach more)
                if episode_actions > 500:
                    complexity_bonus = 20.0  # 20 energy bonus for complex games
                    print(f" Complexity bonus: +{complexity_bonus:.2f} energy for {episode_actions}-action game")
                elif episode_actions > 200:
                    complexity_bonus = 10.0  # 10 energy bonus for medium games
                    print(f" Complexity bonus: +{complexity_bonus:.2f} energy for {episode_actions}-action game")
                else:
                    complexity_bonus = 0.0
                
                total_replenishment = base_replenishment + learning_bonus + complexity_bonus
                # Ensure current_energy is not None before arithmetic operations
                if self.current_energy is None:
                    print(" Warning: current_energy was None, resetting to 100.0")
                    self.current_energy = 100.0
                self.current_energy = min(100.0, self.current_energy + total_replenishment)
                skill_phase = energy_params['skill_phase']
                actions_until_next_sleep = int(self.current_energy / energy_params['action_energy_cost'])
                
                print(f" Energy replenished: {total_replenishment:.2f} total -> {self.current_energy:.2f}")
                print(f" Ready for ~{actions_until_next_sleep} actions until next sleep cycle ({skill_phase} phase)")
            else:
                print(f" Sleep not needed: Energy {self.current_energy:.2f} > threshold {energy_params['sleep_trigger_threshold']:.2f} ({energy_params['skill_phase']} phase)")
            
            # Update energy level in system - use current energy from per-action depletion
            # The per-action energy system has already managed energy during gameplay
            final_energy = max(0.0, self.current_energy)  # Use energy from per-action management
            print(f" Energy level updated to {final_energy:.2f}")
            self._update_energy_level(final_energy)
            
            # Update game complexity history for future energy allocation
            # Ensure we have valid values to prevent NoneType errors
            episode_actions = episode_actions if episode_actions is not None else 0
            effective_actions = effective_actions if effective_actions is not None else []
            effectiveness_ratio = min(1.0, len(effective_actions) / max(1, episode_actions))  # Cap at 100%
            self._update_game_complexity_history(game_id, episode_actions, effectiveness_ratio)
            print(f" Updated complexity history for {game_id}: {episode_actions} actions, {effectiveness_ratio:.2%} effective")
            
            # Build comprehensive episode result
            episode_duration = time.time() - episode_start_time
            
            # Extract grid dimensions from final output
            grid_dims = self._extract_grid_dimensions_from_output(stdout_text, stderr_text)
            
            result = {
                'success': final_state == 'WIN',
                'final_score': total_score,
                'best_score': best_score,
                'game_state': final_state,
                'game_id': game_id,
                'session': session_count,  # Updated key and variable name
                'actions_taken': episode_actions,
                'episode_duration': episode_duration,
                'timestamp': time.time(),
                'sleep_cycle_executed': bool(sleep_cycle_results),
                'sleep_cycle_results': sleep_cycle_results,
                'reset_decision': reset_decision,
                'memory_consolidation_status': self._get_memory_consolidation_status(),
                'sleep_state_info': self._get_current_sleep_state_info(),
                # ENHANCED: Action sequence tracking for continuous learning
                'effective_actions': effective_actions,
                'action_sequences': self._analyze_action_sequences(effective_actions),
                'mid_game_consolidations': self.training_state.get('mid_game_consolidations', 0),
                'success_weighted_memories': len([a for a in effective_actions if a.get('success', False)]),
                'continuous_learning_metrics': {
                    'consolidation_points': max(1, episode_actions // 150),
                    'memory_strength_ops': len([a for a in effective_actions if a.get('effectiveness', 0) > 0.2]),
                    'pattern_discoveries': len(effective_actions),
                    'learning_velocity': len(effective_actions) / max(1, episode_duration / 60)  # Patterns per minute
                }
            }
            
            if grid_dims:
                result['grid_dimensions'] = grid_dims
            
            # Evaluate reset effectiveness if reset was used
            if reset_decision['should_reset']:
                reset_effectiveness = self._evaluate_reset_effectiveness(reset_decision, result)
                result['reset_effectiveness'] = reset_effectiveness
            
            # Record action sequence outcome and save intelligence
            if self.available_actions_memory['current_game_id'] == game_id:
                sequence = self.available_actions_memory['sequence_in_progress']
                success = result.get('success', False)
                self._record_sequence_outcome(sequence, success)
                
                # Save action intelligence for this game
                self._save_game_action_intelligence(game_id)
                
                # Log intelligence summary
                intel_summary = self._get_action_intelligence_summary(game_id)
                logger.info(f" Action Intelligence for {game_id}: "
                        f"{intel_summary.get('effective_actions', 0)}/{intel_summary.get('total_actions_learned', 0)} effective actions, "
                        f"{intel_summary.get('coordinate_patterns_learned', 0)} coordinate patterns learned")
                
            return result
            
        except Exception as e:
            return {'success': False, 'final_score': 0, 'error': str(e), 'game_id': game_id}
    
    def _count_memory_files(self) -> int:
        """Count memory and checkpoint files for verbose monitoring."""
        try:
            memory_paths = [
                Path("checkpoints"),
                Path("data/meta_learning_data"),
                Path("data"),
                Path("test_meta_learning_data")
            ]
            
            total_files = 0
            for path in memory_paths:
                if path.exists():
                    total_files += len(list(path.rglob("*")))
            
            return total_files
        except Exception:
            return 0
    
    async def run_swarm_mode(
        self,
        games: List[str],
        max_concurrent: int = 3,
        max_episodes_per_game: int = 20
    ) -> Dict[str, Any]:
        """Run multiple games concurrently for faster learning (SWARM mode)."""
        print(f"\n SWARM MODE ACTIVATED")
        print(f"Concurrent Games: {min(max_concurrent, len(games))}")
        print(f"Total Games: {len(games)}")
        print("="*60)
        
        #  CRITICAL FIX: Set swarm mode flag to enable per-game scorecard isolation
        self._swarm_mode_active = True
        print(f" [SWARM-FIX] Swarm mode activated for per-game scorecard isolation")
        
        swarm_results = {
            'mode': 'swarm',
            'total_games': len(games),
            'max_concurrent': max_concurrent,
            'games_completed': {},
            'overall_performance': {},
            'swarm_efficiency': {}
        }
        
        # Split games into batches for concurrent processing
        game_batches = [games[i:i+max_concurrent] for i in range(0, len(games), max_concurrent)]
        
        total_start_time = time.time()
        
        for batch_idx, batch in enumerate(game_batches, 1):
            print(f"\n SWARM BATCH {batch_idx}/{len(game_batches)}")
            print(f"Games: {', '.join(batch)}")
            
            # Run games in this batch concurrently
            batch_tasks = []
            for game_id in batch:
                task = asyncio.create_task(
                    self._train_on_game_swarm(game_id, max_episodes_per_game)
                )
                batch_tasks.append(task)
            
            # Wait for all games in batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for game_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in swarm game {game_id}: {result}")
                    swarm_results['games_completed'][game_id] = {
                        'error': str(result),
                        'success': False
                    }
                else:
                    swarm_results['games_completed'][game_id] = result
                    win_rate = result.get('final_performance', {}).get('win_rate', 0)
                    grid_size = result.get('final_performance', {}).get('grid_size', 'unknown')
                    print(f" {game_id}: {win_rate:.1%} win rate | Grid: {grid_size}")
        
        total_duration = time.time() - total_start_time
        
        # Calculate swarm performance metrics
        successful_games = [r for r in swarm_results['games_completed'].values() if not r.get('error')]
        
        if successful_games:
            total_episodes = sum(len(game.get('episodes', [])) for game in successful_games)
            total_wins = sum(
                sum(1 for ep in game.get('episodes', []) if ep.get('success', False))
                for game in successful_games
            )
            
            swarm_results['overall_performance'] = {
                'games_completed': len(successful_games),
                'total_episodes': total_episodes,
                'overall_win_rate': total_wins / max(1, total_episodes),
                'games_per_hour': len(successful_games) / (total_duration / 3600),
                'episodes_per_hour': total_episodes / (total_duration / 3600)
            }
            
            swarm_results['swarm_efficiency'] = {
                'total_duration_hours': total_duration / 3600,
                'concurrent_speedup': len(games) / (total_duration / 3600) if total_duration > 0 else 0,
                'batch_count': len(game_batches),
                'average_batch_size': len(games) / len(game_batches)
            }
        
        print(f"\n SWARM MODE COMPLETE")
        print(f"Duration: {total_duration/60:.1f} minutes")
        print(f"Games/Hour: {swarm_results['overall_performance'].get('games_per_hour', 0):.1f}")
        print(f"Overall Win Rate: {swarm_results['overall_performance'].get('overall_win_rate', 0):.1%}")
        
        # Show rate limiting statistics
        self.print_rate_limit_status()
        
        #  CRITICAL FIX: Cleanup swarm mode flag
        self._swarm_mode_active = False
        print(f" [SWARM-FIX] Swarm mode deactivated, scorecard isolation disabled")
        
        # Reset game completion tracking
        if hasattr(self, '_game_completed'):
            self._game_completed = False
        
        return swarm_results
    
    def _mark_game_completed(self, game_id: str) -> None:
        """Mark a game as completed for proper scorecard lifecycle management."""
        self._game_completed = True
        print(f" [GAME-COMPLETE] Game {game_id} marked as completed")
    
    def _reset_game_completion(self) -> None:
        """Reset game completion flag when starting a new game."""
        self._game_completed = False
    
    def _determine_action_success(self, response_data: Dict[str, Any]) -> bool:
        """Determine if an action was successful based on response data."""
        try:
            # Check for explicit success indicators
            if 'reward' in response_data:
                return response_data['reward'] > 0
            elif 'score' in response_data:
                return response_data['score'] > 0
            elif 'done' in response_data:
                return response_data['done']
            elif 'success' in response_data:
                return response_data['success']
            
            # Check for state improvements
            if 'new_state' in response_data:
                new_state = response_data['new_state']
                if isinstance(new_state, str):
                    return new_state.upper() in ['WIN', 'SUCCESS', 'COMPLETE']
            
            # Default to False if no clear success indicator
            return False
            
        except Exception as e:
            logger.warning(f"Error determining action success: {e}")
            return False
    
    def get_enhanced_simulation_status(self) -> Dict[str, Any]:
        """Get status of the enhanced simulation system."""
        if not self.simulation_agent:
            return {'enabled': False, 'message': 'Enhanced simulation agent not available'}
        
        try:
            return {
                'enabled': True,
                'imagination_status': self.simulation_agent.get_imagination_status(),
                'simulation_statistics': self.simulation_agent.get_simulation_statistics(),
                'method_recommendations': self.simulation_agent.get_method_recommendations()
            }
        except Exception as e:
            return {'enabled': True, 'error': str(e)}
    
    def set_learning_mode(self, mode: str) -> bool:
        """Set the learning mode for the enhanced simulation system."""
        if not self.simulation_agent:
            return False
        
        try:
            from src.core.enhanced_simulation_config import LearningMode
            learning_mode = LearningMode(mode.lower())
            self.simulation_agent.enable_learning_mode(learning_mode)
            return True
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to set learning mode {mode}: {e}")
            return False
    
    def start_ab_test(self, test_name: str, method_a: str, method_b: str) -> Optional[str]:
        """Start an A/B test between two simulation methods."""
        if not self.simulation_agent:
            return None
        
        try:
            from src.core.path_generator import SearchMethod
            method_a_enum = SearchMethod(method_a.lower())
            method_b_enum = SearchMethod(method_b.lower())
            return self.simulation_agent.start_ab_test(test_name, method_a_enum, method_b_enum)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to start A/B test: {e}")
            return None
    
    def _learn_from_action_outcome(self, game_id: str, action_number: int, x: Optional[int], y: Optional[int], 
                                 response_data: Dict[str, Any], before_state: Dict[str, Any], 
                                 after_state: Dict[str, Any]):
        """Learn patterns from action outcomes and store them for future use."""
        if not self.governor or not hasattr(self.governor, 'learning_manager') or not self.governor.learning_manager:
            return
        
        try:
            # Calculate success metrics
            score_change = after_state.get('score', 0) - before_state.get('score', 0)
            game_state = response_data.get('state', 'UNKNOWN')
            success = game_state in ['WIN', 'LEVEL_WIN', 'FULL_GAME_WIN'] or score_change > 0
            
            # Create context for pattern learning
            context = {
                'game_id': game_id,
                'action_number': action_number,
                'coordinates': (x, y) if x is not None and y is not None else None,
                'grid_size': (response_data.get('grid_width', 64), response_data.get('grid_height', 64)),
                'game_state': game_state,
                'score_change': score_change,
                'available_actions': response_data.get('available_actions', [])
            }
            
            # Create pattern data
            pattern_data = {
                'action_type': f'ACTION{action_number}',
                'coordinates': (x, y) if x is not None and y is not None else None,
                'success': success,
                'score_change': score_change,
                'game_state': game_state,
                'context_hash': hash(str(sorted(context.items())))
            }
            
            # Learn the pattern
            from src.core.cross_session_learning import KnowledgeType, PersistenceLevel
            
            # Determine persistence level based on success
            persistence_level = PersistenceLevel.PERMANENT if success and score_change > 5 else PersistenceLevel.SESSION
            
            pattern_id = self.governor.learning_manager.learn_pattern(
                KnowledgeType.ACTION_PATTERN,
                pattern_data,
                context,
                success_rate=1.0 if success else 0.0,
                persistence_level=persistence_level
            )
            
            if pattern_id:
                logger.debug(f"🧠 Learned action pattern {pattern_id}: ACTION{action_number} {'successful' if success else 'unsuccessful'}")
                
                # Also learn coordinate patterns for ACTION6
                if action_number == 6 and x is not None and y is not None:
                    coord_pattern_data = {
                        'action_type': 'ACTION6_COORDINATE',
                        'x': x,
                        'y': y,
                        'success': success,
                        'score_change': score_change,
                        'grid_size': context['grid_size']
                    }
                    
                    coord_pattern_id = self.governor.learning_manager.learn_pattern(
                        KnowledgeType.SPATIAL_PATTERN,
                        coord_pattern_data,
                        context,
                        success_rate=1.0 if success else 0.0,
                        persistence_level=persistence_level
                    )
                    
                    if coord_pattern_id:
                        logger.debug(f"🧠 Learned coordinate pattern {coord_pattern_id}: ({x},{y}) {'successful' if success else 'unsuccessful'}")
            
        except Exception as e:
            logger.error(f"Error learning from action outcome: {e}")

    def _get_pattern_recommendations(self, game_id: str, context: Dict[str, Any], available_actions: List[int]) -> Dict[int, float]:
        """Get pattern-based recommendations for action selection."""
        if not self.governor or not hasattr(self.governor, 'learning_manager') or not self.governor.learning_manager:
            return {}
        
        try:
            from src.core.cross_session_learning import KnowledgeType
            
            # Create context for pattern retrieval
            pattern_context = {
                'game_id': game_id,
                'grid_size': context.get('response_data', {}).get('grid_width', 64),
                'available_actions': available_actions,
                'frame_analysis': context.get('frame_analysis', {})
            }
            
            # Get action patterns
            action_patterns = self.governor.learning_manager.retrieve_applicable_patterns(
                KnowledgeType.ACTION_PATTERN,
                pattern_context,
                min_confidence=0.3,
                max_results=10
            )
            
            # Get spatial patterns for ACTION6
            spatial_patterns = self.governor.learning_manager.retrieve_applicable_patterns(
                KnowledgeType.SPATIAL_PATTERN,
                pattern_context,
                min_confidence=0.3,
                max_results=5
            )
            
            recommendations = {}
            
            # Process action patterns
            for pattern in action_patterns:
                if pattern.pattern_data.get('success', False):
                    action_type = pattern.pattern_data.get('action_type', '')
                    if action_type.startswith('ACTION'):
                        try:
                            action_num = int(action_type.replace('ACTION', ''))
                            if action_num in available_actions:
                                # Boost based on pattern confidence and success rate
                                boost = pattern.confidence * pattern.success_rate * 0.5
                                recommendations[action_num] = recommendations.get(action_num, 0) + boost
                        except ValueError:
                            continue
            
            # Process spatial patterns for ACTION6
            if 6 in available_actions:
                for pattern in spatial_patterns:
                    if pattern.pattern_data.get('success', False):
                        # Boost ACTION6 based on spatial pattern success
                        boost = pattern.confidence * pattern.success_rate * 0.3
                        recommendations[6] = recommendations.get(6, 0) + boost
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting pattern recommendations: {e}")
            return {}

    def _update_action_probabilities(self):
        """Dynamically update action probabilities based on success rates and recent usage."""
        if not hasattr(self, 'available_actions_memory') or 'action_effectiveness' not in self.available_actions_memory:
            return

    async def _train_on_game_swarm(
        self,
        game_id: str,
        max_episodes: int
    ) -> Dict[str, Any]:
        """Train on a single game in swarm mode with concurrency."""
        game_results = {
            'game_id': game_id,
            'episodes': [],
            'performance_metrics': {},
            'final_performance': {},
            'grid_dimensions': (64, 64),  # Will be updated dynamically
            'error_count': 0,
            'error_messages': []
        }
        
        try:
            # Initialize action tracking
            if not hasattr(self, 'available_actions_memory'):
                self.available_actions_memory = {
                    'action_effectiveness': {
                        i: {
                            'successes': 0, 
                            'attempts': 0,
                            'last_used': 0,  # Track when this action was last used
                            'success_rate': 0.0  # Track success rate explicitly
                        } for i in range(1, 8)
                    },
                    'last_actions': [],
                    'action_history': [],
                    'game_context': {},
                    'total_episodes': 0,
                    'consecutive_failures': 0,
                    'last_successful_action': None
                }
            
            episode_count = 1
            
            while episode_count <= max_episodes:
                try:
                    # Run the enhanced mastery session with detailed logging
                    logger.info(f" Starting episode {episode_count}/{max_episodes} for {game_id}")
                    
                    # Reset game completion flag for new episode
                    if episode_count == 1:
                        self._reset_game_completion()
                    
                    episode_result = await self._run_real_arc_mastery_session_enhanced(
                        game_id, episode_count
                    )
                    
                    # Process the episode results
                    if episode_result and isinstance(episode_result, dict) and 'error' not in episode_result:
                        # Add to results
                        game_results['episodes'].append(episode_result)
                        
                        # Update performance metrics
                        if hasattr(self, '_update_performance_metrics'):
                            self._update_performance_metrics(game_results, episode_result)
                        
                        # Update action tracking
                        if hasattr(self, '_update_action_tracking'):
                            self._update_action_tracking(episode_result)
                        
                        logger.info(f" Episode {episode_count} completed successfully for {game_id}")
                    else:
                        # Log the error with proper type checking
                        if isinstance(episode_result, dict):
                            error_msg = episode_result.get('error', 'Unknown error')
                        elif isinstance(episode_result, str):
                            error_msg = f"Episode result was string instead of dict: {episode_result}"
                        else:
                            error_msg = f"Episode result was unexpected type {type(episode_result)}: {episode_result}"
                        
                        logger.error(f" Episode {episode_count} failed for {game_id}: {error_msg}")
                        
                        # Track errors
                        game_results['error_count'] += 1
                        game_results['error_messages'].append(f"Episode {episode_count}: {error_msg}")
                        
                except asyncio.TimeoutError as e:
                    error_msg = f"Episode {episode_count} timed out after 5 minutes: {str(e)}"
                    logger.error(f"⏱ {error_msg}", exc_info=True)
                    
                    # Track errors
                    game_results['error_count'] += 1
                    game_results['error_messages'].append(error_msg)
                    
                except Exception as e:
                    error_msg = f"Unexpected error in episode {episode_count}: {str(e)}"
                    logger.error(f" {error_msg}", exc_info=True)
                    
                    # Track errors
                    game_results['error_count'] += 1
                    game_results['error_messages'].append(error_msg)
                        
                # Increment episode counter
                episode_count += 1
                
                # Small delay between episodes to prevent overwhelming the system
                await asyncio.sleep(1.0)
            
            # Calculate final performance metrics if the method exists
            if hasattr(self, '_calculate_final_metrics'):
                self._calculate_final_metrics(game_results)
            
            # Log final results for this game
            if game_results['episodes']:
                last_episode = game_results['episodes'][-1]
                logger.info(f" Completed training on {game_id} - Final Score: {last_episode.get('final_score', 0)} | "
                        f"Episodes: {len(game_results['episodes'])} | Errors: {game_results['error_count']}")
                
                # Mark game as completed for proper scorecard lifecycle management
                self._mark_game_completed(game_id)
            else:
                logger.warning(f" No successful episodes for {game_id} - {game_results['error_count']} errors")
            
            return game_results
            
        except Exception as e:
            error_msg = f"Critical error in _train_on_game_swarm for game {game_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Add error to results
            game_results['error'] = str(e)
            game_results['traceback'] = traceback.format_exc()
            game_results['success'] = False
            
            # Log detailed error information
            logger.error(f" Training failed for {game_id} after {len(game_results.get('episodes', []))} episodes")
            if game_results.get('error_messages'):
                logger.error(f"Previous errors: {', '.join(game_results['error_messages'][-5:])}")
                
            return game_results

    async def run_continuous_learning(self, session_id: str) -> Dict[str, Any]:
        """
        Run the continuous learning loop for a session with SWARM mode support.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session results with ARC-3 scoreboard URL and win highlights
        """
        if not self.current_session or self.current_session.session_id != session_id:
            raise ValueError(f"No active session with ID {session_id}")
            
        session = self.current_session
        session_results = {
            'session_id': session_id,
            'start_time': time.time(),
            'games_played': {},
            'overall_performance': {},
            'learning_insights': [],
            'knowledge_transfers': [],
            'detailed_metrics': {
                'salience_mode': session.salience_mode.value,
                'sleep_cycles': 0,
                'memory_operations': 0,
                'high_salience_experiences': 0,
                'compressed_memories': 0,
                'grid_sizes_encountered': set()
            },
            'arc3_scoreboard_url': ARC3_SCOREBOARD_URL,
            'win_highlighted': False,
            # 🧠 Meta-Cognitive Systems Monitoring
            'governor_decisions': [],
            'architect_evolutions': [],
            'meta_cognitive_active': self.governor is not None and self.architect is not None
        }
        
        logger.info(f"Running continuous learning for session {session_id}")
        
        # Start cross-session learning session
        if self.governor and hasattr(self.governor, 'learning_manager') and self.governor.learning_manager:
            session_context = {
                'session_id': session_id,
                'games_to_play': session.games_to_play,
                'max_mastery_sessions_per_game': session.max_mastery_sessions_per_game,
                'target_performance': session.target_performance,
                'salience_mode': session.salience_mode.value,
                'swarm_enabled': session.swarm_enabled
            }
            self.learning_session_id = self.governor.start_learning_session(session_context)
            logger.info(f"🧠 Started cross-session learning: {self.learning_session_id}")
        
        try:
            # Check if SWARM mode should be used based on configuration
            if session.swarm_enabled and len(session.games_to_play) > 2:
                print(f"\n SWARM MODE ENABLED for {len(session.games_to_play)} games")
                swarm_results = await self.run_swarm_mode(
                    session.games_to_play,
                    max_concurrent=3,
                    max_episodes_per_game=session.max_mastery_sessions_per_game  # Fixed parameter name
                )
                
                # Convert SWARM results to session format
                for game_id, game_result in swarm_results['games_completed'].items():
                    session_results['games_played'][game_id] = game_result
                    
                    # Track grid dimensions encountered
                    if 'grid_dimensions' in game_result:
                        grid_dims = game_result['grid_dimensions']
                        session_results['detailed_metrics']['grid_sizes_encountered'].add(f"{grid_dims[0]}x{grid_dims[1]}")
                
                # Use SWARM performance metrics
                session_results['swarm_mode_used'] = True
                session_results['swarm_results'] = swarm_results
                
            else:
                # Use sequential training for smaller sessions
                print(f"\n SEQUENTIAL MODE for {len(session.games_to_play)} games")
                session_results['swarm_mode_used'] = False
                
                # CONTINUOUS GAME CYCLING - Keep playing games until 6 hours is up
                total_games = len(session.games_to_play)
                game_cycle_count = 0
                current_game_index = 0
                
                while True:
                    # Check if we've exceeded the session duration (6 hours)
                    elapsed_time = time.time() - session_results['start_time']
                    if elapsed_time >= (session.session_duration * 60):  # Convert minutes to seconds
                        print(f"\n⏰ SESSION TIME COMPLETE: {elapsed_time/3600:.1f} hours elapsed")
                        break
                    
                    # Select next game (cycle through available games)
                    game_id = session.games_to_play[current_game_index]
                    game_cycle_count += 1
                    
                    print(f"\n🔄 GAME CYCLE {game_cycle_count} - Game: {game_id} (Index: {current_game_index + 1}/{total_games})")
                    print(f"⏱️  Time remaining: {(session.session_duration * 60 - elapsed_time)/60:.1f} minutes")
                    
                    game_results = await self._train_on_game(
                        game_id,
                        session.max_mastery_sessions_per_game,  # Updated attribute name
                        session.target_performance
                    )
                    
                    # Check if game ended and we should continue
                    game_state = game_results.get('final_performance', {}).get('final_state', 'UNKNOWN')
                    if game_state == 'GAME_OVER':
                        print(f"🎮 Game {game_id} ended with GAME_OVER - continuing to next game")
                        # Don't break, just continue to next game
                    elif game_state == 'FULL_GAME_WIN':
                        print(f"🏆 Game {game_id} ended with FULL GAME WIN - continuing to next game")
                        # Don't break, just continue to next game
                    elif game_state == 'LEVEL_WIN':
                        print(f"🎯 Game {game_id} ended with LEVEL WIN - continuing to next game")
                        # Don't break, just continue to next game
                    elif game_state == 'WIN':
                        print(f"✅ Game {game_id} ended with WIN - continuing to next game")
                        # Don't break, just continue to next game
                    
                    session_results['games_played'][game_id] = game_results
                    
                    # Track grid dimensions
                    if 'grid_dimensions' in game_results:
                        grid_dims = game_results['grid_dimensions']
                        session_results['detailed_metrics']['grid_sizes_encountered'].add(f"{grid_dims[0]}x{grid_dims[1]}")
                    
                    # Update detailed metrics
                    self._update_detailed_metrics(session_results['detailed_metrics'], game_results)
                    
                    # Display compact completion status
                    self._display_game_completion_status(game_id, game_results, session_results['detailed_metrics'])
                    
                    # Apply insights (background processing)
                    await self._apply_learning_insights(game_id, game_results)
                    
                    # Move to next game (cycle back to start if we reach the end)
                    current_game_index = (current_game_index + 1) % total_games
                    
            # Finalize session
            session_results['end_time'] = time.time()
            session_results['duration'] = session_results['end_time'] - session_results['start_time']
            session_results['overall_performance'] = self._calculate_session_performance(session_results)
            
            # Add continuous cycling information
            if 'game_cycle_count' in locals():
                session_results['continuous_cycling'] = {
                    'total_game_cycles': game_cycle_count,
                    'games_per_hour': game_cycle_count / (session_results['duration'] / 3600),
                    'cycling_mode': 'continuous_6hour'
                }
            
            # Add grid size summary
            grid_sizes = list(session_results['detailed_metrics']['grid_sizes_encountered'])
            session_results['grid_sizes_summary'] = {
                'unique_sizes': len(grid_sizes),
                'sizes_encountered': grid_sizes,
                'dynamic_sizing_verified': len(grid_sizes) > 1
            }
            
            # Generate final insights (top 3 only)
            final_insights = self._generate_session_insights(session_results)
            session_results['learning_insights'].extend(final_insights[:3])
            
            # Check if this qualifies as a winning session
            overall_win_rate = session_results['overall_performance'].get('overall_win_rate', 0)
            session_results['win_highlighted'] = overall_win_rate > 0.3
            
            # Display compact final results with ARC-3 URL
            self._display_final_session_results(session_results)
            
            # Update global metrics and save (background)
            self._update_global_metrics(session_results)
            self._save_session_results(session_results)
            
            # End cross-session learning session
            if self.governor and hasattr(self.governor, 'learning_manager') and self.governor.learning_manager and self.learning_session_id:
                performance_summary = {
                    'total_games': len(session_results['games_played']),
                    'overall_win_rate': overall_win_rate,
                    'total_actions': sum(game.get('total_actions', 0) for game in session_results['games_played'].values()),
                    'learning_insights': len(session_results.get('learning_insights', [])),
                    'session_duration': time.time() - session_results['start_time']
                }
                self.governor.end_learning_session(performance_summary)
                logger.info(f"🧠 Ended cross-session learning: {self.learning_session_id}")
                self.learning_session_id = None
            
            logger.info(f"Completed training session {session_id} - Win rate: {overall_win_rate:.1%}")
            return session_results
            
        except Exception as e:
            logger.error(f"Error in continuous learning session: {e}")
            session_results['error'] = str(e)
            session_results['end_time'] = time.time()
            return session_results
        finally:
            # Clean up any active HTTP sessions
            await self.cleanup_sessions()

    def _get_memory_consolidation_status(self) -> Dict[str, Any]:
        """Return the current status of memory consolidation."""
        # Initialize trackers if they don't exist
        if not hasattr(self, 'memory_consolidation_tracker'):
            self.memory_consolidation_tracker = {
                'memory_operations_per_cycle': 0,
                'consolidation_operations_count': 0,
                'high_salience_memories_strengthened': 0,
                'low_salience_memories_decayed': 0,
                'memory_compression_active': False,
                'total_memory_operations': 0
            }
        return self.memory_consolidation_tracker

    def _get_current_sleep_state_info(self) -> Dict[str, Any]:
        """Return the current sleep state information."""
        if not hasattr(self, 'sleep_state_tracker'):
            self.sleep_state_tracker = {
                'sleep_cycles_this_session': 0,
                'current_energy_level': 100.0,
                'last_sleep_cycle': 0,
                'total_sleep_cycles': 0,
                'last_sleep_trigger': [],
                'sleep_quality_scores': [],
                'is_currently_sleeping': False,
                'total_sleep_time': 0.0
            }
        return self.sleep_state_tracker

    def _count_memory_files(self) -> int:
        """Count the number of memory files in the memory directory."""
        memory_dir = os.path.join(os.path.dirname(__file__), '..', 'memory')
        if not os.path.exists(memory_dir):
            return 0
        return len([f for f in os.listdir(memory_dir) if f.endswith('.pkl')])
        
    def _estimate_game_complexity(self, game_id: str) -> str:
        """Estimate the complexity of a game based on its ID and previous performance.
        
        Args:
            game_id: The ID of the game to estimate complexity for
            
        Returns:
            str: 'low', 'medium', or 'high' complexity
        """
        # Check if we have a cached complexity for this game
        for complexity, games in self.game_complexity.items():
            if game_id in games:
                return complexity
                
        # Default to medium complexity for unknown games
        return 'medium'

    def _update_detailed_metrics(self, detailed_metrics: Dict[str, Any], game_results: Dict[str, Any]):
        """Update detailed metrics with game results including sleep and memory operations."""
        # Initialize trackers if they don't exist
        if not hasattr(self, 'memory_consolidation_tracker'):
            self._get_memory_consolidation_status()
        if not hasattr(self, 'sleep_state_tracker'):
            self._get_current_sleep_state_info()
            
        # Count sleep cycles (simulated)
        episodes_count = len(game_results.get('episodes', []))
        # Ensure sleep_state_tracker is initialized
        if not hasattr(self, 'sleep_state_tracker'):
            self.sleep_state_tracker = {
                'is_currently_sleeping': False,
                'sleep_cycles_this_session': 0,
                'total_sleep_time': 0.0,
                'memory_operations_per_cycle': 0,
                'last_sleep_trigger': [],
                'sleep_quality_scores': [],
                'current_energy_level': 100.0,
                'last_sleep_cycle': 0,
                'total_sleep_cycles': 0
            }
        detailed_metrics['sleep_cycles'] = self.sleep_state_tracker.get('sleep_cycles_this_session', 0)
        
        # Count high salience experiences
        detailed_metrics['high_salience_experiences'] = detailed_metrics.get('high_salience_experiences', 0)
        for episode in game_results.get('episodes', []):
            if episode.get('final_score', 0) > 75:  # High score = high salience
                detailed_metrics['high_salience_experiences'] += 1
        
        # Memory operations tracking
        mem_tracker = self.memory_consolidation_tracker
        detailed_metrics['memory_operations'] = mem_tracker.get('memory_operations_per_cycle', 0)
        detailed_metrics['consolidation_operations'] = mem_tracker.get('consolidation_operations_count', 0)
        detailed_metrics['memories_strengthened'] = mem_tracker.get('high_salience_memories_strengthened', 0)
        detailed_metrics['memories_decayed'] = mem_tracker.get('low_salience_memories_decayed', 0)
        
        # Initialize other metrics if they don't exist
        detailed_metrics.setdefault('compressed_memories', 0)
        detailed_metrics.setdefault('compression_active', False)
        
        # Update global counters
        if not hasattr(self, 'global_counters'):
            self.global_counters = {
                'total_memory_operations': 0,
                'total_sleep_cycles': 0
            }
        
        # Reset decision tracking
        detailed_metrics['reset_decisions'] = self.game_reset_tracker['reset_decisions_made']
        detailed_metrics['reset_success_rate'] = self.game_reset_tracker['reset_success_rate']

    def _display_game_completion_status(self, game_id: str, game_results: Dict[str, Any], detailed_metrics: Dict[str, Any]):
        """Display compact game completion status with sleep and memory info."""
        performance = game_results.get('performance_metrics', {})
        win_rate = performance.get('win_rate', 0)
        avg_score = performance.get('average_score', 0)
        best_score = performance.get('best_score', 0)
        grid_size = game_results.get('final_performance', {}).get('final_grid_size', 'unknown')
        
        # Highlight wins
        if win_rate > 0.5:
            status = "WINNER"
        elif win_rate > 0.3:
            status = "STRONG"
        elif win_rate > 0.1:
            status = "LEARNING"
        else:
            status = "TRAINING"
            
        print(f"\n{status}: {game_id} | Grid: {grid_size}")
        print(f"Episodes: {performance.get('total_episodes', 0)} | Win: {win_rate:.1%} | Avg: {avg_score:.0f} | Best: {best_score}")
        
        # Enhanced system metrics with sleep and memory info
        sleep_cycles = detailed_metrics.get('sleep_cycles', 0)
        memory_ops = detailed_metrics.get('memory_operations', 0)
        high_sal = detailed_metrics.get('high_salience_experiences', 0)
        reset_decisions = self.game_reset_tracker['reset_decisions_made']
        
        print(f"Sleep: {sleep_cycles} | Memory Ops: {memory_ops} | High-Sal: {high_sal} | Resets: {reset_decisions}")
        
        # Show current system status flags
        status_flags = self.get_system_status_flags()
        active_systems = []
        if status_flags.get('is_consolidating_memories', False):
            active_systems.append("CONSOLIDATING")
        if status_flags.get('is_prioritizing_memories', False):
            active_systems.append("PRIORITIZING")
        if status_flags.get('memory_compression_active', False):
            active_systems.append("COMPRESSING")
        if status_flags.get('is_sleeping', False):
            active_systems.append("SLEEPING")
            
        if active_systems:
            print(f"Active: {' | '.join(active_systems)}")

    def _display_final_session_results(self, session_results: Dict[str, Any]):
        """Display compact final session results with ARC-3 URL and grid size info."""
        overall_perf = session_results.get('overall_performance', {})
        detailed_metrics = session_results.get('detailed_metrics', {})
        grid_summary = session_results.get('grid_sizes_summary', {})
        
        # Determine overall performance status
        overall_win_rate = overall_perf.get('overall_win_rate', 0)
        if overall_win_rate > 0.5:
            status = "CHAMPIONSHIP PERFORMANCE"
        elif overall_win_rate > 0.3:
            status = "STRONG PERFORMANCE"  
        elif overall_win_rate > 0.1:
            status = "LEARNING PROGRESS"
        else:
            status = "TRAINING COMPLETE"
            
        print(f"\n{'='*60}")
        print(f"{status}")
        print(f"{'='*60}")
        
        # Essential metrics only
        print(f"Duration: {session_results.get('duration', 0):.0f}s | Games: {overall_perf.get('games_trained', 0)} | Episodes: {overall_perf.get('total_episodes', 0)}")
        print(f"Win Rate: {overall_win_rate:.1%} | Avg Score: {overall_perf.get('overall_average_score', 0):.0f}")
        
        # Continuous cycling information
        continuous_info = session_results.get('continuous_cycling', {})
        if continuous_info:
            print(f"🔄 CONTINUOUS CYCLING: {continuous_info.get('total_game_cycles', 0)} total cycles | {continuous_info.get('games_per_hour', 0):.1f} games/hour")
        
        # Grid size information - CRITICAL NEW FEATURE
        if grid_summary.get('dynamic_sizing_verified'):
            print(f" Dynamic Grid Sizing: {grid_summary.get('unique_sizes', 0)} different sizes detected")
            print(f"   Sizes: {', '.join(grid_summary.get('sizes_encountered', []))}")
        else:
            sizes = grid_summary.get('sizes_encountered', ['64x64'])
            print(f"Grid Sizes: {', '.join(sizes)}")
        
        # SWARM mode info if used
        if session_results.get('swarm_mode_used'):
            swarm_info = session_results.get('swarm_results', {}).get('swarm_efficiency', {})
            print(f" SWARM Mode: {swarm_info.get('games_per_hour', 0):.1f} games/hour | Concurrent speedup enabled")
        
        print(f"Learning Efficiency: {overall_perf.get('learning_efficiency', 0):.2f} | Knowledge Transfer: {overall_perf.get('knowledge_transfer_score', 0):.2f}")
        
        # System performance (compact)
        print(f"Sleep: {detailed_metrics.get('sleep_cycles', 0)} | Memory: {detailed_metrics.get('memory_operations', 0)} | High-Sal: {detailed_metrics.get('high_salience_experiences', 0)}")
        
        if detailed_metrics.get('salience_mode') == 'decay_compression':
            compression_ratio = detailed_metrics.get('compressed_memories', 0) / max(1, detailed_metrics.get('memory_operations', 1))
            print(f"Compression: {compression_ratio:.0%}")
        
        # Highlight top insights (max 3)
        insights = session_results.get('learning_insights', [])
        if insights:
            print(f"\nTop Insights:")
            for i, insight in enumerate(insights[:3], 1):
                print(f"  {i}. {insight.get('content', 'No content')}")
        
        # Always show ARC-3 scoreboard URL
        print(f"\nARC-3 Test Scoreboard: {ARC3_SCOREBOARD_URL}")
        
        # Show comprehensive system status
        system_status = self.get_system_status_flags()
        memory_status = self.get_sleep_and_memory_status()
        
        print(f"\n SYSTEM STATUS:")
        # Ensure sleep_state_tracker is initialized
        if not hasattr(self, 'sleep_state_tracker'):
            self.sleep_state_tracker = {
                'is_currently_sleeping': False,
                'sleep_cycles_this_session': 0,
                'total_sleep_time': 0.0,
                'memory_operations_per_cycle': 0,
                'last_sleep_trigger': [],
                'sleep_quality_scores': [],
                'current_energy_level': 100.0,
                'last_sleep_cycle': 0,
                'total_sleep_cycles': 0
            }
        sleep_cycles = memory_status.get('sleep_status', {}).get('sleep_cycles_this_session', 0)
        consolidating = system_status.get('is_consolidating_memories', False)
        prioritizing = system_status.get('is_prioritizing_memories', False)
        compression = system_status.get('memory_compression_active', False)
        
        print(f"Sleep Cycles: {sleep_cycles} | Memory Consolidations: {consolidating}")
        print(f"Memory Prioritization: {prioritizing} | Compression: {compression}")
        
        if system_status.get('has_made_reset_decisions', False):
            reset_stats = memory_status.get('game_reset_status', {})
            total_resets = reset_stats.get('total_reset_decisions', 0)
            success_rate = reset_stats.get('reset_success_rate', 0.0)
            last_reason = reset_stats.get('last_reset_reason')
            
            print(f" Game Resets: {total_resets} decisions | Success Rate: {success_rate:.1%}")
            if last_reason:
                print(f"   Last Reset: {last_reason}")
        
        # Highlight if this was a winning session
        if overall_win_rate > 0.3:
            print(" SUBMIT TO LEADERBOARD - STRONG PERFORMANCE DETECTED!")
        
        # Display memory hierarchy status
        self._display_memory_hierarchy_status()
        
        print("="*60)

    async def _apply_learning_insights(self, game_id: str, game_results: Dict[str, Any]):
        """Apply learning insights to improve future performance."""
        # This would involve updating the agent's configuration based on learned patterns
        insights = self.arc_meta_learning.get_strategic_recommendations(game_id)
        
        if insights:
            logger.info(f"Applying {len(insights)} insights for {game_id}")
            # In a full implementation, you would modify agent parameters here
            
    def _calculate_game_performance(self, game_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for a game."""
        episodes = game_results.get('episodes', [])
        if not episodes:
            return {}
            
        return {
            'total_episodes': len(episodes),
            'win_rate': sum(1 for ep in episodes if ep.get('success', False)) / len(episodes),
            'average_score': sum((ep.get('final_score') or 0) for ep in episodes) / len(episodes),
            'average_actions': sum(ep.get('actions_taken', 0) for ep in episodes) / len(episodes),
            'best_score': max((ep.get('final_score') or 0) for ep in episodes),
            'patterns_per_episode': len(game_results.get('patterns_discovered', [])) / len(episodes)
        }
        
    def _calculate_session_performance(self, session_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall session performance."""
        games_played = session_results['games_played']
        if not games_played:
            return {}
            
        total_episodes = sum(len(game.get('episodes', [])) for game in games_played.values())
        total_wins = sum(sum(1 for ep in game.get('episodes', []) if ep.get('success', False)) 
                        for game in games_played.values())
        # Enhanced null safety for score calculations
        total_score = 0
        for game in games_played.values():
            episodes = game.get('episodes', []) if game else []
            for ep in episodes:
                if ep:
                    score = ep.get('final_score')
                    if score is not None and isinstance(score, (int, float)):
                        total_score += score
        
        learning_efficiency = self._calculate_learning_efficiency(session_results)
        knowledge_transfer = self._calculate_knowledge_transfer_score(session_results)
        
        return {
            'games_trained': len(games_played),
            'total_episodes': total_episodes,
            'overall_win_rate': total_wins / max(1, total_episodes),
            'overall_average_score': total_score / max(1, total_episodes),
            'learning_efficiency': learning_efficiency or 0.0,
            'knowledge_transfer_score': knowledge_transfer or 0.0
        }
        
    def _calculate_learning_efficiency(self, session_results: Dict[str, Any]) -> float:
        """Calculate how efficiently the agent learned during the session."""
        # Simplified efficiency metric based on improvement over time
        games_played = session_results['games_played']
        efficiency_scores = []
        
        for game_results in games_played.values():
            episodes = game_results.get('episodes', [])
            if len(episodes) >= 10:
                # Enhanced null safety for performance calculations
                early_scores = []
                late_scores = []
                
                for ep in episodes[:5]:
                    if ep:
                        score = ep.get('final_score')
                        if score is not None and isinstance(score, (int, float)):
                            early_scores.append(score)
                
                for ep in episodes[-5:]:
                    if ep:
                        score = ep.get('final_score')
                        if score is not None and isinstance(score, (int, float)):
                            late_scores.append(score)
                
                early_performance = sum(early_scores) / max(len(early_scores), 1)
                late_performance = sum(late_scores) / max(len(late_scores), 1)
                improvement = late_performance - early_performance
                efficiency_scores.append(max(0, improvement / 100))  # Normalize
                
        return sum(efficiency_scores) / max(1, len(efficiency_scores))
        
    def _calculate_knowledge_transfer_score(self, session_results: Dict[str, Any]) -> float:
        """Calculate how well knowledge transferred between games."""
        # Simplified metric based on performance on later games vs earlier games
        games_played = list(session_results['games_played'].values())
        if len(games_played) < 2:
            return 0.0
            
        early_games = games_played[:len(games_played)//2]
        late_games = games_played[len(games_played)//2:]
        
        # Enhanced null safety for performance metrics
        def safe_score(game):
            if not game or not isinstance(game, dict):
                return 0
            metrics = game.get('performance_metrics')
            if not metrics or not isinstance(metrics, dict):
                return 0
            score = metrics.get('average_score')
            return score if isinstance(score, (int, float)) else 0
        
        early_avg = sum(safe_score(game) for game in early_games) / max(1, len(early_games))
        late_avg = sum(safe_score(game) for game in late_games) / max(1, len(late_games))
        
        return max(0, (late_avg - early_avg) / 100)  # Normalize
        
    def _generate_session_insights(self, session_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from the completed session."""
        insights = []
        
        # Performance trend insight
        overall_perf = session_results['overall_performance']
        if overall_perf.get('learning_efficiency', 0) > 0.3:
            insights.append({
                'type': 'positive_learning_trend',
                'content': f"Agent showed good learning efficiency: {overall_perf['learning_efficiency']:.2f}",
                'confidence': 0.8
            })
            
        # Knowledge transfer insight
        if overall_perf.get('knowledge_transfer_score', 0) > 0.2:
            insights.append({
                'type': 'knowledge_transfer',
                'content': f"Successful knowledge transfer between games: {overall_perf['knowledge_transfer_score']:.2f}",
                'confidence': 0.7
            })

        # Grid size insight
        grid_summary = session_results.get('grid_sizes_summary', {})
        if grid_summary.get('dynamic_sizing_verified'):
            insights.append({
                'type': 'dynamic_grid_adaptation',
                'content': f"Successfully adapted to {grid_summary.get('unique_sizes', 0)} different grid sizes",
                'confidence': 0.9
            })
            
        return insights
        
    def _update_global_metrics(self, session_results: Dict[str, Any]):
        """Update global performance metrics."""
        overall_perf = session_results.get('overall_performance', {})
        
        # Update cumulative metrics with null safety
        self.global_performance_metrics['total_games_played'] = (self.global_performance_metrics.get('total_games_played') or 0) + (overall_perf.get('games_trained') or 0)
        self.global_performance_metrics['total_episodes'] = (self.global_performance_metrics.get('total_episodes') or 0) + (overall_perf.get('total_episodes') or 0)
        
        # Update averages (simplified) with comprehensive null protection
        current_avg_score = self.global_performance_metrics.get('average_score') or 0.0
        new_avg_score = overall_perf.get('overall_average_score') or 0.0
        self.global_performance_metrics['average_score'] = (
            (current_avg_score or 0.0) * 0.8 + (new_avg_score or 0.0) * 0.2
        )
        
        current_win_rate = self.global_performance_metrics.get('win_rate') or 0.0
        new_win_rate = overall_perf.get('overall_win_rate') or 0.0
        self.global_performance_metrics['win_rate'] = (
            (current_win_rate or 0.0) * 0.8 + (new_win_rate or 0.0) * 0.2
        )
        
    def _save_session_results(self, session_results: Dict[str, Any]):
        """Save final session results."""
        # Default to data/sessions for session results to centralize artifacts
        session_dir = Path("data") / "sessions"
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        filename = session_dir / f"session_{session_results['session_id']}_final.json"
        try:
            # Convert grid_sizes_encountered set to list for JSON serialization
            if 'detailed_metrics' in session_results and 'grid_sizes_encountered' in session_results['detailed_metrics']:
                session_results['detailed_metrics']['grid_sizes_encountered'] = list(session_results['detailed_metrics']['grid_sizes_encountered'])
            
            with open(filename, 'w') as f:
                json.dump(session_results, f, indent=2, default=str)
            
            # Also save meta-learning state under data/meta_learning_sessions
            meta_learning_dir = Path("data") / "meta_learning_sessions"
            try:
                meta_learning_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            meta_learning_file = meta_learning_dir / f"meta_learning_{session_results['session_id']}.json"
            self.arc_meta_learning.save_learning_state(str(meta_learning_file))
            
        except Exception as e:
            logger.error(f"Failed to save session results: {e}")

    def _load_game_action_intelligence(self, game_id: str) -> Dict[str, Any]:
        """Load learned action patterns for this specific game."""
        intelligence_file = self.save_directory / f"action_intelligence_{game_id}.json"
        
        if intelligence_file.exists():
            try:
                with open(intelligence_file, 'r') as f:
                    intelligence = json.load(f)
                logger.info(f" Loaded action intelligence for {game_id}: {len(intelligence.get('effective_actions', {}))} effective actions")
                return intelligence
            except Exception as e:
                logger.warning(f"Failed to load action intelligence for {game_id}: {e}")
                
        return {
            'game_id': game_id,
            'initial_actions': [],
            'effective_actions': {},
            'winning_sequences': [],
            'coordinate_patterns': {},
            'action_transitions': {},
            'last_updated': 0
        }
    
    def _save_game_action_intelligence(self, game_id: str):
        """Save learned action patterns for this specific game."""
        intelligence_file = self.save_directory / f"action_intelligence_{game_id}.json"
        
        # Calculate effectiveness rates
        effective_actions = {}
        for action, data in self.available_actions_memory['action_effectiveness'].items():
            if data['attempts'] > 0:
                success_rate = data['successes'] / data['attempts']
                if success_rate > 0.1:  # Only save actions with some success
                    effective_actions[action] = {
                        'success_rate': success_rate,
                        'attempts': data['attempts'],
                        'successes': data['successes']
                    }
        
        # Process coordinate patterns
        coordinate_patterns = {}
        for action, coords in self.available_actions_memory['coordinate_patterns'].items():
            coordinate_patterns[action] = {}
            for (x, y), data in coords.items():
                if data['attempts'] > 0:
                    success_rate = data['successes'] / data['attempts']
                    coordinate_patterns[action][f"{x},{y}"] = {
                        'success_rate': success_rate,
                        'attempts': data['attempts'],
                        'successes': data['successes']
                    }
        
        game_intelligence = {
            'game_id': game_id,
            'initial_actions': self.available_actions_memory['initial_actions'],
            'effective_actions': effective_actions,
            'winning_sequences': self.available_actions_memory['winning_action_sequences'][-20:],  # Keep best 20
            'coordinate_patterns': coordinate_patterns,
            'action_transitions': dict(self.available_actions_memory['action_transitions']),
            'total_sessions_learned': len(self.available_actions_memory.get('action_history', [])),
            'last_updated': time.time()
        }
        
        try:
            with open(intelligence_file, 'w') as f:
                json.dump(game_intelligence, f, indent=2)
            logger.info(f" Saved action intelligence for {game_id}: {len(effective_actions)} effective actions")
        except Exception as e:
            logger.error(f"Failed to save action intelligence for {game_id}: {e}")

    def display_action_intelligence_summary(self, game_id: Optional[str] = None):
        """Display a summary of learned action intelligence."""
        print(f"\n ACTION INTELLIGENCE SUMMARY")
        print("="*50)
        
        learning_stats = self.available_actions_memory.get('action_learning_stats', {})
        print(f" Global Learning Stats:")
        print(f"   Total Observations: {learning_stats.get('total_observations', 0)}")
        print(f"   Movements Tracked: {learning_stats.get('movements_tracked', 0)}")
        print(f"   Effects Catalogued: {learning_stats.get('effects_catalogued', 0)}")
        print(f"   Game Contexts Learned: {learning_stats.get('game_contexts_learned', 0)}")
        
        print(f"\n ACTION MAPPINGS:")
        for action, mapping in self.available_actions_memory['action_semantic_mapping'].items():
            print(f"\n  ACTION{action}:")
            print(f"    Description: {self.get_action_description(action, game_id)}")
            
            # Show movement patterns
            movements = mapping['grid_movement_patterns']
            if movements and any(movements.values()):
                dominant_move = max(movements.items(), key=lambda x: x[1])
                if dominant_move[1] > 0:
                    print(f"    Dominant Movement: {dominant_move[0]} ({dominant_move[1]} observations)")
            
            # Show common effects
            effects = mapping['common_effects']
            if effects:
                top_effects = sorted(effects.items(), key=lambda x: x[1], reverse=True)[:3]
                effects_str = ", ".join([f"{effect} ({count})" for effect, count in top_effects])
                print(f"    Common Effects: {effects_str}")
            
            # Show game-specific roles
            if game_id and game_id in mapping['game_specific_roles']:
                role_info = mapping['game_specific_roles'][game_id]
                print(f"    Role in {game_id}: {role_info['role']} (confidence: {role_info['confidence']:.2f})")
            
            # Show coordinate success zones for ACTION6
            if action == 6 and 'coordinate_success_zones' in mapping and mapping['coordinate_success_zones']:
                best_zones = []
                for zone_key, zone_data in mapping['coordinate_success_zones'].items():
                    if zone_data['attempts'] > 1:
                        success_rate = zone_data['successes'] / zone_data['attempts']
                        if success_rate > 0.5:
                            best_zones.append(f"Zone{zone_key}: {success_rate:.1%}")
                
                if best_zones:
                    print(f"    Best Coordinate Zones: {', '.join(best_zones[:3])}")
        
        print("="*50)

    def _update_available_actions(self, response_data: Dict[str, Any], game_id: str, action_taken: Optional[int] = None):
        """Update available actions based on API response metadata."""
        available_actions = response_data.get('available_actions', [])
        
        # Initialize for new game
        if self.available_actions_memory['current_game_id'] != game_id:
            self._initialize_game_actions(game_id, available_actions)
        
        # Track action transitions if we took an action
        if action_taken is not None and self.available_actions_memory['current_actions']:
            self._record_action_transition(action_taken, available_actions)
        
        # Update current available actions
        prev_actions = set(self.available_actions_memory['current_actions'])
        self.available_actions_memory['current_actions'] = available_actions
        
        # Track action sequence in progress
        if action_taken is not None:
            self.available_actions_memory['sequence_in_progress'].append(action_taken)
        
        # Learn from action effectiveness
        if action_taken is not None:
            self._analyze_action_effectiveness(action_taken, response_data)
        
        logger.debug(f" Available actions updated for {game_id}: {available_actions}")
    
    def _initialize_game_actions(self, game_id: str, initial_actions: List[int]):
        """Initialize action memory for a new game."""
        # Ensure available_actions_memory is properly initialized
        if not hasattr(self, 'available_actions_memory'):
            self.available_actions_memory = {
                'current_game_id': None,
                'action_history': [],
                'effectiveness_tracking': {},
                'coordinate_patterns': {},
                'winning_action_sequences': [],
                'failed_action_patterns': [],
                'game_intelligence_cache': {},
                'last_action_result': None,
                'sequence_in_progress': [],
                'session_start_time': time.time(),
                'total_actions_taken': 0
            }
        
        # Load cached intelligence if available
        # Ensure game_intelligence_cache key exists
        if 'game_intelligence_cache' not in self.available_actions_memory:
            self.available_actions_memory['game_intelligence_cache'] = {}
        
        if game_id not in self.available_actions_memory['game_intelligence_cache']:
            try:
                self.available_actions_memory['game_intelligence_cache'][game_id] = self._load_game_action_intelligence(game_id)
            except Exception as e:
                # If loading fails, use empty intelligence
                self.available_actions_memory['game_intelligence_cache'][game_id] = {}
        
        # Reset current session data but preserve learned intelligence
        cached_intel = self.available_actions_memory.get('game_intelligence_cache', {}).get(game_id, {})
        
        self.available_actions_memory.update({
            'current_game_id': game_id,
            'initial_actions': initial_actions,
            'current_actions': initial_actions,
            'action_history': [],
            'sequence_in_progress': [],
            'last_action_result': None
        })
        
        # Initialize effectiveness tracking with cached data
        self.available_actions_memory['action_effectiveness'] = {}
        for action in initial_actions:
            if str(action) in cached_intel.get('effective_actions', {}):
                cached_data = cached_intel['effective_actions'][str(action)]
                self.available_actions_memory['action_effectiveness'][action] = {
                    'attempts': cached_data.get('attempts', 0),
                    'successes': cached_data.get('successes', 0),
                    'success_rate': cached_data.get('success_rate', 0.0)
                }
            else:
                self.available_actions_memory['action_effectiveness'][action] = {
                    'attempts': 0,
                    'successes': 0,
                    'success_rate': 0.0
                }
        
        # Load coordinate patterns
        self.available_actions_memory['coordinate_patterns'] = {}
        for action in initial_actions:
            self.available_actions_memory['coordinate_patterns'][action] = {}
            if str(action) in cached_intel.get('coordinate_patterns', {}):
                cached_coords = cached_intel['coordinate_patterns'][str(action)]
                for coord_str, data in cached_coords.items():
                    x, y = map(int, coord_str.split(','))
                    self.available_actions_memory['coordinate_patterns'][action][(x, y)] = {
                        'attempts': data.get('attempts', 0),
                        'successes': data.get('successes', 0),
                        'success_rate': data.get('success_rate', 0.0)
                    }
        
        # Load winning sequences and transitions
        self.available_actions_memory['winning_action_sequences'] = cached_intel.get('winning_sequences', [])
        self.available_actions_memory['action_transitions'] = cached_intel.get('action_transitions', {})
        
        logger.info(f" Initialized actions for {game_id}: {initial_actions} (loaded {len(cached_intel.get('effective_actions', {}))} cached patterns)")

    def _record_action_transition(self, action_taken: int, new_available_actions: List[int]):
        """Record transition from one action to available next actions."""
        prev_state = tuple(sorted(self.available_actions_memory['current_actions']))
        new_state = tuple(sorted(new_available_actions))
        transition_key = (prev_state, action_taken, new_state)
        
        self.available_actions_memory['action_transitions'][str(transition_key)] = \
            self.available_actions_memory['action_transitions'].get(str(transition_key), 0) + 1

    def _analyze_action_effectiveness(self, action_taken: int, response_data: Dict[str, Any]):
        """Analyze if the action was effective based on response."""
        if action_taken not in self.available_actions_memory['action_effectiveness']:
            self.available_actions_memory['action_effectiveness'][action_taken] = {
                'attempts': 0, 'successes': 0, 'success_rate': 0.0
            }
        
        # Increment attempts
        self.available_actions_memory['action_effectiveness'][action_taken]['attempts'] += 1
        
        # Determine if action was successful based on multiple indicators
        success_indicators = [
            response_data.get('state') == 'WIN',
            response_data.get('score', 0) > 0,
            len(response_data.get('available_actions', [])) > 0,  # Game didn't end
            'error' not in response_data
        ]
        
        # Action is successful if it meets multiple criteria
        is_successful = sum(success_indicators) >= 2
        
        if is_successful:
            self.available_actions_memory['action_effectiveness'][action_taken]['successes'] += 1
        
        # Recalculate success rate
        attempts = self.available_actions_memory['action_effectiveness'][action_taken]['attempts']
        successes = self.available_actions_memory['action_effectiveness'][action_taken]['successes']
        self.available_actions_memory['action_effectiveness'][action_taken]['success_rate'] = successes / attempts if attempts > 0 else 0.0
        
        # Store result for sequence analysis
        self.available_actions_memory['last_action_result'] = {
            'action': action_taken,
            'success': is_successful,
            'response': response_data
        }
        
        # NEW: Learn action semantics from grid behavior
        self._learn_action_semantics(action_taken, response_data, is_successful)

    def _learn_action_semantics(self, action: int, response_data: Dict[str, Any], success: bool):
        """Learn what each action actually does based on grid behavior and effects."""
        # Ensure action mapping exists
        if action not in self.available_actions_memory['action_semantic_mapping']:
            return
        
        mapping = self.available_actions_memory['action_semantic_mapping'][action]
        game_id = self.available_actions_memory.get('current_game_id')
        
        # Update learning statistics
        if 'action_learning_stats' not in self.available_actions_memory:
            self.available_actions_memory['action_learning_stats'] = {
                'total_observations': 0,
                'pattern_confidence_threshold': 0.7,
                'movements_tracked': 0,
                'effects_catalogued': 0,
                'game_contexts_learned': 0
            }
        self.available_actions_memory['action_learning_stats']['total_observations'] += 1
        
        # Analyze grid movement patterns (for actions 1-5, 7)
        if action != 6:  # Non-coordinate actions
            movement = self._detect_grid_movement(response_data)
            if movement:
                if movement in mapping['grid_movement_patterns']:
                    mapping['grid_movement_patterns'][movement] += 1
                    self.available_actions_memory.get('action_learning_stats', {}).setdefault('movements_tracked', 0)
                    self.available_actions_memory['action_learning_stats']['movements_tracked'] += 1
                    print(f" ACTION{action} learned movement: {movement} (total: {mapping['grid_movement_patterns'][movement]})")
        else:
            # For ACTION6, track coordinate-based behavior
            if 'coordinate_based' in mapping['grid_movement_patterns']:
                mapping['grid_movement_patterns']['coordinate_based'] += 1
        
        # Analyze action effects
        effects = self._detect_action_effects(action, response_data, success)
        for effect in effects:
            if effect in mapping['common_effects']:
                mapping['common_effects'][effect] += 1
            else:
                mapping['common_effects'][effect] = 1
                self.available_actions_memory.get('action_learning_stats', {}).setdefault('effects_catalogued', 0)
                self.available_actions_memory['action_learning_stats']['effects_catalogued'] += 1
                print(f" ACTION{action} learned new effect: {effect}")
        
        # Learn game-specific roles
        if game_id and success:
            role = self._infer_action_role(action, response_data, effects)
            if role:
                if game_id not in mapping['game_specific_roles']:
                    mapping['game_specific_roles'][game_id] = {'role': role, 'confidence': 0.1}
                    self.available_actions_memory.get('action_learning_stats', {}).setdefault('game_contexts_learned', 0)
                    self.available_actions_memory['action_learning_stats']['game_contexts_learned'] += 1
                    print(f" ACTION{action} role in {game_id}: {role}")
                else:
                    # Increase confidence if consistent
                    current_role = mapping['game_specific_roles'][game_id]
                    if current_role['role'] == role:
                        current_role['confidence'] = min(1.0, current_role['confidence'] + 0.1)
                    else:
                        # Conflicting role - reduce confidence and potentially update
                        current_role['confidence'] = max(0.0, current_role['confidence'] - 0.05)
                        if current_role['confidence'] < 0.3:
                            current_role['role'] = role
                            current_role['confidence'] = 0.1
        
        # For ACTION6, learn coordinate success zones
        if action == 6 and 'coordinates' in response_data:
            x, y = response_data['coordinates']
            zone_key = (x // 16, y // 16)  # Divide grid into 4x4 zones
            
            if 'coordinate_success_zones' not in mapping:
                mapping['coordinate_success_zones'] = {}
            
            if zone_key not in mapping['coordinate_success_zones']:
                mapping['coordinate_success_zones'][zone_key] = {'attempts': 0, 'successes': 0}
            
            zone_data = mapping['coordinate_success_zones'][zone_key]
            zone_data['attempts'] += 1
            if success:
                zone_data['successes'] += 1

    def _detect_grid_movement(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Detect if an action caused movement on the grid."""
        # Analyze frame data changes to detect movement
        frame = response_data.get('frame', [])
        
        # Simple heuristics for movement detection
        # In real implementation, would compare before/after frames
        
        # For now, use state changes and score improvements as proxy
        if response_data.get('score', 0) > 0:
            # Score improved - likely positive action
            state = response_data.get('state', 'NOT_FINISHED')
            if state == 'WIN':
                return 'success_move'
            else:
                return 'progress_move'
        elif len(response_data.get('available_actions', [])) < 4:
            # Fewer actions available - might have reached boundary
            return 'boundary_hit'
        else:
            return 'none'

    def _detect_action_effects(self, action: int, response_data: Dict[str, Any], success: bool) -> List[str]:
        """Detect what effects an action had."""
        effects = []
        
        # Basic effect detection based on response
        if success:
            effects.append('successful_action')
            
        if response_data.get('state') == 'WIN':
            effects.append('game_winning')
            effects.append('terminal_action')
        elif response_data.get('state') == 'GAME_OVER':
            effects.append('game_ending')
            effects.append('terminal_action')
        
        if response_data.get('score', 0) > 0:
            effects.append('score_increase')
        
        available_actions = response_data.get('available_actions', [])
        if len(available_actions) == 0:
            effects.append('no_actions_remaining')
        elif len(available_actions) < 4:
            effects.append('limited_actions')
        
        # Action-specific effect detection
        if action == 7:  # Undo action
            effects.append('undo_attempted')
            if success:
                effects.append('undo_successful')
        elif action == 5:  # Interaction action
            effects.append('interaction_attempted')
            if success:
                effects.append('interaction_successful')
        
        return effects

    def _infer_action_role(self, action: int, response_data: Dict[str, Any], effects: List[str]) -> Optional[str]:
        """Infer the role/purpose of an action in the current game context."""
        if 'game_winning' in effects:
            return 'victory_action'
        elif 'game_ending' in effects and 'successful_action' not in effects:
            return 'failure_action'
        elif 'score_increase' in effects:
            return 'progress_action'
        elif 'interaction_successful' in effects:
            return 'interaction_action'
        elif 'undo_successful' in effects:
            return 'undo_action'
        elif action in [1, 2, 3, 4] and 'progress_move' in effects:
            return 'movement_action'
        elif action == 6 and 'successful_action' in effects:
            return 'coordinate_action'
        else:
            return 'exploration_action'

    def get_action_description(self, action: int, game_id: Optional[str] = None) -> str:
        """Get intelligent description of what an action does based on learned behavior."""
        if action not in self.available_actions_memory['action_semantic_mapping']:
            return f"ACTION{action} - Unknown"
        
        mapping = self.available_actions_memory['action_semantic_mapping'][action]
        
        # Start with default description
        description = mapping['default_description']
        
        # Add learned behaviors if confident enough
        confidence_threshold = self.available_actions_memory.get('action_learning_stats', {}).get('pattern_confidence_threshold', 0.7)
        
        # Add movement pattern info
        movements = mapping['grid_movement_patterns']
        if movements and any(movements.values()):
            dominant_movement = max(movements.items(), key=lambda x: x[1])
            if dominant_movement[1] > 5:  # At least 5 observations
                description += f" | Learned: {dominant_movement[0]} movement"
        
        # Add game-specific role if available
        if game_id and game_id in mapping['game_specific_roles']:
            role_info = mapping['game_specific_roles'][game_id]
            if role_info['confidence'] > confidence_threshold:
                description += f" | Role in {game_id}: {role_info['role']}"
        
        # Add common effects
        effects = mapping['common_effects']
        if effects and any(effects.values()):
            top_effect = max(effects.items(), key=lambda x: x[1])
            if top_effect[1] > 3:  # At least 3 observations
                description += f" | Common effect: {top_effect[0]}"
        
        return description

    def _select_intelligent_action(self, available_actions: List[int], game_context: Dict[str, Any] = None) -> int:
        """Select action based on learned effectiveness and patterns with enhanced strategy refinement."""
        if not available_actions:
            return 6  # Fallback to action 6
        
        # Filter out invalid actions (never select unavailable actions)
        valid_actions = [a for a in available_actions if a in available_actions]
        if not valid_actions:
            return available_actions[0]
        
        # Get effectiveness data and recent performance context
        action_scores = {}
        total_actions_taken = self.global_counters.get('total_actions', 0)
        recent_failures = self.global_counters.get('recent_consecutive_failures', 0)
        
        for action in valid_actions:
            effectiveness = self.available_actions_memory['action_effectiveness'].get(action, {'success_rate': 0.0, 'attempts': 0})
            base_score = effectiveness['success_rate']
            
            # Enhanced scoring factors
            
            # 1. Bonus for actions that appear in winning sequences
            sequence_bonus = 0.0
            for seq in self.available_actions_memory['winning_action_sequences'][-5:]:  # Recent wins
                if action in seq:
                    sequence_bonus += 0.3  # Increased from 0.2
            
            # 2. Penalty for actions in failed patterns  
            pattern_penalty = 0.0
            current_sequence = self.available_actions_memory['sequence_in_progress'][-3:]  # Last 3 actions
            for failed_pattern in self.available_actions_memory['failed_action_patterns'][-5:]:
                if current_sequence + [action] == failed_pattern:
                    pattern_penalty = 0.4  # Reduced from 0.5 to be less restrictive
                    break
            
            # 3. NEW: Diversity bonus - encourage trying underutilized but available actions
            diversity_bonus = 0.0
            if effectiveness['attempts'] < 3:  # Haven't tried this action much
                diversity_bonus = 0.15
            
            # 4. NEW: Context-aware scoring based on game state
            context_bonus = 0.0
            if game_context and 'game_id' in game_context:
                game_id = game_context['game_id']
                # Load game-specific intelligence if available
                game_intel = self._load_game_action_intelligence(game_id)
                if game_intel and action in game_intel.get('effective_actions', {}):
                    context_bonus = 0.2
            
            # 5. NEW: Anti-stagnation bonus - if we've had many recent failures, try different actions
            stagnation_bonus = 0.0
            if recent_failures > 5:
                # Prefer actions we haven't used recently
                recent_actions = self.available_actions_memory['sequence_in_progress'][-10:]
                if action not in recent_actions:
                    stagnation_bonus = 0.25
            
            # 6. CRITICAL: Anti-repetition penalty - heavily penalize recently overused actions
            repetition_penalty = 0.0
            if hasattr(self, '_last_selected_actions'):
                recent_actions = self._last_selected_actions[-8:]  # Last 8 actions
                if len(recent_actions) >= 5:
                    action_frequency = recent_actions.count(action) / len(recent_actions)
                    if action_frequency > 0.6:  # Used more than 60% of the time
                        repetition_penalty = 0.5 + (action_frequency - 0.6) * 2.0  # Heavy penalty
                        print(f" REPETITION PENALTY: Action {action} used {action_frequency:.1%} recently (-{repetition_penalty:.2f})")
                    elif action_frequency > 0.4:  # Used more than 40% of the time  
                        repetition_penalty = 0.2 + (action_frequency - 0.4) * 1.0  # Moderate penalty
            
            # 7. NEW: Semantic intelligence bonus - use learned action behavior
            semantic_bonus = self._calculate_semantic_action_score(action, game_context)
            
            final_score = max(0.05, base_score + sequence_bonus + diversity_bonus + context_bonus + stagnation_bonus + semantic_bonus - pattern_penalty - repetition_penalty)
            action_scores[action] = final_score
        
        # Enhanced selection strategy with adaptive exploration
        exploration_rate = 0.1  # Base exploration rate
        
        # Increase exploration if we're stuck (many failures)
        if recent_failures > 8:
            exploration_rate = 0.25  # 25% exploration when stuck
        elif total_actions_taken < 50:  # Early in training
            exploration_rate = 0.2   # 20% exploration early on
        
        is_exploration = random.random() < exploration_rate
        
        if is_exploration:  # Adaptive exploration
            # Smart exploration: prefer less-tried actions
            unexplored = [a for a in valid_actions if self.available_actions_memory['action_effectiveness'].get(a, {'attempts': 0})['attempts'] < 2]
            if unexplored:
                selected = random.choice(unexplored)
            else:
                selected = random.choice(valid_actions)
        else:
            # Intelligent selection with enhanced weighting
            weights = [max(0.05, action_scores[action]) for action in valid_actions]
            selected = random.choices(valid_actions, weights=weights)[0]
        
        # Update selection tracking
        self.available_actions_memory['sequence_in_progress'].append(selected)
        # Keep only last 20 actions in sequence
        if len(self.available_actions_memory['sequence_in_progress']) > 20:
            self.available_actions_memory['sequence_in_progress'] = self.available_actions_memory['sequence_in_progress'][-20:]
        
        return selected

    def _calculate_semantic_action_score(self, action: int, game_context: Dict[str, Any] = None) -> float:
        """Calculate semantic intelligence bonus based on learned action behavior."""
        if action not in self.available_actions_memory['action_semantic_mapping']:
            return 0.0
        
        mapping = self.available_actions_memory['action_semantic_mapping'][action]
        
        # FIXED: Handle both dictionary and string game_context
        if isinstance(game_context, dict):
            game_id = game_context.get('game_id')
        elif isinstance(game_context, str):
            game_id = game_context
        else:
            game_id = None
            
        semantic_score = 0.0
        
        # 1. Game-specific role bonus
        if game_id and game_id in mapping['game_specific_roles']:
            role_info = mapping['game_specific_roles'][game_id]
            confidence_threshold = self.available_actions_memory.get('action_learning_stats', {}).get('pattern_confidence_threshold', 0.7)
            
            if role_info['confidence'] > confidence_threshold:
                role_bonuses = {
                    'victory_action': 0.4,      # High bonus for actions that win games
                    'progress_action': 0.25,    # Good bonus for actions that make progress
                    'movement_action': 0.15,    # Moderate bonus for movement
                    'interaction_action': 0.2,  # Good bonus for interactions
                    'coordinate_action': 0.15,  # Moderate bonus for coordinate actions
                    'undo_action': 0.05,        # Small bonus for undo (can be useful)
                    'exploration_action': 0.1,  # Small exploration bonus
                    'failure_action': -0.3      # Penalty for actions that cause failure
                }
                role_bonus = role_bonuses.get(role_info['role'], 0.0)
                semantic_score += role_bonus * role_info['confidence']
        
        # 2. Movement pattern confidence bonus
        movements = mapping['grid_movement_patterns']
        if movements:
            total_movements = sum(movements.values())
            if total_movements > 5:  # Need sufficient observations
                # Bonus for consistent movement patterns
                if movements:
                    dominant_pattern = max(movements.items(), key=lambda x: x[1])
                    consistency = dominant_pattern[1] / total_movements
                
                if consistency > 0.6:  # 60%+ consistency
                    pattern_bonuses = {
                        'success_move': 0.2,
                        'progress_move': 0.15,
                        'up': 0.1, 'down': 0.1, 'left': 0.1, 'right': 0.1,
                        'coordinate_based': 0.1,
                        'undo': 0.05,
                        'none': -0.05,  # Small penalty for no movement
                        'boundary_hit': -0.1  # Penalty for hitting boundaries
                    }
                    pattern_bonus = pattern_bonuses.get(dominant_pattern[0], 0.0)
                    semantic_score += pattern_bonus * consistency
        
        # 3. Effect pattern bonus
        effects = mapping['common_effects']
        if effects:
            total_effects = sum(effects.values())
            if total_effects > 3:  # Need sufficient observations
                effect_bonuses = {
                    'successful_action': 0.2,
                    'score_increase': 0.25,
                    'game_winning': 0.4,
                    'interaction_successful': 0.2,
                    'undo_successful': 0.1,
                    'progress_action': 0.15,
                    'game_ending': -0.2,     # Penalty for ending games unsuccessfully
                    'no_actions_remaining': -0.15
                }
                
                for effect, frequency in effects.items():
                    if frequency / total_effects > 0.3:  # At least 30% frequency
                        effect_bonus = effect_bonuses.get(effect, 0.0)
                        semantic_score += effect_bonus * (frequency / total_effects)
        
        # 4. Coordinate success zone bonus (for ACTION6)
        if action == 6 and 'coordinate_success_zones' in mapping:
            zones = mapping['coordinate_success_zones']
            if zones:
                # Find the most successful zones
                zone_success_rates = []
                for zone_data in zones.values():
                    if zone_data['attempts'] > 2:  # Need sufficient attempts
                        success_rate = zone_data['successes'] / zone_data['attempts']
                        zone_success_rates.append(success_rate)
                
                if zone_success_rates:
                    avg_success_rate = sum(zone_success_rates) / len(zone_success_rates)
                    semantic_score += avg_success_rate * 0.15  # Moderate bonus for coordinate success
        
        # Cap the semantic score to prevent over-weighting
        return max(-0.5, min(0.5, semantic_score))

    def _log_action_details_from_output(self, stdout_text: str, game_id: str):
        """Extract and log action details from game session output."""
        try:
            print(f"\n ACTION ANALYSIS for {game_id}:")
            
            # Look for API action calls in the output
            api_action_pattern = r'(ACTION[1-7].*?(?:success|failed|error))'
            api_actions = re.findall(api_action_pattern, stdout_text, re.IGNORECASE | re.DOTALL)
            
            if api_actions:
                print(f"    API Actions Found: {len(api_actions)}")
                for i, action in enumerate(api_actions[:10]):  # Show first 10 actions
                    action_clean = re.sub(r'\s+', ' ', action.strip())
                    if len(action_clean) > 100:
                        action_clean = action_clean[:97] + "..."
                    print(f"      {i+1}. {action_clean}")
                if len(api_actions) > 10:
                    print(f"      ... and {len(api_actions)-10} more actions")
            
            # Look for available actions in the output  
            available_pattern = r'available.*?actions?[:\s]*\[([^\]]+)\]'
            available_matches = re.findall(available_pattern, stdout_text, re.IGNORECASE)
            
            if available_matches:
                print(f"    Available Actions Seen:")
                unique_available = list(set(available_matches))
                for i, actions in enumerate(unique_available[:5]):  # Show first 5 unique sets
                    print(f"      {i+1}. [{actions}]")
                if len(unique_available) > 5:
                    print(f"      ... and {len(unique_available)-5} more sets")
            
            # Look for game state changes
            state_pattern = r'(?:state|status)[:\s]*(\w+)'
            states = re.findall(state_pattern, stdout_text, re.IGNORECASE)
            
            if states:
                unique_states = list(set(states))
                print(f"    Game States: {', '.join(unique_states)}")
            
            # Look for scores
            score_pattern = r'score[:\s]*(\d+)'
            scores = re.findall(score_pattern, stdout_text, re.IGNORECASE)
            
            if scores:
                score_progression = [int(s) for s in scores[-10:]]  # Last 10 scores
                print(f"    Score Progression: {' → '.join(map(str, score_progression))}")
            
            print()  # Add spacing
            
        except Exception as e:
            print(f"    Error analyzing action details: {e}")

    def _optimize_coordinates_for_action(self, action: int, grid_dims: Tuple[int, int], game_id: str = 'unknown') -> Tuple[int, int]:
        """Universal coordinate optimization with enhanced intelligence when available."""
        grid_width, grid_height = grid_dims
        
        # ENHANCED: Use intelligent coordinate selection if available
        if hasattr(self, 'enhanced_coordinate_intelligence') and hasattr(self, 'use_enhanced_coordinate_selection'):
            if self.use_enhanced_coordinate_selection:
                try:
                    intelligent_coord = self.enhanced_coordinate_intelligence.get_intelligent_coordinates(action, grid_dims, game_id)
                    return intelligent_coord
                except Exception as e:
                    print(f" Enhanced coordinate intelligence failed: {e}, falling back to standard system")
        
        # Special strategic coordinate selection for ACTION 6 with full directional system
        if action == 6:
            return self._get_strategic_action6_coordinates(grid_dims, game_id)
        
        # Universal boundary-aware coordinate selection for other actions
        return self._get_universal_boundary_aware_coordinates(action, grid_dims)
    
    def _get_universal_boundary_aware_coordinates(self, action: int, grid_dims: Tuple[int, int]) -> Tuple[int, int]:
        """
        Universal boundary-aware coordinate selection for all coordinate-based actions.
        
        Features:
        1. Avoids known boundaries discovered by any action
        2. Prefers success zones discovered by any action
        3. Uses intelligent exploration patterns
        4. Tracks coordinate effectiveness per action
        """
        grid_width, grid_height = grid_dims
        game_id = self.available_actions_memory.get('current_game_id', 'unknown')
        
        # Get universal boundary system
        boundary_system = self.available_actions_memory['universal_boundary_detection']
        
        # CRITICAL: Ensure boundary system is initialized for this game
        self._ensure_boundary_system_initialized(game_id)
        
        # Get action-specific settings
        action_settings = boundary_system['directional_systems'].get(action, {'boundary_avoidance_radius': 2})
        avoidance_radius = action_settings.get('boundary_avoidance_radius', 2)
        
        # Get known boundaries for this game
        known_boundaries = set(boundary_system['boundary_data'][game_id].keys())
        
        # Get success zones for this game
        success_zones = boundary_system['success_zone_mapping'][game_id]
        
        # Get success zones for this game
        success_zones = boundary_system['success_zone_mapping'][game_id]
        
        # ENHANCED: Intelligent surveying strategy - avoid slow traversal through safe zones
        safe_regions = boundary_system.get('safe_regions', {}).get(game_id, {})
        if safe_regions:
            # Instead of continuing through safe regions, jump to explore boundaries!
            best_survey_target = self._get_intelligent_survey_target(game_id, grid_dims, known_boundaries, safe_regions)
            
            if best_survey_target:
                survey_x, survey_y, survey_reason = best_survey_target
                print(f" ACTION {action} INTELLIGENT SURVEY: Jumping to {(survey_x, survey_y)} - {survey_reason}")
                return (survey_x, survey_y)
            
            # Only use safe regions if no interesting survey targets exist
            best_region = None
            best_region_score = 0
            
            for region_id, region_data in safe_regions.items():
                region_score = region_data['safety_score']
                if region_score > best_region_score:
                    # Check if region is not too close to boundaries
                    region_center = region_data['center']
                    too_close_to_boundary = False
                    
                    for boundary_coord in known_boundaries:
                        boundary_x, boundary_y = boundary_coord
                        distance = abs(region_center[0] - boundary_x) + abs(region_center[1] - boundary_y)
                        if distance < avoidance_radius:
                            too_close_to_boundary = True
                            break
                    
                    if not too_close_to_boundary:
                        best_region = region_data
                        best_region_score = region_score
            
            if best_region:
                # Use edge of safe region to push boundaries, not center
                edge_coord = self._get_safe_region_edge_coordinate(best_region, grid_dims, known_boundaries)
                print(f" ACTION {action} SAFE REGION EDGE: Using boundary coordinate {edge_coord} from region with "
                    f"{len(best_region['coordinates'])} safe squares to explore new territory")
                return edge_coord
        
        # NEW: Check coordinate clusters
        clusters = boundary_system.get('coordinate_clusters', {}).get(game_id, {})
        if clusters:
            best_cluster = None
            best_cluster_rate = 0
            
            for cluster_id, cluster_data in clusters.items():
                if cluster_data['avg_success_rate'] > best_cluster_rate and cluster_data['avg_success_rate'] > 0.2:  # LOWERED - Less strict cluster success rate
                    # Check if cluster center is not too close to boundaries
                    cluster_center = cluster_data['center']
                    too_close_to_boundary = False
                    
                    for boundary_coord in known_boundaries:
                        boundary_x, boundary_y = boundary_coord
                        distance = abs(cluster_center[0] - boundary_x) + abs(cluster_center[1] - boundary_y)
                        if distance < avoidance_radius:
                            too_close_to_boundary = True
                            break
                    
                    if not too_close_to_boundary:
                        best_cluster = cluster_data
                        best_cluster_rate = cluster_data['avg_success_rate']
            
            if best_cluster:
                # Use cluster center or nearby coordinate
                cluster_center = best_cluster['center']
                print(f" ACTION {action} CLUSTER: Using cluster center {cluster_center} "
                    f"(success rate: {best_cluster_rate:.1%}, {len(best_cluster['members'])} members)")
                return cluster_center
        
        # ORIGINAL: Find best individual success zone if available
        if success_zones:
            best_zone = None
            best_success_rate = 0
            
            for coords, zone_data in success_zones.items():
                success_rate = zone_data['success_count'] / max(1, zone_data['total_attempts'])
                if success_rate > best_success_rate and success_rate > 0.15:  # LOWERED - Less strict success rate requirement
                    # Check if this zone is not too close to boundaries
                    zone_x, zone_y = coords
                    too_close_to_boundary = False
                    
                    for boundary_coord in known_boundaries:
                        boundary_x, boundary_y = boundary_coord
                        distance = abs(zone_x - boundary_x) + abs(zone_y - boundary_y)  # Manhattan distance
                        if distance < avoidance_radius:
                            too_close_to_boundary = True
                            break
                    
                    if not too_close_to_boundary:
                        best_zone = coords
                        best_success_rate = success_rate
            
            if best_zone:
                print(f" ACTION {action} SUCCESS ZONE: Using proven coordinates {best_zone} (success rate: {best_success_rate:.1%})")
                return best_zone
        
        # Generate exploration coordinates avoiding boundaries
        max_attempts = 50  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            # Intelligent coordinate generation based on action type
            if action in [1, 2, 3, 4]:  # Directional actions
                # Use directional bias for movement actions
                center_x, center_y = grid_width // 2, grid_height // 2
                
                if action == 1:  # Up
                    x = center_x + random.randint(-5, 5)
                    y = random.randint(0, center_y)
                elif action == 2:  # Down  
                    x = center_x + random.randint(-5, 5)
                    y = random.randint(center_y, grid_height - 1)
                elif action == 3:  # Left
                    x = random.randint(0, center_x)
                    y = center_y + random.randint(-5, 5)
                elif action == 4:  # Right
                    x = random.randint(center_x, grid_width - 1)
                    y = center_y + random.randint(-5, 5)
            elif action == 5:  # Interaction action
                # Focus on central regions for interactions
                x = random.randint(grid_width // 4, 3 * grid_width // 4)
                y = random.randint(grid_height // 4, 3 * grid_height // 4)
            elif action == 7:  # Undo action
                # Use recent coordinate history for undo
                action_history = boundary_system['action_coordinate_history'][game_id]
                if action_history and len(action_history) > 0:
                    # Get recent coordinates from any action
                    recent_coords = []
                    for action_num, coord_history in action_history.items():
                        recent_coords.extend([coord[0] for coord in coord_history[-3:]])  # Last 3 coordinates
                    
                    if recent_coords:
                        x, y = random.choice(recent_coords)
                    else:
                        x, y = random.randint(0, grid_width-1), random.randint(0, grid_height-1)
                else:
                    x, y = random.randint(0, grid_width-1), random.randint(0, grid_height-1)
            else:
                # Default exploration
                x, y = random.randint(0, grid_width-1), random.randint(0, grid_height-1)
            
            # Ensure coordinates are in bounds
            x = max(0, min(x, grid_width - 1))
            y = max(0, min(y, grid_height - 1))
            
            # Check if too close to known boundaries AND danger zones
            coordinates = (x, y)
            too_close = False
            
            # Check proximity to individual boundaries
            for boundary_coord in known_boundaries:
                boundary_x, boundary_y = boundary_coord
                distance = abs(x - boundary_x) + abs(y - boundary_y)
                if distance < avoidance_radius:
                    too_close = True
                    break
            
            # NEW: Check proximity to danger zones
            if not too_close:
                danger_zones = boundary_system.get('danger_zones', {}).get(game_id, {})
                for danger_id, danger_data in danger_zones.items():
                    danger_center = danger_data['center']
                    danger_radius = danger_data['avoidance_radius']
                    distance = abs(x - danger_center[0]) + abs(y - danger_center[1])
                    if distance < danger_radius:
                        too_close = True
                        break
            
            if not too_close:
                spatial_summary = self.get_spatial_intelligence_summary(game_id)
                total_safe_areas = spatial_summary.get('safe_regions', 0) + spatial_summary.get('coordinate_clusters', 0)
                print(f" ACTION {action} SPATIAL-AWARE: Selected ({x},{y}) avoiding {len(known_boundaries)} boundaries "
                    f"and {len(boundary_system.get('danger_zones', {}).get(game_id, {}))} danger zones | "
                    f"Safe areas available: {total_safe_areas}")
                return coordinates
            
            attempts += 1
        
        # Fallback - use center if all else fails
        fallback_x, fallback_y = grid_width // 2, grid_height // 2
        print(f" ACTION {action} FALLBACK: Using center ({fallback_x},{fallback_y}) after boundary avoidance")
        return (fallback_x, fallback_y)

    def _get_strategic_action6_coordinates(self, grid_dims: Tuple[int, int], game_id: str = None) -> Tuple[int, int]:
        """
        INTELLIGENT SURVEYING SYSTEM for ACTION 6 - Fast grid exploration instead of slow directional crawling.
        
        Key Features:
        1. Jumps intelligently across the grid to map regions quickly
        2. Uses safe regions to launch exploration into unknown territory
        3. Avoids slow line-by-line traversal through known safe areas
        4. Prioritizes boundary detection and territory expansion
        """
        grid_width, grid_height = grid_dims
        if game_id is None:
            game_id = self.available_actions_memory.get('current_game_id', 'unknown')
        
        # Use universal boundary detection system
        boundary_system = self.available_actions_memory['universal_boundary_detection']
        
        # CRITICAL: Ensure boundary system is initialized for this game
        self._ensure_boundary_system_initialized(game_id)
        
        known_boundaries = set(boundary_system['boundary_data'][game_id].keys())
        safe_regions = boundary_system.get('safe_regions', {}).get(game_id, {})
        
        # INTELLIGENT SURVEYING: Instead of slow directional movement, make strategic jumps
        if safe_regions and len(list(safe_regions.values())[0]['coordinates']) > 10:
            # We have established safe regions - time to survey efficiently!
            survey_target = self._get_intelligent_survey_target(game_id, grid_dims, known_boundaries, safe_regions)
            
            if survey_target:
                survey_x, survey_y, survey_reason = survey_target
                print(f" ACTION 6 INTELLIGENT SURVEY: {survey_reason}")
                
                # Update position tracking for future moves  
                setattr(self, '_current_game_x', survey_x)
                setattr(self, '_current_game_y', survey_y)
                
                return (survey_x, survey_y)
        
        # FALLBACK: If no safe regions yet, use improved initial exploration
        # Get current position from game state (stored by previous ACTION 6 or start at center)
        current_x = getattr(self, '_current_game_x', grid_width // 2)
        current_y = getattr(self, '_current_game_y', 0)  # Start at top for systematic mapping
        
        # Use universal boundary detection directional system for ACTION 6
        directional_system = boundary_system['directional_systems'].get(6, {})
        current_direction_data = directional_system.get('current_direction', {}).get(game_id, 'right')
        direction_progression = directional_system.get('direction_progression', {
            'right': {'next': 'down', 'coordinate_delta': (1, 0)},
            'down': {'next': 'left', 'coordinate_delta': (0, 1)}, 
            'left': {'next': 'up', 'coordinate_delta': (-1, 0)},
            'up': {'next': 'right', 'coordinate_delta': (0, -1)}
        })
        
        # Get current direction (initialize if needed)
        current_direction = current_direction_data if isinstance(current_direction_data, str) else 'right'
        if game_id not in directional_system.get('current_direction', {}):
            if 'current_direction' not in directional_system:
                directional_system['current_direction'] = {}
            directional_system['current_direction'][game_id] = current_direction
        
        direction_info = direction_progression[current_direction]
        dx, dy = direction_info['coordinate_delta']
        
        # Calculate next coordinate in current direction
        new_x = current_x + dx
        new_y = current_y + dy
        
        # Check grid bounds and adjust if we hit the edge
        hit_boundary = False
        if new_x < 0 or new_x >= grid_width or new_y < 0 or new_y >= grid_height:
            hit_boundary = True
            boundary_type = f"grid_edge_{current_direction}"
            
            # Clamp to grid bounds
            new_x = max(0, min(new_x, grid_width - 1))
            new_y = max(0, min(new_y, grid_height - 1))
            
            # Mark this as a boundary using universal system
            boundary_coord = (new_x, new_y)
            boundary_system['boundary_data'][game_id][boundary_coord] = {
                'boundary_type': boundary_type,
                'detection_count': boundary_system['boundary_data'][game_id].get(boundary_coord, {}).get('detection_count', 0) + 1,
                'timestamp': time.time(),
                'action': 6
            }
            
            print(f" ACTION 6 BOUNDARY: Hit {boundary_type} at ({new_x},{new_y}) - pivoting direction")
        
        # PIVOT TO NEW DIRECTION if boundary hit
        if hit_boundary:
            # Pivot to next semantic direction
            next_direction = direction_info['next']
            directional_system['current_direction'][game_id] = next_direction
            
            # Calculate coordinates in new direction from current position
            next_direction_info = direction_progression[next_direction]
            dx, dy = next_direction_info['coordinate_delta']
            
            pivot_x = current_x + dx
            pivot_y = current_y + dy
            
            # Ensure pivot coordinates are within bounds
            pivot_x = max(0, min(pivot_x, grid_width - 1))
            pivot_y = max(0, min(pivot_y, grid_height - 1))
            
            new_x, new_y = pivot_x, pivot_y
            print(f" ACTION 6 PIVOT: Direction {current_direction} → {next_direction}, coordinates ({current_x},{current_y}) → ({new_x},{new_y})")
        
        # Update tracking data - ensure last_coordinates is initialized for this game
        if game_id not in boundary_system['last_coordinates']:
            boundary_system['last_coordinates'][game_id] = None
        boundary_system['last_coordinates'][game_id] = (new_x, new_y)
        
        # Track coordinate attempt history - ensure game_id is initialized
        if game_id not in boundary_system['coordinate_attempts']:
            boundary_system['coordinate_attempts'][game_id] = {}
        coord_key = (new_x, new_y)
        if coord_key not in boundary_system['coordinate_attempts'][game_id]:
            boundary_system['coordinate_attempts'][game_id][coord_key] = {'attempts': 0, 'consecutive_stuck': 0}
        boundary_system['coordinate_attempts'][game_id][coord_key]['attempts'] += 1
        
        # Update current position tracking
        self._current_game_x = new_x
        self._current_game_y = new_y
        
        # CRITICAL: Detect coordinate stagnation and force movement
        coord_key = (new_x, new_y)
        if coord_key in boundary_system['coordinate_attempts'][game_id]:
            consecutive_stuck = boundary_system['coordinate_attempts'][game_id][coord_key].get('consecutive_stuck', 0)
            if consecutive_stuck > 10:  # Stuck at same coordinates for 10+ attempts
                print(f" COORDINATE STAGNATION DETECTED at ({new_x},{new_y}) - FORCING MOVEMENT")
                
                # Force jump to a completely different region
                import random
                jump_regions = [
                    (grid_width // 8, grid_height // 8),      # Far corner
                    (7 * grid_width // 8, grid_height // 8),  # Opposite corner  
                    (grid_width // 2, grid_height // 8),      # Top center
                    (grid_width // 8, grid_height // 2),      # Left center
                    (7 * grid_width // 8, 7 * grid_height // 8), # Far bottom right
                ]
                new_x, new_y = random.choice(jump_regions)
                
                # Reset stagnation counter
                boundary_system['coordinate_attempts'][game_id][coord_key]['consecutive_stuck'] = 0
                
                # Reset direction to explore from new position
                directional_system['current_direction'][game_id] = random.choice(['right', 'down', 'left', 'up'])
                
                print(f" EMERGENCY JUMP: Moved to ({new_x},{new_y}), new direction: {directional_system['current_direction'][game_id]}")
                
                # Update tracking for new position
                self._current_game_x = new_x
                self._current_game_y = new_y
            else:
                boundary_system['coordinate_attempts'][game_id][coord_key]['consecutive_stuck'] = consecutive_stuck + 1
        
        # Display boundary intelligence
        num_boundaries = len(boundary_system['boundary_data'][game_id])
        direction_display = current_direction.upper()
        if hit_boundary:
            direction_display = f"{current_direction.upper()}→{boundary_system['current_direction'][game_id].upper()}"
        
        print(f" ACTION 6 BOUNDARY-AWARE: {direction_display} from ({current_x},{current_y}) → ({new_x},{new_y}) | Boundaries mapped: {num_boundaries}")
        
        return (new_x, new_y)
    
    def _get_strategic_coordinates(self, action: int, grid_dims: Tuple[int, int]) -> Tuple[int, int]:
        """Generate strategic default coordinates for actions with exploration."""
        grid_width, grid_height = grid_dims
        
        # Add some randomization and exploration to coordinate selection
        import random
        
        # Strategic coordinate selection based on action type and grid analysis
        if action == 1:  # Often drawing/placing
            # Explore upper left quadrant with some variation
            base_x, base_y = grid_width // 4, grid_height // 4
            x = max(1, min(grid_width - 1, base_x + random.randint(-3, 3)))
            y = max(1, min(grid_height - 1, base_y + random.randint(-3, 3)))
            return (x, y)
        elif action == 2:  # Often modifying  
            # Explore center area with variation
            base_x, base_y = grid_width // 2, grid_height // 2
            x = max(1, min(grid_width - 1, base_x + random.randint(-5, 5)))
            y = max(1, min(grid_height - 1, base_y + random.randint(-5, 5)))
            return (x, y)
        elif action == 3:  # Often erasing/removing
            # Explore lower right quadrant
            base_x, base_y = 3 * grid_width // 4, 3 * grid_height // 4
            x = max(1, min(grid_width - 1, base_x + random.randint(-3, 3)))
            y = max(1, min(grid_height - 1, base_y + random.randint(-3, 3)))
            return (x, y)
        elif action == 4:  # Often pattern-related
            # Explore lower left quadrant
            base_x, base_y = grid_width // 4, 3 * grid_height // 4
            x = max(1, min(grid_width - 1, base_x + random.randint(-3, 3)))
            y = max(1, min(grid_height - 1, base_y + random.randint(-3, 3)))
            return (x, y)
        elif action == 5:  # Often transformation
            # Explore upper right quadrant
            base_x, base_y = 3 * grid_width // 4, grid_height // 4
            x = max(1, min(grid_width - 1, base_x + random.randint(-3, 3)))
            y = max(1, min(grid_height - 1, base_y + random.randint(-3, 3)))
            return (x, y)
        elif action == 6:  # Special coordinate-based action - explore more broadly
            # For action 6, explore different grid regions more systematically
            regions = [
                (grid_width // 4, grid_height // 4),      # Upper left
                (3 * grid_width // 4, grid_height // 4),  # Upper right  
                (grid_width // 4, 3 * grid_height // 4),  # Lower left
                (3 * grid_width // 4, 3 * grid_height // 4), # Lower right
                (grid_width // 2, grid_height // 2),      # Center
                (grid_width // 8, grid_height // 8),      # Far upper left
                (7 * grid_width // 8, 7 * grid_height // 8), # Far lower right
            ]
            base_x, base_y = random.choice(regions)
            x = max(1, min(grid_width - 1, base_x + random.randint(-4, 4)))
            y = max(1, min(grid_height - 1, base_y + random.randint(-4, 4)))
            return (x, y)
        else:
            # For any other actions, explore the entire grid more broadly
            x = random.randint(2, grid_width - 2)
            y = random.randint(2, grid_height - 2)
            return (x, y)

    def _record_coordinate_effectiveness(self, action: int, x: int, y: int, success: bool):
        """Universal coordinate effectiveness recording with boundary detection system."""
        coordinates = (x, y)
        
        # Get current score improvement (will be 0 if we don't have it)
        score_improvement = 0.0
        if hasattr(self, '_last_score_improvement'):
            score_improvement = self._last_score_improvement
        
        # Get current game ID
        game_id = self.available_actions_memory.get('current_game_id', 'unknown')
        
        # ENHANCED: Record coordinate attempt in frame analyzer for learning
        if hasattr(self, 'frame_analyzer') and self.frame_analyzer:
            try:
                self.frame_analyzer.record_coordinate_attempt(
                    x, y, success, score_improvement
                )
            except Exception as e:
                print(f" Frame analyzer tracking failed: {e}")
        
        # Record using universal coordinate intelligence system
        self.record_action_result(action, coordinates, success, score_improvement, game_id)
        
        # Legacy coordinate_patterns for backward compatibility
        if action not in self.available_actions_memory['coordinate_patterns']:
            self.available_actions_memory['coordinate_patterns'][action] = {}
            
        coord_key = (x, y)
        if coord_key not in self.available_actions_memory['coordinate_patterns'][action]:
            self.available_actions_memory['coordinate_patterns'][action][coord_key] = {
                'attempts': 0, 'successes': 0, 'success_rate': 0.0
            }
        
        coord_data = self.available_actions_memory['coordinate_patterns'][action][coord_key]
        coord_data['attempts'] += 1
        if success:
            coord_data['successes'] += 1
        coord_data['success_rate'] = coord_data['successes'] / coord_data['attempts']
    
    def record_action_result(self, action_num: int, coordinates: tuple, success: bool, score_improvement: float, game_id: str = None):
        """Record action results with universal boundary detection and success zone mapping."""
        if not game_id:
            game_id = self.available_actions_memory.get('current_game_id', 'unknown')
        
        boundary_system = self.available_actions_memory['universal_boundary_detection']
        
        # CRITICAL: Ensure boundary system is initialized for this game
        self._ensure_boundary_system_initialized(game_id)
        
        # Record coordinate attempt
        coord_str = str(coordinates)
        if coord_str not in boundary_system['coordinate_attempts'][game_id]:
            boundary_system['coordinate_attempts'][game_id][coord_str] = {
                'attempts': 0,
                'successes': 0,
                'success_rate': 0.0,
                'actions_used': set(),
                'last_successful_action': None,
                'score_improvements': []
            }
        
        coord_data = boundary_system['coordinate_attempts'][game_id][coord_str]
        coord_data['attempts'] += 1
        coord_data['actions_used'].add(action_num)
        coord_data['score_improvements'].append(score_improvement)
        
        if success:
            coord_data['successes'] += 1
            coord_data['last_successful_action'] = action_num
            
            # Update success zone mapping
            if coordinates not in boundary_system['success_zone_mapping'][game_id]:
                boundary_system['success_zone_mapping'][game_id][coordinates] = {
                    'success_count': 0,
                    'total_attempts': 0,
                    'last_success_score': score_improvement,
                    'best_action': action_num
                }
            
            zone_data = boundary_system['success_zone_mapping'][game_id][coordinates]
            zone_data['success_count'] += 1
            zone_data['total_attempts'] = coord_data['attempts']
            zone_data['last_success_score'] = score_improvement
            if score_improvement > 0:
                zone_data['best_action'] = action_num
        
        coord_data['success_rate'] = coord_data['successes'] / coord_data['attempts']
        
        # Record action coordinate history
        if action_num not in boundary_system['action_coordinate_history'][game_id]:
            boundary_system['action_coordinate_history'][game_id][action_num] = []
        
        boundary_system['action_coordinate_history'][game_id][action_num].append((coordinates, success, score_improvement))
        
        # Keep only last 10 coordinates per action to prevent memory bloat
        if len(boundary_system['action_coordinate_history'][game_id][action_num]) > 10:
            boundary_system['action_coordinate_history'][game_id][action_num] = \
                boundary_system['action_coordinate_history'][game_id][action_num][-10:]
        
        # Universal boundary detection - check if any action is getting stuck
        self._detect_stuck_coordinates_universal(action_num, coordinates, game_id)
        
        # Print universal coordinate intelligence update with spatial analysis
        total_boundaries = len(boundary_system['boundary_data'][game_id])
        total_success_zones = len(boundary_system['success_zone_mapping'][game_id])
        
        # Get spatial intelligence summary
        spatial_summary = self.get_spatial_intelligence_summary(game_id)
        safe_regions = spatial_summary.get('safe_regions', 0)
        clusters = spatial_summary.get('coordinate_clusters', 0)
        danger_zones = spatial_summary.get('danger_zones', 0)
        
        print(f" COORDINATE INTELLIGENCE: Action {action_num} at {coordinates} ({'' if success else ''}) | "
            f"Boundaries: {total_boundaries} | Success Zones: {total_success_zones} | "
            f"Safe Regions: {safe_regions} | Clusters: {clusters} | Danger Zones: {danger_zones}")
              
        # Update global intelligence for successful coordinates
        if success:
            self._update_global_coordinate_intelligence(coordinates, action_num, 'success')
    
    def _detect_stuck_coordinates_universal(self, action_num: int, coordinates: tuple, game_id: str):
        """Universal stuck coordinate detection for all actions."""
        boundary_system = self.available_actions_memory['universal_boundary_detection']
        coord_str = str(coordinates)
        
        # Count recent attempts at this coordinate by this action
        action_history = boundary_system['action_coordinate_history'][game_id].get(action_num, [])
        recent_attempts = [coord for coord, success, score in action_history[-5:] if coord == coordinates]
        
        # If this action has attempted this coordinate 3+ times recently without major success
        if len(recent_attempts) >= 3:
            recent_successes = [success for coord, success, score in action_history[-5:] 
                            if coord == coordinates and success]
            recent_score_improvements = [score for coord, success, score in action_history[-5:] 
                                    if coord == coordinates and score > 0.1]
            
            # Detect boundary if multiple attempts with low success
            if len(recent_successes) == 0 and len(recent_score_improvements) == 0:
                if coord_str not in boundary_system['boundary_data'][game_id]:
                    boundary_system['boundary_data'][game_id][coord_str] = {
                        'detected_by_actions': [],
                        'attempts_when_detected': 0,
                        'discovery_game': game_id,
                        'confidence': 0.0
                    }
                
                boundary_data = boundary_system['boundary_data'][game_id][coord_str]
                if action_num not in boundary_data['detected_by_actions']:
                    boundary_data['detected_by_actions'].append(action_num)
                    boundary_data['attempts_when_detected'] += len(recent_attempts)
                    boundary_data['confidence'] = min(1.0, len(recent_attempts) / 5.0)
                    
                    print(f" UNIVERSAL BOUNDARY: Action {action_num} detected boundary at {coordinates} "
                        f"(confidence: {boundary_data['confidence']:.1%})")
                    
                    # Update global coordinate intelligence
                    self._update_global_coordinate_intelligence(coordinates, action_num, 'boundary')

    def _ensure_boundary_system_initialized(self, game_id: str):
        """Ensure all boundary system components are initialized for a game_id."""
        boundary_system = self.available_actions_memory['universal_boundary_detection']
        
        if game_id not in boundary_system['boundary_data']:
            boundary_system['boundary_data'][game_id] = {}
        if game_id not in boundary_system['coordinate_attempts']:
            boundary_system['coordinate_attempts'][game_id] = {}
        if game_id not in boundary_system['action_coordinate_history']:
            boundary_system['action_coordinate_history'][game_id] = {}
        if game_id not in boundary_system['stuck_patterns']:
            boundary_system['stuck_patterns'][game_id] = {}
        if game_id not in boundary_system['success_zone_mapping']:
            boundary_system['success_zone_mapping'][game_id] = {}
        if game_id not in boundary_system['last_coordinates']:
            boundary_system['last_coordinates'][game_id] = None
        if 'current_direction' not in boundary_system:
            boundary_system['current_direction'] = {}
        if game_id not in boundary_system['current_direction']:
            boundary_system['current_direction'][game_id] = 'right'  # Default direction

    def _get_intelligent_survey_target(self, game_id: str, grid_dims: Tuple[int, int], 
                                    known_boundaries: set, safe_regions: dict) -> Optional[Tuple[int, int, str]]:
        """
        Intelligent surveying: Instead of slow traversal through safe zones, jump to explore boundaries.
        
        Strategy:
        1. Find edges of safe regions and jump outward to test new territory  
        2. Make large coordinate jumps to quickly map the grid
        3. Target unexplored quadrants
        4. Push the bounds of understanding by testing boundary extensions
        
        Returns: (x, y, reason) or None if no good survey target
        """
        grid_width, grid_height = grid_dims
        
        # Strategy 1: Jump from safe region edges to unexplored territory
        for region_id, region_data in safe_regions.items():
            region_coords = region_data['coordinates']
            
            # Find the extremes of this safe region
            if region_coords:
                min_x = min(coord[0] for coord in region_coords)
                max_x = max(coord[0] for coord in region_coords)
                min_y = min(coord[1] for coord in region_coords)
                max_y = max(coord[1] for coord in region_coords)
            else:
                min_x = max_x = min_y = max_y = 0
            
            # Create jump targets that extend beyond the safe region
            jump_targets = [
                (min_x - 5, min_y - 3, f"Jump LEFT from safe region {region_id}"),
                (max_x + 5, min_y - 3, f"Jump RIGHT from safe region {region_id}"),
                (min_x - 3, min_y - 5, f"Jump UP from safe region {region_id}"),
                (min_x - 3, max_y + 5, f"Jump DOWN from safe region {region_id}"),
                # Diagonal jumps for comprehensive mapping
                (min_x - 4, min_y - 4, f"Jump UP-LEFT from safe region {region_id}"),
                (max_x + 4, min_y - 4, f"Jump UP-RIGHT from safe region {region_id}"),
                (min_x - 4, max_y + 4, f"Jump DOWN-LEFT from safe region {region_id}"),
                (max_x + 4, max_y + 4, f"Jump DOWN-RIGHT from safe region {region_id}")
            ]
            
            # Find valid jump targets that are in bounds and not near known boundaries
            for target_x, target_y, reason in jump_targets:
                # Clamp to grid bounds
                target_x = max(0, min(target_x, grid_width - 1))
                target_y = max(0, min(target_y, grid_height - 1))
                
                # Check if this is far enough from known boundaries
                if self._is_good_survey_target((target_x, target_y), known_boundaries, min_distance=3):
                    return (target_x, target_y, reason)
        
        # Strategy 2: Quadrant exploration - jump to unexplored grid quadrants
        quadrants = [
            (grid_width // 4, grid_height // 4, "Explore TOP-LEFT quadrant"),
            (3 * grid_width // 4, grid_height // 4, "Explore TOP-RIGHT quadrant"),  
            (grid_width // 4, 3 * grid_height // 4, "Explore BOTTOM-LEFT quadrant"),
            (3 * grid_width // 4, 3 * grid_height // 4, "Explore BOTTOM-RIGHT quadrant")
        ]
        
        for quad_x, quad_y, reason in quadrants:
            if self._is_good_survey_target((quad_x, quad_y), known_boundaries, min_distance=5):
                # Check if this quadrant is underexplored
                quadrant_explored = any(
                    abs(coord[0] - quad_x) < 8 and abs(coord[1] - quad_y) < 8 
                    for safe_coords in [region_data['coordinates'] for region_data in safe_regions.values()]
                    for coord in safe_coords
                )
                
                if not quadrant_explored:
                    return (quad_x, quad_y, reason)
        
        # Strategy 3: Boundary extension - test just beyond known boundaries to find limits
        boundary_extensions = []
        for boundary_coord in known_boundaries:
            bx, by = boundary_coord
            
            # Try coordinates just beyond the boundary in multiple directions
            extensions = [
                (bx - 2, by, f"Test boundary extension LEFT of {boundary_coord}"),
                (bx + 2, by, f"Test boundary extension RIGHT of {boundary_coord}"),
                (bx, by - 2, f"Test boundary extension UP of {boundary_coord}"),
                (bx, by + 2, f"Test boundary extension DOWN of {boundary_coord}")
            ]
            
            for ext_x, ext_y, reason in extensions:
                # Clamp to grid bounds  
                ext_x = max(0, min(ext_x, grid_width - 1))
                ext_y = max(0, min(ext_y, grid_height - 1))
                
                if self._is_good_survey_target((ext_x, ext_y), known_boundaries, min_distance=2):
                    boundary_extensions.append((ext_x, ext_y, reason))
        
        if boundary_extensions:
            return random.choice(boundary_extensions)
        
        return None

    def _is_good_survey_target(self, target_coord: Tuple[int, int], known_boundaries: set, min_distance: int = 3) -> bool:
        """Check if a coordinate is a good survey target (not too close to known boundaries)."""
        target_x, target_y = target_coord
        
        for boundary_coord in known_boundaries:
            bx, by = boundary_coord
            distance = abs(target_x - bx) + abs(target_y - by)
            if distance < min_distance:
                return False
                
        return True

    def _get_safe_region_edge_coordinate(self, safe_region: dict, grid_dims: Tuple[int, int], 
                                    known_boundaries: set) -> Tuple[int, int]:
        """
        Get a coordinate at the edge of a safe region to push outward and explore new territory.
        Instead of using the center, find the edge that's most promising for expansion.
        """
        grid_width, grid_height = grid_dims
        region_coords = safe_region['coordinates']
        
        # Find the bounding box of the safe region
        if region_coords:
            min_x = min(coord[0] for coord in region_coords)
            max_x = max(coord[0] for coord in region_coords)
            min_y = min(coord[1] for coord in region_coords)
            max_y = max(coord[1] for coord in region_coords)
        else:
            min_x = max_x = min_y = max_y = 0
        
        # Define edge candidates - coordinates just outside the safe region
        edge_candidates = []
        
        # Add edges in all directions
        for coord in region_coords:
            x, y = coord
            candidates = [
                (x - 1, y, "LEFT edge"),
                (x + 1, y, "RIGHT edge"), 
                (x, y - 1, "UP edge"),
                (x, y + 1, "DOWN edge")
            ]
            
            for candidate_x, candidate_y, direction in candidates:
                # Check bounds
                if 0 <= candidate_x < grid_width and 0 <= candidate_y < grid_height:
                    # Check if this edge coordinate is not already in the safe region
                    if (candidate_x, candidate_y) not in region_coords:
                        # Check if it's not too close to known boundaries
                        if self._is_good_survey_target((candidate_x, candidate_y), known_boundaries, min_distance=2):
                            edge_candidates.append((candidate_x, candidate_y, direction))
        
        # Return a random edge candidate, or fall back to region center
        if edge_candidates:
            selected_edge = random.choice(edge_candidates)
            return (selected_edge[0], selected_edge[1])  # Return just x,y
        else:
            # Fallback to region center if no good edges found
            return safe_region['center']
    
    def _update_global_coordinate_intelligence(self, coordinates: tuple, action_num: int, event_type: str):
        """Update global coordinate intelligence across all games and actions."""
        boundary_system = self.available_actions_memory['universal_boundary_detection']
        
        # Initialize global intelligence if needed
        if 'global_coordinate_intelligence' not in boundary_system:
            boundary_system['global_coordinate_intelligence'] = {
                'universal_boundaries': {},  # Coordinates that are problematic across games
                'universal_success_zones': {},  # Coordinates that work well across games
                'action_coordinate_preferences': {}  # Which actions work best at which coordinate types
            }
        
        global_intel = boundary_system['global_coordinate_intelligence']
        coord_str = str(coordinates)
        
        if event_type == 'boundary':
            if coord_str not in global_intel['universal_boundaries']:
                global_intel['universal_boundaries'][coord_str] = {
                    'games_detected_in': [],
                    'actions_that_detected': set(),
                    'total_detections': 0
                }
            
            current_game = self.available_actions_memory.get('current_game_id', 'unknown')
            boundary_data = global_intel['universal_boundaries'][coord_str]
            if current_game not in boundary_data['games_detected_in']:
                boundary_data['games_detected_in'].append(current_game)
            boundary_data['actions_that_detected'].add(action_num)
            boundary_data['total_detections'] += 1
            
            print(f" GLOBAL BOUNDARY: {coordinates} detected across {len(boundary_data['games_detected_in'])} games by {len(boundary_data['actions_that_detected'])} actions")
            
        elif event_type == 'success':
            if coord_str not in global_intel['universal_success_zones']:
                global_intel['universal_success_zones'][coord_str] = {
                    'games_succeeded_in': [],
                    'actions_that_succeeded': set(),
                    'total_successes': 0,
                    'cross_game_success_rate': 0.0
                }
            
            current_game = self.available_actions_memory.get('current_game_id', 'unknown')
            success_data = global_intel['universal_success_zones'][coord_str]
            if current_game not in success_data['games_succeeded_in']:
                success_data['games_succeeded_in'].append(current_game)
            success_data['actions_that_succeeded'].add(action_num)
            success_data['total_successes'] += 1
            success_data['cross_game_success_rate'] = success_data['total_successes'] / max(1, len(success_data['games_succeeded_in']))
            
            # Trigger spatial analysis for connected regions
            self._analyze_spatial_connections(coordinates, current_game)

    def _analyze_spatial_connections(self, new_coordinate: tuple, game_id: str):
        """Analyze spatial connections between success zones to find safe regions and clusters."""
        boundary_system = self.available_actions_memory['universal_boundary_detection']
        
        # Initialize spatial data structures for this game if needed
        if game_id not in boundary_system['safe_regions']:
            boundary_system['safe_regions'][game_id] = {}
        if game_id not in boundary_system['connected_zones']:
            boundary_system['connected_zones'][game_id] = {}
        if game_id not in boundary_system['coordinate_clusters']:
            boundary_system['coordinate_clusters'][game_id] = {}
        if game_id not in boundary_system['danger_zones']:
            boundary_system['danger_zones'][game_id] = {}
        
        # Get all success zones for this game
        success_zones = boundary_system['success_zone_mapping'].get(game_id, {})
        if not success_zones:
            return
        
        # Find connected coordinates (adjacent successful zones)
        connected_groups = self._find_connected_coordinate_groups(success_zones, game_id)
        
        # Analyze each connected group for safety
        for group_id, group_data in connected_groups.items():
            if len(group_data['coordinates']) >= 3:  # Minimum cluster size
                # Calculate center point
                coords_list = list(group_data['coordinates'])
                center_x = sum(coord[0] for coord in coords_list) / len(coords_list)
                center_y = sum(coord[1] for coord in coords_list) / len(coords_list)
                center = (int(center_x), int(center_y))
                
                # Calculate average success rate for this region
                total_success_rate = sum(
                    success_zones[coord]['success_count'] / max(1, success_zones[coord]['total_attempts'])
                    for coord in coords_list if coord in success_zones
                )
                avg_success_rate = total_success_rate / len(coords_list)
                
                # Create or update safe region
                region_id = f"region_{group_id}"
                boundary_system['safe_regions'][game_id][region_id] = {
                    'coordinates': group_data['coordinates'],
                    'success_rate': avg_success_rate,
                    'center': center,
                    'size': len(group_data['coordinates']),
                    'safety_score': avg_success_rate * len(group_data['coordinates']),  # Size + quality
                    'last_updated': time.time()
                }
                
                print(f" SAFE REGION DETECTED: {region_id} with {len(group_data['coordinates'])} coordinates, "
                    f"success rate: {avg_success_rate:.1%}, center: {center}")
        
        # Find coordinate clusters using distance-based clustering
        self._update_coordinate_clusters(new_coordinate, game_id)
        
        # Update danger zones from boundary data
        self._update_danger_zones(game_id)
    
    def _find_connected_coordinate_groups(self, success_zones: dict, game_id: str) -> dict:
        """Find groups of connected (adjacent) successful coordinates."""
        visited = set()
        connected_groups = {}
        group_id = 0
        
        def get_neighbors(coord):
            """Get adjacent coordinates (8-directional connectivity)."""
            x, y = coord
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if neighbor in success_zones:
                        neighbors.append(neighbor)
            return neighbors
        
        def dfs_connected_component(start_coord, group_coords):
            """Depth-first search to find all connected coordinates."""
            if start_coord in visited:
                return
            
            visited.add(start_coord)
            group_coords.add(start_coord)
            
            for neighbor in get_neighbors(start_coord):
                if neighbor not in visited:
                    dfs_connected_component(neighbor, group_coords)
        
        # Find all connected components
        for coord in success_zones:
            if coord not in visited:
                group_coords = set()
                dfs_connected_component(coord, group_coords)
                
                if len(group_coords) >= 2:  # Only keep groups with 2+ coordinates
                    connected_groups[group_id] = {
                        'coordinates': group_coords,
                        'size': len(group_coords)
                    }
                    group_id += 1
        
        return connected_groups
    
    def _update_coordinate_clusters(self, new_coordinate: tuple, game_id: str):
        """Update coordinate clusters using distance-based analysis."""
        boundary_system = self.available_actions_memory['universal_boundary_detection']
        clusters = boundary_system['coordinate_clusters'][game_id]
        success_zones = boundary_system['success_zone_mapping'].get(game_id, {})
        
        # Parameters for clustering
        cluster_radius = 5  # Maximum distance to be in same cluster
        min_cluster_size = 3
        
        # Find nearest cluster for new coordinate
        nearest_cluster = None
        min_distance = float('inf')
        
        for cluster_id, cluster_data in clusters.items():
            center = cluster_data['center']
            distance = abs(new_coordinate[0] - center[0]) + abs(new_coordinate[1] - center[1])  # Manhattan distance
            if distance < min_distance and distance <= cluster_radius:
                min_distance = distance
                nearest_cluster = cluster_id
        
        if nearest_cluster is not None:
            # Add to existing cluster
            clusters[nearest_cluster]['members'].add(new_coordinate)
            
            # Recalculate cluster center
            members = list(clusters[nearest_cluster]['members'])
            center_x = sum(coord[0] for coord in members) / len(members)
            center_y = sum(coord[1] for coord in members) / len(members)
            clusters[nearest_cluster]['center'] = (int(center_x), int(center_y))
            
            # Recalculate average success rate
            total_success_rate = sum(
                success_zones[coord]['success_count'] / max(1, success_zones[coord]['total_attempts'])
                for coord in members if coord in success_zones
            )
            clusters[nearest_cluster]['avg_success_rate'] = total_success_rate / len(members)
            
            print(f" CLUSTER UPDATED: cluster_{nearest_cluster} now has {len(members)} members, "
                f"success rate: {clusters[nearest_cluster]['avg_success_rate']:.1%}")
        else:
            # Create new cluster
            cluster_id = len(clusters)
            clusters[f"cluster_{cluster_id}"] = {
                'center': new_coordinate,
                'radius': cluster_radius,
                'members': {new_coordinate},
                'avg_success_rate': success_zones.get(new_coordinate, {}).get('success_count', 0) / max(1, success_zones.get(new_coordinate, {}).get('total_attempts', 1)),
                'created_time': time.time()
            }
            
            print(f" NEW CLUSTER: cluster_{cluster_id} created at {new_coordinate}")
    
    def _update_danger_zones(self, game_id: str):
        """Update danger zones from boundary data."""
        boundary_system = self.available_actions_memory['universal_boundary_detection']
        boundary_data = boundary_system['boundary_data'].get(game_id, {})
        danger_zones = boundary_system['danger_zones'][game_id]
        
        # Group boundaries into danger zones
        danger_coords = set()
        for coord_str, boundary_info in boundary_data.items():
            if boundary_info['confidence'] > 0.7:  # High confidence boundaries
                coord = eval(coord_str)  # Convert string back to tuple
                danger_coords.add(coord)
        
        if len(danger_coords) >= 3:
            # Find connected danger regions
            danger_groups = self._find_connected_coordinate_groups({coord: {} for coord in danger_coords}, game_id)
            
            for group_id, group_data in danger_groups.items():
                if len(group_data['coordinates']) >= 3:
                    # Calculate center of danger zone
                    coords_list = list(group_data['coordinates'])
                    center_x = sum(coord[0] for coord in coords_list) / len(coords_list)
                    center_y = sum(coord[1] for coord in coords_list) / len(coords_list)
                    center = (int(center_x), int(center_y))
                    
                    danger_id = f"danger_{group_id}"
                    danger_zones[danger_id] = {
                        'coordinates': group_data['coordinates'],
                        'failure_rate': 0.95,  # High failure rate for boundaries
                        'avoidance_radius': max(3, int(len(coords_list) * 0.5)),  # Dynamic avoidance radius
                        'center': center,
                        'last_updated': time.time()
                    }
                    
                    print(f" DANGER ZONE: {danger_id} with {len(coords_list)} dangerous coordinates, "
                        f"avoidance radius: {danger_zones[danger_id]['avoidance_radius']}")
    
    def get_spatial_intelligence_summary(self, game_id: str) -> dict:
        """Get comprehensive spatial intelligence summary for a game."""
        boundary_system = self.available_actions_memory['universal_boundary_detection']
        
        summary = {
            'safe_regions': len(boundary_system.get('safe_regions', {}).get(game_id, {})),
            'connected_zones': len(boundary_system.get('connected_zones', {}).get(game_id, {})),
            'coordinate_clusters': len(boundary_system.get('coordinate_clusters', {}).get(game_id, {})),
            'danger_zones': len(boundary_system.get('danger_zones', {}).get(game_id, {})),
            'individual_success_zones': len(boundary_system.get('success_zone_mapping', {}).get(game_id, {})),
            'boundaries_detected': len(boundary_system.get('boundary_data', {}).get(game_id, {}))
        }
        
        # Calculate safety metrics
        safe_regions = boundary_system.get('safe_regions', {}).get(game_id, {})
        if safe_regions:
            total_safe_coords = sum(len(region['coordinates']) for region in safe_regions.values())
            avg_region_safety = sum(region['success_rate'] for region in safe_regions.values()) / len(safe_regions)
            summary['total_safe_coordinates'] = total_safe_coords
            summary['average_region_safety'] = avg_region_safety
        
        return summary

    def _record_sequence_outcome(self, sequence: List[int], success: bool):
        """Record whether an action sequence was successful."""
        if success and len(sequence) >= 2:
            # Add to winning sequences
            self.available_actions_memory['winning_action_sequences'].append(sequence.copy())
            # Keep only recent successful sequences
            if len(self.available_actions_memory['winning_action_sequences']) > 50:
                self.available_actions_memory['winning_action_sequences'] = \
                    self.available_actions_memory['winning_action_sequences'][-50:]
        elif not success and len(sequence) >= 2:
            # Add to failed patterns (shorter sequences to avoid overfitting)
            failed_pattern = sequence[-3:] if len(sequence) >= 3 else sequence
            self.available_actions_memory['failed_action_patterns'].append(failed_pattern)
            # Keep only recent failed patterns
            if len(self.available_actions_memory['failed_action_patterns']) > 30:
                self.available_actions_memory['failed_action_patterns'] = \
                    self.available_actions_memory['failed_action_patterns'][-30:]

    def _get_action_intelligence_summary(self, game_id: str) -> Dict[str, Any]:
        """Get summary of learned action intelligence for a game."""
        if self.available_actions_memory['current_game_id'] != game_id:
            return {}
            
        effectiveness = self.available_actions_memory['action_effectiveness']
        effective_actions = {k: v for k, v in effectiveness.items() if v['success_rate'] > 0.5}
        
        return {
            'total_actions_learned': len(effectiveness),
            'effective_actions': len(effective_actions),
            'best_action': max(effectiveness.keys(), key=lambda x: effectiveness[x]['success_rate']) if effectiveness else None,
            'winning_sequences_count': len(self.available_actions_memory['winning_action_sequences']),
            'coordinate_patterns_learned': sum(len(coords) for coords in self.available_actions_memory['coordinate_patterns'].values()),
            'intelligence_available': game_id in self.available_actions_memory['game_intelligence_cache']
        }
        """Load previous state if available."""
        state_file = self.save_directory / "continuous_learning_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    
                self.global_performance_metrics.update(state_data.get('global_performance_metrics', {}))
                self.session_history = state_data.get('session_history', [])
                
                logger.info("Loaded previous continuous learning state")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    def _load_state(self):
        """Load previous training state if available."""
        state_file = self.save_directory / "continuous_learning_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.session_history = state.get('session_history', [])
                    self.global_performance_metrics = state.get('global_performance_metrics', self.global_performance_metrics)
                    logger.info(f" Loaded previous state: {len(self.session_history)} sessions")
            except Exception as e:
                logger.warning(f"Failed to load previous state: {e}")

    def _save_state(self):
        """Save current training state."""
        state = {
            'session_history': self.session_history[-50:],  # Keep last 50 sessions
            'global_performance_metrics': self.global_performance_metrics,
            'last_updated': time.time()
        }
        
        state_file = self.save_directory / "continuous_learning_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _should_stop_training(self, game_results: Dict[str, Any], target_performance: Dict[str, float]) -> bool:
        """Check if we should stop training on this game based on target performance."""
        episodes = game_results.get('episodes', [])
        if len(episodes) < 5:  # Need at least 5 episodes to evaluate
            return False
        
        # Calculate current performance
        recent_episodes = episodes[-5:]  # Look at last 5 episodes
        recent_wins = sum(1 for ep in recent_episodes if ep.get('success', False))
        recent_win_rate = recent_wins / len(recent_episodes)
        recent_avg_score = sum((ep.get('final_score') or 0) for ep in recent_episodes) / len(recent_episodes)
        
        # Check if we've reached target performance
        target_win_rate = target_performance.get('win_rate', 0.3)
        target_avg_score = target_performance.get('avg_score', 50.0)
        
        if recent_win_rate >= target_win_rate and recent_avg_score >= target_avg_score:
            return True
            
        return False

    def start_training_session(
        self,
        games: List[str],
        max_mastery_sessions_per_game: int = 50,  # Updated parameter name
        max_actions_per_session: int = ActionLimits.get_max_actions_per_session(),  # Default action limit per game session
        enable_contrarian_mode: bool = False,  # New parameter
        target_win_rate: float = 0.3,
        target_avg_score: float = 50.0,
        salience_mode: SalienceMode = SalienceMode.LOSSLESS,
        enable_salience_comparison: bool = False,
        swarm_enabled: bool = True
    ) -> str:
        """
        Start a new continuous learning session.
        
        Args:
            games: List of ARC game IDs to train on
            max_mastery_sessions_per_game: Maximum mastery sessions per game (renamed from episodes)
            target_win_rate: Target win rate to achieve
            target_avg_score: Target average score
            salience_mode: Which salience mode to use (LOSSLESS or DECAY_COMPRESSION)
            enable_salience_comparison: Whether to run comparison between modes
            
        Returns:
            session_id: Unique identifier for this session
        """
        self._ensure_initialized()
        
        session_id = f"session_{int(time.time())}"
        
        self.current_session = TrainingSession(
            session_id=session_id,
            games_to_play=games,
            max_mastery_sessions_per_game=max_mastery_sessions_per_game,  # Updated parameter name
            learning_rate_schedule={
                'initial': 0.001,
                'mid': 0.0005,
                'final': 0.0002
            },
            save_interval=10,
            target_performance={
                'win_rate': target_win_rate,
                'avg_score': target_avg_score
            },
            max_actions_per_session=max_actions_per_session,  # New parameter
            enable_contrarian_strategy=enable_contrarian_mode,  # New parameter
            salience_mode=salience_mode,
            enable_salience_comparison=enable_salience_comparison,
            swarm_enabled=swarm_enabled
        )
        
        # Set session ID on main object for display
        self.session_id = session_id
        
        # Reset scorecard state for new session
        self._reset_scorecard_state()
        
        # Initialize salience calculator for this session
        self.salience_calculator = SalienceCalculator(
            mode=salience_mode,
            decay_rate=0.01 if salience_mode == SalienceMode.DECAY_COMPRESSION else 0.0,
            salience_min=0.05,
            compression_threshold=0.15
        )
        
        #  CONSOLIDATED: Using unified energy system configuration
        logger.info(f" Unified energy system configured: current={self.current_energy:.1f}/100, "
                f"sleep_threshold=40.0, energy_costs=standardized")
        
        # Display session startup information
        self._display_session_startup_info(session_id, games, salience_mode)
        
        logger.info(f"Started training session {session_id} with {len(games)} games using {salience_mode.value} mode")
        if enable_salience_comparison:
            logger.info("Salience mode comparison enabled - will test both modes")
        return session_id
    
    def _display_session_startup_info(self, session_id: str, games: List[str], salience_mode: SalienceMode):
        """Display compact session startup information."""
        print(f"\nARC-3 Training Session: {session_id}")
        print(f"Games: {', '.join(games)} | Mode: {salience_mode.value}")
        print(f"Scoreboard: {ARC3_SCOREBOARD_URL}")
        print("-" * 60)

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of learning progress."""
        return {
            'global_metrics': self.global_performance_metrics,
            'meta_learning_summary': self.arc_meta_learning.get_learning_summary(),
            'session_count': len(self.session_history),
            'current_session': self.current_session.session_id if self.current_session else None,
            'salience_performance_history': self.salience_performance_history,
            'current_salience_mode': self.current_session.salience_mode.value if self.current_session else None,
            'last_updated': datetime.now().isoformat()
        }

    async def _take_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take an action in the game and return the result.
        
        Args:
            state: Current game state dictionary
            
        Returns:
            Dictionary containing reward, game_state, and other action results
        """
        try:
            # Simple action selection - choose a random action
            import random
            action_id = random.randint(1, 7)  # Actions 1-7
            
            # Calculate reward based on action effectiveness
            # This is a simplified reward calculation
            base_reward = random.uniform(0.1, 1.0)
            
            # Determine game state (simplified)
            if state['actions_taken'] >= state.get('max_actions', 100):
                game_state = 'GAME_OVER'
            elif random.random() < 0.1:  # 10% chance of winning
                game_state = 'WIN'
            else:
                game_state = 'IN_PROGRESS'
            
            return {
                'reward': base_reward,
                'game_state': game_state,
                'action_id': action_id,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in _take_action: {e}")
            return {
                'reward': 0.0,
                'game_state': 'GAME_OVER',
                'action_id': 1,
                'success': False,
                'error': str(e)
            }

    @property
    def game_complexity(self) -> str:
        """
        Get the current game complexity level.
        
        Returns:
            String indicating game complexity level
        """
        return getattr(self, '_game_complexity', 'medium')
    
    def set_game_complexity(self, complexity: str):
        """
        Set the game complexity level.
        
        Args:
            complexity: Complexity level ('low', 'medium', 'high')
        """
        self._game_complexity = complexity



    # ====== SLEEP STATE AND MEMORY CONSOLIDATION METHODS ======
    
    def _get_current_agent_state(self) -> Dict[str, Any]:
        """Get current agent state for sleep and reset decisions."""
        # Simulate agent state - in real implementation this would come from the actual agent
        return {
            'energy': 50.0,  # Current energy level
            'memory_usage': 0.7,  # Memory utilization percentage
            'learning_progress': 0.1,  # Recent learning progress
            'consecutive_failures': 0,  # Count of recent failures
            'performance_trend': 'stable'  # 'improving', 'stable', 'declining'
        }
    
    def _should_agent_sleep(self, agent_state: Dict[str, Any], episode_count: int) -> bool:
        """Determine if agent should enter sleep cycle."""
        # Sleep triggers - adjusted for more frequent sleep cycles
        sleep_triggers = {
            'low_energy': agent_state['energy'] < 40.0,  # Increased from 20.0 to 40.0
            'high_memory_usage': agent_state['memory_usage'] > 0.7,  # Reduced from 0.9 to 0.7
            'periodic_sleep': episode_count % 5 == 0 and episode_count > 0,  # Reduced from 10 to 5 episodes
            'low_learning_progress': agent_state['learning_progress'] < 0.1  # Increased from 0.05 to 0.1
        }
        
        should_sleep = any(sleep_triggers.values())
        
        if should_sleep:
            trigger_reasons = [k for k, v in sleep_triggers.items() if v]
            self.sleep_state_tracker['last_sleep_trigger'] = trigger_reasons
            logger.info(f"Sleep triggered by: {', '.join(trigger_reasons)}")
        
        return should_sleep
    
    def _check_and_handle_boredom(self, episode_count: int) -> Dict[str, Any]:
        """Enhanced boredom detection that triggers strategy switching and action experimentation.
        
        Detects when agent gets stuck in action loops or repeated failures and triggers:
        1. New action combination strategies
        2. Random exploration using available actions
        3. Strategy switching to break bad cycles
        
        Returns:
            Dict containing boredom detection results and strategy changes made.
        """
        boredom_results = {
            'boredom_detected': False,
            'boredom_type': 'none',
            'strategy_switched': False,
            'new_strategy': None,
            'action_experimentation_triggered': False,
            'curriculum_advanced': False,
            'new_complexity': None,
            'reason': 'no_boredom'
        }
            
        # Only check after enough episodes for meaningful pattern detection
        if episode_count < 5:
            return boredom_results
        
        # Get recent LP history and action patterns
        recent_lp_history = self.training_state.get('lp_history', [])
        recent_action_patterns = self._analyze_recent_action_patterns()
        
        if len(recent_lp_history) < 3:
            return boredom_results
        
        # Calculate LP trend and action repetition patterns
        recent_lp_values = [entry.get('learning_progress', 0) for entry in recent_lp_history]
        lp_variance = np.var(recent_lp_values) if recent_lp_values else 0
        mean_lp = np.mean(recent_lp_values) if recent_lp_values else 0
        
        # Enhanced boredom detection criteria
        is_lp_stagnant = lp_variance < 0.001  # Very low variance = stagnation
        is_lp_low = mean_lp < 0.03  # Consistently low LP
        consecutive_low_episodes = sum(1 for lp in recent_lp_values[-5:] if lp < 0.05)
        action_loop_detected = recent_action_patterns.get('repetitive_loops', 0) > 2
        strategy_effectiveness_declining = recent_action_patterns.get('effectiveness_trend', 0) < -0.1
        
        # Detect different types of boredom
        if action_loop_detected and consecutive_low_episodes >= 2:
            boredom_results['boredom_detected'] = True
            boredom_results['boredom_type'] = 'action_loops'
            boredom_results['reason'] = f'Detected {recent_action_patterns["repetitive_loops"]} action loops with low LP'
            
            # Trigger action experimentation using available actions
            self._trigger_action_experimentation()
            boredom_results['action_experimentation_triggered'] = True
            
        elif is_lp_stagnant and is_lp_low and consecutive_low_episodes >= 3:
            boredom_results['boredom_detected'] = True
            boredom_results['boredom_type'] = 'lp_stagnation'
            boredom_results['reason'] = f'LP stagnation: variance={lp_variance:.6f}, mean={mean_lp:.3f}'
            
            # Advance curriculum complexity
            current_complexity = self.training_config.get('curriculum_complexity', 1)
            new_complexity = min(current_complexity + 1, 10)
            
            if new_complexity > current_complexity:
                self.training_config['curriculum_complexity'] = new_complexity
                boredom_results['curriculum_advanced'] = True
                boredom_results['new_complexity'] = new_complexity
                
        elif strategy_effectiveness_declining:
            boredom_results['boredom_detected'] = True
            boredom_results['boredom_type'] = 'strategy_ineffective'
            boredom_results['reason'] = f'Strategy effectiveness declining: {recent_action_patterns["effectiveness_trend"]:.3f}'
            
            # Switch to new strategy
            new_strategy = self._switch_action_strategy()
            boredom_results['strategy_switched'] = True
            boredom_results['new_strategy'] = new_strategy
        
        if boredom_results['boredom_detected']:
            logger.info(f" Boredom detected ({boredom_results['boredom_type']}): {boredom_results['reason']}")
        
        return boredom_results
    
    def _analyze_recent_action_patterns(self) -> Dict[str, Any]:
        """Analyze recent action patterns to detect loops and ineffective strategies."""
        lp_history = self.training_state.get('lp_history', [])
        if len(lp_history) < 3:
            return {'repetitive_loops': 0, 'effectiveness_trend': 0}
        
        # Get recent episodes
        recent_episodes = lp_history[-5:]
        
        # Check for repetitive action patterns (simplified detection)
        action_sequences = []
        effectiveness_values = []
        
        for episode in recent_episodes:
            # Track effectiveness trend
            effectiveness_values.append(episode.get('effectiveness', 0))
        
        # Calculate effectiveness trend (positive = improving, negative = declining)
        if len(effectiveness_values) >= 3:
            early_avg = np.mean(effectiveness_values[:2])
            late_avg = np.mean(effectiveness_values[-2:])
            effectiveness_trend = late_avg - early_avg
        else:
            effectiveness_trend = 0
        
        # Simple loop detection based on similar effectiveness scores
        loop_count = 0
        if len(effectiveness_values) >= 3:
            for i in range(len(effectiveness_values) - 2):
                if abs(effectiveness_values[i] - effectiveness_values[i+1]) < 0.01:
                    loop_count += 1
        
        return {
            'repetitive_loops': loop_count,
            'effectiveness_trend': effectiveness_trend,
            'recent_effectiveness': effectiveness_values[-1] if effectiveness_values else 0
        }
    
    def _trigger_action_experimentation(self):
        """Trigger experimentation with available actions to break out of bad cycles."""
        current_game = self.available_actions_memory.get('current_game_id')
        if current_game:
            # Mark that we should try random/experimental actions
            self.available_actions_memory['experiment_mode'] = True
            self.available_actions_memory['experiment_started'] = time.time()
            logger.info(f" Action experimentation triggered for {current_game}")
    
    def _switch_action_strategy(self) -> str:
        """Switch to a new action strategy to break ineffective patterns."""
        strategies = ['focused_exploration', 'random_sampling', 'pattern_matching', 'conservative_moves']
        current_strategy = self.training_state.get('current_strategy', 'focused_exploration')
        
        # Pick a different strategy
        available_strategies = [s for s in strategies if s != current_strategy]
        new_strategy = random.choice(available_strategies)
        
        self.training_state['current_strategy'] = new_strategy
        logger.info(f" Strategy switched: {current_strategy} → {new_strategy}")
        
        return new_strategy

    def _should_trigger_mid_game_sleep(self, step_count: int, recent_actions: List[Dict]) -> bool:
        """Determine if mid-game sleep should be triggered for pattern consolidation.
        
        Mid-game sleep enables continuous learning within episodes, matching top performers.
        Triggers on:
        - Pattern accumulation (every 100-200 actions)
        - Significant learning signals
        - Energy threshold with consolidation opportunities
        """
        # Don't sleep too early or too frequently
        last_sleep = getattr(self, '_last_mid_game_sleep_step', 0)
        min_sleep_interval = 50  # Reduced from 100 to 50 minimum actions between sleep cycles
        
        if step_count - last_sleep < min_sleep_interval:
            return False
        
        # Trigger conditions for mid-game sleep - more frequent for better learning
        pattern_accumulation_trigger = step_count % 75 == 0  # Reduced from 150 to 75 for more frequent sleep
        
        # Check for significant learning signals in recent actions
        if len(recent_actions) >= 10:
            recent_scores = [action.get('effectiveness', 0) for action in recent_actions[-10:]]
            high_learning_signal = np.mean(recent_scores) > 0.2  # Reduced from 0.3 to 0.2 for easier trigger
            
            if pattern_accumulation_trigger or high_learning_signal:
                return True
        
        # Energy-based trigger with learning opportunity - more frequent
        current_energy = getattr(self.energy_system, 'current_energy', 100.0)
        if current_energy < 70.0 and len(recent_actions) >= 15:  # Increased from 60.0 to 70.0 and reduced from 20 to 15 actions
            # Check if we have patterns worth consolidating
            effectiveness_values = [action.get('effectiveness', 0) for action in recent_actions[-15:]]  # Reduced from 20 to 15
            if len([e for e in effectiveness_values if e > 0.15]) >= 3:  # Reduced thresholds: 0.2→0.15, 5→3 actions
                return True
        
        return False

    async def _execute_mid_game_sleep(self, recent_actions: List[Dict], step_count: int) -> None:
        """Execute mid-game sleep cycle for pattern consolidation during gameplay.
        
        This is CRITICAL for matching top performer capabilities - consolidates patterns
        without ending the episode, enabling continuous learning within games.
        """
        self._last_mid_game_sleep_step = step_count
        
        # Extract high-value actions for consolidation
        effective_actions = [
            action for action in recent_actions[-50:]  # Last 50 actions
            if action.get('effectiveness', 0) > 0.2
        ]
        
        if len(effective_actions) > 0:
            logger.info(f" Mid-game consolidating {len(effective_actions)} effective actions")
            
            # Quick memory consolidation (shorter than post-episode sleep)
            try:
                # Sleep system not available in ContinuousLearningLoop - skip consolidation
                if False:  # Disabled: hasattr(self.agent, 'sleep_system')
                    # Trigger short consolidation cycle
                    consolidation_duration = min(10, len(effective_actions) * 0.5)  # 0.5s per action
                    
                    sleep_input = {
                        'effective_actions': effective_actions,
                        'consolidation_type': 'mid_game',
                        'duration': consolidation_duration
                    }
                    
                    # Quick memory strengthening without full sleep cycle
                    for i, action in enumerate(effective_actions):
                        # Memory system not available in ContinuousLearningLoop - skip salience update  
                        if False:  # Disabled: memory system not available
                            # Use action index as memory index for salience update
                            pass  # Memory operations disabled
                    
                    logger.info(f" Mid-game consolidation completed in {consolidation_duration:.1f}s")
                
            except Exception as e:
                logger.warning(f"Mid-game sleep error: {e}")
        
        # Partial energy restoration (not full like post-episode)
        if hasattr(self, 'energy_system'):
            #  CONSOLIDATED: Use unified energy system
            restored_energy = min(100.0, self.current_energy + 20.0)  # Small restoration
            self.current_energy = restored_energy
            logger.info(f" Energy restored: {self.current_energy - 20.0:.2f} → {restored_energy:.2f}")

    async def _simulate_mid_game_consolidation(self, effective_actions: List[Dict], total_actions: int) -> None:
        """Simulate mid-game consolidation points as if sleep occurred during gameplay.
        
        Since we run complete game sessions, this retroactively applies the consolidation
        that would have happened if we had mid-game sleep cycles. This simulates the
        continuous learning that top performers achieve.
        """
        if not effective_actions:
            return
        
        # Calculate consolidation points based on action count
        consolidation_points = max(1, total_actions // 150)  # Every 150 actions
        actions_per_consolidation = len(effective_actions) // max(1, consolidation_points)
        
        logger.info(f" Simulating {consolidation_points} mid-game consolidation points for {total_actions} actions")
        
        # Process effective actions in chunks as if consolidated during gameplay
        for i in range(consolidation_points):
            start_idx = i * actions_per_consolidation
            end_idx = min((i + 1) * actions_per_consolidation, len(effective_actions))
            action_chunk = effective_actions[start_idx:end_idx]
            
            if not action_chunk:
                continue
                
            logger.info(f"   Consolidation point {i+1}: {len(action_chunk)} actions")
            
            # Strengthen memories for this chunk with success weighting
            try:
                for action in action_chunk:
                    effectiveness = action.get('effectiveness', 0)
                    
                    # Apply success weighting (10x for wins)
                    if action.get('success', False):
                        weight = effectiveness * 10.0  # SUCCESS-WEIGHTED PRIORITY
                    else:
                        weight = effectiveness
                    
                    # Strengthen memory if above threshold
                    if hasattr(self.demo_agent, 'memory') and hasattr(self.demo_agent.memory, 'update_memory_salience') and weight > 0.2:
                        # Use action index as memory index for salience update
                        memory_idx = torch.tensor([i % self.demo_agent.memory.memory_size])
                        salience_val = torch.tensor([weight])
                        self.demo_agent.memory.update_memory_salience(memory_idx, salience_val)
                
                # Record consolidation in training state
                if 'mid_game_consolidations' not in self.training_state:
                    self.training_state['mid_game_consolidations'] = 0
                self.training_state['mid_game_consolidations'] += 1
                
            except Exception as e:
                logger.warning(f"Consolidation point {i+1} error: {e}")

    def _analyze_action_sequences(self, effective_actions: List[Dict]) -> Dict[str, Any]:
        """Analyze action sequences to identify learning patterns and strategies.
        
        This transforms the traditional episode structure into action sequences,
        enabling continuous learning analysis like top performers.
        """
        if not effective_actions:
            return {'sequences': [], 'patterns': [], 'strategy_effectiveness': {}}
        
        # Group actions into sequences based on effectiveness patterns
        sequences = []
        current_sequence = []
        
        for i, action in enumerate(effective_actions):
            effectiveness = action.get('effectiveness', 0)
            
            # Start new sequence on effectiveness jumps or strategy changes
            if (current_sequence and 
                abs(effectiveness - current_sequence[-1].get('effectiveness', 0)) > 0.3):
                sequences.append(current_sequence)
                current_sequence = []
            
            current_sequence.append(action)
        
        if current_sequence:
            sequences.append(current_sequence)
        
        # Analyze patterns within sequences
        patterns = []
        for seq in sequences:
            if len(seq) >= 3:
                pattern = {
                    'length': len(seq),
                    'avg_effectiveness': np.mean([a.get('effectiveness', 0) for a in seq]),
                    'effectiveness_trend': self._calculate_trend([a.get('effectiveness', 0) for a in seq]),
                    'action_types': [a.get('action_type', 'unknown') for a in seq],
                    'success_rate': len([a for a in seq if a.get('success', False)]) / len(seq)
                }
                patterns.append(pattern)
        
        # Strategy effectiveness analysis
        strategy_effectiveness = {}
        for action in effective_actions:
            strategy = action.get('strategy', 'unknown')
            if strategy not in strategy_effectiveness:
                strategy_effectiveness[strategy] = {'total': 0, 'count': 0}
            strategy_effectiveness[strategy]['total'] += action.get('effectiveness', 0)
            strategy_effectiveness[strategy]['count'] += 1
        
        # Calculate average effectiveness per strategy
        for strategy in strategy_effectiveness:
            count = strategy_effectiveness[strategy]['count']
            if count > 0:
                strategy_effectiveness[strategy]['avg'] = strategy_effectiveness[strategy]['total'] / count
        
        return {
            'sequences': sequences,
            'patterns': patterns,
            'strategy_effectiveness': strategy_effectiveness,
            'total_sequences': len(sequences),
            'avg_sequence_length': np.mean([len(seq) for seq in sequences]) if sequences else 0
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a list of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return slope
        except:
            return 0.0
    
    def _calculate_session_effectiveness(self, session_result: Dict[str, Any], game_results: Dict[str, Any]) -> float:  # Renamed method
        """Calculate mastery session effectiveness with SUCCESS-WEIGHTED PRIORITY for memory retention.
        
        WIN attempts get 10x higher effectiveness for memory prioritization.
        This ensures successful strategies are strongly retained.
        """
        base_effectiveness = 0.0
        
        # SUCCESS MULTIPLIER - Critical for memory retention
        success = session_result.get('success', False)  # Updated parameter name
        if success:
            success_multiplier = 10.0  # WIN attempts get 10x priority
            base_effectiveness += 0.5   # Base bonus for success
        else:
            success_multiplier = 1.0    # Normal priority for failures
        
        # Score-based effectiveness
        current_score = session_result.get('final_score', 0)  # Updated parameter name
        previous_sessions = game_results.get('episodes', [])  # Will eventually rename this key too
        
        if len(previous_sessions) > 1:  # Updated variable name
            previous_scores = [session.get('final_score', 0) for session in previous_sessions[-5:]]  # Updated variable name
            avg_previous_score = sum(previous_scores) / len(previous_scores)
            
            if avg_previous_score > 0:
                score_improvement = (current_score - avg_previous_score) / avg_previous_score
                base_effectiveness += max(0, score_improvement * 0.3)
        
        # Action efficiency effectiveness
        actions_taken = session_result.get('actions_taken', 0)  # Updated parameter name
        if actions_taken > 0 and current_score > 0:
            efficiency = current_score / actions_taken  # Score per action
            base_effectiveness += min(efficiency * 0.1, 0.2)
        
        # Pattern discovery effectiveness
        patterns_found = len(session_result.get('patterns_discovered', []))  # Updated parameter name
        base_effectiveness += min(patterns_found * 0.05, 0.1)
        
        # Apply success multiplier for memory prioritization
        final_effectiveness = base_effectiveness * success_multiplier
        
        #  CONSOLIDATED: Update unified energy system with session performance
        success = session_result.get('success', False)
        score_improvement = current_score - session_result.get('initial_score', 0)
        
        # Use unified energy consumption system
        action_effective = success or score_improvement > 0
        is_exploration = actions_taken > 50  # Consider longer sessions as exploration
        is_repetitive = actions_taken > 200  # Penalize very long sessions
        
        final_energy = self._unified_energy_consumption(
            action_effective=action_effective,
            is_exploration=is_exploration, 
            is_repetitive=is_repetitive
        )
        
        # Check if unified energy system suggests sleep
        if self._should_trigger_sleep_cycle(actions_taken, final_effectiveness):
            logger.info(f" Unified energy system suggests sleep: energy={final_energy:.1f}/100")
        
        return min(final_effectiveness, 5.0)  # Cap but allow high values for wins
    
    def _convert_effectiveness_to_lp(self, effectiveness: float) -> float:
        """Convert episode effectiveness to learning progress using Tabula Rasa principles.
        
        LP should reflect intrinsic motivation and genuine learning advancement,
        not just external rewards.
        """
        # Base conversion with diminishing returns
        lp = effectiveness * 0.8  # Base conversion factor
        
        # Add noise for exploration
        noise = random.uniform(-0.05, 0.05)
        lp += noise
        
        # Ensure positive but realistic LP
    def _calculate_session_performance(self, session_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive session performance using Tabula Rasa metrics."""
        games_played = session_results['games_played']
        if not games_played:
            return {}
            
        total_episodes = sum(len(game.get('episodes', [])) for game in games_played.values())
        total_wins = sum(sum(1 for ep in game.get('episodes', []) if ep.get('success', False)) 
                        for game in games_played.values())
        # Enhanced null safety for score calculations
        total_score = 0
        for game in games_played.values():
            episodes = game.get('episodes', []) if game else []
            for ep in episodes:
                if ep:
                    score = ep.get('final_score')
                    if score is not None and isinstance(score, (int, float)):
                        total_score += score
        
        # Standard performance metrics
        standard_metrics = {
            'games_trained': len(games_played),
            'total_episodes': total_episodes,
            'overall_win_rate': total_wins / max(1, total_episodes),
            'overall_average_score': total_score / max(1, total_episodes),
            'learning_efficiency': self._calculate_learning_efficiency(session_results),
            'knowledge_transfer_score': self._calculate_knowledge_transfer_score(session_results)
        }
        
        # Comprehensive Tabula Rasa metrics
        tabula_rasa_metrics = self._calculate_tabula_rasa_metrics()
        
        # Combine all metrics
        comprehensive_metrics = {**standard_metrics, **tabula_rasa_metrics}
        
        return comprehensive_metrics
    
    def _calculate_tabula_rasa_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive metrics for all Tabula Rasa principles."""
        metrics = {}
        
        # 1. Learning Progress Stability
        lp_history = self.training_state.get('lp_history', [])
        if len(lp_history) >= 5:
            recent_lp = [entry['learning_progress'] for entry in lp_history[-10:]]
            lp_variance = np.var(recent_lp)
            lp_mean = np.mean(recent_lp)
            metrics['lp_stability'] = max(0, 1.0 - lp_variance)  # Lower variance = higher stability
            metrics['lp_magnitude'] = lp_mean
        else:
            metrics['lp_stability'] = 0.0
            metrics['lp_magnitude'] = 0.0
        
        # 2. Memory System Effectiveness
        memory_tracker = self.memory_consolidation_tracker
        metrics['memory_consolidation_rate'] = memory_tracker.get('last_consolidation_score', 0.0)
        metrics['memory_operations_efficiency'] = min(
            memory_tracker.get('memory_operations_per_cycle', 0) / 10.0, 1.0
        )
        
        # 3. Sleep Cycle Quality
        # Ensure sleep_state_tracker is initialized
        if not hasattr(self, 'sleep_state_tracker'):
            self.sleep_state_tracker = {
                'is_currently_sleeping': False,
                'sleep_cycles_this_session': 0,
                'total_sleep_time': 0.0,
                'memory_operations_per_cycle': 0,
                'last_sleep_trigger': [],
                'sleep_quality_scores': [],
                'current_energy_level': 100.0,
                'last_sleep_cycle': 0,
                'total_sleep_cycles': 0
            }
        
        sleep_tracker = self.sleep_state_tracker
        sleep_scores = sleep_tracker.get('sleep_quality_scores', [])
        if sleep_scores:
            metrics['sleep_quality'] = np.mean(sleep_scores)
            metrics['sleep_efficiency'] = sleep_tracker.get('sleep_efficiency', 0.0)
        else:
            metrics['sleep_quality'] = 0.0
            metrics['sleep_efficiency'] = 0.0
        
        # 4. Energy System Health
        metrics['energy_management'] = 1.0  # Placeholder - would need energy state tracking
        
        # 5. Boredom Detection Effectiveness
        boredom_detections = sum(1 for entry in lp_history if entry.get('boredom_detected', False))
        metrics['boredom_detection_rate'] = boredom_detections / max(1, len(lp_history))
        
        # 6. Curriculum Complexity Progress
        complexity_level = self.training_config.get('curriculum_complexity', 1)
        max_complexity = self.training_config.get('max_complexity', 10)
        metrics['curriculum_progress'] = complexity_level / max_complexity
        
        # 7. Goal System Progress
        discovered_goals = len(self.training_state.get('discovered_goals', []))
        metrics['goal_discovery_rate'] = min(discovered_goals / 10.0, 1.0)  # Normalize to 10 goals
        metrics['goal_system_phase'] = {'survival': 0.33, 'template': 0.67, 'emergent': 1.0}.get(
            self.goal_system.current_phase.value, 0.0
        )
        
        # 8. Overall Tabula Rasa Health Score
        core_scores = [
            metrics['lp_stability'],
            metrics['memory_consolidation_rate'],
            metrics['sleep_quality'],
            metrics['curriculum_progress']
        ]
        metrics['tabula_rasa_health'] = np.mean([s for s in core_scores if s > 0])
        
        return metrics
    
    def _process_emergent_goals(self, episode_result: Dict[str, Any], game_results: Dict[str, Any], learning_progress: float) -> Dict[str, Any]:
        """Process emergent goal discovery from episode patterns and outcomes.
        
        This integrates the goal invention system to automatically discover new goals
        based on patterns, failures, and learning progress signals.
        """
        goal_results = {
            'new_goals_discovered': False,
            'new_goals': [],
            'goal_updates': [],
            'phase_transition': False
        }
        
        # Extract patterns and context for goal discovery
        patterns = episode_result.get('patterns_discovered', [])
        success = episode_result.get('success', False)
        score = episode_result.get('final_score', 0)
        game_state = episode_result.get('game_state', 'NOT_FINISHED')
        
        # Create agent state for goal system
        agent_state = {
            'learning_progress': learning_progress,
            'score': score,
            'success': success,
            'game_state': game_state,
            'patterns': patterns,
            'episode_count': self.training_state['episode_count']
        }
        
        # Process goals through the goal invention system
        try:
            # Create proper agent state for goal system
            from core.data_models import AgentState
            goal_agent_state = AgentState(
                position=torch.tensor([0.0, 0.0, 1.0]),
                orientation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
                energy=self.current_energy,
                hidden_state=torch.zeros(64),
                active_goals=[],
                memory_state=None,
                timestamp=self.training_state.get('episode_count', 0)
            )
            
            # Update current goals with learning progress (with error handling)
            current_goals = self.goal_system.get_active_goals(goal_agent_state)
            for goal in current_goals:
                if learning_progress is not None and hasattr(goal, 'progress') and goal.progress is not None:
                    self.goal_system.update_goal_progress(goal, learning_progress)
            
            # Check for emergent goal discovery by adding high-LP experience
            if learning_progress is not None and learning_progress > 0.1 and self.goal_system.current_phase == GoalPhase.EMERGENT:
                # Add experience to emergent goals for clustering
                state_repr = torch.randn(64)  # Placeholder state representation
                self.goal_system.emergent_goals.add_experience(state_repr, learning_progress, goal_agent_state)
                
                # Check if new goals were discovered
                new_goals = self.goal_system.emergent_goals.get_active_goals(goal_agent_state)
                if len(new_goals) > len(current_goals):
                    goal_results['new_goals_discovered'] = True
                    goal_results['new_goals'] = new_goals[len(current_goals):]
                    
                    # Store discovered goals for tracking
                    if 'discovered_goals' not in self.training_state:
                        self.training_state['discovered_goals'] = []
                    
                    for goal in goal_results['new_goals']:
                        goal_entry = {
                            'goal_id': goal.goal_id,
                        'description': goal.description,
                        'priority': goal.priority,
                        'discovered_at_episode': self.training_state['episode_count'],
                        'discovered_at_score': score,
                        'timestamp': time.time()
                    }
                    self.training_state['discovered_goals'].append(goal_entry)
            
            # Check for phase transitions
            if self.goal_system.check_phase_transition():
                goal_results['phase_transition'] = True
                new_phase = self.goal_system.current_phase
                print(f" Goal system transitioned to {new_phase.value} phase!")
                
        except Exception as e:
            logger.warning(f"Goal processing failed: {e}")
        
        return goal_results
    
    async def _execute_sleep_cycle(self, game_id: str, episode_count: int) -> Dict[str, Any]:
        """Execute a sleep cycle with memory consolidation."""
        # Ensure sleep_state_tracker is initialized
        if not hasattr(self, 'sleep_state_tracker'):
            self.sleep_state_tracker = {
                'is_currently_sleeping': False,
                'sleep_cycles_this_session': 0,
                'total_sleep_time': 0.0,
                'memory_operations_per_cycle': 0,
                'last_sleep_trigger': [],
                'sleep_quality_scores': [],
                'current_energy_level': 100.0,
                'last_sleep_cycle': 0,
                'total_sleep_cycles': 0
            }
        
        self.sleep_state_tracker['is_currently_sleeping'] = True
        self.sleep_state_tracker['sleep_cycles_this_session'] = self.sleep_state_tracker.get('sleep_cycles_this_session', 0) + 1
        
        sleep_start_time = time.time()
        
        try:
            # Simulate sleep cycle execution
            sleep_results = {
                'sleep_duration': 0.0,
                'memory_consolidation_performed': False,
                'memories_prioritized': False,
                'high_salience_strengthened': 0,
                'low_salience_decayed': 0,
                'compression_applied': False,
                'consolidation_score': 0.0
            }
            
            # Phase 1: Memory Prioritization
            prioritization_results = self._prioritize_memories_by_salience()
            sleep_results.update(prioritization_results)
            
            # Phase 2: Memory Consolidation
            consolidation_results = self._consolidate_prioritized_memories()
            sleep_results.update(consolidation_results)
            
            # Phase 3: Memory Compression (if using decay mode)
            compression_results = self._apply_memory_compression()
            sleep_results.update(compression_results)
            
            sleep_end_time = time.time()
            sleep_results['sleep_duration'] = sleep_end_time - sleep_start_time
            
            # Update sleep state tracker
            self.sleep_state_tracker['total_sleep_time'] = self.sleep_state_tracker.get('total_sleep_time', 0.0) + sleep_results['sleep_duration']
            if 'sleep_quality_scores' not in self.sleep_state_tracker:
                self.sleep_state_tracker['sleep_quality_scores'] = []
            self.sleep_state_tracker['sleep_quality_scores'].append(sleep_results['consolidation_score'])
            
            logger.info(f"Sleep cycle completed for {game_id} episode {episode_count}: {sleep_results['sleep_duration']:.2f}s")
            
            return sleep_results
            
        finally:
            self.sleep_state_tracker['is_currently_sleeping'] = False
    
    def _prioritize_memories_by_salience(self) -> Dict[str, Any]:
        """Prioritize memories based on salience values."""
        # Ensure memory consolidation tracker is initialized
        if not hasattr(self, 'memory_consolidation_tracker'):
            self.memory_consolidation_tracker = {
                'memory_operations_per_cycle': 0,
                'consolidation_operations_count': 0,
                'is_consolidating_memories': False,
                'is_prioritizing_memories': False,
                'memory_compression_active': False,
                'high_salience_memories_strengthened': 0,
                'low_salience_memories_decayed': 0,
                'last_consolidation_score': 0.0,
                'total_memory_operations': 0
            }
        
        self.memory_consolidation_tracker['is_prioritizing_memories'] = True
        
        try:
            # Simulate memory prioritization
            if self.salience_calculator:
                # Get high salience experiences for priority processing
                high_salience_count = len(self.salience_calculator.get_high_salience_experiences(threshold=0.4))  # Reduced from 0.6 to 0.4
                medium_salience_count = len(self.salience_calculator.get_high_salience_experiences(threshold=0.3)) - high_salience_count
                
                return {
                    'memories_prioritized': True,
                    'high_priority_memories': high_salience_count,
                    'medium_priority_memories': medium_salience_count,
                    'prioritization_operations': high_salience_count + medium_salience_count
                }
            else:
                return {'memories_prioritized': False}
                
        finally:
            self.memory_consolidation_tracker['is_prioritizing_memories'] = False
    
    def _consolidate_prioritized_memories(self) -> Dict[str, Any]:
        """Consolidate prioritized memories with strengthening and decay."""
        # Ensure memory consolidation tracker is initialized
        if not hasattr(self, 'memory_consolidation_tracker'):
            self.memory_consolidation_tracker = {
                'memory_operations_per_cycle': 0,
                'consolidation_operations_count': 0,
                'is_consolidating_memories': False,
                'is_prioritizing_memories': False,
                'memory_compression_active': False,
                'high_salience_memories_strengthened': 0,
                'low_salience_memories_decayed': 0,
                'last_consolidation_score': 0.0,
                'total_memory_operations': 0
            }
        
        self.memory_consolidation_tracker['is_consolidating_memories'] = True
        
        try:
            # Simulate memory consolidation operations
            consolidation_ops = 0
            high_salience_strengthened = 0
            low_salience_decayed = 0
            
            if self.salience_calculator:
                # Strengthen high-salience memories
                high_salience_experiences = self.salience_calculator.get_high_salience_experiences(threshold=0.7)
                high_salience_strengthened = len(high_salience_experiences)
                consolidation_ops += high_salience_strengthened
                
                # Decay low-salience memories (but protect winning memories)
                all_experiences = self.salience_calculator.get_high_salience_experiences(threshold=0.0)
                low_salience_candidates = [exp for exp in all_experiences if exp['salience'] < 0.3]
                
                # Filter out protected memories from decay
                low_salience_experiences = []
                protected_count = 0
                for exp in low_salience_candidates:
                    if self._is_experience_protected(exp):
                        protected_count += 1
                        # Apply salience floor instead of decay
                        exp['salience'] = max(exp['salience'], self._get_protection_floor(exp))
                    else:
                        low_salience_experiences.append(exp)
                
                low_salience_decayed = len(low_salience_experiences)
                if protected_count > 0:
                    print(f"    Protected {protected_count} winning memories from salience decay")
                
                consolidation_ops += low_salience_decayed
                
                # Calculate consolidation effectiveness
                consolidation_score = (high_salience_strengthened * 2 - low_salience_decayed * 0.5) / max(1, consolidation_ops)
            else:
                consolidation_score = 0.0
            
            # Update tracking
            self.memory_consolidation_tracker['consolidation_operations_count'] = self.memory_consolidation_tracker.get('consolidation_operations_count', 0) + consolidation_ops
            self.memory_consolidation_tracker['high_salience_memories_strengthened'] = self.memory_consolidation_tracker.get('high_salience_memories_strengthened', 0) + high_salience_strengthened
            self.memory_consolidation_tracker['low_salience_memories_decayed'] = self.memory_consolidation_tracker.get('low_salience_memories_decayed', 0) + low_salience_decayed
            self.memory_consolidation_tracker['last_consolidation_score'] = consolidation_score
            self.memory_consolidation_tracker['memory_operations_per_cycle'] = consolidation_ops
            
            return {
                'memory_consolidation_performed': True,
                'consolidation_operations': consolidation_ops,
                'high_salience_strengthened': high_salience_strengthened,
                'low_salience_decayed': low_salience_decayed,
                'consolidation_score': consolidation_score
            }
            
        finally:
            self.memory_consolidation_tracker['is_consolidating_memories'] = False
    
    def _apply_memory_compression(self) -> Dict[str, Any]:
        """Apply memory compression if using decay mode."""
        if not self.salience_calculator or self.salience_calculator.mode == SalienceMode.LOSSLESS:
            return {'compression_applied': False}
        
        # Ensure memory consolidation tracker is initialized
        if not hasattr(self, 'memory_consolidation_tracker'):
            self.memory_consolidation_tracker = {
                'memory_operations_per_cycle': 0,
                'consolidation_operations_count': 0,
                'is_consolidating_memories': False,
                'is_prioritizing_memories': False,
                'memory_compression_active': False,
                'high_salience_memories_strengthened': 0,
                'low_salience_memories_decayed': 0,
                'last_consolidation_score': 0.0,
                'total_memory_operations': 0
            }
        
        self.memory_consolidation_tracker['memory_compression_active'] = True
        
        try:
            # Simulate compression operations
            compression_operations = 0
            compressed_memories_count = 0
            
            if hasattr(self.salience_calculator, 'compressed_memories'):
                # Apply compression to low-salience memories
                all_experiences = self.salience_calculator.get_high_salience_experiences(threshold=0.0)
                compressible_experiences = [exp for exp in all_experiences if exp['salience'] < 0.2]
                
                # Group similar experiences for compression
                compression_groups = self._group_experiences_for_compression(compressible_experiences)
                
                for group in compression_groups:
                    if len(group) >= 3:  # Only compress if we have enough similar experiences
                        compressed_memory = self._compress_experience_group(group)
                        self.salience_calculator.compressed_memories.append(compressed_memory)
                        compressed_memories_count += 1
                        compression_operations += len(group)
            
            return {
                'compression_applied': True,
                'compression_operations': compression_operations,
                'compressed_memories_count': compressed_memories_count,
                'compression_ratio': compressed_memories_count / max(1, compression_operations)
            }
            
        finally:
            self.memory_consolidation_tracker['memory_compression_active'] = False
    
    def _should_activate_contrarian_strategy(self, game_id: str, consecutive_failures: int) -> Dict[str, Any]:
        """
        Determine if contrarian strategy should be activated when consistently reaching GAME_OVER.
        
        Contrarian strategy: If normal actions consistently fail, try the opposite approach
        based on memory of previous failed patterns.
        """
        contrarian_decision = {
            'activate': False,
            'reason': None,
            'confidence': 0.0,
            'strategy_type': 'standard'
        }
        
        # Activate contrarian if:
        # 1. Multiple consecutive failures (3+ GAME_OVER states)
        # 2. Same game repeatedly failing
        # 3. No learning progress being made
        
        if consecutive_failures >= 3:
            # Check if this specific game has been consistently failing
            game_history = self.game_complexity_history.get(game_id, {})
            recent_effectiveness = game_history.get('recent_effectiveness', [])
            
            if len(recent_effectiveness) >= 3 and all(eff < 0.1 for eff in recent_effectiveness[-3:]):
                contrarian_decision.update({
                    'activate': True,
                    'reason': f'Consistent GAME_OVER on {game_id} after {consecutive_failures} failures',
                    'confidence': min(0.9, consecutive_failures * 0.2),
                    'strategy_type': 'memory_inverse'
                })
                print(f" CONTRARIAN MODE ACTIVATED: {contrarian_decision['reason']}")
                print(f"   Confidence: {contrarian_decision['confidence']:.1%}")
        
        return contrarian_decision

    def _apply_contrarian_strategy_to_command(self, cmd: List[str], contrarian_decision: Dict[str, Any]) -> List[str]:
        """
        Modify the command to implement contrarian strategy based on memory analysis.
        """
        if not contrarian_decision['activate']:
            return cmd
        
        # Add contrarian flags to the command
        if contrarian_decision['strategy_type'] == 'memory_inverse':
            cmd.extend(['--strategy', 'contrarian'])
            cmd.extend(['--invert-memory-patterns', 'true'])
            
        print(f" Applied contrarian strategy: {contrarian_decision['strategy_type']}")
        return cmd
    
    def _evaluate_game_reset_decision(self, game_id: str, episode_count: int, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate whether the model should decide to reset the game."""
        reset_decision = {
            'should_reset': False,
            'reason': None,
            'confidence': 0.0,
            'expected_benefit': 0.0
        }
        
        criteria = self.game_reset_tracker['reset_decision_criteria']
        
        # Evaluate reset conditions
        reset_conditions = {
            'consecutive_failures': agent_state.get('consecutive_failures', 0) >= criteria['consecutive_failures'],
            'performance_degradation': agent_state.get('learning_progress', 1.0) < -criteria['performance_degradation'],
            'memory_overflow': agent_state.get('memory_usage', 0.0) >= criteria['memory_overflow'],
            'energy_depletion': agent_state.get('energy', 100.0) <= criteria['energy_depletion'],
            'negative_learning_trend': agent_state.get('performance_trend') == 'declining'
        }
        
        # Calculate reset probability based on conditions
        active_conditions = [k for k, v in reset_conditions.items() if v]
        reset_probability = len(active_conditions) / len(reset_conditions)
        
        # Additional factors that influence reset decision
        if episode_count > 20:  # Don't reset too early
            # Calculate expected benefit of reset
            recent_performance = self._get_recent_game_performance(game_id)
            expected_benefit = self._calculate_reset_expected_benefit(recent_performance, active_conditions)
            
            # Make reset decision if conditions and expected benefit are high enough
            if reset_probability > 0.4 and expected_benefit > 0.3:
                reset_decision.update({
                    'should_reset': True,
                    'reason': f"Multiple conditions met: {', '.join(active_conditions)}",
                    'confidence': reset_probability,
                    'expected_benefit': expected_benefit
                })
        
        return reset_decision
    
    def _record_reset_decision(self, reset_decision: Dict[str, Any]):
        """Record that a reset decision was made."""
        self.game_reset_tracker['reset_decisions_made'] += 1
        self.game_reset_tracker['reset_reasons'].append(reset_decision['reason'])
        self.game_reset_tracker['last_reset_decision'] = {
            'timestamp': time.time(),
            'reason': reset_decision['reason'],
            'confidence': reset_decision['confidence'],
            'expected_benefit': reset_decision['expected_benefit']
        }
        
        logger.info(f"GAME RESET DECISION: {reset_decision['reason']} (confidence: {reset_decision['confidence']:.2f})")
    
    def _evaluate_reset_effectiveness(self, episode_result: Dict[str, Any], reset_decision: Dict[str, Any]) -> float:
        """Evaluate how effective the reset decision was."""
        # Compare performance after reset
        post_reset_score = episode_result.get('final_score', 0)
        expected_benefit = reset_decision.get('expected_benefit', 0)
        
        # Calculate effectiveness (how well the reset performed vs expectations)
        if expected_benefit > 0:
            effectiveness = min(post_reset_score / (expected_benefit * 100), 2.0)  # Cap at 2x expected
        else:
            effectiveness = 1.0 if post_reset_score > 20 else 0.0
        
        # Update reset success rate
        if self.game_reset_tracker['reset_effectiveness_scores']:
            current_avg = sum(self.game_reset_tracker['reset_effectiveness_scores']) / len(self.game_reset_tracker['reset_effectiveness_scores'])
            self.game_reset_tracker['reset_success_rate'] = (current_avg + effectiveness) / 2
        else:
            self.game_reset_tracker['reset_success_rate'] = effectiveness
        
        logger.info(f"Reset effectiveness: {effectiveness:.2f} (expected: {expected_benefit:.2f}, actual score: {post_reset_score})")
        return effectiveness
    
    def _get_recent_game_performance(self, game_id: str) -> Dict[str, float]:
        """Get recent performance metrics for a specific game."""
        # This would normally query historical data
        return {
            'recent_win_rate': 0.2,
            'recent_avg_score': 35.0,
            'performance_trend': -0.1,  # Negative = declining
            'episodes_played': 15
        }
    
    def _calculate_reset_expected_benefit(self, recent_performance: Dict[str, float], active_conditions: List[str]) -> float:
        """Calculate expected benefit of resetting the game."""
        base_benefit = 0.1  # Base benefit of a fresh start
        
        # Increase benefit based on how poorly things are going
        if 'performance_degradation' in active_conditions:
            base_benefit += 0.2
        if 'consecutive_failures' in active_conditions:
            base_benefit += 0.2
        if 'energy_depletion' in active_conditions:
            base_benefit += 0.1
        if 'memory_overflow' in active_conditions:
            base_benefit += 0.3  # Fresh memory state can be very beneficial
        
        # Factor in current performance level
        current_win_rate = recent_performance.get('recent_win_rate', 0.0)
        if current_win_rate < 0.1:  # Very poor performance
            base_benefit += 0.2
        
        return min(base_benefit, 1.0)  # Cap at 100% expected benefit
    
    def _group_experiences_for_compression(self, experiences: List[Dict]) -> List[List[Dict]]:
        """Group similar experiences for compression."""
        # Simple grouping by similarity - in real implementation would be more sophisticated
        groups = []
        similarity_threshold = 0.3
        
        for exp in experiences:
            added_to_group = False
            for group in groups:
                if len(group) > 0:
                    # Simple similarity check
                    similarity = self._calculate_experience_similarity(exp, group[0])
                    if similarity > similarity_threshold:
                        group.append(exp)
                        added_to_group = True
                        break
            
            if not added_to_group:
                groups.append([exp])
        
        return [group for group in groups if len(group) >= 2]  # Only return groups with multiple experiences
    
    def _calculate_experience_similarity(self, exp1: Dict, exp2: Dict) -> float:
        """Calculate similarity between two experiences."""
        # Simplified similarity calculation
        score_diff = abs(exp1.get('final_score', 0) - exp2.get('final_score', 0))
        salience_diff = abs(exp1.get('salience', 0.5) - exp2.get('salience', 0.5))
        
        # Higher similarity = lower differences
        similarity = 1.0 - (score_diff / 100.0 + salience_diff) / 2.0
        return max(0.0, similarity)
    
    def _compress_experience_group(self, group: List[Dict]) -> Dict[str, Any]:
        """Compress a group of similar experiences into a single compressed memory."""
        if not group:
            return {}
        
        # Create compressed representation
        avg_score = sum((exp.get('final_score') or 0) for exp in group) / len(group)
        avg_salience = sum(exp.get('salience', 0.5) for exp in group) / len(group)
        
        return {
            'type': 'compressed_memory',
            'original_count': len(group),
            'avg_score': avg_score,
            'avg_salience': avg_salience,
            'compression_timestamp': time.time(),
            'pattern_summary': f"Similar experiences with avg score {avg_score:.1f}"
        }
    
    def _get_memory_consolidation_status(self) -> Dict[str, Any]:
        """Get current memory consolidation status."""
        # Ensure memory consolidation tracker is initialized
        if not hasattr(self, 'memory_consolidation_tracker'):
            self.memory_consolidation_tracker = {
                'memory_operations_per_cycle': 0,
                'consolidation_operations_count': 0,
                'is_consolidating_memories': False,
                'is_prioritizing_memories': False,
                'memory_compression_active': False,
                'high_salience_memories_strengthened': 0,
                'low_salience_memories_decayed': 0,
                'last_consolidation_score': 0.0,
                'total_memory_operations': 0
            }
        
        return {
            'is_consolidating': self.memory_consolidation_tracker.get('is_consolidating_memories', False),
            'is_prioritizing': self.memory_consolidation_tracker.get('is_prioritizing_memories', False),
            'compression_active': self.memory_consolidation_tracker.get('memory_compression_active', False),
            'total_consolidation_ops': self.memory_consolidation_tracker.get('consolidation_operations_count', 0),
            'last_consolidation_score': self.memory_consolidation_tracker.get('last_consolidation_score', 0.0),
            'high_salience_strengthened': self.memory_consolidation_tracker.get('high_salience_memories_strengthened', 0),
            'low_salience_decayed': self.memory_consolidation_tracker.get('low_salience_memories_decayed', 0)
        }
    
    def _get_current_sleep_state_info(self) -> Dict[str, Any]:
        """Get current sleep state information."""
        # Ensure sleep_state_tracker is initialized
        if not hasattr(self, 'sleep_state_tracker'):
            self.sleep_state_tracker = {
                'is_currently_sleeping': False,
                'sleep_cycles_this_session': 0,
                'total_sleep_time': 0.0,
                'memory_operations_per_cycle': 0,
                'last_sleep_trigger': [],
                'sleep_quality_scores': [],
                'current_energy_level': 100.0,
                'last_sleep_cycle': 0,
                'total_sleep_cycles': 0
            }
        
        return {
            'is_sleeping': self.sleep_state_tracker.get('is_currently_sleeping', False),
            'sleep_cycles_completed': self.sleep_state_tracker.get('sleep_cycles_this_session', 0),
            'total_sleep_time': self.sleep_state_tracker.get('total_sleep_time', 0.0),
            'last_sleep_trigger': self.sleep_state_tracker.get('last_sleep_trigger', []),
            'current_energy_level': getattr(self, 'current_energy', 1.0),  # Include actual energy level
            'average_sleep_quality': (
                sum(self.sleep_state_tracker.get('sleep_quality_scores', [])) / 
                max(1, len(self.sleep_state_tracker.get('sleep_quality_scores', [])))
            ) if self.sleep_state_tracker.get('sleep_quality_scores', []) else 0.0
        }
    
    # ====== PUBLIC STATUS METHODS FOR USER QUERIES ======
    
    def is_consolidating_memories(self) -> bool:
        """Return True if currently consolidating memories."""
        # Ensure memory consolidation tracker is initialized
        if not hasattr(self, 'memory_consolidation_tracker'):
            self.memory_consolidation_tracker = {
                'memory_operations_per_cycle': 0,
                'consolidation_operations_count': 0,
                'is_consolidating_memories': False,
                'is_prioritizing_memories': False,
                'memory_compression_active': False,
                'high_salience_memories_strengthened': 0,
                'low_salience_memories_decayed': 0,
                'last_consolidation_score': 0.0,
                'total_memory_operations': 0
            }
        return self.memory_consolidation_tracker.get('is_consolidating_memories', False)
    
    def is_prioritizing_memories(self) -> bool:
        """Return True if currently prioritizing memories."""
        # Ensure memory consolidation tracker is initialized
        if not hasattr(self, 'memory_consolidation_tracker'):
            self.memory_consolidation_tracker = {
                'memory_operations_per_cycle': 0,
                'consolidation_operations_count': 0,
                'is_consolidating_memories': False,
                'is_prioritizing_memories': False,
                'memory_compression_active': False,
                'high_salience_memories_strengthened': 0,
                'low_salience_memories_decayed': 0,
                'last_consolidation_score': 0.0,
                'total_memory_operations': 0
            }
        return self.memory_consolidation_tracker.get('is_prioritizing_memories', False)
    
    def is_sleeping(self) -> bool:
        """Return True if agent is currently in sleep state."""
        # Ensure sleep_state_tracker is initialized
        if not hasattr(self, 'sleep_state_tracker'):
            self.sleep_state_tracker = {
                'is_currently_sleeping': False,
                'sleep_cycles_this_session': 0,
                'total_sleep_time': 0.0,
                'memory_operations_per_cycle': 0,
                'last_sleep_trigger': [],
                'sleep_quality_scores': [],
                'current_energy_level': 100.0,
                'last_sleep_cycle': 0,
                'total_sleep_cycles': 0
            }
        return self.sleep_state_tracker.get('is_currently_sleeping', False)
    
    def is_memory_compression_active(self) -> bool:
        """Return True if memory compression is currently active."""
        # Ensure memory consolidation tracker is initialized
        if not hasattr(self, 'memory_consolidation_tracker'):
            self.memory_consolidation_tracker = {
                'memory_operations_per_cycle': 0,
                'consolidation_operations_count': 0,
                'is_consolidating_memories': False,
                'is_prioritizing_memories': False,
                'memory_compression_active': False,
                'high_salience_memories_strengthened': 0,
                'low_salience_memories_decayed': 0,
                'last_consolidation_score': 0.0,
                'total_memory_operations': 0
            }
        return self.memory_consolidation_tracker.get('memory_compression_active', False)
    
    def has_made_reset_decisions(self) -> bool:
        """Return True if model has made any game reset decisions."""
        # Ensure game_reset_tracker is initialized
        if not hasattr(self, 'game_reset_tracker'):
            self.game_reset_tracker = {
                'reset_decisions_made': 0,
                'successful_resets': 0,
                'failed_resets': 0,
                'reset_success_rate': 0.0,
                'last_reset_decision': None,
                'reset_decision_criteria': {}
            }
        return self.game_reset_tracker.get('reset_decisions_made', 0) > 0
    
    def get_sleep_and_memory_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of sleep states and memory operations.
        
        Returns:
            Complete status including True/False flags for all operations
        """
        # Ensure sleep_state_tracker is initialized
        if not hasattr(self, 'sleep_state_tracker'):
            self.sleep_state_tracker = {
                'is_currently_sleeping': False,
                'sleep_cycles_this_session': 0,
                'total_sleep_time': 0.0,
                'memory_operations_per_cycle': 0,
                'last_sleep_trigger': [],
                'sleep_quality_scores': [],
                'current_energy_level': 100.0,
                'last_sleep_cycle': 0,
                'total_sleep_cycles': 0
            }
        
        return {
            # Sleep state status
            'sleep_status': {
                'is_currently_sleeping': self.is_sleeping(),
                'sleep_cycles_this_session': self.sleep_state_tracker.get('sleep_cycles_this_session', 0),
                'total_sleep_time_minutes': self.sleep_state_tracker.get('total_sleep_time', 0.0) / 60.0,
                'sleep_efficiency': self._calculate_sleep_efficiency()
            },
            
            # Memory consolidation status
            'memory_consolidation_status': {
                'is_consolidating_memories': self.is_consolidating_memories(),
                'is_prioritizing_memories': self.is_prioritizing_memories(),
                'consolidation_operations_completed': self.memory_consolidation_tracker.get('consolidation_operations_count', 0),
                'high_salience_memories_strengthened': self.memory_consolidation_tracker.get('high_salience_memories_strengthened', 0),
                'low_salience_memories_decayed': self.memory_consolidation_tracker.get('low_salience_memories_decayed', 0),
                'last_consolidation_effectiveness': self.memory_consolidation_tracker.get('last_consolidation_score', 0.0)
            },
            
            # Memory compression status
            'memory_compression_status': {
                'compression_active': self.is_memory_compression_active(),
                'compression_mode': getattr(self.salience_calculator, 'mode', None).value if hasattr(self, 'salience_calculator') and self.salience_calculator and hasattr(self.salience_calculator, 'mode') else 'none',
                'total_compressed_memories': len(getattr(self.salience_calculator, 'compressed_memories', [])) if hasattr(self, 'salience_calculator') and self.salience_calculator else 0
            },
            
            # Game reset decision status
            'game_reset_status': {
                'has_made_reset_decisions': self.has_made_reset_decisions(),
                'total_reset_decisions': self.game_reset_tracker.get('reset_decisions_made', 0),
                'reset_success_rate': self.game_reset_tracker.get('reset_success_rate', 0.0),
                'last_reset_reason': self.game_reset_tracker.get('last_reset_decision', {}).get('reason') if self.game_reset_tracker.get('last_reset_decision') else None,
                'reset_decision_criteria': self.game_reset_tracker.get('reset_decision_criteria', {})
            }
        }
    
    def _calculate_sleep_efficiency(self) -> float:
        """Calculate how efficiently sleep cycles are being used."""
        # Ensure sleep_state_tracker is initialized
        if not hasattr(self, 'sleep_state_tracker'):
            self.sleep_state_tracker = {
                'is_currently_sleeping': False,
                'sleep_cycles_this_session': 0,
                'total_sleep_time': 0.0,
                'memory_operations_per_cycle': 0,
                'last_sleep_trigger': [],
                'sleep_quality_scores': [],
                'current_energy_level': 100.0,
                'last_sleep_cycle': 0,
                'total_sleep_cycles': 0
            }
        
        quality_scores = self.sleep_state_tracker.get('sleep_quality_scores', [])
        if not quality_scores:
            return 0.0
        
        sleep_cycles = self.sleep_state_tracker.get('sleep_cycles_this_session', 0)
        
        # Efficiency = average quality * frequency factor
        avg_quality = sum(quality_scores) / len(quality_scores)
        frequency_factor = min(sleep_cycles / 10.0, 1.0)  # Optimal frequency around 10 cycles per session
        
        return avg_quality * frequency_factor

    def _is_reset_decision_pending(self) -> bool:
        """Check if a reset decision is currently being evaluated."""
        last_decision = self.game_reset_tracker.get('last_reset_decision')
        if not last_decision:
            return False
        
        # Consider decision "pending" if it was made very recently (within 30 seconds)
        time_since_decision = time.time() - last_decision.get('timestamp', 0)
        return time_since_decision < 30.0
    
    def _is_memory_system_healthy(self) -> bool:
        """Check if memory system is operating within healthy parameters."""
        # Ensure memory consolidation tracker is initialized
        if not hasattr(self, 'memory_consolidation_tracker'):
            self.memory_consolidation_tracker = {
                'memory_operations_per_cycle': 0,
                'consolidation_operations_count': 0,
                'is_consolidating_memories': False,
                'is_prioritizing_memories': False,
                'memory_compression_active': False,
                'high_salience_memories_strengthened': 0,
                'low_salience_memories_decayed': 0,
                'last_consolidation_score': 0.0,
                'total_memory_operations': 0
            }
        
        consolidation_count = self.memory_consolidation_tracker.get('consolidation_operations_count', 0)
        last_score = self.memory_consolidation_tracker.get('last_consolidation_score', 0.0)
        
        # Healthy if we've had successful consolidations with positive scores
        return consolidation_count > 0 and last_score > 0.1

    def _parse_complete_game_session(self, stdout_text: str, stderr_text: str) -> Dict[str, Any]:
        """Parse complete game session output to extract results and effective actions."""
        import re
        
        # Enhanced null safety - ensure we never return None values
        result = {
            'final_score': 0,
            'total_actions': 0,
            'final_state': 'UNKNOWN',
            'effective_actions': []
        }
        
        # Null safety for input parameters
        stdout_text = stdout_text if stdout_text is not None else ""
        stderr_text = stderr_text if stderr_text is not None else ""
        
        try:
            # Look for final scorecard in output
            if '"won": 1' in stdout_text:
                result['final_state'] = 'WIN'
            elif '"won": 0' in stdout_text:
                result['final_state'] = 'GAME_OVER'
            elif 'TIMEOUT' in stdout_text.upper():
                result['final_state'] = 'TIMEOUT'
            
            # Extract score from scorecard
            score_match = re.search(r'"score":\s*(\d+)', stdout_text)
            if score_match:
                result['final_score'] = int(score_match.group(1))
            
            # Extract total actions from scorecard
            actions_match = re.search(r'"total_actions":\s*(\d+)', stdout_text)
            if actions_match:
                result['total_actions'] = int(actions_match.group(1))
            
            # Extract effective actions by looking for action patterns (more permissive)
            effective_actions = []
            
            # Look for ACTION patterns - treat all attempted actions as potentially effective for learning
            action_pattern = r'ACTION(\d+)'
            action_matches = re.findall(action_pattern, stdout_text, re.IGNORECASE)
            
            # Count action attempts even if they didn't score
            action_counts = {}
            for action_num in action_matches:
                action_counts[action_num] = action_counts.get(action_num, 0) + 1
            
            # Create effective actions from attempts (for learning purposes)
            for action_num, count in action_counts.items():
                if count >= 2:  # Actions used multiple times might be more significant
                    effectiveness = min(0.5, count / 10.0)  # Lower baseline effectiveness
                else:
                    effectiveness = 0.1  # Minimal effectiveness for single attempts
                
                effective_actions.append({
                    'action_number': int(action_num),
                    'action_type': f'ACTION{action_num}',
                    'score_achieved': result['final_score'],
                    'effectiveness': effectiveness,
                    'attempt_count': count
                })
            
            # Look for score improvements (original logic preserved)
            action_score_pattern = r'ACTION(\d+).*?score[:\s]+(\d+)'
            action_matches = re.findall(action_score_pattern, stdout_text, re.IGNORECASE)
            
            seen_actions = set()  # Prevent duplicates
            for action_num, score_str in action_matches:
                score = int(score_str)
                action_key = f"ACTION{action_num}_{score}"
                
                if score > 0 and action_key not in seen_actions:  # This action had a positive effect
                    seen_actions.add(action_key)
                    # Update existing action or add new high-effectiveness action
                    existing_action = next((a for a in effective_actions if a['action_number'] == int(action_num)), None)
                    if existing_action:
                        existing_action['effectiveness'] = min(1.0, score / 20.0)  # More lenient effectiveness scoring
                        existing_action['score_achieved'] = score
                    else:
                        effective_actions.append({
                            'action_number': int(action_num),
                            'action_type': f'ACTION{action_num}',
                            'score_achieved': score,
                            'effectiveness': min(1.0, score / 20.0)  # More lenient effectiveness scoring
                        })
            
            # Look for successful RESET sequences (avoid double-counting)
            reset_score_pattern = r'RESET.*?score[:\s]+(\d+)'
            reset_matches = re.findall(reset_score_pattern, stdout_text, re.IGNORECASE)
            for score_str in reset_matches:
                score = int(score_str)
                reset_key = f"RESET_{score}"
                
                if score > 0 and reset_key not in seen_actions:
                    seen_actions.add(reset_key)
                    effective_actions.append({
                        'action_type': 'RESET_SEQUENCE',
                        'score_achieved': score,
                        'effectiveness': min(1.0, score / 10.0)  # Reset sequences are highly valuable, even lower threshold
                    })
            
            result['effective_actions'] = effective_actions
            
            print(f" Parsed game session: {result['total_actions']} actions, {len(effective_actions)} effective, final score {result['final_score']}")
            try:
                write_session_trace(result.get('game_id', 'unknown'), result, stdout_text)
            except Exception:
                pass
            
        except Exception as e:
            print(f" Error parsing complete game session: {e}")
        
        return result

    async def _trigger_sleep_cycle(self, effective_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Trigger a sleep cycle to process and consolidate effective actions with real memory operations."""
        print(f" SLEEP CYCLE INITIATED - Processing {len(effective_actions)} effective actions...")
        
        sleep_result = {
            'actions_processed': len(effective_actions),
            'memories_strengthened': 0,
            'memories_deleted': 0,
            'memories_combined': 0,
            'energy_restored': 70.0,  # Restore 70 energy points on sleep
            'insights_generated': 0,
            'priority_memories_loaded': 0
        }
        
        try:
            # 1. LOAD AND PRIORITIZE EXISTING MEMORIES
            print(f" Phase 1: Loading and prioritizing existing memories...")
            relevant_memories = await self._load_prioritized_memories()
            sleep_result['priority_memories_loaded'] = len(relevant_memories)
            print(f"   Loaded {len(relevant_memories)} prioritized memories")
            
            # 2. GARBAGE COLLECT IRRELEVANT MEMORIES  
            print(f" Phase 2: Garbage collecting irrelevant memories...")
            deleted_count = await self._garbage_collect_memories()
            sleep_result['memories_deleted'] = deleted_count
            print(f"   Deleted {deleted_count} irrelevant memories")
            
            # 3. COMBINE/COMPRESS SIMILAR MEMORIES
            print(f" Phase 3: Combining similar memory patterns...")
            combined_count = await self._combine_similar_memories()
            sleep_result['memories_combined'] = combined_count
            print(f"   Combined {combined_count} similar memory clusters")
            
            # 4. STRENGTHEN EFFECTIVE ACTION MEMORIES
            print(f" Phase 4: Strengthening effective action memories...")
            for action in effective_actions:
                effectiveness = action.get('effectiveness', 0)
                if effectiveness > 0.1:  # Much lower threshold to capture more learnings - from 0.3 to 0.1
                    await self._strengthen_action_memory_with_context(action, relevant_memories)
                    sleep_result['memories_strengthened'] += 1
            
            # 5. GENERATE INSIGHTS FROM MEMORY PATTERNS
            print(f" Phase 5: Generating insights from memory patterns...")
            if len(effective_actions) >= 2 or len(relevant_memories) >= 5:
                insights = await self._generate_memory_insights(effective_actions, relevant_memories)
                sleep_result['insights_generated'] = len(insights)
                for insight in insights:
                    print(f" Insight: {insight}")
            
            # 6. PREPARE MEMORY GUIDANCE FOR NEXT GAME
            print(f" Phase 6: Preparing memory-informed guidance...")
            guidance = await self._prepare_memory_guidance(relevant_memories, effective_actions)
            sleep_result['guidance_prepared'] = len(guidance)
            print(f"   Prepared {len(guidance)} guidance points for next game")
            
            print(f" SLEEP CYCLE COMPLETED:")
            print(f"    {sleep_result['priority_memories_loaded']} memories loaded")
            print(f"    {sleep_result['memories_deleted']} memories deleted")  
            print(f"    {sleep_result['memories_combined']} memories combined")
            print(f"    {sleep_result['memories_strengthened']} memories strengthened")
            print(f"    {sleep_result['insights_generated']} insights generated")
            
            # UPDATE GLOBAL COUNTERS - THIS IS THE FIX!
            self._update_global_counters(
                sleep_cycle_completed=True,
                memory_ops=sleep_result['priority_memories_loaded'],
                memories_deleted=sleep_result['memories_deleted'],
                memories_combined=sleep_result['memories_combined'],
                memories_strengthened=sleep_result['memories_strengthened']
            )
            
        except Exception as e:
            print(f" Error during sleep cycle: {e}")
        
        return sleep_result

    def _strengthen_action_memory(self, action: Dict[str, Any]) -> None:
        """Strengthen memory of an effective action."""
        action_type = action.get('action_type', 'unknown')
        effectiveness = action.get('effectiveness', 0)
        score = action.get('score_achieved', 0)
        
        print(f" Strengthening memory: {action_type} (effectiveness: {effectiveness:.2f}, score: {score})")
        
        # This would integrate with the existing memory system
        # For now, we track it for the learning system
        if not hasattr(self, 'effective_action_memories'):
            self.effective_action_memories = []
        
        self.effective_action_memories.append({
            'action_type': action_type,
            'effectiveness': effectiveness,
            'score': score,
            'timestamp': time.time()
        })

    def _generate_action_insights(self, actions: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from patterns in effective actions."""
        insights = []
        
        try:
            # Look for patterns in action types
            action_types = [a.get('action_type', '') for a in actions]
            type_counts = {}
            for action_type in action_types:
                type_counts[action_type] = type_counts.get(action_type, 0) + 1
            
            # Find most effective action type
            if type_counts:
                most_common = max(type_counts, key=type_counts.get)
                if type_counts[most_common] >= 2:
                    insights.append(f"Most effective action type: {most_common} (used {type_counts[most_common]} times)")
            
            # Look for effectiveness trends
            effectiveness_scores = [a.get('effectiveness', 0) for a in actions]
            if len(effectiveness_scores) >= 3:
                avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
                if avg_effectiveness > 0.7:
                    insights.append("High overall action effectiveness detected")
                elif effectiveness_scores[-1] > effectiveness_scores[0]:
                    insights.append("Action effectiveness improving over time")
            
            # Look for score achievements
            scores = [a.get('score_achieved', 0) for a in actions]
            total_score = sum(scores)
            if total_score > 0:
                insights.append(f"Total effective score contribution: {total_score}")
            
        except Exception as e:
            print(f" Error generating insights: {e}")
        
        return insights

    def _update_energy_level(self, new_energy: float) -> None:
        """Update the system's energy level (0-100.0 scale)."""
        self.current_energy = max(0.0, min(100.0, new_energy))
        
        # Persist energy level to global counters
        self.global_counters['persistent_energy_level'] = self.current_energy
        self._save_global_counters()
        
        # Update in sleep state info if available
        if hasattr(self, 'sleep_system') and self.sleep_system:
            try:
                # This would integrate with the actual energy system
                print(f" Energy level updated to {self.current_energy:.2f}")
            except:
                pass

    def _estimate_game_complexity(self, game_id: str) -> str:
        """Estimate game complexity based on game ID pattern and historical data."""
        
        # Check if we have historical data for this game
        if hasattr(self, 'game_complexity_history'):
            if game_id in self.game_complexity_history:
                historical_actions = self.game_complexity_history[game_id]['avg_actions']
                if historical_actions > 500:
                    return 'high'
                elif historical_actions > 200:
                    return 'medium'
                else:
                    return 'low'
        
        # Estimate based on game ID patterns (some game types are known to be complex)
        try:
            # Games with certain patterns tend to be more complex
            if any(pattern in game_id.lower() for pattern in ['complex', 'multi', 'transform', 'sequence']):
                return 'high'
            elif any(pattern in game_id.lower() for pattern in ['simple', 'basic', 'easy']):
                return 'low'
            else:
                return 'medium'  # Default assumption
        except:
            return 'medium'

    def _update_game_complexity_history(self, game_id: str, actions_taken: int, effectiveness_ratio: float):
        """Update historical complexity data for a game."""
        if not hasattr(self, 'game_complexity_history'):
            self.game_complexity_history = {}
        
        if game_id not in self.game_complexity_history:
            self.game_complexity_history[game_id] = {
                'total_plays': 0,
                'total_actions': 0,
                'avg_actions': 0,
                'effectiveness_history': []
            }
        
        # Comprehensive null safety - ensure no None values in calculations
        actions_taken = actions_taken if actions_taken is not None else 0
        effectiveness_ratio = effectiveness_ratio if effectiveness_ratio is not None else 0.0
        
        # Additional safety checks
        if not isinstance(actions_taken, (int, float)):
            actions_taken = 0
        if not isinstance(effectiveness_ratio, (int, float)):
            effectiveness_ratio = 0.0
        
        history = self.game_complexity_history[game_id]
        
        # Ensure history values are never None
        history['total_plays'] = (history.get('total_plays') or 0) + 1
        history['total_actions'] = (history.get('total_actions') or 0) + actions_taken
        history['avg_actions'] = history['total_actions'] / max(history['total_plays'], 1)
        history['effectiveness_history'].append(effectiveness_ratio)
        
        # Keep only last 10 effectiveness scores
        if len(history['effectiveness_history']) > 10:
            history['effectiveness_history'] = history['effectiveness_history'][-10:]

    async def _load_prioritized_memories(self) -> List[Dict[str, Any]]:
        """Load and prioritize existing memories based on relevance and recency."""
        try:
            import os
            import json
            from pathlib import Path
            
            memories = []
            
            # Look for memory files in the continuous learning data directory
            memory_dirs = [
                Path("data"),
                Path("data/meta_learning_data"), 
                Path("test_meta_learning_data")
            ]
            
            for memory_dir in memory_dirs:
                if memory_dir.exists():
                    for memory_file in memory_dir.glob("*.json"):
                        try:
                            with open(memory_file, 'r') as f:
                                memory_data = json.load(f)
                                # Add metadata
                                memory_data['file_path'] = str(memory_file)
                                memory_data['file_age'] = os.path.getmtime(memory_file)
                                memories.append(memory_data)
                        except Exception as e:
                            print(f"    Failed to load {memory_file}: {e}")
            
            # Prioritize memories by relevance (recent + effective)
            def priority_score(memory):
                recency_score = memory.get('file_age', 0) / 1000000  # Recent files score higher
                effectiveness_score = memory.get('final_score', 0) * 10  # Effective memories score higher
                return recency_score + effectiveness_score
            
            memories.sort(key=priority_score, reverse=True)
            
            # Return top 10 most relevant memories
            return memories[:10]
            
        except Exception as e:
            print(f"    Error loading memories: {e}")
            return []

    async def _garbage_collect_memories(self) -> int:
        """Delete irrelevant, old, or low-value memories to free up space."""
        try:
            import os
            import time
            from pathlib import Path
            
            deleted_count = 0
            current_time = time.time()
            
            memory_dirs = [
                Path("data"),
                Path("data/meta_learning_data"),
                Path("test_meta_learning_data") 
            ]
            
            for memory_dir in memory_dirs:
                if memory_dir.exists():
                    for memory_file in memory_dir.glob("*.json"):
                        try:
                            file_age_days = (current_time - os.path.getmtime(memory_file)) / 86400
                            file_size_kb = os.path.getsize(memory_file) / 1024
                            
                            # Check if this memory is protected from deletion
                            is_protected = self._is_memory_protected(memory_file.name)
                            
                            # Delete if: very old (>7 days) OR very small (<1KB) OR temp files
                            # BUT NOT if protected
                            should_delete = (
                                not is_protected and (
                                    file_age_days > 7 or
                                    file_size_kb < 1 or
                                    'temp_' in memory_file.name or
                                    'test_' in memory_file.name
                                )
                            )
                            
                            if should_delete:
                                os.remove(memory_file)
                                deleted_count += 1
                                print(f"    Deleted {memory_file.name} (age: {file_age_days:.1f}d, size: {file_size_kb:.1f}KB)")
                            elif is_protected:
                                print(f"    Protected {memory_file.name} from deletion (winning memory)")
                                
                        except Exception as e:
                            print(f"    Failed to process {memory_file}: {e}")
            
            return deleted_count
            
        except Exception as e:
            print(f"    Error during garbage collection: {e}")
            return 0

    async def _combine_similar_memories(self) -> int:
        """Combine similar memory patterns to reduce redundancy."""
        try:
            import json
            import time
            import os
            from pathlib import Path
            from collections import defaultdict
            
            combined_count = 0
            memory_clusters = defaultdict(list)
            
            # Group memories by similarity patterns
            memory_dir = Path("data")
            if memory_dir.exists():
                for memory_file in memory_dir.glob("*.json"):
                    try:
                        with open(memory_file, 'r') as f:
                            memory_data = json.load(f)
                            
                            # Cluster by game patterns and scores
                            cluster_key = f"score_{memory_data.get('final_score', 0)//10}_actions_{memory_data.get('actions_taken', 0)//50}"
                            memory_clusters[cluster_key].append({
                                'file': memory_file,
                                'data': memory_data
                            })
                    except:
                        continue
            
            # Combine clusters with multiple similar memories
            for cluster_key, memories in memory_clusters.items():
                if len(memories) > 2:  # Combine if 3+ similar memories
                    # Create combined memory
                    combined_memory = {
                        'type': 'combined_memory_cluster',
                        'cluster_key': cluster_key,
                        'original_count': len(memories),
                        'combined_timestamp': time.time(),
                        'combined_data': {
                            'avg_score': sum((m['data'].get('final_score') or 0) for m in memories) / len(memories),
                            'avg_actions': sum(m['data'].get('actions_taken', 0) for m in memories) / len(memories),
                            'patterns': [m['data'] for m in memories[:3]]  # Keep top 3 examples
                        }
                    }
                    
                    # Save combined memory
                    combined_file = memory_dir / f"combined_{cluster_key}_{int(time.time())}.json"
                    with open(combined_file, 'w') as f:
                        json.dump(combined_memory, f, indent=2)
                    
                    # Delete original files
                    for memory in memories:
                        try:
                            os.remove(memory['file'])
                        except:
                            pass
                    
                    combined_count += 1
                    print(f"    Combined {len(memories)} memories into {combined_file.name}")
            
            return combined_count
            
        except Exception as e:
            print(f"    Error combining memories: {e}")
            return 0

    async def _strengthen_action_memory_with_context(self, action: Dict[str, Any], context_memories: List[Dict[str, Any]]) -> None:
        """Strengthen memory of an effective action with contextual information."""
        try:
            import time
            
            action_type = action.get('action_type', 'unknown')
            effectiveness = action.get('effectiveness', 0)
            score = action.get('score_achieved', 0)
            
            print(f"    Strengthening: {action_type} (effectiveness: {effectiveness:.2f}, score: {score})")
            
            # Create enhanced memory entry
            enhanced_memory = {
                'action_type': action_type,
                'effectiveness': effectiveness,
                'score': score,
                'timestamp': time.time(),
                'context': {
                    'similar_patterns': len([m for m in context_memories if m.get('final_score', 0) > 0]),
                    'learning_context': f"Enhanced during sleep cycle with {len(context_memories)} context memories"
                },
                'strengthened': True
            }
            
            # Add to effective action memories
            if not hasattr(self, 'effective_action_memories'):
                self.effective_action_memories = []
            
            self.effective_action_memories.append(enhanced_memory)
            
            # Keep only last 50 strengthened memories
            if len(self.effective_action_memories) > 50:
                self.effective_action_memories = self.effective_action_memories[-50:]
                
        except Exception as e:
            print(f"    Error strengthening memory: {e}")

    async def _generate_memory_insights(self, effective_actions: List[Dict[str, Any]], memories: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from memory patterns and current effective actions."""
        insights = []
        
        try:
            # Pattern analysis from memories
            if memories:
                successful_games = [m for m in memories if m.get('final_score', 0) > 0]
                if len(successful_games) >= 2:
                    avg_successful_actions = sum(m.get('actions_taken', 0) for m in successful_games) / len(successful_games)
                    insights.append(f"Historical pattern: Successful games average {avg_successful_actions:.0f} actions")
                
                # Look for game type patterns
                game_types = [m.get('game_id', '')[:4] for m in memories if m.get('game_id')]
                if game_types:
                    most_common_type = max(set(game_types), key=game_types.count)
                    insights.append(f"Most practiced game type: {most_common_type}* patterns")
            
            # Current action effectiveness
            if effective_actions:
                action_types = [a.get('action_type', '') for a in effective_actions]
                if action_types:
                    most_effective = max(set(action_types), key=action_types.count)
                    insights.append(f"Current session: {most_effective} actions showing highest effectiveness")
                    
                avg_effectiveness = sum(a.get('effectiveness', 0) for a in effective_actions) / len(effective_actions)
                if avg_effectiveness > 0.7:
                    insights.append("High effectiveness session - patterns worth reinforcing")
            
            # Cross-reference current actions with memory patterns
            if effective_actions and memories:
                insights.append("Memory consolidation: Linking current learnings with historical patterns")
                
        except Exception as e:
            print(f"    Error generating insights: {e}")
        
        return insights

    async def _prepare_memory_guidance(self, memories: List[Dict[str, Any]], effective_actions: List[Dict[str, Any]]) -> List[str]:
        """Prepare memory-informed guidance for the next game session."""
        guidance = []
        
        try:
            # Guidance from historical successes
            successful_memories = [m for m in memories if m.get('final_score', 0) > 0]
            if successful_memories:
                guidance.append(f"Historical success: Focus on patterns from {len(successful_memories)} successful games")
                
                # Extract successful action patterns
                if hasattr(self, 'effective_action_memories') and self.effective_action_memories:
                    top_actions = sorted(self.effective_action_memories, key=lambda x: x.get('effectiveness', 0), reverse=True)[:3]
                    for action in top_actions:
                        guidance.append(f"Prioritize: {action.get('action_type', 'unknown')} actions (proven effective)")
            
            # Guidance from current session
            if effective_actions:
                guidance.append(f"Current learnings: {len(effective_actions)} effective action patterns identified")
                
            # Energy and complexity guidance
            guidance.append("Energy management: Monitor action effectiveness to optimize sleep timing")
            
        except Exception as e:
            print(f"    Error preparing guidance: {e}")
        
        return guidance

    def _load_global_counters(self) -> Dict[str, int]:
        """Load global counters that persist across sessions."""
        try:
            import json
            counter_file = self.save_directory / "global_counters.json"
            if counter_file.exists():
                with open(counter_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Could not load global counters: {e}")
        
        # Default counters
        return {
            'total_sleep_cycles': 0,
            'total_memory_operations': 0,
            'total_sessions': 0,
            'cumulative_energy_spent': 0.0,
            'total_memories_deleted': 0,
            'total_memories_combined': 0,
            'total_memories_strengthened': 0,
            'persistent_energy_level': 1.0  # Track energy across sessions
        }
    
    def _save_global_counters(self):
        """Save global counters to persist across sessions."""
        try:
            import json
            counter_file = self.save_directory / "global_counters.json"
            with open(counter_file, 'w') as f:
                json.dump(self.global_counters, f, indent=2)
        except Exception as e:
            print(f"Could not save global counters: {e}")
    
    def _update_global_counters(self, sleep_cycle_completed: bool = False, memory_ops: int = 0, memories_deleted: int = 0, memories_combined: int = 0, memories_strengthened: int = 0):
        """Update global counters and save to disk."""
        if sleep_cycle_completed:
            self.global_counters['total_sleep_cycles'] += 1
        
        if memory_ops > 0:
            self.global_counters['total_memory_operations'] += memory_ops
            
        if memories_deleted > 0:
            self.global_counters['total_memories_deleted'] += memories_deleted
            
        if memories_combined > 0:
            self.global_counters['total_memories_combined'] += memories_combined
            
        if memories_strengthened > 0:
            self.global_counters['total_memories_strengthened'] += memories_strengthened
            
        # Always update current energy level for persistence
        self.global_counters['persistent_energy_level'] = getattr(self, 'current_energy', 1.0)
        
        # Auto-save counters
        self._save_global_counters()
    
    def _preserve_ultimate_win_memories(self, session_result: Dict[str, Any], score: int, game_id: str):
        """
        Preserve memories from full game wins with IMMORTAL priority.
        These memories are never deleted and get maximum salience.
        """
        try:
            # Memory system not available in ContinuousLearningLoop - skip memory preservation
            if True:  # Changed: not hasattr(self.agent, 'memory')
                return
                
            # Ultimate win gets maximum preservation
            preservation_strength = 1.0  # Maximum strength
            min_salience_floor = 0.95   # Nearly maximum minimum
            protection_duration = 9999  # Effectively permanent
            print(f"🏆 IMMORTAL MEMORY PRESERVATION: Full game win memories (strength: {preservation_strength})")
            
            # Mark in training state for ultimate protection
            if not hasattr(self, 'ultimate_wins'):
                self.ultimate_wins = []
            
            self.ultimate_wins.append({
                'game_id': game_id,
                'score': score,
                'timestamp': time.time(),
                'preservation_strength': preservation_strength,
                'memory_priority': 'IMMORTAL'
            })
            
            print(f"🏆 ULTIMATE WIN RECORDED: {game_id} with score {score} - IMMORTAL memory priority")
            
        except Exception as e:
            print(f"Error preserving ultimate win memories: {e}")

    def _preserve_winning_memories(self, session_result: Dict[str, Any], score: int, game_id: str, is_level_progression: bool = False):
        """
        Preserve memories that led to wins or level progression.
        These memories get super-high salience and are protected from deletion.
        """
        try:
            # Memory system not available in ContinuousLearningLoop - skip memory preservation
            if True:  # Changed: not hasattr(self.agent, 'memory')
                return
                
            # Determine preservation strength based on achievement type
            if is_level_progression:
                preservation_strength = 0.95  # Nearly permanent
                min_salience_floor = 0.8  # Very high minimum
                protection_duration = 1000  # Much longer protection
                print(f" CRITICAL MEMORY PRESERVATION: Level progression memories (strength: {preservation_strength})")
            elif score >= 4:  # High score achievement
                preservation_strength = 0.85
                min_salience_floor = 0.6
                protection_duration = 500
                print(f" HIGH-VALUE MEMORY PRESERVATION: Score {score} memories (strength: {preservation_strength})")
            elif score >= 1:  # Any positive score
                preservation_strength = 0.75
                min_salience_floor = 0.4
                protection_duration = 200
                print(f" WINNING MEMORY PRESERVATION: Score {score} memories (strength: {preservation_strength})")
            else:
                return  # No preservation for zero scores
            
            # Extract effective actions from this session
            effective_actions = []
            if 'effective_actions' in session_result:
                effective_actions = session_result['effective_actions']
            elif 'action_sequences' in session_result:
                # Extract from action sequences if available
                sequences = session_result.get('action_sequences', {})
                for seq_key, seq_data in sequences.items():
                    if seq_data.get('effectiveness', 0) > 0.3:
                        effective_actions.append({
                            'action': seq_key,
                            'effectiveness': seq_data['effectiveness'],
                            'score': score
                        })
            
            # Apply memory preservation with super-high salience
            winning_memory_count = 0
            for i, action in enumerate(effective_actions):
                # Memory system not available in ContinuousLearningLoop - skip salience update
                if False:  # Disabled: memory system not available
                    # Create high-salience preservation
                    pass  # Memory operations disabled
                    winning_memory_count += 1
                    
                # Mark in training state for long-term protection
                if 'protected_memories' not in self.training_state:
                    self.training_state['protected_memories'] = []
                    
                protection_entry = {
                    'memory_id': f"{game_id}_{i}",
                    'action': action,
                    'score_achieved': score,
                    'salience_floor': min_salience_floor,
                    'protection_expires': time.time() + protection_duration,
                    'is_level_progression': is_level_progression,
                    'game_id': game_id,
                    'timestamp': time.time()
                }
                self.training_state['protected_memories'].append(protection_entry)
            
            # Clean up expired protections
            current_time = time.time()
            self.training_state['protected_memories'] = [
                mem for mem in self.training_state['protected_memories']
                if mem['protection_expires'] > current_time
            ]
            
            print(f" Protected {winning_memory_count} winning memories from deletion (expires in {protection_duration}s)")
            
        except Exception as e:
            logger.warning(f"Failed to preserve winning memories: {e}")

    def _preserve_breakthrough_memories(self, session_result: Dict[str, Any], score: int, game_id: str, new_level: int, previous_level: int):
        """
        Progressive memory hierarchy system - only preserves TRUE breakthroughs with escalating priority.
        Higher levels get stronger protection and demote previous level memories.
        """
        try:
            # Memory system not available in ContinuousLearningLoop - skip memory hierarchy
            if True:  # Changed: not hasattr(self.agent, 'memory')
                return
                
            # Update game level records
            if game_id not in self.game_level_records:
                self.game_level_records[game_id] = {'highest_level': 0, 'breakthroughs': []}
            
            self.game_level_records[game_id]['highest_level'] = new_level
            
            # Calculate tier-based protection (1-5 scale)
            tier = min(new_level, 5)  # Cap at tier 5 for extreme breakthroughs
            
            # Progressive strength: Tier 1=0.75, Tier 2=0.80, Tier 3=0.85, Tier 4=0.90, Tier 5=0.95
            preservation_strength = 0.70 + (tier * 0.05)
            
            # Progressive floor: Tier 1=0.4, Tier 2=0.5, Tier 3=0.6, Tier 4=0.7, Tier 5=0.8  
            min_salience_floor = 0.30 + (tier * 0.10)
            
            # Progressive duration: Higher tiers last longer
            protection_duration = 200 + (tier * 200)  # 400, 600, 800, 1000, 1200
            
            print(f" LEVEL {new_level} BREAKTHROUGH! Tier {tier} Protection (strength: {preservation_strength:.2f}, floor: {min_salience_floor:.1f})")
            
            # Demote previous level memories if this is a higher tier
            if previous_level > 0:
                self._demote_previous_level_memories(game_id, previous_level, tier)
            
            # Extract and preserve breakthrough memories
            effective_actions = []
            if 'effective_actions' in session_result:
                effective_actions = session_result['effective_actions']
            elif 'action_sequences' in session_result:
                sequences = session_result.get('action_sequences', {})
                for seq_key, seq_data in sequences.items():
                    if seq_data.get('effectiveness', 0) > 0.3:
                        effective_actions.append({
                            'action': seq_key,
                            'effectiveness': seq_data['effectiveness'],
                            'score': score
                        })
            
            # Apply hierarchical memory preservation
            breakthrough_memory_count = 0
            breakthrough_memory_ids = []
            
            for i, action in enumerate(effective_actions):
                memory_id = f"{game_id}_L{new_level}_{i}"
                
                # Memory system not available in ContinuousLearningLoop - skip salience update
                if False:  # Disabled: memory system not available
                    # Apply tier-based salience boost
                    pass  # Memory operations disabled
                    
                # Store in hierarchical memory system
                protection_entry = {
                    'memory_id': memory_id,
                    'action': action,
                    'score_achieved': score,
                    'level_achieved': new_level,
                    'tier': tier,
                    'salience_strength': preservation_strength,
                    'salience_floor': min_salience_floor,
                    'protection_expires': time.time() + protection_duration,
                    'game_id': game_id,
                    'breakthrough_timestamp': time.time(),
                    'is_breakthrough': True
                }
                self.memory_hierarchy['protected_memories'].append(protection_entry)
                self.memory_hierarchy['tier_assignments'][memory_id] = tier
            
            # Record this breakthrough
            breakthrough_record = {
                'level': new_level,
                'tier': tier,
                'timestamp': time.time(),
                'memory_ids': breakthrough_memory_ids,
                'score': score
            }
            self.game_level_records[game_id]['breakthroughs'].append(breakthrough_record)
            
            print(f" Preserved {breakthrough_memory_count} Tier {tier} breakthrough memories (Level {new_level})")
            
        except Exception as e:
            print(f" Error in breakthrough memory preservation: {e}")

    def _demote_previous_level_memories(self, game_id: str, previous_level: int, new_tier: int):
        """
        Demote memories from previous levels when a higher level is achieved.
        Previous level memories get reduced priority but aren't deleted.
        """
        try:
            demoted_count = 0
            
            # Find memories from previous levels for this game
            for protection_entry in self.memory_hierarchy.get('protected_memories', []):
                if (protection_entry.get('game_id') == game_id and 
                    protection_entry.get('level_achieved', 0) == previous_level and
                    protection_entry.get('tier', 0) < new_tier):
                    
                    old_tier = protection_entry.get('tier', 1)
                    
                    # Reduce protection strength (but keep some protection)
                    new_strength = max(0.60, protection_entry.get('salience_strength', 0.75) - 0.15)
                    new_floor = max(0.25, protection_entry.get('salience_floor', 0.4) - 0.15)
                    
                    protection_entry['salience_strength'] = new_strength
                    protection_entry['salience_floor'] = new_floor
                    protection_entry['demoted_from_tier'] = old_tier
                    protection_entry['demoted_timestamp'] = time.time()
                    
                    demoted_count += 1
            
            if demoted_count > 0:
                print(f" Demoted {demoted_count} Level {previous_level} memories (still protected but lower priority)")
                
        except Exception as e:
            print(f" Error in memory demotion: {e}")
    
    def _is_memory_protected(self, memory_filename: str) -> bool:
        """Check if a memory file is protected from deletion using hierarchical system."""
        try:
            # Check both old and new memory protection systems
            legacy_protected_memories = self.training_state.get('protected_memories', [])
            hierarchical_protected_memories = self.memory_hierarchy.get('protected_memories', [])
            current_time = time.time()
            
            # Check legacy protection system
            for protection in legacy_protected_memories:
                if protection['protection_expires'] > current_time:
                    memory_id = protection['memory_id']
                    game_id = protection['game_id']
                    
                    if (game_id in memory_filename or 
                        memory_id in memory_filename or
                        'session_' in memory_filename):
                        return True
            
            # Check hierarchical protection system
            for protection in hierarchical_protected_memories:
                if protection['protection_expires'] > current_time:
                    memory_id = protection['memory_id']
                    game_id = protection['game_id']
                    tier = protection.get('tier', 1)
                    
                    if (game_id in memory_filename or 
                        memory_id in memory_filename or
                        'session_' in memory_filename):
                        # Higher tier memories get extra protection
                        protection_multiplier = 1.0 + (tier * 0.2)  # Tier 5 = 2x protection time
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking memory protection: {e}")
            return False

    def _display_memory_hierarchy_status(self):
        """Display current memory hierarchy and protection status."""
        try:
            hierarchical_memories = self.memory_hierarchy.get('protected_memories', [])
            current_time = time.time()
            
            # Group by tier and game
            tier_summary = {}
            active_count = 0
            
            for memory in hierarchical_memories:
                if memory['protection_expires'] > current_time:
                    active_count += 1
                    tier = memory.get('tier', 1)
                    game_id = memory.get('game_id', 'unknown')
                    level = memory.get('level_achieved', 0)
                    
                    if tier not in tier_summary:
                        tier_summary[tier] = {'count': 0, 'games': set(), 'levels': set()}
                    
                    tier_summary[tier]['count'] += 1
                    tier_summary[tier]['games'].add(game_id)
                    tier_summary[tier]['levels'].add(level)
            
            if active_count > 0:
                print(f"\n MEMORY HIERARCHY STATUS ({active_count} protected memories):")
                for tier in sorted(tier_summary.keys(), reverse=True):
                    info = tier_summary[tier]
                    strength = 0.70 + (tier * 0.05)
                    floor = 0.30 + (tier * 0.10)
                    games = ', '.join(list(info['games'])[:3])  # Show first 3 games
                    levels = ', '.join(map(str, sorted(info['levels'])))
                    
                    print(f"   Tier {tier}: {info['count']} memories | Strength: {strength:.2f} | Floor: {floor:.1f}")
                    print(f"            Games: {games} | Levels: {levels}")
                print()
            
        except Exception as e:
            logger.warning(f"Error displaying memory hierarchy: {e}")
    
    def _is_experience_protected(self, experience: Dict[str, Any]) -> bool:
        """Check if a specific experience/memory is protected from decay using hierarchical system."""
        try:
            # Check both legacy and hierarchical protection systems
            legacy_protected_memories = self.training_state.get('protected_memories', [])
            hierarchical_protected_memories = self.memory_hierarchy.get('protected_memories', [])
            current_time = time.time()
            
            # Check legacy protection
            for protection in legacy_protected_memories:
                if protection['protection_expires'] > current_time:
                    protected_action = protection.get('action', {})
                    exp_action = experience.get('action', {})
                    
                    if (protected_action.get('type') == exp_action.get('type') and
                        protection.get('score_achieved', 0) > 0):
                        return True
            
            # Check hierarchical protection (stronger for higher tiers)
            for protection in hierarchical_protected_memories:
                if protection['protection_expires'] > current_time:
                    protected_action = protection.get('action', {})
                    exp_action = experience.get('action', {})
                    tier = protection.get('tier', 1)
                    
                    if (protected_action.get('type') == exp_action.get('type') and
                        protection.get('score_achieved', 0) > 0):
                        # Higher tier memories are more protected
                        protection_strength = tier / 5.0  # Tier 5 = 100% protection
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking experience protection: {e}")
            return False
    
    def _get_protection_floor(self, experience: Dict[str, Any]) -> float:
        """Get the minimum salience floor for a protected experience using hierarchical system."""
        try:
            # Check both legacy and hierarchical systems for protection floors
            legacy_protected_memories = self.training_state.get('protected_memories', [])
            hierarchical_protected_memories = self.memory_hierarchy.get('protected_memories', [])
            current_time = time.time()
            max_floor = 0.0
            
            # Check legacy protection floors
            for protection in legacy_protected_memories:
                if protection['protection_expires'] > current_time:
                    protected_action = protection.get('action', {})
                    exp_action = experience.get('action', {})
                    
                    if (protected_action.get('type') == exp_action.get('type') and
                        protection.get('score_achieved', 0) > 0):
                        max_floor = max(max_floor, protection.get('salience_floor', 0.0))
            
            # Check hierarchical protection floors (higher tiers = higher floors)
            for protection in hierarchical_protected_memories:
                if protection['protection_expires'] > current_time:
                    protected_action = protection.get('action', {})
                    exp_action = experience.get('action', {})
                    tier = protection.get('tier', 1)
                    
                    if (protected_action.get('type') == exp_action.get('type') and
                        protection.get('score_achieved', 0) > 0):
                        # Hierarchical floor: Tier 1=0.4, Tier 2=0.5, ..., Tier 5=0.8
                        tier_floor = protection.get('salience_floor', 0.30 + (tier * 0.10))
                        max_floor = max(max_floor, tier_floor)
            
            return max_floor
            
        except Exception as e:
            logger.warning(f"Error getting protection floor: {e}")
            return 0.1

    async def _investigate_game_state(self, game_id: str, guid: Optional[str] = None) -> Dict[str, Any]:
        """Investigate game state to recover available actions."""
        return await self.investigate_api_available_actions(game_id)
    
    async def investigate_api_available_actions(self, game_id: str) -> Dict[str, Any]:
        """Investigate what available actions the API actually provides."""
        try:
            # Start a game session to get initial state
            session_data = await self._start_game_session(game_id)
            if not session_data:
                return {"error": "Failed to start game session"}
            
            guid = session_data.get('guid')
            if not guid:
                return {"error": "No GUID received"}
            
            # Extract information from the session data (no additional API call needed)
            print(f"\n API INVESTIGATION for {game_id}:")
            print(f"    Session Data Keys: {list(session_data.keys())}")
            print(f"    Game State: {session_data.get('state', 'UNKNOWN')}")
            print(f"    Available Actions: {session_data.get('available_actions', [])}")
            print(f"    Score: {session_data.get('score', 0)}")
            print(f"    GUID: {guid}")
            
            # Return the investigation results
            investigation_result = {
                "game_id": game_id,
                "guid": guid,
                "state": session_data.get('state', 'NOT_FINISHED'),
                "available_actions": session_data.get('available_actions', [1,2,3,4,5,6,7]),
                "score": session_data.get('score', 0),
                "frame": session_data.get('frame', []),
                "scorecard_id": session_data.get('scorecard_id'),
                "reset_type": session_data.get('reset_type', 'new_game'),
                "investigation_successful": True,
                "api_endpoint_used": "RESET command (no additional call needed)"
            }
            
            print(f"    Investigation successful - ready for direct control")
            return investigation_result
            
        except Exception as e:
            error_msg = f"API investigation failed: {str(e)}"
            print(f"    {error_msg}")
            return {"error": error_msg}

    async def start_training_with_direct_control(
        self, 
        game_id: str,
        max_actions_per_game: int = ActionLimits.get_max_actions_per_game(),
        session_count: int = 0
    ) -> Dict[str, Any]:
        """Run training session with direct API action control instead of external main.py."""
        print(f"\n STARTING DIRECT CONTROL TRAINING for {game_id}")
        print(f"   Max Actions: {max_actions_per_game}, Session: {session_count}")
        
        #  CRITICAL FIX: Handle per-game scorecard for swarm mode
        original_scorecard_id = self.current_scorecard_id
        
        # Track if we created a new scorecard for this session
        created_new_scorecard = False
        
        try:
            # Check if we have an existing session for this game
            existing_guid = self.current_game_sessions.get(game_id)
            
            # Decide whether to reuse session or start fresh
            should_reuse_session = (existing_guid and 
                                  session_count > 0 and 
                                  session_count % 10 != 0)  # Start fresh every 10 sessions
            
            print(f" Session decision: existing_guid={existing_guid}, session_count={session_count}, should_reuse={should_reuse_session}")
            
            if should_reuse_session:
                # Use level reset for subsequent sessions of the same game
                print(f" Using LEVEL RESET for session {session_count} of {game_id} (reusing session)")
                session_data = await self._start_game_session(game_id, existing_guid=existing_guid)
            else:
                # Use new game reset for the first session or when starting fresh
                if existing_guid:
                    print(f" Starting FRESH SESSION for {game_id} (clearing previous session)")
                    # Clear the old session
                    del self.current_game_sessions[game_id]
                else:
                    print(f" Using NEW GAME RESET for session {session_count} of {game_id}")
                session_data = await self._start_game_session(game_id)
            
            if not session_data:
                return {"error": "Failed to start game session", "actions_taken": 0}
            
            # Extract investigation data from the session data we just got
            investigation = {
                'state': session_data.get('state', 'NOT_STARTED'),
                'score': session_data.get('score', 0),
                'available_actions': session_data.get('available_actions', [1,2,3,4,5,6,7]),
                'frame': session_data.get('frame', [])
            }
            
            #  CRITICAL FIX: Initialize frame data from session start
            initial_frame = session_data.get('frame', [])
            if initial_frame:
                self._last_frame = initial_frame
                try:
                    arr_init, (iw, ih) = self._normalize_frame(initial_frame)
                    if arr_init is not None:
                        print(f" Initialized frame data: {arr_init.shape[0]}x{arr_init.shape[1]}")
                    else:
                        print(f" Initialized frame data: unknown (fallback)")
                except Exception:
                    print(f" Initialized frame data: unknown (error)")
            
            guid = session_data.get('guid')
            current_state = investigation.get('state', 'NOT_STARTED')
            current_score = investigation.get('score', 0)
            available_actions = investigation.get('available_actions', [1,2,3,4,5,6,7])  # Default fallback
            
            print(f"   SESSION STARTED:")
            print(f"   GUID: {guid}")
            print(f"   Initial State: {current_state}")
            print(f"   Initial Score: {current_score}")
            print(f"   Available Actions: {available_actions}")
            print(f"   TARGET: Win (score ≥ 100) or reach terminal state")
            
            # Direct action control loop
            actions_taken = 0
            effective_actions = []
            action_history = []
            last_score_check = 0  # Track when we last displayed score progress
            
            #  SMART ACTION CAP - Calculate dynamic limit based on game complexity
            if hasattr(self, '_action_cap_system') and self._action_cap_system['enabled']:
                dynamic_action_cap = self._calculate_dynamic_action_cap(available_actions, max_actions_per_game)
                # Always use dynamic cap as it's now properly scaled
                actual_max_actions = min(max_actions_per_game, dynamic_action_cap)
                print(f" SMART LIMIT: {actual_max_actions} actions (dynamic cap: {dynamic_action_cap}, configured: {max_actions_per_game})")
            else:
                actual_max_actions = max_actions_per_game
                print(f" USING CONFIGURED LIMIT: {actual_max_actions} actions (configured: {max_actions_per_game})")
            
            # Reset progress tracker for this game
            if hasattr(self, '_progress_tracker'):
                self._progress_tracker.update({
                    'actions_taken': 0,
                    'last_score': current_score,
                    'actions_without_progress': 0,
                    'last_meaningful_change': 0,
                    'action_pattern_history': [],
                    'score_history': [current_score],
                    'termination_reason': None
                })
            
            while (actions_taken < actual_max_actions and 
                current_state not in ['WIN', 'GAME_OVER'] and
                current_score < 100):  # Reasonable win condition
                
                # CRITICAL: Always check current available actions from the latest response
                # Increment action counter (use local session counter, not global)
                actions_taken += 1
                
                # PROGRESS-BASED CAP EXTENSION: Check if we should extend the cap due to progress
                if hasattr(self, '_action_cap_system') and self._action_cap_system['enabled']:
                    extended_cap = self._check_progress_and_extend_cap(current_score, actions_taken, actual_max_actions, max_actions_per_game)
                    if extended_cap > actual_max_actions:
                        actual_max_actions = extended_cap
                        print(f" ACTION CAP UPDATED: Now {actual_max_actions} actions (progress-based extension)")
                
                print("=" * 80)
                print(f" ACTION {actions_taken}/{actual_max_actions} | Game: {game_id} | Score: {current_score}")
                print("=" * 80)
                
                # Check if we've reached the action limit
                if actions_taken >= actual_max_actions:
                    print(f" REACHED SMART ACTION LIMIT ({actual_max_actions}) - Stopping session")
                    if hasattr(self, '_progress_tracker'):
                        self._progress_tracker['termination_reason'] = f"Reached smart action cap ({actual_max_actions})"
                    break
                
                if not available_actions:
                    print("  No available actions - attempting recovery before stopping")
                    # Try to get fresh actions from API investigation
                    try:
                        investigation = await self._investigate_game_state(game_id, session_data.get('guid'))
                        if investigation and 'available_actions' in investigation:
                            fresh_actions = investigation['available_actions']
                            if fresh_actions:
                                print(f"  Recovery successful: Got fresh actions {fresh_actions}")
                                available_actions = fresh_actions
                            else:
                                print("  Recovery failed: Still no actions available - stopping game loop")
                                break
                        else:
                            print("  Recovery failed: Could not investigate game state - stopping game loop")
                            break
                    except Exception as e:
                        print(f"  Recovery failed: {e} - stopping game loop")
                        break
                    
                print(f" Available: {available_actions}")
                
                #  SMART EARLY TERMINATION - Check if we should stop due to lack of progress
                if hasattr(self, '_should_terminate_early'):
                    should_terminate, termination_reason = self._should_terminate_early(current_score, actions_taken)
                    if should_terminate:
                        print(f" EARLY TERMINATION: {termination_reason}")
                        if hasattr(self, '_progress_tracker'):
                            self._progress_tracker['termination_reason'] = termination_reason
                        
                        # Analyze why we got stuck
                        if hasattr(self, '_analyze_stagnation_cause'):
                            stagnation_analysis = self._analyze_stagnation_cause(game_id, action_history)
                            print(f" STAGNATION ANALYSIS:")
                            print(f"   Patterns: {stagnation_analysis.get('stagnation_patterns', [])}")
                            print(f"   Effectiveness: {stagnation_analysis.get('action_effectiveness', {})}")
                            print(f"   Suggested Fixes: {stagnation_analysis.get('suggested_fixes', [])}")
                        
                        break
                
                # Use our intelligent action selection with current available actions
                try:
                    action_selection_response = {
                        'available_actions': available_actions,
                        'state': current_state,
                        'score': current_score,
                        'frame': session_data.get('frame', [])  # Get frame from session data
                    }
                    selected_action = self._select_next_action(action_selection_response, game_id)
                    
                    # Track memory operations for action selection
                    if hasattr(self, 'global_counters'):
                        self.global_counters['total_memory_operations'] = self.global_counters.get('total_memory_operations', 0) + 1
                    
                except Exception as e:
                    print(f" Error in action selection: {e}")
                    selected_action = None
                
                if selected_action is None:
                    print(" Action selection failed - stopping game loop")
                    break
                
                # Update action pattern history for loop detection
                if hasattr(self, '_progress_tracker'):
                    self._progress_tracker['action_pattern_history'].append(selected_action)
                    # Keep only last 15 actions for pattern detection
                    if len(self._progress_tracker['action_pattern_history']) > 15:
                        self._progress_tracker['action_pattern_history'] = self._progress_tracker['action_pattern_history'][-15:]
                
                if selected_action is None:
                    print(" Action selection failed - stopping game loop")
                    break
                
                # Get frame analysis for enhanced action execution
                if hasattr(self, '_last_frame_analysis'):
                    current_frame_analysis = getattr(self, '_last_frame_analysis', {}).get(game_id, {})
                else:
                    current_frame_analysis = {}
                
                #  CRITICAL FIX: Get actual frame dimensions from latest action result with proper structure handling
                actual_frame = None
                
                # First try to get frame from latest action result (most current)
                if hasattr(self, '_last_frame') and self._last_frame:
                    actual_frame = self._last_frame
                    print(f" Using frame from last action result")
                # Fallback to session data if no recent frame
                elif session_data.get('frame'):
                    actual_frame = session_data.get('frame', [])
                    print(f" Using frame from session data")
                # Last resort: investigation data
                elif investigation.get('frame'):
                    actual_frame = investigation.get('frame', [])
                    print(f" Using frame from investigation data")
                
                #  CRITICAL FIX: Normalize frame into a 2D numpy array and derive grid dims
                normalized_arr, dims = self._normalize_frame(actual_frame)
                if normalized_arr is not None:
                    # Store canonical current frame data as 2D numpy array (height, width)
                    # Note: normalize returns (width, height) dims tuple
                    self.current_frame_data = normalized_arr
                    actual_grid_dims = dims
                    print(f" Using actual frame dimensions: {actual_grid_dims} (W×H) - normalized frame stored")
                else:
                    actual_grid_dims = (64, 64)
                    print(f" No frame data available after normalization, using fallback dimensions: {actual_grid_dims}")
                
                # Optimize coordinates if needed (for ACTION6)
                x, y = None, None
                if selected_action == 6:
                    if current_frame_analysis:
                        x, y = self._enhance_coordinate_selection_with_frame_analysis(
                            selected_action, actual_grid_dims, game_id, current_frame_analysis
                        )
                    else:
                        x, y = self._optimize_coordinates_for_action(selected_action, actual_grid_dims, game_id)
                    
                    #  CRITICAL FIX: Validate coordinates before action execution
                    if x is not None and y is not None:
                        if not self._verify_grid_bounds(x, y, actual_grid_dims[0], actual_grid_dims[1]):
                            print(f" COORDINATE ERROR: ({x},{y}) out of bounds for {actual_grid_dims}, using safe fallback")
                            # Use safe fallback coordinates
                            x, y = self._safe_coordinate_fallback(actual_grid_dims[0], actual_grid_dims[1], "coordinate out of bounds")
                    elif selected_action == 6:
                        # If no coordinates were generated for ACTION6, create safe ones
                        print(f" No coordinates generated for ACTION6, using safe fallback")
                        x, y = self._safe_coordinate_fallback(actual_grid_dims[0], actual_grid_dims[1], "no coordinates generated")
                
                coord_display = f" at ({x},{y})" if x is not None else ""
                print(f" EXECUTING: Action {selected_action}{coord_display}")
                
                # Show intelligent action description (more concise)
                action_desc = self.get_action_description(selected_action, game_id)
                if "Learned:" in action_desc:
                    # Extract just the key learning info
                    learned_info = action_desc.split("Learned:")[-1].strip()
                    print(f"    {learned_info[:80]}..." if len(learned_info) > 80 else f"    {learned_info}")
                elif self.available_actions_memory.get('action_learning_stats', {}).get('total_observations', 0) > 0:
                    # Show basic stats
                    total_attempts = self.available_actions_memory.get('action_effectiveness', {}).get(selected_action, {}).get('attempts', 0)
                    if total_attempts > 0:
                        success_rate = self.available_actions_memory.get('action_effectiveness', {}).get(selected_action, {}).get('success_rate', 0.0)
                        print(f"    Success: {success_rate:.1%} ({total_attempts} tries)")
                
                # Execute the action with actual grid dimensions
                try:
                    action_result = await self._send_enhanced_action(
                        game_id, selected_action, x, y, actual_grid_dims[0], actual_grid_dims[1], current_frame_analysis
                    )
                    
                    # Track memory operations for each action
                    if hasattr(self, 'global_counters'):
                        self.global_counters['total_memory_operations'] = self.global_counters.get('total_memory_operations', 0) + 1
                    
                except Exception as e:
                    print(f" Action execution error: {e}")
                    action_result = None
                
                # Ensure tracking lists exist
                if 'effective_actions' not in locals():
                    effective_actions = []
                if 'action_history' not in locals():
                    action_history = []

                # Initialize was_effective default to False to avoid unbound local errors
                was_effective = False

                if action_result:
                    # 🧠 GOVERNOR ERROR HANDLING - Check for API errors that need Governor intervention
                    if self.governor and 'error' in action_result:
                        governor_error_response = self.governor.handle_api_error(
                            error_response=action_result,
                            game_id=game_id,
                            context={
                                'game_id': game_id,
                                'current_state': current_state,
                                'current_score': current_score,
                                'selected_action': selected_action,
                                'coordinates': (x, y) if x is not None else None
                            }
                        )
                        
                        if governor_error_response:
                            print(f"🧠 GOVERNOR ERROR HANDLING: {governor_error_response['reasoning']}")
                            
                            # Handle RESET recommendation
                            if governor_error_response.get('recommended_action') == 'RESET':
                                print(f"🔄 GOVERNOR RECOMMENDING RESET for {game_id}")
                                try:
                                    reset_result = await self._start_game_session(game_id)
                                    if reset_result and 'error' not in reset_result:
                                        print(f"✅ RESET successful - game {game_id} started")
                                        # Update state with reset result
                                        new_state = reset_result.get('state', current_state)
                                        new_score = reset_result.get('score', current_score)
                                        new_available = reset_result.get('available_actions', available_actions)
                                        # Update frame data
                                        if 'frame' in reset_result:
                                            session_data['frame'] = reset_result['frame']
                                            self._last_frame = reset_result['frame']
                                        # Continue with the session
                                        continue
                                    else:
                                        print(f"❌ RESET failed: {reset_result.get('error', 'Unknown error')}")
                                        # Fall through to normal error handling
                                except Exception as e:
                                    print(f"❌ RESET execution error: {e}")
                                    # Fall through to normal error handling
                            
                            # Handle WAIT recommendation (rate limiting)
                            elif governor_error_response.get('recommended_action') == 'WAIT':
                                wait_time = 5.0  # Default wait time
                                print(f"⏳ GOVERNOR RECOMMENDING WAIT for {wait_time}s due to rate limiting")
                                await asyncio.sleep(wait_time)
                                continue
                            
                            # Handle STOP recommendation (authentication error)
                            elif governor_error_response.get('recommended_action') == 'STOP':
                                print(f"🛑 GOVERNOR RECOMMENDING STOP due to authentication error")
                                return {
                                    'error': 'Authentication failed - stopping session',
                                    'state': 'ERROR',
                                    'score': 0,
                                    'available_actions': []
                                }
                    
                    try:
                        # Update state from response
                        new_state = action_result.get('state', current_state)
                        new_score = action_result.get('score', current_score)
                        # CRITICAL: Extract available actions from API response
                        new_available = action_result.get('available_actions', available_actions)
                    except Exception as e:
                        print(f" Error processing action result: {e}")
                        new_state = current_state
                        new_score = current_score
                        new_available = available_actions
                    
                    #  CRITICAL FIX: Update frame data from action result with proper dimension handling
                    new_frame = action_result.get('frame') or action_result.get('grid', [])
                    if new_frame:
                        session_data['frame'] = new_frame
                        self._last_frame = new_frame  # Keep latest frame for dimension extraction

                        # Normalize and store canonical frame data
                        normalized_new, new_dims = self._normalize_frame(new_frame)
                        if normalized_new is not None:
                            self.current_frame_data = normalized_new
                            frame_width, frame_height = new_dims
                            print(f" Updated frame data: {frame_width}x{frame_height} (W×H) - normalized and stored")
                        else:
                            print(f" Updated frame data: Invalid frame structure after normalization")
                    
                    # Track effectiveness
                    score_improvement = new_score - current_score
                    was_effective = (score_improvement > 0 or new_state in ['WIN'])
                    
                    # Update simulation agent with action outcome
                    if self.simulation_agent:
                        try:
                            coordinates = (x, y) if x is not None and y is not None else None
                            # Determine success based on response data
                            success = self._determine_action_success(action_result)
                            self.simulation_agent.update_with_outcome(
                                action=selected_action,
                                coordinates=coordinates,
                                actual_outcome=success,
                                context=current_state
                            )
                        except Exception as e:
                            logger.warning(f"Failed to update simulation agent: {e}")
                    
                    # Clean result display
                    if score_improvement > 0:
                        print(f" RESULT: Score {current_score} → {new_score} (+{score_improvement:.1f}) | State: {new_state}")
                    elif score_improvement < 0:
                        print(f" RESULT: Score {current_score} → {new_score} ({score_improvement:.1f}) | State: {new_state}")
                    else:
                        print(f"  RESULT: Score unchanged ({new_score}) | State: {new_state}")
                    
                    # Update available actions for next iteration
                    try:
                        if new_available != available_actions:
                            print(f" Actions: {available_actions} → {new_available}")
                            available_actions = new_available  # CRITICAL FIX: Update available_actions with fresh API data
                    except ValueError as e:
                        # Handle array comparison issues
                        if "ambiguous" in str(e):
                            # Convert to lists for comparison
                            new_list = list(new_available) if hasattr(new_available, '__iter__') else new_available
                            old_list = list(available_actions) if hasattr(available_actions, '__iter__') else available_actions
                            if new_list != old_list:
                                print(f" Actions: {available_actions} → {new_available}")
                                available_actions = new_available
                        else:
                            raise
                    
                # CRITICAL: Validate that action_result is a dictionary before processing
                if not isinstance(action_result, dict):
                    print(f" CRITICAL ERROR: action_result is not a dict, it's {type(action_result)}: {action_result}")
                    print(f" This will cause 'str' object has no attribute 'items' error")
                    # Convert string error to proper error dictionary
                    if isinstance(action_result, str):
                        action_result = {
                            'error': action_result,
                            'state': 'ERROR',
                            'score': 0,
                            'available_actions': []
                        }
                    else:
                        # Skip processing this action result
                        continue
                    
                    #  CRITICAL FIX: Use unified energy consumption for consistency
                    # Determine if this was exploration or repetitive behavior
                    is_exploration = selected_action == 6 and hasattr(self, '_progress_tracker')  # ACTION6 is typically exploration
                    is_repetitive = (not was_effective and actions_taken > 100)
                    
                    # Consume energy using unified method
                    remaining_energy = self._unified_energy_consumption(
                        action_effective=was_effective,
                        is_exploration=is_exploration,
                        is_repetitive=is_repetitive
                    )
                    
                    # Show energy status more concisely
                    if remaining_energy < 50:
                        print(f" Energy: {remaining_energy:.1f}/100 {'' if remaining_energy < 20 else '🟡' if remaining_energy < 40 else '🟢'}")
                    
                    #  CRITICAL FIX: Intelligent sleep trigger system
                    # Calculate recent effectiveness for smart sleep decisions
                    recent_effective_count = sum(1 for action in action_history[-20:] if action.get('effective', False))
                    recent_effectiveness = recent_effective_count / max(1, len(action_history[-20:]))
                    
                    # Check if sleep cycle should trigger using intelligent method
                    should_sleep = self._should_trigger_sleep_cycle(actions_taken, recent_effectiveness)
                    
                    if should_sleep:
                        print(f" SLEEP TRIGGER: Low energy ({remaining_energy:.1f}) after {actions_taken} actions")
                        
                        # Execute enhanced sleep cycle with current data
                        sleep_result = await self._trigger_enhanced_sleep_with_arc_data(
                            action_history, effective_actions, game_id
                        )
                        print(f" Sleep completed: {sleep_result}")
                        
                        # Restore energy using unified system
                        energy_restoration = 25.0
                        self.current_energy = min(100.0, self.current_energy + energy_restoration)
                        print(f" Energy restored: +{energy_restoration:.1f} → {self.current_energy:.1f}/100")
                    
                    # Track effectiveness for analysis
                try:
                    if was_effective:
                        effective_actions.append({
                            'action_number': selected_action,
                            'action_type': f'ACTION{selected_action}',
                            'score_achieved': new_score,
                            'score_improvement': score_improvement,
                            'effectiveness': min(1.0, score_improvement / 20.0),
                            'coordinates': (x, y) if x is not None else None
                        })
                    
                    # Update tracking
                    action_history.append({
                        'action': selected_action,
                        'coordinates': (x, y) if x is not None else None,
                        'before_score': current_score,
                        'after_score': new_score,
                        'effective': was_effective,
                        'state_change': f"{current_state} → {new_state}"
                    })
                except Exception as e:
                    print(f" Error in action tracking: {e}")
                    
                    # Update current state for next iteration
                try:
                    current_state = new_state
                    current_score = new_score
                    # available_actions already updated above
                    
                    # Update session_data with new frame information for next iteration
                    if 'frame' in action_result:
                        session_data['frame'] = action_result['frame']
                except Exception as e:
                    print(f" Error updating state: {e}")
                    
                else:
                    # Record failed action
                    try:
                        action_history.append({
                            'action': selected_action,
                            'coordinates': (x, y) if x is not None else None,
                            'before_score': current_score,
                            'after_score': current_score,
                            'effective': False,
                            'state_change': 'FAILED',
                            'error': True
                        })
                    except Exception as e:
                        print(f" Error recording failed action: {e}")
                
                # WIN RATE-BASED PER-ACTION ENERGY DEPLETION
                try:
                    # Get current energy parameters based on win rate
                    energy_params = self._calculate_win_rate_adaptive_energy_parameters()
                    action_cost = energy_params['action_energy_cost']
                    
                    # Apply effectiveness modifier to energy cost
                    if was_effective:
                        action_cost *= 0.8  # Reward effective actions with 20% energy savings
                    else:
                        action_cost *= energy_params.get('effectiveness_multiplier', 1.0)  # Penalize ineffective actions
                
                    # Deplete energy
                    self.current_energy = max(0.0, self.current_energy - action_cost)
                except Exception as e:
                    print(f" Error in energy depletion: {e}")
                
                # Display energy status periodically
                if actions_taken % 5 == 0:  # Every 5 actions
                    energy_emoji = "🟢" if self.current_energy > 70 else "🟡" if self.current_energy > 40 else ""
                    print(f" Energy: {self.current_energy:.1f}/100 {energy_emoji}")
                
                # Check for immediate sleep trigger due to low energy
                sleep_threshold = energy_params['sleep_trigger_threshold']
                if self.current_energy <= sleep_threshold:
                    print(f" SLEEP TRIGGER: Low energy ({self.current_energy:.1f}) after {actions_taken} actions")
                    print(f" ENHANCED SLEEP CONSOLIDATION STARTING...")
                    
                    # Trigger sleep consolidation
                    if hasattr(self, 'sleep_system') and self.sleep_system:
                        try:
                            # Create a mini sleep experience for rapid consolidation
                            sleep_experiences = [{
                                'action': selected_action,
                                'coordinates': (x, y) if x is not None else None,
                                'effectiveness': was_effective,
                                'score_change': new_score - current_score if action_result else 0,
                                'context': f"Action during {energy_params['learning_phase']} phase"
                            }]
                            
                            sleep_result = self._enhanced_sleep_consolidation(
                                experiences=sleep_experiences,
                                sleep_reason=f"Energy depletion during {energy_params['learning_phase']}"
                            )
                            
                            # Restore energy based on sleep quality
                            if sleep_result and sleep_result.get('success', False):
                                # Ensure current_energy is not None before arithmetic operations
                                if self.current_energy is None:
                                    print(" Warning: current_energy was None during restoration, resetting to 60.0")
                                    self.current_energy = 60.0
                                energy_restoration = min(40.0, 100.0 - self.current_energy)  # Restore up to 40 energy
                                self.current_energy += energy_restoration
                                print(f" Energy restored: {self.current_energy - energy_restoration:.1f} → {self.current_energy:.1f}")
                            
                        except Exception as e:
                            print(f" Sleep consolidation error: {e}")
                            # Fallback: restore some energy anyway
                            self.current_energy = min(100.0, self.current_energy + 25.0)
                    else:
                        # Simple energy restoration if no sleep system
                        print(" Skipping memory consolidation - no predictive core available")
                        self.current_energy = min(100.0, self.current_energy + 25.0)
                        print(f" Energy restored to: {self.current_energy:.1f}/100")
                
                
                # Track actions without progress for emergency override
                try:
                    if current_score == investigation.get('score', 0):
                        if not hasattr(self, '_actions_without_progress'):
                            self._actions_without_progress = 0
                        self._actions_without_progress += 1
                    else:
                        # Progress made - reset counter
                        self._actions_without_progress = 0
                except Exception as e:
                    print(f" Error tracking progress: {e}")
                
                # Display score progress every 10 actions
                try:
                    if actions_taken % 10 == 0 or actions_taken - last_score_check >= 10:
                        score_change = current_score - investigation.get('score', 0)
                        effectiveness_pct = len(effective_actions)/max(1,actions_taken)*100
                        print(f" Progress #{actions_taken}: Score {current_score} (+{score_change}) | Effective: {len(effective_actions)}/{actions_taken} ({effectiveness_pct:.0f}%)")
                        if current_score == investigation.get('score', 0):
                            print(f"     No progress in {actions_taken} actions")
                        last_score_check = actions_taken
                except Exception as e:
                    print(f" Error displaying progress: {e}")
                
                # Rate-limit compliant delay between actions
                # With 8 RPS limit, we need at least 0.125s between requests
                # Use 0.15s for safety margin (6.67 RPS actual rate)
                await asyncio.sleep(0.15)
            
            # Session complete - handle scorecard cleanup based on mode and creation status
            if hasattr(self, 'current_scorecard_id') and self.current_scorecard_id:
                # Determine if we should close the scorecard - be more conservative
                should_close_scorecard = False
                close_reason = ""
                
                # Track scorecard usage
                if not hasattr(self, '_scorecard_usage_tracker'):
                    self._scorecard_usage_tracker = {
                        'created_at': time.time(),
                        'games_used': set(),
                        'actions_taken': 0,
                        'sessions_completed': 0
                    }
                
                # Update usage tracking
                self._scorecard_usage_tracker['games_used'].add(game_id)
                self._scorecard_usage_tracker['sessions_completed'] = session_count
                
                # Only close scorecards in very specific circumstances
                
                # 1. Close if we're explicitly in swarm mode AND this is the end of a game
                if hasattr(self, '_swarm_mode_active') and self._swarm_mode_active:
                    # Only close at the end of a complete game, not during gameplay
                    if hasattr(self, '_game_completed') and self._game_completed:
                        should_close_scorecard = True
                        close_reason = f"Swarm mode: Game {game_id} completed"
                        print(f" [SCORECARD] {close_reason}")
                
                # 2. Close if we've used this scorecard for too many games (prevent memory issues)
                elif len(self._scorecard_usage_tracker['games_used']) >= 20:
                    should_close_scorecard = True
                    close_reason = f"Scorecard limit reached: {len(self._scorecard_usage_tracker['games_used'])} games"
                    print(f" [SCORECARD] {close_reason}")
                
                # 3. Close if this is the final session (very high session count)
                elif session_count >= 500:  # Increased threshold from 100 to 500
                    should_close_scorecard = True
                    close_reason = f"Session limit reached: {session_count} sessions"
                    print(f" [SCORECARD] {close_reason}")
                
                # 4. Close if we're explicitly clearing sessions (e.g., switching games)
                elif hasattr(self, '_force_scorecard_close') and self._force_scorecard_close:
                    should_close_scorecard = True
                    close_reason = "Forced close requested"
                    print(f" [SCORECARD] {close_reason}")
                    self._force_scorecard_close = False  # Reset the flag
                
                # 5. Close if scorecard is very old (prevent stale scorecards)
                elif time.time() - self._scorecard_usage_tracker['created_at'] > 3600:  # 1 hour
                    should_close_scorecard = True
                    close_reason = f"Scorecard expired: {int(time.time() - self._scorecard_usage_tracker['created_at'])}s old"
                    print(f" [SCORECARD] {close_reason}")
                
                # Log the decision
                if should_close_scorecard:
                    print(f" [SCORECARD] Closing scorecard {self.current_scorecard_id} - {close_reason}")
                else:
                    print(f" [SCORECARD] Keeping scorecard {self.current_scorecard_id} active (games: {len(self._scorecard_usage_tracker['games_used'])}, sessions: {session_count})")
                
                if should_close_scorecard:
                    try:
                        scorecard_closed = await self._close_scorecard(self.current_scorecard_id)
                        if scorecard_closed:
                            print(f" Closed scorecard: {self.current_scorecard_id}")
                            self.current_scorecard_id = None  # Clear the scorecard ID
                        else:
                            print(f" Failed to close scorecard {self.current_scorecard_id}")
                    except Exception as e:
                        print(f" Error closing scorecard: {e}")
                else:
                    print(f" Keeping scorecard {self.current_scorecard_id} open for continued training")
                    print(f" Scorecard kept: created_new={getattr(self, '_created_new_scorecard', False)}, session_count={session_count}")
            
            try:
                final_result = {
                    'final_score': current_score,
                    'final_state': current_state,
                    'total_actions': actions_taken,
                    'effective_actions': effective_actions,
                    'action_history': action_history,
                    'success': current_state == 'WIN' or current_score > 0,
                    'termination_reason': (
                        'WIN' if current_state == 'WIN' else
                        'GAME_OVER' if current_state == 'GAME_OVER' else
                        'MAX_ACTIONS' if actions_taken >= max_actions_per_game else
                        'HIGH_SCORE' if current_score >= 100 else
                        'UNKNOWN'
                    )
                }
            except Exception as e:
                print(f" Error creating final result: {e}")
                final_result = {
                    'final_score': 0,
                    'final_state': 'ERROR',
                    'total_actions': 0,
                    'effective_actions': [],
                    'action_history': [],
                    'success': False,
                    'termination_reason': 'ERROR'
                }
            
            # Clean session summary
            try:
                print("\n" + "="*80)
                print(" SESSION COMPLETE")
                print("="*80)
                print(f"Game: {game_id}")
                print(f"Final Score: {current_score} | State: {current_state}")
                print(f"Actions: {actions_taken}/{max_actions_per_game}")
                print(f"Effective: {len(effective_actions)} ({len(effective_actions)/max(1,actions_taken):.1%})")
                print(f"Result: {final_result['termination_reason']}")
                
                # Show post-session memory status
                print(f"\n POST-SESSION MEMORY STATUS:")
                print(f"   Memory Operations: {self.global_counters.get('total_memory_operations', 0)}")
                print(f"   Sleep Cycles: {self.global_counters.get('total_sleep_cycles', 0)}")
                print(f"   Energy Level: {self.current_energy:.2f}")
            
                # Show coordinate intelligence summary if available
                if hasattr(self, 'enhanced_coordinate_intelligence'):
                    print(" Coordinate Intelligence: ACTIVE")
                
                print("="*80)
            except Exception as e:
                print(f" Error in session summary: {e}")
            
            return final_result
            
        except Exception as e:
            print(f"    Error in direct control training: {e}")
            print(f"    Error type: {type(e)}")
            import traceback
            print(f"    Traceback: {traceback.format_exc()}")
            return {"error": str(e), "actions_taken": 0}
    
    async def _trigger_enhanced_sleep_with_arc_data(
        self, 
        action_history: List[Dict], 
        effective_actions: List[Dict], 
        game_id: str
    ) -> Dict[str, Any]:
        """Enhanced sleep cycle that integrates with our updated sleep system."""
        try:
            # Prepare ARC-specific data for enhanced sleep system
            arc_data = {
                'action_effectiveness': {
                    action['action']: {
                        'effectiveness': action.get('effective', False),
                        'score_improvement': action.get('after_score', 0) - action.get('before_score', 0),
                        'coordinates': action.get('coordinates'),
                        'context': f"Game {game_id}, Action {action.get('action')}"
                    }
                    for action in action_history[-50:]  # Last 50 actions
                },
                'coordinate_intelligence': {
                    'successful_coordinates': [
                        action.get('coordinates') for action in effective_actions 
                        if action.get('coordinates') is not None
                    ],
                    'failed_patterns': [
                        action.get('coordinates') for action in action_history 
                        if not action.get('effective', False) and action.get('coordinates') is not None
                    ][-20:],  # Last 20 failed coordinates
                    'game_context': game_id
                },
                'action_sequences': {
                    'recent_sequence': [action.get('action') for action in action_history[-10:]],
                    'effective_sequence': [action.get('action_number') for action in effective_actions[-5:]],
                    'pattern_analysis': self._analyze_action_patterns(action_history)
                }
            }
            
            # Get goal system data if available
            goal_data = {
                'active_goals': getattr(self, '_current_active_goals', []),
                'emergent_goals': {
                    'coordinate_exploration': len(set(
                        action.get('coordinates') for action in action_history 
                        if action.get('coordinates') is not None
                    )),
                    'score_improvement': len(effective_actions),
                    'game_completion': game_id
                }
            }
            
            # Execute enhanced sleep cycle using our updated sleep system
            if hasattr(self, 'sleep_system') and self.sleep_system is not None:
                # Initialize sleep system if needed
                if not hasattr(self.sleep_system, 'is_sleeping'):
                    logger.warning("Sleep system not properly initialized, using fallback")
                    await asyncio.sleep(0.5)
                    return await self._fallback_sleep_result()
                
                # Enter sleep mode
                fake_agent_state = type('AgentState', (), {'energy': self.current_energy})()
                self.sleep_system.enter_sleep(fake_agent_state)
                
                # Create minimal replay buffer from recent actions
                replay_buffer = self._create_replay_buffer_from_actions(action_history)
                
                # Execute enhanced sleep cycle
                print(f" ENHANCED SLEEP CONSOLIDATION STARTING...")
                sleep_result = self.sleep_system.execute_sleep_cycle(
                    replay_buffer=replay_buffer,
                    arc_data=arc_data,
                    goal_data=goal_data
                )
                
                # Wake up from sleep
                wake_result = self.sleep_system.wake_up()
                
                # ENHANCED: Run Frame Analyzer Sleep Consolidation
                if self.frame_analyzer and hasattr(self.frame_analyzer, 'consolidate_learning_during_sleep'):
                    print(f" VISUAL INTELLIGENCE CONSOLIDATION...")
                    visual_consolidation = self.frame_analyzer.consolidate_learning_during_sleep()
                    
                    print(f"    Visual Hypotheses Generated: {visual_consolidation['new_hypotheses_generated']}")
                    print(f"    Visual Patterns Discovered: {visual_consolidation['patterns_discovered']}")
                    print(f"    Visual Learning Insights: {len(visual_consolidation['learning_insights'])}")
                    
                    # Log visual insights for debugging
                    for insight in visual_consolidation['learning_insights']:
                        print(f"      • {insight}")
                    
                    # Get and apply visual recommendations for next actions
                    visual_recommendations = self.frame_analyzer.get_actionable_recommendations()
                    if visual_recommendations:
                        print(f"    Top Visual Recommendation: {visual_recommendations[0]['recommendation']}")
                        print(f"      Confidence: {visual_recommendations[0]['confidence']:.1%}")
                        print(f"      Reasoning: {visual_recommendations[0]['reasoning']}")
                else:
                    print(f"    Frame analyzer not available for visual consolidation")
                
                # Update sleep cycle counter
                self.global_counters['total_sleep_cycles'] = self.global_counters.get('total_sleep_cycles', 0) + 1
                
                print(f" ENHANCED SLEEP COMPLETE - Patterns: {sleep_result.get('failed_patterns_identified', 0)} failed, "
                    f"{sleep_result.get('successful_patterns_strengthened', 0)} successful, "
                    f"Strategies: {sleep_result.get('diversification_strategies_created', 0)}")
                
                return {
                    'success': True,
                    'sleep_type': 'enhanced_consolidation',
                    'sleep_cycles_completed': self.global_counters['total_sleep_cycles'],
                    'arc_data_processed': len(arc_data['action_effectiveness']),
                    'coordinate_patterns_analyzed': len(arc_data['coordinate_intelligence']['successful_coordinates']),
                    'sleep_system_result': sleep_result,
                    'wake_result': wake_result,
                    'failed_patterns_identified': sleep_result.get('failed_patterns_identified', 0),
                    'successful_patterns_strengthened': sleep_result.get('successful_patterns_strengthened', 0),
                    'diversification_strategies_created': sleep_result.get('diversification_strategies_created', 0)
                }
            else:
                return await self._fallback_sleep_result()
                
        except Exception as e:
            logger.error(f"Enhanced sleep cycle error: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_executed': True
            }
    
    async def _fallback_sleep_result(self) -> Dict[str, Any]:
        """ CRITICAL FIX: Enhanced fallback sleep with memory consolidation when advanced system is unavailable."""
        
        print(f" ENHANCED FALLBACK SLEEP STARTING...")
        
        # Perform basic memory consolidation operations
        consolidation_results = {
            'patterns_identified': 0,
            'successful_patterns_strengthened': 0,
            'failed_patterns_weakened': 0,
            'coordinate_patterns_discovered': 0,
            'action_effectiveness_updated': 0
        }
        
        try:
            # 1. Consolidate action effectiveness memory
            if hasattr(self, 'available_actions_memory'):
                action_memory = self.available_actions_memory
                total_observations = action_memory.get('action_learning_stats', {}).get('total_observations', 0)
                
                if total_observations > 0:
                    print(f"    Consolidating {total_observations} action observations...")
                    
                    # Strengthen successful patterns
                    effectiveness_data = action_memory.get('action_effectiveness', {})
                    for action, data in effectiveness_data.items():
                        if data.get('success_rate', 0) > 0.6:  # High success rate
                            # Strengthen successful patterns (increase confidence)
                            data['confidence'] = min(1.0, data.get('confidence', 0.5) + 0.1)
                            consolidation_results['successful_patterns_strengthened'] += 1
                        elif data.get('success_rate', 0) < 0.2:  # Low success rate
                            # Weaken failed patterns (decrease confidence)
                            data['confidence'] = max(0.1, data.get('confidence', 0.5) - 0.05)
                            consolidation_results['failed_patterns_weakened'] += 1
                    
                    consolidation_results['action_effectiveness_updated'] = len(effectiveness_data)
            
            # 2. Consolidate coordinate intelligence if frame analyzer exists
            if hasattr(self, 'frame_analyzer') and self.frame_analyzer:
                print(f"    Running basic visual consolidation...")
                
                # Simple pattern discovery in explored coordinates
                if hasattr(self, '_progress_tracker'):
                    explored_coords = getattr(self._progress_tracker, 'explored_coordinates', set())
                    if len(explored_coords) > 10:
                        print(f"       Analyzed {len(explored_coords)} coordinate patterns")
                        consolidation_results['coordinate_patterns_discovered'] = len(explored_coords) // 10
            
            # 3. Update global memory counters
            if hasattr(self, 'global_counters'):
                self.global_counters['total_memory_operations'] = self.global_counters.get('total_memory_operations', 0) + 1
                self.global_counters['total_sleep_cycles'] = self.global_counters.get('total_sleep_cycles', 0) + 1
            
            # 4. Basic rest period for system consolidation
            await asyncio.sleep(0.2)  # Brief consolidation pause
            
            print(f" FALLBACK SLEEP COMPLETE - Strengthened: {consolidation_results['successful_patterns_strengthened']}, "
                f"Weakened: {consolidation_results['failed_patterns_weakened']}, "
                f"Coordinates: {consolidation_results['coordinate_patterns_discovered']}")
            
            return {
                'success': True,
                'sleep_type': 'enhanced_fallback',
                'memory_consolidation_performed': True,
                'sleep_cycles_completed': self.global_counters.get('total_sleep_cycles', 0),
                **consolidation_results
            }
            
        except Exception as e:
            logger.warning(f"Fallback sleep consolidation error: {e}")
            return {
                'success': True,
                'sleep_type': 'basic_fallback',
                'note': 'Basic sleep with minimal consolidation',
                'error': str(e)
            }
    
    def _create_replay_buffer_from_actions(self, action_history: List[Dict]) -> List:
        """Create a minimal replay buffer from action history for sleep system."""
        try:
            from core.data_models import Experience, AgentState, SensoryInput
            import torch
            
            replay_buffer = []
            
            for i, action_data in enumerate(action_history[-50:]):  # Last 50 actions
                try:
                    # Create minimal states for experience
                    current_state = AgentState(
                        visual=torch.randn(3, 64, 64),  # Placeholder visual
                        proprioception=torch.tensor([
                            action_data.get('coordinates', [0, 0])[0] if action_data.get('coordinates') else 0,
                            action_data.get('coordinates', [0, 0])[1] if action_data.get('coordinates') else 0
                        ], dtype=torch.float32),
                        energy_level=action_data.get('energy', 50.0),
                        timestamp=i
                    )
                    
                    next_state = AgentState(
                        visual=torch.randn(3, 64, 64),  # Placeholder visual
                        proprioception=current_state.proprioception + torch.randn(2) * 0.1,
                        energy_level=max(0, current_state.energy_level - 0.5),
                        timestamp=i + 1
                    )
                    
                    # Create experience
                    experience = Experience(
                        state=current_state,
                        action=action_data.get('action', 1),
                        reward=action_data.get('after_score', 0) - action_data.get('before_score', 0),
                        next_state=next_state,
                        learning_progress=action_data.get('effective', 0) * 0.5 if action_data.get('effective') is not None else 0.0
                    )
                    
                    replay_buffer.append(experience)
                    
                except Exception as e:
                    logger.debug(f"Failed to create experience from action {i}: {e}")
                    continue
            
            return replay_buffer
            
        except ImportError:
            logger.warning("Could not import required classes for replay buffer creation")
            return []
    
    def _analyze_action_patterns(self, action_history: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in action history for sleep system integration."""
        if not action_history:
            return {'pattern_type': 'no_data'}
        
        recent_actions = [action.get('action') for action in action_history[-20:]]
        
        # Detect loops and repetitive patterns
        action_frequency = {}
        for action in recent_actions:
            action_frequency[action] = action_frequency.get(action, 0) + 1
        
        most_frequent = max(action_frequency.items(), key=lambda x: x[1]) if action_frequency else (None, 0)
        
        return {
            'pattern_type': 'repetitive_loop' if most_frequent[1] > len(recent_actions) * 0.6 else 'diverse_exploration',                 
            'dominant_action': most_frequent[0],
            'dominance_ratio': most_frequent[1] / max(1, len(recent_actions)),                                                            
            'action_diversity': len(set(recent_actions)),
            'total_actions_analyzed': len(recent_actions)
        }


# ===== Compatibility shim =====
# Some utility methods were accidentally defined at module level (they
# expect 'self' as the first argument). To preserve backward compatibility
# and avoid AttributeError at runtime, attach those functions as methods
# onto the ContinuousLearningLoop class at import time.
try:
    import inspect as _inspect
    for _name, _obj in list(globals().items()):
        if _inspect.isfunction(_obj) or _inspect.iscoroutinefunction(_obj):
            try:
                _sig = _inspect.signature(_obj)
                _params = list(_sig.parameters.keys())
                if _params and _params[0] == 'self':
                    # Only attach if the class doesn't already provide it
                    if not hasattr(ContinuousLearningLoop, _name):
                        setattr(ContinuousLearningLoop, _name, _obj)
            except Exception:
                # Ignore any functions we cannot introspect
                continue
except Exception:
    # Best-effort shim - failure here is non-fatal; missing methods will
    # continue to raise AttributeError which will be logged elsewhere.
    pass


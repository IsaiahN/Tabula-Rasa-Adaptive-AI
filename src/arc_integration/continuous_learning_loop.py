"""
Continuous Learning Loop for ARC-AGI-3 Training

This module implements a continuous learning system that runs the Adaptive Learning Agent
against ARC-AGI-3 tasks, collecting insights and improving performance over time.
"""
import asyncio
import aiohttp
import json
import logging
import numpy as np
import os
import random
import re  # Added for enhanced parsing
import sys
import time
import torch  # Added for tensor operations
from typing import Dict, List, Any, Optional, Tuple  # Added Tuple for grid dimensions
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor  # Added for SWARM mode
from collections import deque  # Added for rate limiting

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to the path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Now import with absolute imports within the package
from arc_integration.arc_meta_learning import ARCMetaLearningSystem
from core.meta_learning import MetaLearningSystem
from core.salience_system import SalienceCalculator, SalienceMode, SalienceWeightedReplayBuffer
from core.sleep_system import SleepCycle
from core.agent import AdaptiveLearningAgent
from core.predictive_core import PredictiveCore
from goals.goal_system import GoalInventionSystem, GoalPhase
from core.energy_system import EnergySystem
from memory.dnc import DNCMemory

try:
    # Import for salience mode comparison if available  
    from salience_mode_comparison import SalienceModeComparator
except ImportError:
    # Fallback if comparison module is not available
    class SalienceModeComparator:
        @staticmethod
        def compare_modes(*args, **kwargs):
            return {'comparison_available': False}
            return {'comparison_available': False}
    SALIENCE_COMPARATOR_AVAILABLE = True
except ImportError:
    SALIENCE_COMPARATOR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.warning("SalienceModeComparator not available - comparison features disabled")

logger = logging.getLogger(__name__)

# ARC-3 API Configuration
ARC3_BASE_URL = "https://three.arcprize.org"
ARC3_SCOREBOARD_URL = "https://arcprize.org/leaderboard"

# Rate Limiting Configuration for ARC-AGI-3 API
# Official limit: 600 requests per minute (RPM)
ARC3_RATE_LIMIT = {
    'requests_per_minute': 600,
    'requests_per_second': 10,  # 600/60 = 10 RPS max
    'safe_requests_per_second': 8,  # Conservative limit with 20% buffer
    'backoff_base_delay': 1.0,  # Base delay for exponential backoff
    'backoff_max_delay': 60.0,  # Maximum backoff delay
    'request_timeout': 30.0     # Timeout per request
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
                print(f"⏸️ Rate limit: Waiting {wait_time:.1f}s (at {len(self.request_times)}/600 RPM)")
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
        
        print(f"🚫 Rate limit exceeded (429) - backing off {self.backoff_delay:.1f}s")
        
    def handle_success_response(self):
        """Handle a successful response - reset backoff."""
        if self.consecutive_429s > 0:
            print(f"✅ Request succeeded - resetting backoff (was {self.backoff_delay:.1f}s)")
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
    max_actions_per_session: int = 500000  # Increased default action limit for deeper exploration
    enable_contrarian_strategy: bool = False  # New contrarian mode
    salience_mode: SalienceMode = SalienceMode.LOSSLESS
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
    
    def __init__(
        self,
        arc_agents_path: str,
        tabula_rasa_path: str,
        api_key: Optional[str] = None,
        save_directory: str = "continuous_learning_data"
    ):
        self.arc_agents_path = Path(arc_agents_path)
        self.tabula_rasa_path = Path(tabula_rasa_path)
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.getenv('ARC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ARC_API_KEY not found. Please set ARC_API_KEY environment variable "
                "or copy .env.template to .env and add your API key."
            )
        
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        
        # Initialize meta-learning systems
        base_meta_learning = MetaLearningSystem(
            memory_capacity=2000,
            insight_threshold=0.15,
            consolidation_interval=50,
            save_directory=str(self.save_directory / "base_meta_learning")
        )
        
        self.arc_meta_learning = ARCMetaLearningSystem(
            base_meta_learning=base_meta_learning,
            pattern_memory_size=1500,
            insight_threshold=0.6,
            cross_validation_threshold=3
        )
        
        # Training state
        self.current_session: Optional[TrainingSession] = None
        self.session_history: List[Dict[str, Any]] = []
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
                6: {'base_relevance': 0.1, 'current_modifier': 0.1, 'recent_success_rate': 0.1, 'last_used': 0, 'consecutive_failures': 0},  # START VERY LOW - coordinate placement ≠ progress
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
                    1: {'boundary_avoidance_radius': 2},  # Actions 1-5 avoid known boundaries
                    2: {'boundary_avoidance_radius': 2},
                    3: {'boundary_avoidance_radius': 2}, 
                    4: {'boundary_avoidance_radius': 2},
                    5: {'boundary_avoidance_radius': 3},  # ACTION 5 gets more avoidance
                    7: {'boundary_avoidance_radius': 2}
                },
                'boundary_stuck_threshold': 3,          # Same coordinates N times = boundary detected
                'success_zone_mapping': {},             # game_id -> {(x,y): {'success_count': int, 'total_attempts': int, 'successful_actions': [int]}}
                'global_coordinate_intelligence': {}    # Cross-game coordinate learning
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
        
        # Rate limiting for ARC-AGI-3 API compliance
        self.rate_limiter = RateLimiter()
        print(f"🛡️ Rate limiter initialized: {ARC3_RATE_LIMIT['safe_requests_per_second']} RPS max, 600 RPM limit")
        
        # Enhanced tracking for sleep states and memory operations
        self.sleep_state_tracker = {
            'is_currently_sleeping': False,
            'sleep_cycles_this_session': 0,
            'total_sleep_time': 0.0,
            'last_sleep_trigger': None,
            'sleep_quality_scores': [],
            'sleep_efficiency': 0.0
        }
        
        # Memory consolidation tracking
        self.memory_consolidation_tracker = {
            'is_consolidating_memories': False,
            'consolidation_operations_count': 0,
            'is_prioritizing_memories': False,
            'high_salience_memories_strengthened': 0,
            'low_salience_memories_decayed': 0,
            'memory_compression_active': False,
            'last_consolidation_score': 0.0,
            'memory_operations_per_cycle': 0
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
            print(f"⚡ Resuming with persistent energy: {self.current_energy:.2f} (after {total_sleep_cycles} sleep cycles)")
        else:
            # Fresh start gets full energy
            self.current_energy = 100.0
            print(f"⚡ Fresh session starting with full energy: {self.current_energy:.2f}")
        
        logger.info("Continuous Learning Loop initialized with ARC-3 API integration")
        
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
        print(f"\n📊 RATE LIMIT STATUS:")
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
        url = "https://three.arcprize.org/api/games"
        headers = {"X-API-Key": self.api_key}
        
        try:
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
                        print(f"🚫 Rate limit exceeded getting games - will retry with backoff")
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
        print("🔍 VERIFYING ARC-AGI-3 API CONNECTION...")
        
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
            print(f"✅ API Connection SUCCESS")
            print(f"   Available Games: {verification_results['total_games_available']}")
            print(f"   Sample Games:")
            for game in verification_results['sample_games']:
                print(f"     - {game['title']} ({game['game_id']})")
        else:
            print(f"❌ API Connection FAILED")
            print(f"   Check API key and internet connection")
        
        # Test 2: Verify ARC-AGI-3-Agents integration
        agents_path_exists = self.arc_agents_path.exists()
        verification_results['arc_agents_available'] = agents_path_exists
        
        if agents_path_exists:
            print(f"✅ ARC-AGI-3-Agents found at: {self.arc_agents_path}")
        else:
            print(f"❌ ARC-AGI-3-Agents NOT found at: {self.arc_agents_path}")
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
                    print(f"❌ Failed to open scorecard")
                    return None
            
            # Step 2: Prepare RESET call
            url = f"{ARC3_BASE_URL}/api/cmd/RESET"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Build payload based on reset type
            payload = {
                "game_id": game_id,
                "card_id": self.current_scorecard_id
            }
            
            if existing_guid:
                # Level Reset - reset within existing game session
                payload["guid"] = existing_guid
                print(f"🔄 LEVEL RESET for {game_id}")
            else:
                # New Game - start fresh game session
                print(f"🔄 NEW GAME RESET for {game_id}")
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=ARC3_RATE_LIMIT['request_timeout'])) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.rate_limiter.handle_success_response()
                        guid = result.get('guid')
                        
                        if guid:
                            # Store the GUID for this game
                            self.current_game_sessions[game_id] = guid
                            reset_type = "LEVEL RESET" if existing_guid else "NEW GAME"
                            print(f"✅ {reset_type} successful: {game_id}")
                            
                            # Initialize position tracking for ACTION 6 directional movement
                            if not existing_guid:  # Only for new games, not level resets
                                # Get grid dimensions from frame if available
                                frame = result.get('frame', [])
                                if frame and len(frame) > 0 and isinstance(frame[0], list):
                                    grid_height, grid_width = len(frame), len(frame[0])
                                else:
                                    grid_width, grid_height = 64, 64  # Default
                                
                                # Start ACTION 6 position at center of grid for directional movement
                                self._current_game_x = grid_width // 2
                                self._current_game_y = grid_height // 2
                                print(f"🎯 Initialized ACTION 6 position at center: ({self._current_game_x},{self._current_game_y}) for {grid_width}x{grid_height} grid")
                                
                                # Clear boundary data for new games - boundaries are game-specific
                                universal_boundary = self.available_actions_memory['universal_boundary_detection']
                                legacy_boundary = self.available_actions_memory['action6_boundary_detection']
                                
                                # Clear universal boundary data
                                if game_id in universal_boundary['boundary_data']:
                                    old_boundaries = len(universal_boundary['boundary_data'][game_id])
                                    print(f"🧹 Cleared {old_boundaries} previous universal boundary mappings for new game {game_id}")
                                
                                # Initialize fresh universal boundary detection for this game
                                universal_boundary['boundary_data'][game_id] = {}
                                universal_boundary['coordinate_attempts'][game_id] = {}
                                universal_boundary['action_coordinate_history'][game_id] = {}
                                universal_boundary['stuck_patterns'][game_id] = {}
                                universal_boundary['success_zone_mapping'][game_id] = {}
                                
                                # Initialize ACTION 6 directional system
                                if 6 in universal_boundary['directional_systems']:
                                    universal_boundary['directional_systems'][6]['current_direction'][game_id] = 'right'
                                
                                # Initialize legacy system for backward compatibility
                                legacy_boundary['boundary_data'][game_id] = {}
                                legacy_boundary['coordinate_attempts'][game_id] = {}
                                legacy_boundary['last_coordinates'][game_id] = None
                                legacy_boundary['stuck_count'][game_id] = 0
                                legacy_boundary['current_direction'][game_id] = 'right'
                                
                                print(f"🌐 Initialized universal boundary detection system for {game_id}")
                                print(f"🔄 All actions will now benefit from boundary awareness and coordinate intelligence")
                            
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
                            print(f"❌ No GUID returned for game {game_id}")
                            return None
                    elif response.status == 429:
                        # Handle rate limit exceeded
                        self.rate_limiter.handle_429_response()
                        print(f"🚫 Rate limit exceeded on RESET {game_id} - will retry with backoff")
                        # Retry after backoff
                        await asyncio.sleep(self.rate_limiter.backoff_delay)
                        return await self._start_game_session(game_id, existing_guid)  # Recursive retry
                    else:
                        self.rate_limiter.handle_success_response()  # Not a rate limit issue
                        error_text = await response.text()
                        print(f"❌ RESET failed: {response.status} - {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            print(f"⏰ API request timeout on RESET {game_id}")
            return None
        except Exception as e:
            print(f"❌ Error starting game session for {game_id}: {e}")
            return None

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
            
            payload = {}  # Empty payload as shown in the API example
            
            print(f"🔄 Opening scorecard...")
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=ARC3_RATE_LIMIT['request_timeout'])) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.rate_limiter.handle_success_response()
                        scorecard_id = result.get('card_id')
                        
                        if scorecard_id:
                            self.current_scorecard_id = scorecard_id
                            print(f"✅ Scorecard opened: {scorecard_id}")
                            return scorecard_id
                        else:
                            print(f"❌ No card_id returned")
                            return None
                    elif response.status == 429:
                        # Handle rate limit exceeded
                        self.rate_limiter.handle_429_response()
                        print(f"🚫 Rate limit exceeded opening scorecard - will retry with backoff")
                        # Retry after backoff
                        await asyncio.sleep(self.rate_limiter.backoff_delay)
                        return await self._open_scorecard()  # Recursive retry
                    else:
                        self.rate_limiter.handle_success_response()  # Not a rate limit issue
                        error_text = await response.text()
                        print(f"❌ Failed to open scorecard: {response.status} - {error_text}")
                        return None
                        
        except asyncio.TimeoutError:
            print(f"⏰ API request timeout opening scorecard")
            return None
        except Exception as e:
            print(f"❌ Error opening scorecard: {e}")
            return None

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
            
            print(f"🔄 Closing scorecard: {scorecard_id}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Closed scorecard: {scorecard_id}")
                        return True
                    else:
                        error_text = await response.text()
                        print(f"❌ Failed to close scorecard: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            print(f"❌ Error closing scorecard: {e}")
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
            print(f"❌ No existing session GUID found for {game_id}, cannot perform level reset")
            return None
        
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
                    'sleep_trigger_energy': 20.0,
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
            frame = response_data.get('frame', [])
            if frame and len(frame) > 0 and isinstance(frame[0], list):
                # Frame is a 2D grid: frame[y][x] format
                height = len(frame[0])  # Number of rows
                width = len(frame[0][0]) if len(frame[0]) > 0 else 64  # Number of columns
                
                # Validate dimensions are reasonable
                if 1 <= width <= 64 and 1 <= height <= 64:
                    return (width, height)
                else:
                    logger.warning(f"Invalid grid dimensions detected: {width}x{height}, using fallback")
            
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
        
        # Infer state from success/failure indicators
        if re.search(r'\b(win|victory|success|solved)\b', combined_output, re.IGNORECASE):
            return 'WIN'
        elif re.search(r'\b(game.*?over|failed|timeout|error)\b', combined_output, re.IGNORECASE):
            return 'GAME_OVER'
        
        return 'NOT_FINISHED'  # Default assumption
    
    def _parse_episode_results_comprehensive(self, stdout: str, stderr: str, game_id: str) -> Dict[str, Any]:
        """Comprehensive parsing of episode results with enhanced pattern detection."""
        result = {'success': False, 'final_score': 0, 'actions_taken': 0, 'level_progressed': False, 'current_level': None}
        combined_output = stdout + "\n" + stderr
        
        # Check for level progression indicators
        level_progression_patterns = [
            r'level.*(\d+).*complete',
            r'passed.*level.*(\d+)', 
            r'advanced.*level.*(\d+)',
            r'next.*level.*(\d+)',
            r'level.*up.*(\d+)',
            r'stage.*(\d+).*complete',
            r'tier.*(\d+).*unlock'
        ]
        
        for pattern in level_progression_patterns:
            match = re.search(pattern, combined_output, re.IGNORECASE)
            if match:
                try:
                    level = int(match.group(1))
                    result['level_progressed'] = True
                    result['current_level'] = level
                    break
                except ValueError:
                    continue
        
        # Enhanced success detection patterns
        success_patterns = [
            r'\b(win|victory|success|solved|correct|passed)\b',
            r'✅',
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
        
        for pattern in action_patterns:
            matches = re.findall(pattern, combined_output, re.IGNORECASE)
            if matches:
                try:
                    result['actions_taken'] = max(int(m) for m in matches)
                except ValueError:
                    result['actions_taken'] = len(matches)
        
        # If success detected but no score, infer reasonable score
        if result['success'] and result['final_score'] == 0:
            result['final_score'] = 100
        
        return result
    
    def _should_continue_game(self, response_data: Dict[str, Any]) -> bool:
        """Check if game should continue based on state - only stop on WIN or GAME_OVER."""
        state = response_data.get('state', 'NOT_FINISHED')
        should_continue = state == 'NOT_FINISHED'
        
        if not should_continue:
            logger.info(f"Game ending with state: {state}")
        
        return should_continue
    
    def _verify_grid_bounds(self, x: int, y: int, grid_width: int, grid_height: int) -> bool:
        """Verify coordinates are within actual grid bounds."""
        return 0 <= x < grid_width and 0 <= y < grid_height
    
    def _select_next_action(self, response_data: Dict[str, Any], game_id: str) -> Optional[int]:
        """Select appropriate action based on learned intelligence and available_actions from API response."""
        available = response_data.get('available_actions', [])
        
        if not available:
            logger.warning(f"No available actions for game {game_id}")
            return None
        
        # Update available actions tracking
        self._update_available_actions(response_data, game_id)
        
        # Track available actions from RESET response
        current_available = available
        if not hasattr(self, '_last_available_actions'):
            self._last_available_actions = {}
        
        last_available = self._last_available_actions.get(game_id, [])
        if current_available != last_available:
            print(f"🎮 Available Actions for {game_id}: {current_available}")
            self._last_available_actions[game_id] = current_available
        
        # Use intelligent action selection with strategic ACTION 6 control
        selected_action = self._select_intelligent_action_with_relevance(available, {'game_id': game_id})
        
        # Add to action history
        self.available_actions_memory['action_history'].append(selected_action)
        
        logger.debug(f"🎯 Selected action {selected_action} from available {available} for {game_id}")
        return selected_action
    
    async def _send_enhanced_action(
        self,
        game_id: str,
        action_number: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        grid_width: int = 64,
        grid_height: int = 64
    ) -> Optional[Dict[str, Any]]:
        """Send action with proper validation, coordinate optimization, effectiveness tracking, and rate limiting."""
        guid = self.current_game_sessions.get(game_id)
        if not guid:
            logger.warning(f"No active session for game {game_id}")
            return None
        
        # Validate action number
        if action_number not in [1, 2, 3, 4, 5, 6, 7]:
            logger.error(f"Invalid action number: {action_number}")
            return None
        
        # Optimize coordinates for coordinate-based actions
        if action_number == 6:
            if x is None or y is None:
                # Use learned coordinate optimization
                x, y = self._optimize_coordinates_for_action(action_number, (grid_width, grid_height))
                logger.info(f"🎯 Optimized coordinates for action {action_number}: ({x},{y})")
            
            # Verify coordinates are within actual grid bounds
            if not self._verify_grid_bounds(x, y, grid_width, grid_height):
                logger.error(f"Coordinates ({x},{y}) out of bounds for grid {grid_width}x{grid_height}")
                return None
        
        try:
            # Apply rate limiting BEFORE making the request
            await self.rate_limiter.acquire()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=ARC3_RATE_LIMIT['request_timeout'])) as session:
                headers = {
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json"
                }
                
                # Determine API endpoint
                url = f"{ARC3_BASE_URL}/api/cmd/ACTION{action_number}"
                
                # Build payload
                payload = {
                    "game_id": game_id,
                    "guid": guid
                }
                
                # ACTION6 uses coordinates, others use reasoning
                if action_number == 6:
                    payload["x"] = x
                    payload["y"] = y
                else:
                    # Add reasoning for actions 1-5,7 (following ARC-3 API pattern)
                    payload["reasoning"] = {
                        "policy": f"intelligent_selection_action_{action_number}",
                        "action_type": f"ACTION{action_number}",
                        "grid_size": f"{grid_width}x{grid_height}",
                        "intelligence_used": self.available_actions_memory['current_game_id'] == game_id,
                        "timestamp": time.time()
                    }
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.rate_limiter.handle_success_response()
                        
                        # Track available actions changes
                        current_available = data.get('actions', [])
                        last_available = getattr(self, '_last_available_actions', {}).get(game_id, [])
                        
                        # Only show available_actions if they changed
                        if current_available != last_available:
                            print(f"🎮 Available Actions Changed for {game_id}: {current_available}")
                            # Store the new available actions
                            if not hasattr(self, '_last_available_actions'):
                                self._last_available_actions = {}
                            self._last_available_actions[game_id] = current_available
                        
                        logger.info(f"Action {action_number} successful for {game_id}")
                        
                        # Update action intelligence with response
                        self._update_available_actions(data, game_id, action_number)
                        
                        # Record coordinate effectiveness if applicable
                        if action_number == 6 and x is not None and y is not None:
                            success = data.get('state') not in ['GAME_OVER'] and 'error' not in data
                            self._record_coordinate_effectiveness(action_number, x, y, success)
                            
                            # Update current position tracking for future directional movement
                            self._current_game_x = x
                            self._current_game_y = y
                        
                        return data
                    elif response.status == 429:
                        # Handle rate limit exceeded - this is the critical case for actions!
                        self.rate_limiter.handle_429_response()
                        error_text = await response.text()
                        print(f"🚫 ACTION {action_number} rate limited (429) - backing off {self.rate_limiter.backoff_delay:.1f}s")
                        logger.warning(f"Rate limit hit on ACTION{action_number} for {game_id}: {error_text}")
                        
                        # Wait for backoff then retry
                        await asyncio.sleep(self.rate_limiter.backoff_delay)
                        return await self._send_enhanced_action(game_id, action_number, x, y, grid_width, grid_height)
                    else:
                        self.rate_limiter.handle_success_response()  # Not a rate limit issue
                        error_text = await response.text()
                        print(f"❌ ACTION {action_number} FAILED: {response.status}")
                        logger.warning(f"Action {action_number} failed for {game_id}: {response.status} - {error_text}")
                        
                        # Record failure if coordinates were used
                        if action_number == 6 and x is not None and y is not None:
                            self._record_coordinate_effectiveness(action_number, x, y, False)
                        
                        return None
                        
        except asyncio.TimeoutError:
            print(f"⏰ ACTION {action_number} timeout for {game_id}")
            return None
        except Exception as e:
            logger.error(f"Error sending action {action_number} to {game_id}: {e}")
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
                    print(f"✅ API Connection OK: {len(games)} games available")
                    return True
                else:
                    print(f"⚠️ API returned empty game list: {games}")
                    return False
            else:
                print(f"❌ API Error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API Connection Failed: {e}")
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
        
        # Validate ARC-AGI-3-Agents path exists
        if not self.arc_agents_path.exists():
            raise ValueError(f"ARC-AGI-3-Agents path does not exist: {self.arc_agents_path}")
        
        if not (self.arc_agents_path / "main.py").exists():
            raise ValueError(f"main.py not found in ARC-AGI-3-Agents: {self.arc_agents_path}")
        
        print(f"Starting ENHANCED ARC-3 training on game: {game_id}")
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
                            
                        # 🏆 PRESERVE WINNING MEMORIES - Mark as high-priority, hard to delete
                        self._preserve_winning_memories(session_result, current_score, game_id)
                    
                    # 📈 CHECK FOR LEVEL PROGRESSION - Only preserve NEW breakthroughs
                    if session_result.get('level_progressed', False):
                        new_level = session_result.get('current_level', 1)
                        previous_best = self.game_level_records.get(game_id, {}).get('highest_level', 0)
                        
                        if new_level > previous_best:
                            print(f"🎉 TRUE LEVEL BREAKTHROUGH! {game_id} advanced from level {previous_best} to {new_level}")
                            # This is a real breakthrough - preserve with hierarchical priority
                            self._preserve_breakthrough_memories(session_result, current_score, game_id, new_level, previous_best)
                        else:
                            print(f"📊 Level {new_level} maintained on {game_id} (no new breakthrough)")
                    else:
                        consecutive_failures += 1
                        
                    print(f"Mastery Session {session_count}: {'WIN' if success else 'LOSS'} | Score: {current_score} | State: {game_state}")  # Updated naming
                    
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
                        print(f"🧠 Boredom detected: {boredom_results['reason']}")
                        if boredom_results['curriculum_advanced']:
                            print(f"📈 Curriculum complexity advanced to level {boredom_results['new_complexity']}")
                    
                    # Integrate goal invention system - discover emergent goals from patterns
                    goal_results = self._process_emergent_goals(session_result, game_results, learning_progress)
                    if goal_results['new_goals_discovered']:
                        print(f"🎯 Discovered {len(goal_results['new_goals'])} new emergent goals")
                        for goal in goal_results['new_goals']:
                            print(f"   New goal: {goal.description} (priority: {goal.priority:.2f})")
                    
                    # CRITICAL: Enhanced terminal state handling with retry logic
                    if game_state in ['WIN', 'GAME_OVER']:
                        print(f"🎯 Game {game_id} reached terminal state: {game_state}")
                        
                        # If it's GAME_OVER and we haven't tried many sessions, try contrarian strategy
                        if game_state == 'GAME_OVER' and session_count < 2 and current_score < 10:
                            print(f"🔄 Early GAME_OVER detected - activating contrarian strategy for retry")
                            self.contrarian_strategy_active = True
                            # Don't break, let it try one more session with different strategy
                        else:
                            break
                    
                    # Check if we should continue based on performance
                    if self._should_stop_training(game_results, target_performance):
                        print(f"Target performance reached for {game_id}")
                        break
                        
                    # Enhanced delay between episodes for rate limit compliance
                    # ARC-3 games involve many API calls, so be conservative
                    episode_delay = 3.0  # Increased from 2.0 to 3.0 seconds
                    print(f"⏸️ Rate limit compliance: waiting {episode_delay}s between episodes")
                    await asyncio.sleep(episode_delay)
                        
                else:
                    print(f"Session {session_count + 1} failed: {session_result.get('error', 'Unknown error')}")
                    consecutive_failures += 1
                    
                    # Stop if too many consecutive API failures
                    if consecutive_failures >= 5:
                        print(f"Stopping training for {game_id} after 5 consecutive API failures")
                        break
                    
                    # Enhanced backoff for failures to respect rate limits
                    failure_delay = min(10.0, 5.0 + (consecutive_failures * 2.0))  # Progressive backoff
                    print(f"⏸️ Failure backoff: waiting {failure_delay}s after {consecutive_failures} failures")
                    await asyncio.sleep(failure_delay)
                    
            except Exception as e:
                logger.error(f"Error in session {session_count + 1} for {game_id}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print(f"Stopping training for {game_id} due to repeated errors")
                    break
                # Enhanced error backoff to prevent rapid retry cycles that could hit rate limits
                error_delay = min(15.0, 5.0 + (consecutive_failures * 3.0))  # Even longer backoff for errors
                print(f"⏸️ Error backoff: waiting {error_delay}s after error #{consecutive_failures}")
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

    def _select_intelligent_action_with_relevance(self, available_actions: List[int], context: Dict[str, Any]) -> int:
        """
        Enhanced action selection with relevance scoring and strategic ACTION 6 control.
        
        Key improvements:
        - ACTION 6 used sparingly as strategic mini-reset only when stuck
        - Dynamic relevance scoring that adapts over time
        - Progress tracking to prevent disruption of good sequences
        - Predictive coordinate selection for ACTION 6
        """
        game_id = context.get('game_id', 'unknown')
        action_count = len(self.available_actions_memory['action_history'])
        
        # Update action relevance scores based on recent performance
        self._update_action_relevance_scores()
        
        # Check for progress stagnation (key insight from your observation)
        progress_stagnant = self._is_progress_stagnant(action_count)
        
        # Calculate weighted action scores with relevance modifiers
        action_scores = {}
        
        # CRITICAL FIX: Hard filter ACTION 6 before scoring to prevent spam
        filtered_available_actions = []
        action6_blocked = False
        
        for action in available_actions:
            if action == 6:
                # Apply strict ACTION 6 availability check
                action6_score = self._calculate_action6_strategic_score(action_count, progress_stagnant)
                if action6_score > 0.01:  # Only allow ACTION 6 if strategic score is meaningful (reduced threshold)
                    filtered_available_actions.append(action)
                    print(f"🎯 ACTION 6 STRATEGIC USE: Progress stagnant={progress_stagnant}, Score={action6_score:.3f}")
                else:
                    action6_blocked = True
                    if action_count % 50 == 0:  # Less frequent logging to reduce spam
                        print(f"🎯 ACTION 6 blocked: Score={action6_score:.3f}, Need progress stagnation")
            else:
                filtered_available_actions.append(action)
        
        # If ACTION 6 was the only option but blocked, allow minimal emergency access
        if not filtered_available_actions and action6_blocked:
            print("🚨 EMERGENCY: ACTION 6 only option available - allowing minimal access")
            filtered_available_actions = [6]
        
        # Score only the filtered available actions
        for action in filtered_available_actions:
            if action in self.available_actions_memory['action_relevance_scores']:
                relevance_data = self.available_actions_memory['action_relevance_scores'][action]
                base_score = relevance_data['base_relevance']
                modifier = relevance_data['current_modifier']
                success_rate = relevance_data['recent_success_rate']
                
                # Special handling for ACTION 6 - your key insight
                if action == 6:
                    action6_score = self._calculate_action6_strategic_score(action_count, progress_stagnant)
                    action_scores[action] = action6_score
                else:
                    # Standard actions get normal scoring with relevance
                    try:
                        semantic_score = self._calculate_semantic_action_score(action, game_id)
                    except:
                        semantic_score = 1.0  # Fallback
                    final_score = base_score * modifier * success_rate * semantic_score
                    
                    # Boost recently successful actions
                    if relevance_data['consecutive_failures'] == 0 and success_rate > 0.6:
                        final_score *= 1.2  # 20% boost for successful actions
                    
                    # Penalize recently failed actions
                    elif relevance_data['consecutive_failures'] > 2:
                        final_score *= 0.7  # 30% penalty for failing actions
                    
                    action_scores[action] = final_score
            else:
                # Fallback for actions without relevance data
                action_scores[action] = 0.5
        
        # Select action based on weighted scores (with some randomness for exploration)
        if not action_scores:
            # Emergency fallback - this should rarely happen
            return random.choice(available_actions) if available_actions else 1
        
        # Use weighted random selection (90% best choice, 10% exploration)
        if random.random() < 0.9:
            selected_action = max(action_scores.keys(), key=lambda k: action_scores[k])
        else:
            # Exploration: weighted random selection
            weights = list(action_scores.values())
            selected_action = random.choices(list(action_scores.keys()), weights=weights)[0]
        
        # Log strategic decisions for ACTION 6
        if selected_action == 6:
            print(f"🎯 ACTION 6 STRATEGIC USE: Progress stagnant={progress_stagnant}, Actions since progress={action_count - self.available_actions_memory['action6_strategy']['last_progress_action']}")
        
        # Update usage tracking
        self._update_action_usage_tracking(selected_action, action_count)
        
        return selected_action

    def _calculate_action6_strategic_score(self, current_action_count: int, progress_stagnant: bool) -> float:
        """
        Calculate strategic score for ACTION 6 based on your insights.
        
        ACTION 6 should only be used when:
        1. Progress has stagnated (no improvement for several actions)
        2. Other actions have been tried sufficiently
        3. Sufficient cooldown has passed since last ACTION 6
        4. As a strategic mini-reset tool
        """
        strategy = self.available_actions_memory['action6_strategy']
        
        # Base score is extremely low - ACTION 6 is strongly discouraged
        base_score = 0.01  # Reduced from 0.1
        
        # Check if we're in emergency mode (truly stuck)
        if not progress_stagnant:
            return 0.001  # Almost never use if progress is being made
        
        # Check minimum actions before considering ACTION 6
        actions_since_start = current_action_count
        if actions_since_start < strategy['min_actions_before_use']:
            return 0.001  # Too early to use ACTION 6
        
        # Check cooldown period
        actions_since_last_action6 = current_action_count - strategy['last_action6_used']
        if actions_since_last_action6 < strategy['action6_cooldown']:
            return 0.001  # Still in cooldown
        
        # Check stagnation duration
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
            print(f"🚨 ACTION 6 EMERGENCY MODE: Stuck for {actions_since_progress} actions")
        
        return min(strategic_score, 0.2)  # Cap at 0.2 to prevent over-use

    def _is_progress_stagnant(self, current_action_count: int) -> bool:
        """
        Determine if progress has stagnated based on recent action effectiveness.
        """
        strategy = self.available_actions_memory['action6_strategy']
        actions_since_progress = current_action_count - strategy['last_progress_action']
        
        return actions_since_progress >= strategy['progress_stagnation_threshold']

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
                        print(f"🎯 ACTION 6 minimal relevance increase: {old_modifier:.3f} → {data['current_modifier']:.3f} (rare meaningful progress detected)")
                    else:
                        # Continuous slow decay to prevent ACTION 6 spam
                        old_modifier = data['current_modifier']
                        data['current_modifier'] = max(0.05, data['current_modifier'] * 0.98)  # Lowered floor to 0.05
                        if old_modifier != data['current_modifier']:
                            print(f"🎯 ACTION 6 relevance decay: {old_modifier:.3f} → {data['current_modifier']:.3f} (preventing coordinate spam)")
                        
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
        action_history = self.available_actions_memory['action_history']
        recent_history = action_history[-window:] if len(action_history) >= window else action_history
        
        attempts = recent_history.count(action_num)
        # For now, assume 50% success rate - this would be updated with actual game feedback
        successes = attempts // 2  # Placeholder - would track actual success/failure
        
        return {'total': attempts, 'successes': successes}

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
                print(f"🎯 Meaningful progress detected: win={has_win}, score={has_score_improvement}, level={has_level_progress}")
            
            return meaningful_progress
        
        # Default to False - no meaningful progress detected
        return False

    def _update_action_usage_tracking(self, selected_action: int, current_action_count: int):
        """Update usage tracking for selected action."""
        if selected_action in self.available_actions_memory['action_relevance_scores']:
            self.available_actions_memory['action_relevance_scores'][selected_action]['last_used'] = current_action_count
            
            # Special tracking for ACTION 6
            if selected_action == 6:
                self.available_actions_memory['action6_strategy']['last_action6_used'] = current_action_count

    def _update_progress_tracking(self, action_result: Dict[str, Any], current_action_count: int):
        """
        Update progress tracking based on action results.
        Call this when you detect that an action led to positive progress.
        """
        if action_result.get('progress_made', False):  # Would be set based on score increase, state improvement, etc.
            self.available_actions_memory['action6_strategy']['last_progress_action'] = current_action_count
            print(f"📈 Progress detected at action {current_action_count}")

    async def _run_real_arc_mastery_session_enhanced(self, game_id: str, session_count: int) -> Dict[str, Any]:  # Renamed method and parameter
        """Enhanced version that runs COMPLETE mastery sessions with up to 100K actions until WIN/GAME_OVER."""
        try:
            # 1. Check for boredom and trigger curriculum advancement
            boredom_results = self._check_and_handle_boredom(session_count)  # Updated parameter name
            
            # 2. Check if agent should sleep before mastery session
            agent_state = self._get_current_agent_state()
            should_sleep_now = self._should_agent_sleep(agent_state, session_count)  # Updated parameter name
            
            sleep_cycle_results = {}
            if should_sleep_now:
                sleep_cycle_results = await self._execute_sleep_cycle(game_id, session_count)  # Updated parameter name
            
            # Check if model decides to reset the game
            reset_decision = self._evaluate_game_reset_decision(game_id, session_count, agent_state)  # Updated parameter name
            
            # Check if contrarian strategy should be activated for consistent GAME_OVER states  
            consecutive_failures = agent_state.get('consecutive_failures', 0)
            contrarian_decision = self._should_activate_contrarian_strategy(game_id, consecutive_failures)
            self.contrarian_strategy_active = contrarian_decision['activate']
            
            # Set up environment with API key
            env = os.environ.copy()
            env['ARC_API_KEY'] = self.api_key
            
            # COMPLETE EPISODE: Run actions in loop until terminal state
            episode_actions = 0
            total_score = 0
            best_score = 0
            final_state = 'NOT_FINISHED'
            episode_start_time = time.time()
            max_actions_per_session = 500000  # Significantly increased from 100K to allow deeper exploration
            
            print(f"🎮 Starting complete mastery session {session_count} for {game_id}")  # Updated naming
            
            # Enhanced option: Choose between external main.py and direct control
            use_direct_control = True  # Set to True to use our enhanced action selection
            
            if use_direct_control:
                print(f"🎯 Using DIRECT API CONTROL with enhanced action selection")
                # Use our direct API control with intelligent action selection
                game_session_result = await self.start_training_with_direct_control(
                    game_id, max_actions_per_session, session_count
                )
                
                if "error" in game_session_result:
                    print(f"❌ Direct control failed: {game_session_result['error']}")
                    print(f"🔄 Falling back to external main.py")
                    use_direct_control = False  # Fall back to external method
                else:
                    # Convert direct control result to expected format
                    total_score = game_session_result.get('final_score', 0)
                    episode_actions = game_session_result.get('total_actions', 0)
                    final_state = game_session_result.get('final_state', 'UNKNOWN')
                    effective_actions = game_session_result.get('effective_actions', [])
                    
                    print(f"🎯 Direct Control Results: Score={total_score}, Actions={episode_actions}, State={final_state}")
                    print(f"🎯 Effective Actions Found: {len(effective_actions)}")
            
            if not use_direct_control:
                print(f"🔄 Using EXTERNAL main.py (fallback mode)")
                # Original external main.py approach
            
            # VERBOSE: Show memory state before mastery session
            print(f"📊 PRE-SESSION MEMORY STATUS:")  # Updated naming
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
                print(f"⚡ Energy boost: +{energy_boost:.2f} for predicted high-complexity game -> {current_energy:.2f}")
                self._update_energy_level(current_energy)
            elif estimated_complexity == 'medium' and current_energy < 60.0:
                energy_boost = 10.0
                current_energy = min(100.0, current_energy + energy_boost)
                print(f"⚡ Energy boost: +{energy_boost:.2f} for predicted medium-complexity game -> {current_energy:.2f}")
                self._update_energy_level(current_energy)
            
            # Reset game at start of episode if needed
            if reset_decision['should_reset']:
                print(f"🔄 Resetting game {game_id}")
                self._record_reset_decision(reset_decision)
            
            # Run complete game session (not individual actions)
            print(f"🎮 Starting complete game session for {game_id}")
            
            # Build command for complete game session
            cmd = [
                'python', 'main.py',
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
                else:
                    # Execute external main.py
                    print(f"🚀 Executing complete game session: {' '.join(cmd)}")
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=str(self.arc_agents_path),
                        env=env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    # Wait for complete game session with longer timeout
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(), 
                            timeout=1800.0  # 30 minutes for deeper exploration
                        )
                        
                        stdout_text = stdout.decode('utf-8', errors='ignore') if stdout else ""
                        stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""
                        
                        print(f"✅ Complete game session finished")
                        
                        # Enhanced logging: Extract and show action details from game output
                        self._log_action_details_from_output(stdout_text, game_id)
                        
                        # Parse complete game results
                        game_results = self._parse_complete_game_session(stdout_text, stderr_text)
                        total_score = game_results.get('final_score', 0)
                        episode_actions = game_results.get('total_actions', 0) 
                        final_state = game_results.get('final_state', 'UNKNOWN')
                        effective_actions = game_results.get('effective_actions', [])
                        
                        print(f"🎯 Game Results: Score={total_score}, Actions={episode_actions}, State={final_state}")
                        print(f"🎯 Effective Actions Found: {len(effective_actions)}")
                        
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
                    
            except Exception as e:
                print(f"❌ Error during complete game session: {e}")
                # Comprehensive error state with null safety
                total_score = 0
                episode_actions = 0
                final_state = 'ERROR'
                effective_actions = []
                stdout_text = ""
                stderr_text = ""
                
                # Log the error for debugging
                print(f"❌ Game session error details: Game={game_id}, Error={str(e)}")
                
                # Check if this is an API connectivity issue
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    print("⚠️ Possible API connectivity issue - validating connection...")
                    api_valid = await self._validate_api_connection()
                    if not api_valid:
                        print("💡 Consider checking ARC_API_KEY and network connectivity")
            
            # Now process the complete game session results
            # Dynamic energy system based on game complexity and learning opportunities
            
            # Calculate adaptive energy cost based on actions and effectiveness
            base_energy_cost = episode_actions * 0.15  # Optimized rate: 0.15 per action (better balance)
            
            # Adjust energy cost based on game complexity
            if episode_actions > 500:  # Very complex game
                energy_multiplier = 1.5  # Higher cost for complex games
                sleep_frequency_target = 200  # Trigger sleep every ~200 actions worth
            elif episode_actions > 200:  # Moderately complex game  
                energy_multiplier = 1.2
                sleep_frequency_target = 150  # Trigger sleep every ~150 actions worth
            else:  # Simple game
                energy_multiplier = 1.0
                sleep_frequency_target = 100  # Trigger sleep every ~100 actions worth
            
            # Additional cost if no effective actions (wasted energy)
            if len(effective_actions) == 0 and episode_actions > 50:
                energy_multiplier *= 1.3  # Penalty for ineffective long games
                print(f"⚡ Energy penalty applied for {episode_actions} ineffective actions")
            
            # Calculate final energy cost
            energy_cost = base_energy_cost * energy_multiplier
            current_energy = pre_sleep_status.get('current_energy_level', self.current_energy)
            remaining_energy = max(0.0, current_energy - energy_cost)
            
            print(f"⚡ Energy: {current_energy:.2f} -> {remaining_energy:.2f}")
            print(f"   Cost: {energy_cost:.3f} (base: {base_energy_cost:.3f} × {energy_multiplier:.1f} multiplier)")
            print(f"   Game complexity: {episode_actions} actions -> {'High' if episode_actions > 500 else 'Medium' if episode_actions > 200 else 'Low'}")
            
            # Dynamic sleep threshold based on learning opportunities
            if len(effective_actions) > 0:
                # Lower threshold when we have effective actions to consolidate
                sleep_threshold = 0.4 + (len(effective_actions) * 0.1)  # More effective actions = higher threshold
                sleep_reason = f"Learning consolidation needed ({len(effective_actions)} effective actions)"
            else:
                # Higher threshold when no learning occurred
                sleep_threshold = 0.2
                sleep_reason = f"Energy depletion (no effective actions found)"
            
            # Trigger sleep based on adaptive thresholds
            sleep_triggered = False
            if remaining_energy <= sleep_threshold:
                print(f"😴 Sleep triggered: {sleep_reason}")
                print(f"   Energy {remaining_energy:.2f} <= threshold {sleep_threshold:.2f}")
                
                sleep_result = await self._trigger_sleep_cycle(effective_actions)
                print(f"🌅 Sleep completed: {sleep_result}")
                sleep_triggered = True
                
                # Adaptive energy replenishment based on learning quality
                base_replenishment = 60.0  # Base 60 energy points restoration (0-100 scale)
                
                # Bonus energy for effective learning
                if len(effective_actions) > 0:
                    learning_bonus = min(30.0, len(effective_actions) * 5.0)  # Up to 30 energy points bonus
                    print(f"⚡ Learning bonus: +{learning_bonus:.2f} energy for {len(effective_actions)} effective actions")
                else:
                    learning_bonus = 0.0
                
                # Bonus energy for complex games (they teach more)
                if episode_actions > 500:
                    complexity_bonus = 20.0  # 20 energy bonus for complex games
                    print(f"⚡ Complexity bonus: +{complexity_bonus:.2f} energy for {episode_actions}-action game")
                elif episode_actions > 200:
                    complexity_bonus = 10.0  # 10 energy bonus for medium games
                    print(f"⚡ Complexity bonus: +{complexity_bonus:.2f} energy for {episode_actions}-action game")
                else:
                    complexity_bonus = 0.0
                
                total_replenishment = base_replenishment + learning_bonus + complexity_bonus
                remaining_energy = min(100.0, remaining_energy + total_replenishment)
                print(f"⚡ Energy replenished: {total_replenishment:.2f} total -> {remaining_energy:.2f}")
            else:
                print(f"⚡ Sleep not needed: Energy {remaining_energy:.2f} > threshold {sleep_threshold:.2f}")
            
            # Update energy level in system
            self._update_energy_level(remaining_energy)
            
            # Update game complexity history for future energy allocation
            # Ensure we have valid values to prevent NoneType errors
            episode_actions = episode_actions if episode_actions is not None else 0
            effective_actions = effective_actions if effective_actions is not None else []
            effectiveness_ratio = min(1.0, len(effective_actions) / max(1, episode_actions))  # Cap at 100%
            self._update_game_complexity_history(game_id, episode_actions, effectiveness_ratio)
            print(f"📈 Updated complexity history for {game_id}: {episode_actions} actions, {effectiveness_ratio:.2%} effective")
            
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
                logger.info(f"🧠 Action Intelligence for {game_id}: "
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
                Path("meta_learning_data"),
                Path("continuous_learning_data"),
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
        print(f"\n🔥 SWARM MODE ACTIVATED")
        print(f"Concurrent Games: {min(max_concurrent, len(games))}")
        print(f"Total Games: {len(games)}")
        print("="*60)
        
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
            print(f"\n🎯 SWARM BATCH {batch_idx}/{len(game_batches)}")
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
                    print(f"✅ {game_id}: {win_rate:.1%} win rate | Grid: {grid_size}")
        
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
        
        print(f"\n🏆 SWARM MODE COMPLETE")
        print(f"Duration: {total_duration/60:.1f} minutes")
        print(f"Games/Hour: {swarm_results['overall_performance'].get('games_per_hour', 0):.1f}")
        print(f"Overall Win Rate: {swarm_results['overall_performance'].get('overall_win_rate', 0):.1%}")
        
        # Show rate limiting statistics
        self.print_rate_limit_status()
        
        return swarm_results
    
    async def _train_on_game_swarm(
        self,
        game_id: str,
        max_episodes: int
    ) -> Dict[str, Any]:
        """Train on a single game for SWARM mode - optimized for concurrent execution."""
        game_results = {
            'game_id': game_id,
            'episodes': [],
            'performance_metrics': {},
            'final_performance': {},
            'grid_dimensions': (64, 64)  # Will be updated dynamically
        }
        
        try:
            episode_count = 1
            consecutive_failures = 0
            
            while episode_count <= max_episodes:
                # Run episode with proper game state checking
                episode_result = await self._run_real_arc_mastery_session_enhanced(game_id, episode_count)
                
                if episode_result and 'error' not in episode_result:
                    game_results['episodes'].append(episode_result)
                    episode_count += 1
                    
                    # Update grid dimensions if available
                    if 'grid_dimensions' in episode_result:
                        game_results['grid_dimensions'] = episode_result['grid_dimensions']
                    
                    success = episode_result.get('success', False)
                    game_state = episode_result.get('game_state', 'NOT_FINISHED')
                    
                    # Only stop if EXPLICITLY told the game is over
                    # Don't assume game over from low scores or timeouts
                    if game_state == 'WIN':
                        logger.info(f"Game {game_id} WON after {episode_count} episodes!")
                        # Continue training even after wins to improve consistency
                    elif game_state == 'GAME_OVER' and episode_count > 10:
                        # Only stop on explicit game over after reasonable attempts
                        logger.info(f"Game {game_id} ended with GAME_OVER after {episode_count} episodes")
                        break
                    
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    # Be more patient with failures - ARC games are hard
                    if consecutive_failures >= 5:  # Increased from 3 to 5
                        logger.warning(f"Game {game_id} had {consecutive_failures} consecutive failures at episode {episode_count}")
                        break
                
                # Rate-limit compliant delay for swarm mode
                # Swarm mode runs multiple games concurrently, so be more conservative
                swarm_delay = 2.0  # Increased from 1.0 to 2.0 seconds
                await asyncio.sleep(swarm_delay)
            
            # Calculate final performance
            game_results['performance_metrics'] = self._calculate_game_performance(game_results)
            game_results['final_performance'] = {
                'episodes_played': len(game_results['episodes']),
                'win_rate': sum(1 for ep in game_results['episodes'] if ep.get('success', False)) / max(1, len(game_results['episodes'])),
                'grid_size': f"{game_results['grid_dimensions'][0]}x{game_results['grid_dimensions'][1]}"
            }
            
            return game_results
            
        except Exception as e:
            logger.error(f"Error in swarm training for {game_id}: {e}")
            return {
                'game_id': game_id,
                'error': str(e),
                'episodes': [],
                'success': False
            }

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
            'win_highlighted': False
        }
        
        logger.info(f"Running continuous learning for session {session_id}")
        
        try:
            # Check if SWARM mode should be used based on configuration
            if session.swarm_enabled and len(session.games_to_play) > 2:
                print(f"\n🔥 SWARM MODE ENABLED for {len(session.games_to_play)} games")
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
                print(f"\n📋 SEQUENTIAL MODE for {len(session.games_to_play)} games")
                session_results['swarm_mode_used'] = False
                
                total_games = len(session.games_to_play)
                for game_idx, game_id in enumerate(session.games_to_play):
                    print(f"\nGame {game_idx + 1}/{total_games}: {game_id}")
                    
                    game_results = await self._train_on_game(
                        game_id,
                        session.max_mastery_sessions_per_game,  # Updated attribute name
                        session.target_performance
                    )
                    
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
                    
            # Finalize session
            session_results['end_time'] = time.time()
            session_results['duration'] = session_results['end_time'] - session_results['start_time']
            session_results['overall_performance'] = self._calculate_session_performance(session_results)
            
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
            
            logger.info(f"Completed training session {session_id} - Win rate: {overall_win_rate:.1%}")
            return session_results
            
        except Exception as e:
            logger.error(f"Error in continuous learning session: {e}")
            session_results['error'] = str(e)
            session_results['end_time'] = time.time()
            return session_results

    def _update_detailed_metrics(self, detailed_metrics: Dict[str, Any], game_results: Dict[str, Any]):
        """Update detailed metrics with game results including sleep and memory operations."""
        # Count sleep cycles (simulated)
        episodes_count = len(game_results.get('episodes', []))
        detailed_metrics['sleep_cycles'] += self.sleep_state_tracker['sleep_cycles_this_session']
        
        # Count high salience experiences
        for episode in game_results.get('episodes', []):
            if episode.get('final_score', 0) > 75:  # High score = high salience
                detailed_metrics['high_salience_experiences'] += 1
        
        # Memory operations tracking
        detailed_metrics['memory_operations'] += self.memory_consolidation_tracker['memory_operations_per_cycle']
        detailed_metrics['consolidation_operations'] = self.memory_consolidation_tracker['consolidation_operations_count']
        detailed_metrics['memories_strengthened'] = self.memory_consolidation_tracker['high_salience_memories_strengthened']
        detailed_metrics['memories_decayed'] = self.memory_consolidation_tracker['low_salience_memories_decayed']
        
        # Compression tracking
        if detailed_metrics.get('salience_mode') == 'decay_compression':
            detailed_metrics['compressed_memories'] = len(getattr(self.salience_calculator, 'compressed_memories', []))
            detailed_metrics['compression_active'] = self.memory_consolidation_tracker['memory_compression_active']
        
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
        if status_flags['is_consolidating_memories']:
            active_systems.append("🧠CONSOLIDATING")
        if status_flags['is_prioritizing_memories']:
            active_systems.append("⚡PRIORITIZING")
        if status_flags['memory_compression_active']:
            active_systems.append("🗜️COMPRESSING")
        if status_flags['is_sleeping']:
            active_systems.append("😴SLEEPING")
            
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
        
        # Grid size information - CRITICAL NEW FEATURE
        if grid_summary.get('dynamic_sizing_verified'):
            print(f"✅ Dynamic Grid Sizing: {grid_summary.get('unique_sizes', 0)} different sizes detected")
            print(f"   Sizes: {', '.join(grid_summary.get('sizes_encountered', []))}")
        else:
            sizes = grid_summary.get('sizes_encountered', ['64x64'])
            print(f"Grid Sizes: {', '.join(sizes)}")
        
        # SWARM mode info if used
        if session_results.get('swarm_mode_used'):
            swarm_info = session_results.get('swarm_results', {}).get('swarm_efficiency', {})
            print(f"🔥 SWARM Mode: {swarm_info.get('games_per_hour', 0):.1f} games/hour | Concurrent speedup enabled")
        
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
        
        print(f"\n🧠 SYSTEM STATUS:")
        print(f"Sleep Cycles: {memory_status['sleep_status']['sleep_cycles_this_session']} | Memory Consolidations: {system_status['is_consolidating_memories']}")
        print(f"Memory Prioritization: {system_status['is_prioritizing_memories']} | Compression: {system_status['memory_compression_active']}")
        
        if system_status['has_made_reset_decisions']:
            reset_stats = memory_status['game_reset_status']
            print(f"🔄 Game Resets: {reset_stats['total_reset_decisions']} decisions | Success Rate: {reset_stats['reset_success_rate']:.1%}")
            if reset_stats['last_reset_reason']:
                print(f"   Last Reset: {reset_stats['last_reset_reason']}")
        
        # Highlight if this was a winning session
        if overall_win_rate > 0.3:
            print("🏆 SUBMIT TO LEADERBOARD - STRONG PERFORMANCE DETECTED!")
        
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
        
        early_avg = sum((game.get('performance_metrics', {}).get('average_score') or 0) for game in early_games) / max(1, len(early_games))
        late_avg = sum((game.get('performance_metrics', {}).get('average_score') or 0) for game in late_games) / max(1, len(late_games))
        
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
        filename = self.save_directory / f"session_{session_results['session_id']}_final.json"
        try:
            # Convert grid_sizes_encountered set to list for JSON serialization
            if 'detailed_metrics' in session_results and 'grid_sizes_encountered' in session_results['detailed_metrics']:
                session_results['detailed_metrics']['grid_sizes_encountered'] = list(session_results['detailed_metrics']['grid_sizes_encountered'])
            
            with open(filename, 'w') as f:
                json.dump(session_results, f, indent=2, default=str)
            
            # Also save meta-learning state
            meta_learning_file = self.save_directory / f"meta_learning_{session_results['session_id']}.json"
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
                logger.info(f"🧠 Loaded action intelligence for {game_id}: {len(intelligence.get('effective_actions', {}))} effective actions")
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
            logger.info(f"💾 Saved action intelligence for {game_id}: {len(effective_actions)} effective actions")
        except Exception as e:
            logger.error(f"Failed to save action intelligence for {game_id}: {e}")

    def display_action_intelligence_summary(self, game_id: Optional[str] = None):
        """Display a summary of learned action intelligence."""
        print(f"\n🧠 ACTION INTELLIGENCE SUMMARY")
        print("="*50)
        
        learning_stats = self.available_actions_memory['action_learning_stats']
        print(f"📊 Global Learning Stats:")
        print(f"   Total Observations: {learning_stats['total_observations']}")
        print(f"   Movements Tracked: {learning_stats['movements_tracked']}")
        print(f"   Effects Catalogued: {learning_stats['effects_catalogued']}")
        print(f"   Game Contexts Learned: {learning_stats['game_contexts_learned']}")
        
        print(f"\n🎯 ACTION MAPPINGS:")
        for action, mapping in self.available_actions_memory['action_semantic_mapping'].items():
            print(f"\n  ACTION{action}:")
            print(f"    Description: {self.get_action_description(action, game_id)}")
            
            # Show movement patterns
            movements = mapping['grid_movement_patterns']
            if any(movements.values()):
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
        
        logger.debug(f"🎯 Available actions updated for {game_id}: {available_actions}")
    
    def _initialize_game_actions(self, game_id: str, initial_actions: List[int]):
        """Initialize action memory for a new game."""
        # Load cached intelligence if available
        if game_id not in self.available_actions_memory['game_intelligence_cache']:
            self.available_actions_memory['game_intelligence_cache'][game_id] = self._load_game_action_intelligence(game_id)
        
        # Reset current session data but preserve learned intelligence
        cached_intel = self.available_actions_memory['game_intelligence_cache'][game_id]
        
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
        
        logger.info(f"🎮 Initialized actions for {game_id}: {initial_actions} (loaded {len(cached_intel.get('effective_actions', {}))} cached patterns)")

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
        self.available_actions_memory['action_learning_stats']['total_observations'] += 1
        
        # Analyze grid movement patterns (for actions 1-5, 7)
        if action != 6:  # Non-coordinate actions
            movement = self._detect_grid_movement(response_data)
            if movement:
                if movement in mapping['grid_movement_patterns']:
                    mapping['grid_movement_patterns'][movement] += 1
                    self.available_actions_memory['action_learning_stats']['movements_tracked'] += 1
                    print(f"🧠 ACTION{action} learned movement: {movement} (total: {mapping['grid_movement_patterns'][movement]})")
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
                self.available_actions_memory['action_learning_stats']['effects_catalogued'] += 1
                print(f"🧠 ACTION{action} learned new effect: {effect}")
        
        # Learn game-specific roles
        if game_id and success:
            role = self._infer_action_role(action, response_data, effects)
            if role:
                if game_id not in mapping['game_specific_roles']:
                    mapping['game_specific_roles'][game_id] = {'role': role, 'confidence': 0.1}
                    self.available_actions_memory['action_learning_stats']['game_contexts_learned'] += 1
                    print(f"🧠 ACTION{action} role in {game_id}: {role}")
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
        confidence_threshold = self.available_actions_memory['action_learning_stats']['pattern_confidence_threshold']
        
        # Add movement pattern info
        movements = mapping['grid_movement_patterns']
        if movements:
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
        if effects:
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
            
            # 6. NEW: Semantic intelligence bonus - use learned action behavior
            semantic_bonus = self._calculate_semantic_action_score(action, game_context)
            
            final_score = max(0.05, base_score + sequence_bonus + diversity_bonus + context_bonus + stagnation_bonus + semantic_bonus - pattern_penalty)
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
        game_id = game_context.get('game_id') if game_context else None
        semantic_score = 0.0
        
        # 1. Game-specific role bonus
        if game_id and game_id in mapping['game_specific_roles']:
            role_info = mapping['game_specific_roles'][game_id]
            confidence_threshold = self.available_actions_memory['action_learning_stats']['pattern_confidence_threshold']
            
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
            print(f"\n🔍 ACTION ANALYSIS for {game_id}:")
            
            # Look for API action calls in the output
            api_action_pattern = r'(ACTION[1-7].*?(?:success|failed|error))'
            api_actions = re.findall(api_action_pattern, stdout_text, re.IGNORECASE | re.DOTALL)
            
            if api_actions:
                print(f"   📡 API Actions Found: {len(api_actions)}")
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
                print(f"   🎮 Available Actions Seen:")
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
                print(f"   🎯 Game States: {', '.join(unique_states)}")
            
            # Look for scores
            score_pattern = r'score[:\s]*(\d+)'
            scores = re.findall(score_pattern, stdout_text, re.IGNORECASE)
            
            if scores:
                score_progression = [int(s) for s in scores[-10:]]  # Last 10 scores
                print(f"   📊 Score Progression: {' → '.join(map(str, score_progression))}")
            
            print()  # Add spacing
            
        except Exception as e:
            print(f"   ⚠️ Error analyzing action details: {e}")

    def _optimize_coordinates_for_action(self, action: int, grid_dims: Tuple[int, int]) -> Tuple[int, int]:
        """Universal coordinate optimization with boundary awareness for all coordinate-based actions."""
        grid_width, grid_height = grid_dims
        
        # Special strategic coordinate selection for ACTION 6 with full directional system
        if action == 6:
            return self._get_strategic_action6_coordinates(grid_dims)
        
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
        
        # Initialize data for this game/action if needed
        if game_id not in boundary_system['boundary_data']:
            boundary_system['boundary_data'][game_id] = {}
            boundary_system['coordinate_attempts'][game_id] = {}
            boundary_system['action_coordinate_history'][game_id] = {}
            boundary_system['stuck_patterns'][game_id] = {}
            boundary_system['success_zone_mapping'][game_id] = {}
        
        # Get action-specific settings
        action_settings = boundary_system['directional_systems'].get(action, {'boundary_avoidance_radius': 2})
        avoidance_radius = action_settings.get('boundary_avoidance_radius', 2)
        
        # Get known boundaries for this game
        known_boundaries = set(boundary_system['boundary_data'][game_id].keys())
        
        # Get success zones for this game
        success_zones = boundary_system['success_zone_mapping'][game_id]
        
        # Find best success zone if available
        if success_zones:
            best_zone = None
            best_success_rate = 0
            
            for coords, zone_data in success_zones.items():
                success_rate = zone_data['success_count'] / max(1, zone_data['total_attempts'])
                if success_rate > best_success_rate and success_rate > 0.3:
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
                print(f"🎯 ACTION {action} SUCCESS ZONE: Using proven coordinates {best_zone} (success rate: {best_success_rate:.1%})")
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
            
            # Check if too close to known boundaries
            coordinates = (x, y)
            too_close = False
            
            for boundary_coord in known_boundaries:
                boundary_x, boundary_y = boundary_coord
                distance = abs(x - boundary_x) + abs(y - boundary_y)
                if distance < avoidance_radius:
                    too_close = True
                    break
            
            if not too_close:
                print(f"🎯 ACTION {action} BOUNDARY-AWARE: Selected ({x},{y}) avoiding {len(known_boundaries)} boundaries")
                return coordinates
            
            attempts += 1
        
        # Fallback - use center if all else fails
        fallback_x, fallback_y = grid_width // 2, grid_height // 2
        print(f"🎯 ACTION {action} FALLBACK: Using center ({fallback_x},{fallback_y}) after boundary avoidance")
        return (fallback_x, fallback_y)

    def _get_strategic_action6_coordinates(self, grid_dims: Tuple[int, int]) -> Tuple[int, int]:
        """
        BOUNDARY-AWARE DIRECTIONAL MOVEMENT for ACTION 6 with intelligent pivoting.
        
        Key Features:
        1. Detects boundaries when same coordinates attempted repeatedly
        2. Saves boundary data as non-deletable until new game starts
        3. Pivots to new semantic directions when boundaries hit
        4. Explores systematically: right -> down -> left -> up -> repeat
        """
        grid_width, grid_height = grid_dims
        game_id = self.available_actions_memory.get('current_game_id', 'unknown')
        
        # Initialize boundary detection data for this game if needed
        boundary_system = self.available_actions_memory['action6_boundary_detection']
        if game_id not in boundary_system['boundary_data']:
            boundary_system['boundary_data'][game_id] = {}
            boundary_system['coordinate_attempts'][game_id] = {}
            boundary_system['last_coordinates'][game_id] = None
            boundary_system['stuck_count'][game_id] = 0
            boundary_system['current_direction'][game_id] = 'right'  # Start moving right
        
        # Get current position from game state (stored by previous ACTION 6 or start at center)
        current_x = getattr(self, '_current_game_x', grid_width // 2)
        current_y = getattr(self, '_current_game_y', grid_height // 2)
        
        # Get current direction for this game
        current_direction = boundary_system['current_direction'][game_id]
        direction_info = boundary_system['direction_progression'][current_direction]
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
            
            # Mark this as a boundary
            boundary_coord = (new_x, new_y)
            boundary_system['boundary_data'][game_id][boundary_coord] = {
                'boundary_type': boundary_type,
                'detection_count': boundary_system['boundary_data'][game_id].get(boundary_coord, {}).get('detection_count', 0) + 1,
                'timestamp': time.time()
            }
            
            print(f"🚧 ACTION 6 BOUNDARY: Hit {boundary_type} at ({new_x},{new_y}) - pivoting direction")
        
        # Check if we're repeating the same coordinates (stuck/boundary detection)
        new_coordinates = (new_x, new_y)
        last_coordinates = boundary_system['last_coordinates'][game_id]
        
        if new_coordinates == last_coordinates:
            # Same coordinates attempted again - boundary detected!
            boundary_system['stuck_count'][game_id] += 1
            stuck_count = boundary_system['stuck_count'][game_id]
            
            if stuck_count >= boundary_system['boundary_stuck_threshold']:
                # BOUNDARY DETECTED - Save as non-deletable boundary data
                boundary_coord = new_coordinates
                boundary_system['boundary_data'][game_id][boundary_coord] = {
                    'boundary_type': f"coordinate_stuck_{current_direction}",
                    'detection_count': boundary_system['boundary_data'][game_id].get(boundary_coord, {}).get('detection_count', 0) + 1,
                    'timestamp': time.time()
                }
                
                print(f"🚧 ACTION 6 BOUNDARY: Detected stuck coordinates ({new_x},{new_y}) after {stuck_count} attempts - saving as boundary")
                hit_boundary = True
                boundary_system['stuck_count'][game_id] = 0  # Reset stuck count
        else:
            # Different coordinates - reset stuck count
            boundary_system['stuck_count'][game_id] = 0
        
        # PIVOT TO NEW DIRECTION if boundary hit or max distance reached
        if hit_boundary:
            # Pivot to next semantic direction
            next_direction = direction_info['next']
            boundary_system['current_direction'][game_id] = next_direction
            
            # Calculate coordinates in new direction from current position
            next_direction_info = boundary_system['direction_progression'][next_direction]
            dx, dy = next_direction_info['coordinate_delta']
            
            pivot_x = current_x + dx
            pivot_y = current_y + dy
            
            # Ensure pivot coordinates are within bounds
            pivot_x = max(0, min(pivot_x, grid_width - 1))
            pivot_y = max(0, min(pivot_y, grid_height - 1))
            
            new_x, new_y = pivot_x, pivot_y
            print(f"🔄 ACTION 6 PIVOT: Direction {current_direction} → {next_direction}, coordinates ({current_x},{current_y}) → ({new_x},{new_y})")
        
        # Update tracking data
        boundary_system['last_coordinates'][game_id] = (new_x, new_y)
        
        # Track coordinate attempt history
        coord_key = (new_x, new_y)
        if coord_key not in boundary_system['coordinate_attempts'][game_id]:
            boundary_system['coordinate_attempts'][game_id][coord_key] = {'attempts': 0, 'consecutive_stuck': 0}
        boundary_system['coordinate_attempts'][game_id][coord_key]['attempts'] += 1
        
        # Update current position tracking
        self._current_game_x = new_x
        self._current_game_y = new_y
        
        # Display boundary intelligence
        num_boundaries = len(boundary_system['boundary_data'][game_id])
        direction_display = current_direction.upper()
        if hit_boundary:
            direction_display = f"{current_direction.upper()}→{boundary_system['current_direction'][game_id].upper()}"
        
        print(f"🎯 ACTION 6 BOUNDARY-AWARE: {direction_display} from ({current_x},{current_y}) → ({new_x},{new_y}) | Boundaries mapped: {num_boundaries}")
        
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
        
        # Initialize game data if needed
        if game_id not in boundary_system['boundary_data']:
            boundary_system['boundary_data'][game_id] = {}
            boundary_system['coordinate_attempts'][game_id] = {}
            boundary_system['action_coordinate_history'][game_id] = {}
            boundary_system['stuck_patterns'][game_id] = {}
            boundary_system['success_zone_mapping'][game_id] = {}
        
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
        
        # Print universal coordinate intelligence update
        total_boundaries = len(boundary_system['boundary_data'][game_id])
        total_success_zones = len(boundary_system['success_zone_mapping'][game_id])
        print(f"📊 COORDINATE INTELLIGENCE: Action {action_num} at {coordinates} ({'✅' if success else '❌'}) | "
              f"Boundaries: {total_boundaries} | Success Zones: {total_success_zones}")
    
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
                    
                    print(f"🚧 UNIVERSAL BOUNDARY: Action {action_num} detected boundary at {coordinates} "
                          f"(confidence: {boundary_data['confidence']:.1%})")
                    
                    # Update global coordinate intelligence
                    self._update_global_coordinate_intelligence(coordinates, action_num, 'boundary')
    
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
            
            print(f"🌍 GLOBAL BOUNDARY: {coordinates} detected across {len(boundary_data['games_detected_in'])} games by {len(boundary_data['actions_that_detected'])} actions")

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
                    logger.info(f"📂 Loaded previous state: {len(self.session_history)} sessions")
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
        max_actions_per_session: int = 500000,  # Increased default parameter
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
        
        # Initialize salience calculator for this session
        self.salience_calculator = SalienceCalculator(
            mode=salience_mode,
            decay_rate=0.01 if salience_mode == SalienceMode.DECAY_COMPRESSION else 0.0,
            salience_min=0.05,
            compression_threshold=0.15
        )
        
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
        # Sleep triggers
        sleep_triggers = {
            'low_energy': agent_state['energy'] < 20.0,
            'high_memory_usage': agent_state['memory_usage'] > 0.9,
            'periodic_sleep': episode_count % 10 == 0 and episode_count > 0,
            'low_learning_progress': agent_state['learning_progress'] < 0.05
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
        recent_lp_history = self.training_state.get('lp_history', [])[-10:]
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
            logger.info(f"🧠 Boredom detected ({boredom_results['boredom_type']}): {boredom_results['reason']}")
        
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
            logger.info(f"🧪 Action experimentation triggered for {current_game}")
    
    def _switch_action_strategy(self) -> str:
        """Switch to a new action strategy to break ineffective patterns."""
        strategies = ['focused_exploration', 'random_sampling', 'pattern_matching', 'conservative_moves']
        current_strategy = self.training_state.get('current_strategy', 'focused_exploration')
        
        # Pick a different strategy
        available_strategies = [s for s in strategies if s != current_strategy]
        new_strategy = random.choice(available_strategies)
        
        self.training_state['current_strategy'] = new_strategy
        logger.info(f"🔄 Strategy switched: {current_strategy} → {new_strategy}")
        
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
        min_sleep_interval = 100  # Minimum actions between sleep cycles
        
        if step_count - last_sleep < min_sleep_interval:
            return False
        
        # Trigger conditions for mid-game sleep
        pattern_accumulation_trigger = step_count % 150 == 0  # Regular pattern consolidation
        
        # Check for significant learning signals in recent actions
        if len(recent_actions) >= 10:
            recent_scores = [action.get('effectiveness', 0) for action in recent_actions[-10:]]
            high_learning_signal = np.mean(recent_scores) > 0.3
            
            if pattern_accumulation_trigger or high_learning_signal:
                return True
        
        # Energy-based trigger with learning opportunity
        current_energy = getattr(self.agent.energy_system, 'current_energy', 100.0)
        if current_energy < 60.0 and len(recent_actions) >= 20:
            # Check if we have patterns worth consolidating
            effectiveness_values = [action.get('effectiveness', 0) for action in recent_actions[-20:]]
            if len([e for e in effectiveness_values if e > 0.2]) >= 5:  # At least 5 effective actions
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
            logger.info(f"🌙 Mid-game consolidating {len(effective_actions)} effective actions")
            
            # Quick memory consolidation (shorter than post-episode sleep)
            try:
                if hasattr(self.agent, 'sleep_system'):
                    # Trigger short consolidation cycle
                    consolidation_duration = min(10, len(effective_actions) * 0.5)  # 0.5s per action
                    
                    sleep_input = {
                        'effective_actions': effective_actions,
                        'consolidation_type': 'mid_game',
                        'duration': consolidation_duration
                    }
                    
                    # Quick memory strengthening without full sleep cycle
                    for i, action in enumerate(effective_actions):
                        if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'update_memory_salience'):
                            # Use action index as memory index for salience update
                            memory_idx = torch.tensor([i % self.agent.memory.memory_size])
                            salience_val = torch.tensor([1.5])
                            self.agent.memory.update_memory_salience(memory_idx, salience_val)
                    
                    logger.info(f"✨ Mid-game consolidation completed in {consolidation_duration:.1f}s")
                
            except Exception as e:
                logger.warning(f"Mid-game sleep error: {e}")
        
        # Partial energy restoration (not full like post-episode)
        if hasattr(self.agent, 'energy_system'):
            current_energy = getattr(self.agent.energy_system, 'current_energy', 100.0)
            restored_energy = min(100.0, current_energy + 20.0)  # Small restoration
            self.agent.energy_system.current_energy = restored_energy
            logger.info(f"⚡ Energy restored: {current_energy:.2f} → {restored_energy:.2f}")

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
        
        logger.info(f"🔄 Simulating {consolidation_points} mid-game consolidation points for {total_actions} actions")
        
        # Process effective actions in chunks as if consolidated during gameplay
        for i in range(consolidation_points):
            start_idx = i * actions_per_consolidation
            end_idx = min((i + 1) * actions_per_consolidation, len(effective_actions))
            action_chunk = effective_actions[start_idx:end_idx]
            
            if not action_chunk:
                continue
                
            logger.info(f"  🌙 Consolidation point {i+1}: {len(action_chunk)} actions")
            
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
                print(f"🚀 Goal system transitioned to {new_phase.value} phase!")
                
        except Exception as e:
            logger.warning(f"Goal processing failed: {e}")
        
        return goal_results
    
    async def _execute_sleep_cycle(self, game_id: str, episode_count: int) -> Dict[str, Any]:
        """Execute a sleep cycle with memory consolidation."""
        self.sleep_state_tracker['is_currently_sleeping'] = True
        self.sleep_state_tracker['sleep_cycles_this_session'] += 1
        
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
            self.sleep_state_tracker['total_sleep_time'] += sleep_results['sleep_duration']
            self.sleep_state_tracker['sleep_quality_scores'].append(sleep_results['consolidation_score'])
            
            logger.info(f"Sleep cycle completed for {game_id} episode {episode_count}: {sleep_results['sleep_duration']:.2f}s")
            
            return sleep_results
            
        finally:
            self.sleep_state_tracker['is_currently_sleeping'] = False
    
    def _prioritize_memories_by_salience(self) -> Dict[str, Any]:
        """Prioritize memories based on salience values."""
        self.memory_consolidation_tracker['is_prioritizing_memories'] = True
        
        try:
            # Simulate memory prioritization
            if self.salience_calculator:
                # Get high salience experiences for priority processing
                high_salience_count = len(self.salience_calculator.get_high_salience_experiences(threshold=0.6))
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
                    print(f"   🛡️ Protected {protected_count} winning memories from salience decay")
                
                consolidation_ops += low_salience_decayed
                
                # Calculate consolidation effectiveness
                consolidation_score = (high_salience_strengthened * 2 - low_salience_decayed * 0.5) / max(1, consolidation_ops)
            else:
                consolidation_score = 0.0
            
            # Update tracking
            self.memory_consolidation_tracker['consolidation_operations_count'] += consolidation_ops
            self.memory_consolidation_tracker['high_salience_memories_strengthened'] += high_salience_strengthened
            self.memory_consolidation_tracker['low_salience_memories_decayed'] += low_salience_decayed
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
                print(f"🔄 CONTRARIAN MODE ACTIVATED: {contrarian_decision['reason']}")
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
            
        print(f"🔄 Applied contrarian strategy: {contrarian_decision['strategy_type']}")
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
        return {
            'is_consolidating': self.memory_consolidation_tracker['is_consolidating_memories'],
            'is_prioritizing': self.memory_consolidation_tracker['is_prioritizing_memories'],
            'compression_active': self.memory_consolidation_tracker['memory_compression_active'],
            'total_consolidation_ops': self.memory_consolidation_tracker['consolidation_operations_count'],
            'last_consolidation_score': self.memory_consolidation_tracker['last_consolidation_score'],
            'high_salience_strengthened': self.memory_consolidation_tracker['high_salience_memories_strengthened'],
            'low_salience_decayed': self.memory_consolidation_tracker['low_salience_memories_decayed']
        }
    
    def _get_current_sleep_state_info(self) -> Dict[str, Any]:
        """Get current sleep state information."""
        return {
            'is_sleeping': self.sleep_state_tracker['is_currently_sleeping'],
            'sleep_cycles_completed': self.sleep_state_tracker['sleep_cycles_this_session'],
            'total_sleep_time': self.sleep_state_tracker['total_sleep_time'],
            'last_sleep_trigger': self.sleep_state_tracker['last_sleep_trigger'],
            'current_energy_level': getattr(self, 'current_energy', 1.0),  # Include actual energy level
            'average_sleep_quality': (
                sum(self.sleep_state_tracker['sleep_quality_scores']) / 
                max(1, len(self.sleep_state_tracker['sleep_quality_scores']))
            ) if self.sleep_state_tracker['sleep_quality_scores'] else 0.0
        }
    
    # ====== PUBLIC STATUS METHODS FOR USER QUERIES ======
    
    def is_consolidating_memories(self) -> bool:
        """Return True if currently consolidating memories."""
        return self.memory_consolidation_tracker['is_consolidating_memories']
    
    def is_prioritizing_memories(self) -> bool:
        """Return True if currently prioritizing memories."""
        return self.memory_consolidation_tracker['is_prioritizing_memories']
    
    def is_sleeping(self) -> bool:
        """Return True if agent is currently in sleep state."""
        return self.sleep_state_tracker['is_currently_sleeping']
    
    def is_memory_compression_active(self) -> bool:
        """Return True if memory compression is currently active."""
        return self.memory_consolidation_tracker['memory_compression_active']
    
    def has_made_reset_decisions(self) -> bool:
        """Return True if model has made any game reset decisions."""
        return self.game_reset_tracker['reset_decisions_made'] > 0
    
    def get_sleep_and_memory_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of sleep states and memory operations.
        
        Returns:
            Complete status including True/False flags for all operations
        """
        return {
            # Sleep state status
            'sleep_status': {
                'is_currently_sleeping': self.is_sleeping(),
                'sleep_cycles_this_session': self.sleep_state_tracker['sleep_cycles_this_session'],
                'total_sleep_time_minutes': self.sleep_state_tracker['total_sleep_time'] / 60.0,
                'sleep_efficiency': self._calculate_sleep_efficiency()
            },
            
            # Memory consolidation status
            'memory_consolidation_status': {
                'is_consolidating_memories': self.is_consolidating_memories(),
                'is_prioritizing_memories': self.is_prioritizing_memories(),
                'consolidation_operations_completed': self.memory_consolidation_tracker['consolidation_operations_count'],
                'high_salience_memories_strengthened': self.memory_consolidation_tracker['high_salience_memories_strengthened'],
                'low_salience_memories_decayed': self.memory_consolidation_tracker['low_salience_memories_decayed'],
                'last_consolidation_effectiveness': self.memory_consolidation_tracker['last_consolidation_score']
            },
            
            # Memory compression status
            'memory_compression_status': {
                'compression_active': self.is_memory_compression_active(),
                'compression_mode': self.salience_calculator.mode.value if self.salience_calculator else 'none',
                'total_compressed_memories': len(getattr(self.salience_calculator, 'compressed_memories', []))
            },
            
            # Game reset decision status
            'game_reset_status': {
                'has_made_reset_decisions': self.has_made_reset_decisions(),
                'total_reset_decisions': self.game_reset_tracker['reset_decisions_made'],
                'reset_success_rate': self.game_reset_tracker['reset_success_rate'],
                'last_reset_reason': self.game_reset_tracker['last_reset_decision']['reason'] if self.game_reset_tracker['last_reset_decision'] else None,
                'reset_decision_criteria': self.game_reset_tracker['reset_decision_criteria']
            }
        }
    
    def _calculate_sleep_efficiency(self) -> float:
        """Calculate how efficiently sleep cycles are being used."""
        if not self.sleep_state_tracker['sleep_quality_scores']:
            return 0.0
        
        quality_scores = self.sleep_state_tracker['sleep_quality_scores']
        sleep_cycles = self.sleep_state_tracker['sleep_cycles_this_session']
        
        # Efficiency = average quality * frequency factor
        avg_quality = sum(quality_scores) / len(quality_scores)
        frequency_factor = min(sleep_cycles / 10.0, 1.0)  # Optimal frequency around 10 cycles per session
        
        return avg_quality * frequency_factor

    def get_system_status_flags(self) -> Dict[str, bool]:
        """
        Get simple True/False flags for all major system operations.
        
        Returns:
            Dictionary with boolean flags for each system operation
        """
        return {
            # Sleep states
            'is_sleeping': self.is_sleeping(),
            'sleep_cycles_active': self.sleep_state_tracker['sleep_cycles_this_session'] > 0,
            
            # Memory consolidation
            'is_consolidating_memories': self.is_consolidating_memories(),
            'is_prioritizing_memories': self.is_prioritizing_memories(), 
            'memory_strengthening_active': self.memory_consolidation_tracker['high_salience_memories_strengthened'] > 0,
            'memory_decay_active': self.memory_consolidation_tracker['low_salience_memories_decayed'] > 0,
            
            # Memory compression
            'memory_compression_active': self.is_memory_compression_active(),
            'has_compressed_memories': len(getattr(self.salience_calculator, 'compressed_memories', [])) > 0,
            
            # Game reset decisions
            'has_made_reset_decisions': self.has_made_reset_decisions(),
            'reset_decision_pending': self._is_reset_decision_pending(),
            
            # System health
            'memory_system_healthy': self._is_memory_system_healthy(),
            'learning_system_active': self.current_session is not None,
            'salience_system_active': self.salience_calculator is not None
        }
    
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
        consolidation_count = self.memory_consolidation_tracker['consolidation_operations_count']
        last_score = self.memory_consolidation_tracker['last_consolidation_score']
        
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
            
            print(f"📊 Parsed game session: {result['total_actions']} actions, {len(effective_actions)} effective, final score {result['final_score']}")
            
        except Exception as e:
            print(f"⚠️ Error parsing complete game session: {e}")
        
        return result

    async def _trigger_sleep_cycle(self, effective_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Trigger a sleep cycle to process and consolidate effective actions with real memory operations."""
        print(f"💤 SLEEP CYCLE INITIATED - Processing {len(effective_actions)} effective actions...")
        
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
            print(f"🧠 Phase 1: Loading and prioritizing existing memories...")
            relevant_memories = await self._load_prioritized_memories()
            sleep_result['priority_memories_loaded'] = len(relevant_memories)
            print(f"   Loaded {len(relevant_memories)} prioritized memories")
            
            # 2. GARBAGE COLLECT IRRELEVANT MEMORIES  
            print(f"🗑️ Phase 2: Garbage collecting irrelevant memories...")
            deleted_count = await self._garbage_collect_memories()
            sleep_result['memories_deleted'] = deleted_count
            print(f"   Deleted {deleted_count} irrelevant memories")
            
            # 3. COMBINE/COMPRESS SIMILAR MEMORIES
            print(f"🔗 Phase 3: Combining similar memory patterns...")
            combined_count = await self._combine_similar_memories()
            sleep_result['memories_combined'] = combined_count
            print(f"   Combined {combined_count} similar memory clusters")
            
            # 4. STRENGTHEN EFFECTIVE ACTION MEMORIES
            print(f"💪 Phase 4: Strengthening effective action memories...")
            for action in effective_actions:
                effectiveness = action.get('effectiveness', 0)
                if effectiveness > 0.1:  # Much lower threshold to capture more learnings - from 0.3 to 0.1
                    await self._strengthen_action_memory_with_context(action, relevant_memories)
                    sleep_result['memories_strengthened'] += 1
            
            # 5. GENERATE INSIGHTS FROM MEMORY PATTERNS
            print(f"🔍 Phase 5: Generating insights from memory patterns...")
            if len(effective_actions) >= 2 or len(relevant_memories) >= 5:
                insights = await self._generate_memory_insights(effective_actions, relevant_memories)
                sleep_result['insights_generated'] = len(insights)
                for insight in insights:
                    print(f"💡 Insight: {insight}")
            
            # 6. PREPARE MEMORY GUIDANCE FOR NEXT GAME
            print(f"🎯 Phase 6: Preparing memory-informed guidance...")
            guidance = await self._prepare_memory_guidance(relevant_memories, effective_actions)
            sleep_result['guidance_prepared'] = len(guidance)
            print(f"   Prepared {len(guidance)} guidance points for next game")
            
            print(f"💤 SLEEP CYCLE COMPLETED:")
            print(f"   📚 {sleep_result['priority_memories_loaded']} memories loaded")
            print(f"   🗑️ {sleep_result['memories_deleted']} memories deleted")  
            print(f"   🔗 {sleep_result['memories_combined']} memories combined")
            print(f"   💪 {sleep_result['memories_strengthened']} memories strengthened")
            print(f"   💡 {sleep_result['insights_generated']} insights generated")
            
            # UPDATE GLOBAL COUNTERS - THIS IS THE FIX!
            self._update_global_counters(
                sleep_cycle_completed=True,
                memory_ops=sleep_result['priority_memories_loaded'],
                memories_deleted=sleep_result['memories_deleted'],
                memories_combined=sleep_result['memories_combined'],
                memories_strengthened=sleep_result['memories_strengthened']
            )
            
        except Exception as e:
            print(f"⚠️ Error during sleep cycle: {e}")
        
        return sleep_result

    def _strengthen_action_memory(self, action: Dict[str, Any]) -> None:
        """Strengthen memory of an effective action."""
        action_type = action.get('action_type', 'unknown')
        effectiveness = action.get('effectiveness', 0)
        score = action.get('score_achieved', 0)
        
        print(f"🧠 Strengthening memory: {action_type} (effectiveness: {effectiveness:.2f}, score: {score})")
        
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
            print(f"⚠️ Error generating insights: {e}")
        
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
                print(f"⚡ Energy level updated to {self.current_energy:.2f}")
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
                Path("continuous_learning_data"),
                Path("meta_learning_data"), 
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
                            print(f"   ⚠️ Failed to load {memory_file}: {e}")
            
            # Prioritize memories by relevance (recent + effective)
            def priority_score(memory):
                recency_score = memory.get('file_age', 0) / 1000000  # Recent files score higher
                effectiveness_score = memory.get('final_score', 0) * 10  # Effective memories score higher
                return recency_score + effectiveness_score
            
            memories.sort(key=priority_score, reverse=True)
            
            # Return top 10 most relevant memories
            return memories[:10]
            
        except Exception as e:
            print(f"   ⚠️ Error loading memories: {e}")
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
                Path("continuous_learning_data"),
                Path("meta_learning_data"),
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
                                print(f"   🗑️ Deleted {memory_file.name} (age: {file_age_days:.1f}d, size: {file_size_kb:.1f}KB)")
                            elif is_protected:
                                print(f"   🛡️ Protected {memory_file.name} from deletion (winning memory)")
                                
                        except Exception as e:
                            print(f"   ⚠️ Failed to process {memory_file}: {e}")
            
            return deleted_count
            
        except Exception as e:
            print(f"   ⚠️ Error during garbage collection: {e}")
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
            memory_dir = Path("continuous_learning_data")
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
                    print(f"   🔗 Combined {len(memories)} memories into {combined_file.name}")
            
            return combined_count
            
        except Exception as e:
            print(f"   ⚠️ Error combining memories: {e}")
            return 0

    async def _strengthen_action_memory_with_context(self, action: Dict[str, Any], context_memories: List[Dict[str, Any]]) -> None:
        """Strengthen memory of an effective action with contextual information."""
        try:
            import time
            
            action_type = action.get('action_type', 'unknown')
            effectiveness = action.get('effectiveness', 0)
            score = action.get('score_achieved', 0)
            
            print(f"   💪 Strengthening: {action_type} (effectiveness: {effectiveness:.2f}, score: {score})")
            
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
            print(f"   ⚠️ Error strengthening memory: {e}")

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
            print(f"   ⚠️ Error generating insights: {e}")
        
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
            print(f"   ⚠️ Error preparing guidance: {e}")
        
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
    
    def _preserve_winning_memories(self, session_result: Dict[str, Any], score: int, game_id: str, is_level_progression: bool = False):
        """
        Preserve memories that led to wins or level progression.
        These memories get super-high salience and are protected from deletion.
        """
        try:
            if not hasattr(self.agent, 'memory'):
                return
                
            # Determine preservation strength based on achievement type
            if is_level_progression:
                preservation_strength = 0.95  # Nearly permanent
                min_salience_floor = 0.8  # Very high minimum
                protection_duration = 1000  # Much longer protection
                print(f"🏆 CRITICAL MEMORY PRESERVATION: Level progression memories (strength: {preservation_strength})")
            elif score >= 4:  # High score achievement
                preservation_strength = 0.85
                min_salience_floor = 0.6
                protection_duration = 500
                print(f"🏆 HIGH-VALUE MEMORY PRESERVATION: Score {score} memories (strength: {preservation_strength})")
            elif score >= 1:  # Any positive score
                preservation_strength = 0.75
                min_salience_floor = 0.4
                protection_duration = 200
                print(f"🏆 WINNING MEMORY PRESERVATION: Score {score} memories (strength: {preservation_strength})")
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
                if hasattr(self.agent.memory, 'update_memory_salience'):
                    # Create high-salience preservation
                    memory_idx = torch.tensor([i % self.agent.memory.memory_size])
                    salience_val = torch.tensor([preservation_strength])
                    self.agent.memory.update_memory_salience(memory_idx, salience_val)
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
            
            print(f"✨ Protected {winning_memory_count} winning memories from deletion (expires in {protection_duration}s)")
            
        except Exception as e:
            logger.warning(f"Failed to preserve winning memories: {e}")

    def _preserve_breakthrough_memories(self, session_result: Dict[str, Any], score: int, game_id: str, new_level: int, previous_level: int):
        """
        Progressive memory hierarchy system - only preserves TRUE breakthroughs with escalating priority.
        Higher levels get stronger protection and demote previous level memories.
        """
        try:
            if not hasattr(self.agent, 'memory'):
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
            
            print(f"🏆 LEVEL {new_level} BREAKTHROUGH! Tier {tier} Protection (strength: {preservation_strength:.2f}, floor: {min_salience_floor:.1f})")
            
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
                
                if hasattr(self.agent.memory, 'update_memory_salience'):
                    # Apply tier-based salience boost
                    memory_idx = torch.tensor([i % self.agent.memory.memory_size])
                    salience_val = torch.tensor([preservation_strength])
                    self.agent.memory.update_memory_salience(memory_idx, salience_val)
                    breakthrough_memory_count += 1
                    breakthrough_memory_ids.append(memory_id)
                    
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
            
            print(f"🎯 Preserved {breakthrough_memory_count} Tier {tier} breakthrough memories (Level {new_level})")
            
        except Exception as e:
            print(f"⚠️ Error in breakthrough memory preservation: {e}")

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
                print(f"📉 Demoted {demoted_count} Level {previous_level} memories (still protected but lower priority)")
                
        except Exception as e:
            print(f"⚠️ Error in memory demotion: {e}")
    
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
                print(f"\n🏛️ MEMORY HIERARCHY STATUS ({active_count} protected memories):")
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
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error getting protection floor: {e}")
            return 0.1

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
            print(f"\n🔍 API INVESTIGATION for {game_id}:")
            print(f"   📡 Session Data Keys: {list(session_data.keys())}")
            print(f"   🎮 Game State: {session_data.get('state', 'UNKNOWN')}")
            print(f"   🎯 Available Actions: {session_data.get('available_actions', [])}")
            print(f"   🏁 Score: {session_data.get('score', 0)}")
            print(f"   � GUID: {guid}")
            
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
            
            print(f"   ✅ Investigation successful - ready for direct control")
            return investigation_result
                        
        except Exception as e:
            error_msg = f"API investigation failed: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {"error": error_msg}

    async def start_training_with_direct_control(
        self, 
        game_id: str,
        max_actions: int = 500,
        session_count: int = 0
    ) -> Dict[str, Any]:
        """Run training session with direct API action control instead of external main.py."""
        print(f"\n🎯 STARTING DIRECT CONTROL TRAINING for {game_id}")
        print(f"   Max Actions: {max_actions}, Session: {session_count}")
        
        try:
            # First investigate API to understand available actions
            investigation = await self.investigate_api_available_actions(game_id)
            if "error" in investigation:
                return {"error": f"API investigation failed: {investigation['error']}", "actions_taken": 0}
            
            # Start game session
            session_data = await self._start_game_session(game_id)
            if not session_data:
                return {"error": "Failed to start game session", "actions_taken": 0}
            
            guid = session_data.get('guid')
            current_state = investigation.get('state', 'NOT_STARTED')
            current_score = investigation.get('score', 0)
            available_actions = investigation.get('available_actions', [1,2,3,4,5,6,7])  # Default fallback
            
            print(f"🚀 SESSION STARTED:")
            print(f"   GUID: {guid}")
            print(f"   Initial State: {current_state}")
            print(f"   Initial Score: {current_score}")
            print(f"   Available Actions: {available_actions}")
            
            # Direct action control loop
            actions_taken = 0
            effective_actions = []
            action_history = []
            
            while (actions_taken < max_actions and 
                   current_state not in ['WIN', 'GAME_OVER'] and
                   current_score < 100):  # Reasonable win condition
                
                # Use our intelligent action selection
                selected_action = self._select_intelligent_action(available_actions, {'game_id': game_id})
                
                # Optimize coordinates if needed (for ACTION6)
                x, y = None, None
                if selected_action == 6:
                    x, y = self._optimize_coordinates_for_action(selected_action, (64, 64))
                
                print(f"🎯 SELECTED ACTION: {selected_action}" + (f" at ({x},{y})" if x is not None else ""))
                
                # Show intelligent action description
                action_desc = self.get_action_description(selected_action, game_id)
                if "Learned:" in action_desc or "Role in" in action_desc or "effect:" in action_desc:
                    print(f"   💡 {action_desc}")
                elif self.available_actions_memory['action_learning_stats']['total_observations'] > 0:
                    # Show basic learned info even if not confident
                    mapping = self.available_actions_memory['action_semantic_mapping'].get(selected_action, {})
                    total_attempts = self.available_actions_memory['action_effectiveness'].get(selected_action, {}).get('attempts', 0)
                    if total_attempts > 0:
                        success_rate = self.available_actions_memory['action_effectiveness'][selected_action]['success_rate']
                        print(f"   📊 Success rate: {success_rate:.1%} ({total_attempts} attempts)")
                
                # Execute the action
                action_result = await self._send_enhanced_action(
                    game_id, selected_action, x, y, 64, 64
                )
                
                if action_result:
                    # Update state from response
                    new_state = action_result.get('state', current_state)
                    new_score = action_result.get('score', current_score)
                    new_available = action_result.get('actions', available_actions)
                    
                    # Track effectiveness
                    score_improvement = new_score - current_score
                    was_effective = (score_improvement > 0 or new_state in ['WIN'])
                    
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
                    
                    # Update current state
                    current_state = new_state
                    current_score = new_score
                    available_actions = new_available
                    
                else:
                    # Record failed action
                    action_history.append({
                        'action': selected_action,
                        'coordinates': (x, y) if x is not None else None,
                        'before_score': current_score,
                        'after_score': current_score,
                        'effective': False,
                        'state_change': 'FAILED',
                        'error': True
                    })
                
                actions_taken += 1
                
                # Rate-limit compliant delay between actions
                # With 8 RPS limit, we need at least 0.125s between requests
                # Use 0.15s for safety margin (6.67 RPS actual rate)
                await asyncio.sleep(0.15)
            
            # Session complete
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
                    'MAX_ACTIONS' if actions_taken >= max_actions else
                    'HIGH_SCORE' if current_score >= 100 else
                    'UNKNOWN'
                )
            }
            
            print(f"\n🏁 DIRECT CONTROL SESSION COMPLETE:")
            print(f"   Final Score: {current_score}")
            print(f"   Final State: {current_state}")
            print(f"   Actions Taken: {actions_taken}")
            print(f"   Effective Actions: {len(effective_actions)}")
            print(f"   Termination: {final_result['termination_reason']}")
            
            return final_result
            
        except Exception as e:
            print(f"   ⚠️ Error in direct control training: {e}")
            return {"error": str(e), "actions_taken": 0}

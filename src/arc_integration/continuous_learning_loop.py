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
from typing import Dict, List, Any, Optional, Tuple  # Added Tuple for grid dimensions
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor  # Added for SWARM mode

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
    """Represents a training session configuration."""
    session_id: str
    games_to_play: List[str]
    max_episodes_per_game: int
    learning_rate_schedule: Dict[str, float]
    save_interval: int
    target_performance: Dict[str, float]
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
        
        logger.info("Continuous Learning Loop initialized with ARC-3 API integration")

    async def get_available_games(self) -> List[Dict[str, str]]:
        """
        Get list of available games from ARC-AGI-3 API.
        
        Returns:
            List of game metadata with game_id and title
        """
        url = "https://three.arcprize.org/api/games"
        headers = {"X-API-Key": self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        games = await response.json()
                        logger.info(f"Retrieved {len(games)} available games from ARC-AGI-3 API")
                        return games
                    elif response.status == 401:
                        raise ValueError("Invalid API key - check ARC_API_KEY in .env file")
                    else:
                        raise ValueError(f"API request failed with status {response.status}")
                        
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
        Create a real scorecard using the ARC-AGI-3 API.
        
        Args:
            games_list: List of game_ids to include in scorecard
            
        Returns:
            scorecard_id if successful, None if failed
        """
        url = "https://three.arcprize.org/api/scorecards"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "title": f"Tabula-Rasa Continuous Training {int(time.time())}",
            "games": games_list
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 201:
                        result = await response.json()
                        scorecard_id = result.get('scorecard_id')
                        scorecard_url = f"https://three.arcprize.org/scorecard/{scorecard_id}"
                        
                        logger.info(f"Created real scorecard: {scorecard_url}")
                        self.current_scorecard_id = scorecard_id
                        return scorecard_id
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to create scorecard: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error creating scorecard: {e}")
            return None
        
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
        result = {'success': False, 'final_score': 0, 'actions_taken': 0}
        combined_output = stdout + "\n" + stderr
        
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
        """Select appropriate action based on available_actions from API response."""
        available = response_data.get('available_actions', [])
        
        if not available:
            logger.warning(f"No available actions for game {game_id}")
            return None
        
        # Simple strategy: prefer ACTION6 if available, otherwise random selection
        if 6 in available:
            return 6  # Coordinate-based action
        elif len(available) > 0:
            return random.choice(available)  # Random from available actions
        
        return None
    
    async def _send_enhanced_action(
        self,
        game_id: str,
        action_number: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        grid_width: int = 64,
        grid_height: int = 64
    ) -> Optional[Dict[str, Any]]:
        """Send action with proper validation and grid bounds checking."""
        guid = self.current_game_sessions.get(game_id)
        if not guid:
            logger.warning(f"No active session for game {game_id}")
            return None
        
        # Validate action number
        if action_number not in [1, 2, 3, 4, 5, 6, 7]:
            logger.error(f"Invalid action number: {action_number}")
            return None
        
        # Special handling for ACTION6 (coordinate-based)
        if action_number == 6:
            if x is None or y is None:
                logger.error("ACTION6 requires x,y coordinates")
                return None
            
            # Verify coordinates are within actual grid bounds
            if not self._verify_grid_bounds(x, y, grid_width, grid_height):
                logger.error(f"Coordinates ({x},{y}) out of bounds for grid {grid_width}x{grid_height}")
                return None
        
        try:
            async with aiohttp.ClientSession() as session:
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
                
                # Add coordinates for ACTION6
                if action_number == 6:
                    payload["x"] = x
                    payload["y"] = y
                
                # Add reasoning for better tracking
                payload["reasoning"] = {
                    "action_type": f"ACTION{action_number}",
                    "grid_size": f"{grid_width}x{grid_height}",
                    "coordinates": f"({x},{y})" if action_number == 6 else "N/A",
                    "timestamp": time.time()
                }
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Action {action_number} successful for {game_id}")
                        return data
                    else:
                        error_text = await response.text()
                        logger.warning(f"Action {action_number} failed for {game_id}: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error sending action {action_number} to {game_id}: {e}")
            return None

    async def _train_on_game(
        self,
        game_id: str,
        max_episodes: int,
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
        
        episode_count = 1
        consecutive_failures = 0
        best_score = 0
        
        while episode_count < max_episodes:
            try:
                print(f"Episode {episode_count + 1}/{max_episodes} for {game_id}")
                
                # Run ENHANCED ARC-3 episode with proper state checking
                episode_result = await self._run_real_arc_episode_enhanced(game_id, episode_count)
                
                if episode_result and 'error' not in episode_result:
                    game_results['episodes'].append(episode_result)
                    episode_count += 1
                    
                    # Update grid dimensions if detected
                    if 'grid_dimensions' in episode_result:
                        game_results['grid_dimensions'] = episode_result['grid_dimensions']
                        print(f"Grid size detected: {episode_result['grid_dimensions'][0]}x{episode_result['grid_dimensions'][1]}")
                    
                    # Track performance
                    current_score = episode_result.get('final_score', 0)
                    success = episode_result.get('success', False)
                    game_state = episode_result.get('game_state', 'NOT_FINISHED')
                    scorecard_url = episode_result.get('scorecard_url')
                    
                    if scorecard_url:
                        game_results['scorecard_urls'].append(scorecard_url)
                    
                    if success:
                        consecutive_failures = 0
                        if current_score > best_score:
                            best_score = current_score
                            print(f"New best score for {game_id}: {best_score}")
                    else:
                        consecutive_failures += 1
                        
                    print(f"Episode {episode_count}: {'WIN' if success else 'LOSS'} | Score: {current_score} | State: {game_state}")
                    
                    # CRITICAL: Stop if game reached terminal state (WIN or GAME_OVER)
                    if game_state in ['WIN', 'GAME_OVER']:
                        print(f"🎯 Game {game_id} reached terminal state: {game_state}")
                        break
                    
                    # Check if we should continue based on performance
                    if self._should_stop_training(game_results, target_performance):
                        print(f"Target performance reached for {game_id}")
                        break
                        
                    # Brief delay between episodes to avoid rate limiting
                    await asyncio.sleep(2.0)
                        
                else:
                    print(f"Episode {episode_count + 1} failed: {episode_result.get('error', 'Unknown error')}")
                    consecutive_failures += 1
                    
                    # Stop if too many consecutive API failures
                    if consecutive_failures >= 5:
                        print(f"Stopping training for {game_id} after 5 consecutive API failures")
                        break
                    
                    await asyncio.sleep(5.0)  # Longer delay after failures
                    
            except Exception as e:
                logger.error(f"Error in episode {episode_count + 1} for {game_id}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print(f"Stopping training for {game_id} due to repeated errors")
                    break
                await asyncio.sleep(5.0)
                
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

    async def _run_real_arc_episode_enhanced(self, game_id: str, episode_count: int) -> Dict[str, Any]:
        """Enhanced version that runs COMPLETE episodes with multiple actions until WIN/GAME_OVER."""
        try:
            # Check if agent should sleep before episode
            agent_state = self._get_current_agent_state()
            should_sleep_now = self._should_agent_sleep(agent_state, episode_count)
            
            sleep_cycle_results = {}
            if should_sleep_now:
                sleep_cycle_results = await self._execute_sleep_cycle(game_id, episode_count)
            
            # Check if model decides to reset the game
            reset_decision = self._evaluate_game_reset_decision(game_id, episode_count, agent_state)
            
            # Set up environment with API key
            env = os.environ.copy()
            env['ARC_API_KEY'] = self.api_key
            
            # COMPLETE EPISODE: Run actions in loop until terminal state
            episode_actions = 0
            total_score = 0
            best_score = 0
            final_state = 'NOT_FINISHED'
            episode_start_time = time.time()
            max_actions_per_episode = 1000  # Allow for complex puzzles
            
            print(f"🎮 Starting complete episode {episode_count} for {game_id}")
            
            # VERBOSE: Show memory state before episode
            print(f"📊 PRE-EPISODE MEMORY STATUS:")
            pre_memory_status = self._get_memory_consolidation_status()
            pre_sleep_status = self._get_current_sleep_state_info()
            
            # SHOW GLOBAL CUMULATIVE COUNTERS - THE FIX!
            print(f"   Memory Operations: {self.global_counters.get('total_memory_operations', 0)}")
            print(f"   Sleep Cycles: {self.global_counters.get('total_sleep_cycles', 0)}")
            print(f"   Energy Level: {pre_sleep_status.get('current_energy_level', 1.0):.2f}")
            
            # VERBOSE: Check existing memory files
            memory_files_before = self._count_memory_files()
            print(f"   Memory Files: {memory_files_before}")
            
            # Adaptive energy allocation based on game history
            current_energy = pre_sleep_status.get('current_energy_level', 1.0)
            estimated_complexity = self._estimate_game_complexity(game_id)
            
            # Boost energy if we predict a complex game
            if estimated_complexity == 'high' and current_energy < 0.8:
                energy_boost = 0.2
                current_energy = min(1.0, current_energy + energy_boost)
                print(f"⚡ Energy boost: +{energy_boost:.2f} for predicted high-complexity game -> {current_energy:.2f}")
                self._update_energy_level(current_energy)
            elif estimated_complexity == 'medium' and current_energy < 0.6:
                energy_boost = 0.1
                current_energy = min(1.0, current_energy + energy_boost)
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
            
            # Add tags to identify this episode
            cmd.extend(['--tags', f'Attempt#{episode_count}'])
            
            # Run the complete game session until WIN/GAME_OVER
            try:
                print(f"🚀 Executing complete game session: {' '.join(cmd)}")
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.arc_agents_path),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for complete game session with longer timeout (games can take thousands of actions)
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), 
                        timeout=600.0  # 10 minutes for complete game session
                    )
                    
                    stdout_text = stdout.decode('utf-8', errors='ignore') if stdout else ""
                    stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""
                    
                    print(f"✅ Complete game session finished")
                    
                    # Parse complete game results
                    game_results = self._parse_complete_game_session(stdout_text, stderr_text)
                    total_score = game_results.get('final_score', 0)
                    episode_actions = game_results.get('total_actions', 0) 
                    final_state = game_results.get('final_state', 'UNKNOWN')
                    effective_actions = game_results.get('effective_actions', [])
                    
                    print(f"🎯 Game Results: Score={total_score}, Actions={episode_actions}, State={final_state}")
                    print(f"🎯 Effective Actions Found: {len(effective_actions)}")
                    
                except asyncio.TimeoutError:
                    print(f"⏰ Complete game session timed out after 10 minutes - killing process")
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
                    stdout_text = ""
                    stderr_text = ""
                    
            except Exception as e:
                print(f"❌ Error during complete game session: {e}")
                total_score = 0
                episode_actions = 0
                final_state = 'ERROR'
                effective_actions = []
                stdout_text = ""
                stderr_text = ""
            
            # Now process the complete game session results
            # Dynamic energy system based on game complexity and learning opportunities
            
            # Calculate adaptive energy cost based on actions and effectiveness
            base_energy_cost = episode_actions * 0.005  # Base rate: 0.005 per action
            
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
            current_energy = pre_sleep_status.get('current_energy_level', 1.0)
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
                base_replenishment = 0.6  # Base 60% restoration
                
                # Bonus energy for effective learning
                if len(effective_actions) > 0:
                    learning_bonus = min(0.3, len(effective_actions) * 0.05)  # Up to 30% bonus
                    print(f"⚡ Learning bonus: +{learning_bonus:.2f} energy for {len(effective_actions)} effective actions")
                else:
                    learning_bonus = 0.0
                
                # Bonus energy for complex games (they teach more)
                if episode_actions > 500:
                    complexity_bonus = 0.2  # 20% bonus for complex games
                    print(f"⚡ Complexity bonus: +{complexity_bonus:.2f} energy for {episode_actions}-action game")
                elif episode_actions > 200:
                    complexity_bonus = 0.1  # 10% bonus for medium games
                    print(f"⚡ Complexity bonus: +{complexity_bonus:.2f} energy for {episode_actions}-action game")
                else:
                    complexity_bonus = 0.0
                
                total_replenishment = base_replenishment + learning_bonus + complexity_bonus
                remaining_energy = min(1.0, remaining_energy + total_replenishment)
                print(f"⚡ Energy replenished: {total_replenishment:.2f} total -> {remaining_energy:.2f}")
            else:
                print(f"⚡ Sleep not needed: Energy {remaining_energy:.2f} > threshold {sleep_threshold:.2f}")
            
            # Update energy level in system
            self._update_energy_level(remaining_energy)
            
            # Update game complexity history for future energy allocation
            effectiveness_ratio = len(effective_actions) / max(1, episode_actions)  # Avoid division by zero
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
                'episode': episode_count,
                'actions_taken': episode_actions,
                'episode_duration': episode_duration,
                'timestamp': time.time(),
                'sleep_cycle_executed': bool(sleep_cycle_results),
                'sleep_cycle_results': sleep_cycle_results,
                'reset_decision': reset_decision,
                'memory_consolidation_status': self._get_memory_consolidation_status(),
                'sleep_state_info': self._get_current_sleep_state_info()
            }
            
            if grid_dims:
                result['grid_dimensions'] = grid_dims
            
            # Evaluate reset effectiveness if reset was used
            if reset_decision['should_reset']:
                reset_effectiveness = self._evaluate_reset_effectiveness(reset_decision, result)
                result['reset_effectiveness'] = reset_effectiveness
                
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
                episode_result = await self._run_real_arc_episode_enhanced(game_id, episode_count)
                
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
                
                # Shorter delay for swarm mode
                await asyncio.sleep(1.0)
            
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
                    max_episodes_per_game=session.max_episodes_per_game
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
                        session.max_episodes_per_game,
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
            'average_score': sum(ep.get('final_score', 0) for ep in episodes) / len(episodes),
            'average_actions': sum(ep.get('actions_taken', 0) for ep in episodes) / len(episodes),
            'best_score': max(ep.get('final_score', 0) for ep in episodes),
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
        total_score = sum(sum(ep.get('final_score', 0) for ep in game.get('episodes', [])) 
                         for game in games_played.values())
        
        return {
            'games_trained': len(games_played),
            'total_episodes': total_episodes,
            'overall_win_rate': total_wins / max(1, total_episodes),
            'overall_average_score': total_score / max(1, total_episodes),
            'learning_efficiency': self._calculate_learning_efficiency(session_results),
            'knowledge_transfer_score': self._calculate_knowledge_transfer_score(session_results)
        }
        
    def _calculate_learning_efficiency(self, session_results: Dict[str, Any]) -> float:
        """Calculate how efficiently the agent learned during the session."""
        # Simplified efficiency metric based on improvement over time
        games_played = session_results['games_played']
        efficiency_scores = []
        
        for game_results in games_played.values():
            episodes = game_results.get('episodes', [])
            if len(episodes) >= 10:
                early_performance = sum(ep.get('final_score', 0) for ep in episodes[:5]) / 5
                late_performance = sum(ep.get('final_score', 0) for ep in episodes[-5:]) / 5
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
        
        early_avg = sum(game.get('performance_metrics', {}).get('average_score', 0) for game in early_games) / len(early_games)
        late_avg = sum(game.get('performance_metrics', {}).get('average_score', 0) for game in late_games) / len(late_games)
        
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
        overall_perf = session_results['overall_performance']
        
        # Update cumulative metrics
        self.global_performance_metrics['total_games_played'] += overall_perf.get('games_trained', 0)
        self.global_performance_metrics['total_episodes'] += overall_perf.get('total_episodes', 0)
        
        # Update averages (simplified)
        self.global_performance_metrics['average_score'] = (
            self.global_performance_metrics['average_score'] * 0.8 + 
            overall_perf.get('overall_average_score', 0) * 0.2
        )
        
        self.global_performance_metrics['win_rate'] = (
            self.global_performance_metrics['win_rate'] * 0.8 + 
            overall_perf.get('overall_win_rate', 0) * 0.2
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

    def _load_state(self):
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

    def _should_stop_training(self, game_results: Dict[str, Any], target_performance: Dict[str, float]) -> bool:
        """Check if we should stop training on this game based on target performance."""
        episodes = game_results.get('episodes', [])
        if len(episodes) < 5:  # Need at least 5 episodes to evaluate
            return False
        
        # Calculate current performance
        recent_episodes = episodes[-5:]  # Look at last 5 episodes
        recent_wins = sum(1 for ep in recent_episodes if ep.get('success', False))
        recent_win_rate = recent_wins / len(recent_episodes)
        recent_avg_score = sum(ep.get('final_score', 0) for ep in recent_episodes) / len(recent_episodes)
        
        # Check if we've reached target performance
        target_win_rate = target_performance.get('win_rate', 0.3)
        target_avg_score = target_performance.get('avg_score', 50.0)
        
        if recent_win_rate >= target_win_rate and recent_avg_score >= target_avg_score:
            return True
            
        return False

    def start_training_session(
        self,
        games: List[str],
        max_episodes_per_game: int = 50,
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
            max_episodes_per_game: Maximum episodes per game
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
            max_episodes_per_game=max_episodes_per_game,
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
                
                # Decay low-salience memories
                all_experiences = self.salience_calculator.get_high_salience_experiences(threshold=0.0)
                low_salience_experiences = [exp for exp in all_experiences if exp['salience'] < 0.3]
                low_salience_decayed = len(low_salience_experiences)
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
        avg_score = sum(exp.get('final_score', 0) for exp in group) / len(group)
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
        
        result = {
            'final_score': 0,
            'total_actions': 0,
            'final_state': 'UNKNOWN',
            'effective_actions': []
        }
        
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
            
            # Extract effective actions by looking for score changes
            effective_actions = []
            
            # Look for ACTION patterns with score improvements
            action_lines = re.findall(r'ACTION\d+:.*?score\s+(\d+)', stdout_text, re.IGNORECASE)
            for i, score_str in enumerate(action_lines):
                score = int(score_str)
                if score > 0:  # This action had a positive effect
                    effective_actions.append({
                        'action_number': i + 1,
                        'action_type': f'ACTION{(i % 8) + 1}',  # ARC has 8 action types
                        'score_achieved': score,
                        'effectiveness': min(1.0, score / 100.0)  # Normalize effectiveness
                    })
            
            # Also look for successful state changes (RESET -> ACTION sequences)
            reset_action_sequences = re.findall(r'RESET.*?ACTION\d+.*?score\s+(\d+)', stdout_text, re.IGNORECASE | re.DOTALL)
            for score_str in reset_action_sequences:
                score = int(score_str)
                if score > 0:
                    effective_actions.append({
                        'action_type': 'RESET_SEQUENCE',
                        'score_achieved': score,
                        'effectiveness': min(1.0, score / 50.0)  # Reset sequences are valuable
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
            'energy_restored': 0.7,
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
                if effectiveness > 0.3:  # Lower threshold to capture more learnings
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
        """Update the system's energy level."""
        self.current_energy = max(0.0, min(1.0, new_energy))
        
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
        
        history = self.game_complexity_history[game_id]
        history['total_plays'] += 1
        history['total_actions'] += actions_taken
        history['avg_actions'] = history['total_actions'] / history['total_plays']
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
                            
                            # Delete if: very old (>7 days) OR very small (<1KB) OR temp files
                            should_delete = (
                                file_age_days > 7 or
                                file_size_kb < 1 or
                                'temp_' in memory_file.name or
                                'test_' in memory_file.name
                            )
                            
                            if should_delete:
                                os.remove(memory_file)
                                deleted_count += 1
                                print(f"   🗑️ Deleted {memory_file.name} (age: {file_age_days:.1f}d, size: {file_size_kb:.1f}KB)")
                                
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
                            'avg_score': sum(m['data'].get('final_score', 0) for m in memories) / len(memories),
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
            'total_memories_strengthened': 0
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
            
        # Auto-save counters
        self._save_global_counters()

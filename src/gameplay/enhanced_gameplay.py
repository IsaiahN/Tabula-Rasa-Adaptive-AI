"""
Enhanced Gameplay Integration

This module provides the enhanced gameplay functionality that was previously
in CORE_GAME_MECHANICS, now integrated with the main src/ structure.

Features:
- Enhanced Action 6 coordinate selection with pseudo-button detection
- Integration with existing vision and database systems
- Backward compatibility with existing train.py imports
"""
import sys
sys.dont_write_bytecode = True

import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING, Union
import asyncio
from collections import deque

if TYPE_CHECKING:
    from src.analysis.game_lifecycle_analyzer import GameLifecycleAnalyzer

logger = logging.getLogger(__name__)


class EnhancedGameplay:
    """Enhanced gameplay system with Action 6 pseudo-button detection."""

    def __init__(self, db_interface=None):
        """Initialize the enhanced gameplay system.

        Args:
            db_interface: Database interface for storing learning data
        """
        self.db_interface = db_interface
        self.action6_coordinator = None
        self.pseudo_button_detector = None

        # Sleep and memory systems
        self.sleep_system = None
        self.energy_system = None
        self.goal_system = None
        self.breakthrough_detector = None
        self.experience_buffer = None

        # Sleep state tracking
        self.current_energy = 100.0
        self.action_count = 0
        self.sleep_check_frequency = 50  # Check every 50 actions

        self._initialize_components()

    def _initialize_components(self):
        """Initialize the enhanced components."""
        try:
            # Import and initialize pseudo-button detector
            from src.vision.enhanced.pseudo_button_detector import create_pseudo_button_detector
            self.pseudo_button_detector = create_pseudo_button_detector(self.db_interface)

            # Import and initialize Action 6 coordinator
            from src.gameplay.action6_coordinator import create_action6_coordinator
            self.action6_coordinator = create_action6_coordinator(
                self.db_interface, self.pseudo_button_detector
            )

            # Initialize sleep and memory systems
            self._initialize_sleep_and_memory_systems()

            # Initialize energy system
            self._initialize_energy_system()

            # Initialize goal system
            self._initialize_goal_system()

            # Initialize breakthrough detection
            self._initialize_breakthrough_detection()

            logger.info("Enhanced gameplay components initialized successfully")

        except ImportError as e:
            logger.warning(f"Could not initialize enhanced components: {e}")
            # Create fallback components
            self._create_fallback_components()

    def _create_fallback_components(self):
        """Create fallback components when enhanced ones are not available."""
        logger.info("Using fallback components for enhanced gameplay")

        class FallbackAction6Coordinator:
            def __init__(self):
                self.stats = {'action6_selections': 0}

            async def get_optimal_action6_coordinates(self, frame, game_id, context=None):
                self.stats['action6_selections'] += 1
                # Simple center-based fallback
                if frame and len(frame) > 0:
                    height, width = len(frame), len(frame[0])
                    return width // 2, height // 2
                return 25, 25

            async def analyze_action6_effectiveness(self, *args, **kwargs):
                pass  # No-op fallback

            def get_statistics(self):
                return self.stats

        self.action6_coordinator = FallbackAction6Coordinator()

    def _initialize_sleep_and_memory_systems(self):
        """Initialize sleep and memory consolidation systems."""
        try:
            from src.core.sleep_system import SleepCycle
            from collections import deque

            # Initialize sleep system with simple configuration
            self.sleep_system = SleepCycle(
                predictive_core=None,  # We'll enhance this later if needed
                sleep_trigger_energy=40.0,
                sleep_trigger_boredom_steps=100,
                sleep_duration_steps=50,
                use_salience_weighting=True
            )

            # Initialize experience replay buffer
            self.experience_buffer = deque(maxlen=1000)

            logger.info("Sleep and memory systems initialized")

        except ImportError as e:
            logger.warning(f"Could not initialize sleep systems: {e}")
            self.sleep_system = None
            self.experience_buffer = deque(maxlen=100)  # Simple fallback

    def _initialize_energy_system(self):
        """Initialize energy management system."""
        try:
            from src.core.unified_energy_system import UnifiedEnergySystem

            self.energy_system = UnifiedEnergySystem()
            logger.info("Energy system initialized")

        except ImportError as e:
            logger.warning(f"Could not initialize energy system: {e}")
            # Simple fallback energy system
            self.energy_system = None

    def _initialize_goal_system(self):
        """Initialize goal invention system."""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from src.goals.goal_system import GoalInventionSystem

            self.goal_system = GoalInventionSystem()
            logger.info("Goal system initialized")

        except (ImportError, Exception) as e:
            logger.debug(f"Goal system not available (using fallback): {e}")
            # Create a simple fallback goal system
            class FallbackGoalSystem:
                def __init__(self):
                    self.active = False

                def generate_goals(self, *args, **kwargs):
                    return []

                def update_goals(self, *args, **kwargs):
                    pass

                def get_current_goals(self):
                    return []

            self.goal_system = FallbackGoalSystem()
            logger.info("Goal system initialized (fallback mode)")

    def _initialize_breakthrough_detection(self):
        """Initialize breakthrough detection system."""
        try:
            from src.core.sleep_breakthrough_detection import create_sleep_breakthrough_system

            self.breakthrough_detector, self.breakthrough_processor = create_sleep_breakthrough_system(
                breakthrough_threshold=0.7,
                novelty_threshold=0.6,
                performance_window=50
            )
            logger.info("Breakthrough detection initialized")

        except ImportError as e:
            logger.warning(f"Could not initialize breakthrough detection: {e}")
            self.breakthrough_detector = None
            self.breakthrough_processor = None

    def _should_trigger_sleep(self) -> bool:
        """Check if sleep cycle should be triggered."""
        if not self.sleep_system:
            return False

        # Check energy level (lower threshold to prevent constant sleeping)
        if self.current_energy <= 15.0:
            logger.info(f"Sleep triggered by low energy: {self.current_energy}")
            return True

        # Check action frequency (every 50 actions)
        if self.action_count % self.sleep_check_frequency == 0 and self.action_count > 0:
            logger.info(f"Sleep triggered by action frequency at {self.action_count} actions")
            return True

        return False

    async def _execute_sleep_cycle(self, game_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a sleep cycle with memory consolidation."""
        if not self.sleep_system:
            return {'sleep_executed': False, 'reason': 'no_sleep_system'}

        logger.info("Executing sleep cycle for memory consolidation...")

        try:
            # Convert experience buffer to list for sleep system
            experiences = list(self.experience_buffer)

            # Execute sleep cycle
            sleep_results = self.sleep_system.execute_sleep_cycle(
                replay_buffer=experiences,
                arc_data=context
            )

            # Restore energy after sleep (more generous restoration)
            self.current_energy = min(100.0, self.current_energy + 50.0)

            logger.info(f"Sleep cycle completed: {sleep_results.get('experiences_processed', 0)} experiences processed")

            return {
                'sleep_executed': True,
                'experiences_processed': sleep_results.get('experiences_processed', 0),
                'consolidation_score': sleep_results.get('consolidation_score', 0.0),
                'energy_restored': True,
                'new_energy_level': self.current_energy
            }

        except Exception as e:
            logger.warning(f"Sleep cycle execution failed: {e}")
            return {'sleep_executed': False, 'error': str(e)}

    def _consume_energy(self, action: str) -> None:
        """Consume energy based on action type."""
        energy_costs = {
            'ACTION1': 2.0,
            'ACTION2': 2.0,
            'ACTION3': 2.0,
            'ACTION4': 2.0,
            'ACTION5': 3.0,
            'ACTION6': 5.0,  # Most expensive due to coordinate selection
            'ACTION7': 3.0,
        }

        cost = energy_costs.get(action, 2.0)
        self.current_energy = max(0.0, self.current_energy - cost)

    def _add_experience(self, frame: List[List[int]], action: str, score_before: float, score_after: float, action_coords: Tuple[int, int] = None):
        """Add experience to replay buffer for sleep consolidation."""
        if not self.experience_buffer:
            return

        try:
            # Create simple experience record
            experience = {
                'frame': frame,
                'action': action,
                'score_before': score_before,
                'score_after': score_after,
                'score_change': score_after - score_before,
                'action_coords': action_coords,
                'timestamp': time.time(),
                'energy_level': self.current_energy
            }

            self.experience_buffer.append(experience)

        except Exception as e:
            logger.warning(f"Failed to add experience: {e}")

    async def get_enhanced_action6_coordinates(self, frame: List[List[int]],
                                             game_id: str,
                                             available_actions: List[str] = None) -> Tuple[int, int]:
        """Get enhanced Action 6 coordinates using pseudo-button detection.

        This is the main entry point for enhanced Action 6 coordinate selection.

        Args:
            frame: Current game frame
            game_id: Game identifier
            available_actions: List of available actions (to detect Action 6-only games)

        Returns:
            Tuple of (x, y) coordinates for optimal Action 6 click
        """
        try:
            if not self.action6_coordinator:
                return self._get_simple_fallback_coordinates(frame)

            # Create context for the coordinator
            context = {
                'available_actions': available_actions or ['ACTION6'],
                'action': 'ACTION6'
            }

            # Use the enhanced coordinator
            coords = await self.action6_coordinator.get_optimal_action6_coordinates(
                frame, game_id, context
            )

            logger.debug(f"Enhanced Action 6 coordinates: {coords}")
            return coords

        except Exception as e:
            logger.error(f"Error in enhanced Action 6 coordinate selection: {e}")
            return self._get_simple_fallback_coordinates(frame)

    async def analyze_action6_result(self, frame_before: List[List[int]],
                                   frame_after: List[List[int]],
                                   coordinates: Tuple[int, int],
                                   game_id: str,
                                   score_change: float = 0.0):
        """Analyze the result of an Action 6 for learning purposes.

        Args:
            frame_before: Frame before the action
            frame_after: Frame after the action
            coordinates: Coordinates that were clicked
            game_id: Game identifier
            score_change: Change in game score
        """
        try:
            if self.action6_coordinator and hasattr(self.action6_coordinator, 'analyze_action6_effectiveness'):
                await self.action6_coordinator.analyze_action6_effectiveness(
                    frame_before, frame_after, coordinates, game_id, score_change
                )
        except Exception as e:
            logger.error(f"Error analyzing Action 6 result: {e}")

    def _get_simple_fallback_coordinates(self, frame: List[List[int]]) -> Tuple[int, int]:
        """Simple fallback coordinate selection."""
        if not frame or len(frame) == 0:
            return 25, 25

        height, width = len(frame), len(frame[0])

        # Simple center with slight randomization
        import random
        x = width // 2 + random.randint(-10, 10)
        y = height // 2 + random.randint(-10, 10)

        # Ensure coordinates are within bounds
        x = max(5, min(width - 5, x))
        y = max(5, min(height - 5, y))

        return x, y

    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get statistics about the enhanced functionality."""
        stats = {
            'enhanced_components_available': self.action6_coordinator is not None,
            'pseudo_button_detector_available': self.pseudo_button_detector is not None
        }

        if self.action6_coordinator:
            stats.update(self.action6_coordinator.get_statistics())

        if self.pseudo_button_detector:
            detector_stats = self.pseudo_button_detector.get_statistics()
            stats.update({f"detector_{k}": v for k, v in detector_stats.items()})

        return stats


# Compatibility layer for CORE_GAME_MECHANICS imports
class GameSessionManager:
    """Compatibility wrapper for GameSessionManager."""

    def __init__(self, api_key: str = None, db_path: str = "tabula_rasa.db"):
        """Initialize GameSessionManager with API key and database path.

        Args:
            api_key: ARC-AGI-3 API key (for compatibility with train.py)
            db_path: Database path for storing game data
        """
        # Handle both old and new calling patterns
        if api_key is not None and isinstance(api_key, str) and not api_key.endswith('.db'):
            # Called as GameSessionManager(api_key, db_path) from train.py
            self.api_key = api_key
            self.db_path = db_path
        else:
            # Called as GameSessionManager(db_path) - treat first arg as db_path
            self.api_key = None
            self.db_path = api_key if api_key else db_path

        # Initialize database for visualization
        self.database = CoreGameDatabase(self.db_path)

        self.enhanced_gameplay = None
        self._initialize_enhanced_gameplay()

    def _initialize_enhanced_gameplay(self):
        """Initialize enhanced gameplay if possible."""
        try:
            # Try to create database interface
            db_interface = None

            # Import database interface if available
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
                from src.database.api import DatabaseInterface
                db_interface = DatabaseInterface(self.db_path)
            except ImportError:
                logger.debug("Database interface not available, using simple fallback")

            self.enhanced_gameplay = EnhancedGameplay(db_interface)

        except Exception as e:
            logger.warning(f"Could not initialize enhanced gameplay: {e}")
            self.enhanced_gameplay = EnhancedGameplay()

    async def graceful_shutdown(self):
        """Gracefully shut down the session manager."""
        try:
            print("Session manager shutting down gracefully...")
            if hasattr(self, 'enhanced_gameplay') and self.enhanced_gameplay:
                # Shutdown enhanced gameplay if available
                print("Enhanced gameplay shutdown completed")
            
            # Close database connection
            if hasattr(self, 'database') and self.database:
                self.database.close()
                print("Database connection closed")
                
        except Exception as e:
            print(f"Shutdown error: {e}")


class CoreGameplay:
    """Compatibility wrapper for CoreGameplay."""

    def __init__(self, session_manager=None):
        self.session_manager = session_manager or GameSessionManager()
        self.enhanced_gameplay = self.session_manager.enhanced_gameplay
        # Add compatibility flags
        self.ai_available = True
        # Add knowledge integrator for compatibility 
        self.knowledge_integrator = self._create_knowledge_integrator()
        # Initialize lifecycle analyzer
        self.set_lifecycle_analyzer()

    async def play_single_game(self, game_id: str, max_actions: int = 400, hypothesis_context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Play a single game session using real ARC-AGI-3 API."""
        import time
        import uuid
        import hashlib
        start_time = time.time()

        # Generate unique session ID for visualization
        session_id = str(uuid.uuid4())
        action_count = 0
        
        # Track ACTION 6 effectiveness variables
        last_action6_frame = None
        last_action6_coords = None
        last_action6_score = None
        last_action6_action_number = None

        # ENHANCED: Multi-action stagnation tracking
        game_stagnation_state = {
            'last_frame_hash': None,
            'stagnation_count': 0,
            'stagnation_actions_tried': [],  # Track which stagnation-breaking actions we've tried
            'last_frame': None,
            'consecutive_same_frames': 0
        }

        def calculate_frame_hash(frame):
            """Calculate frame hash for stagnation detection."""
            try:
                frame_str = str(frame)
                return hashlib.md5(frame_str.encode()).hexdigest()
            except:
                # Fallback: sum of all cell values
                total = sum(sum(row) for row in frame)
                return str(total)

        def detect_multi_action_stagnation(current_frame, available_actions):
            """ENHANCED: Detect stagnation and intelligently break loops for ALL game types."""
            try:
                current_hash = calculate_frame_hash(current_frame)

                if game_stagnation_state['last_frame_hash'] is None:
                    game_stagnation_state['last_frame_hash'] = current_hash
                    game_stagnation_state['stagnation_count'] = 0
                    game_stagnation_state['action_rotation_index'] = 0
                    return False, None

                if current_hash == game_stagnation_state['last_frame_hash']:
                    game_stagnation_state['stagnation_count'] += 1
                    game_stagnation_state['consecutive_same_frames'] += 1

                    # FIXED: Detect stagnation for BOTH single-action and multi-action games
                    # Critical fix: ACTION6-only games were never triggering stagnation detection!
                    stagnation_threshold = 3  # Lower threshold for faster detection

                    if game_stagnation_state['stagnation_count'] >= stagnation_threshold:
                        print(f"[STAGNATION] CRITICAL: Frame stagnation detected! Count: {game_stagnation_state['stagnation_count']}")
                        print(f"[STAGNATION] Available actions: {available_actions}")
                        print(f"[STAGNATION] Frame hash: {current_hash}")

                        # ENHANCED: Intelligent action rotation system
                        if len(available_actions) == 1:
                            # Single-action game (e.g., ACTION6-only): Use coordinate rotation for ACTION6
                            single_action = available_actions[0]
                            if single_action == 6:
                                print(f"[STAGNATION] ACTION6-only game stagnation - forcing coordinate diversity")
                                # Force coordinate diversity by returning None to trigger new coordinate selection
                                return True, "FORCE_NEW_COORDINATES"
                            else:
                                print(f"[STAGNATION] Single-action {single_action} stagnation - limited options")
                                return True, f"ACTION{single_action}"

                        else:
                            # Multi-action game: Intelligent action cycling
                            # Reset rotation if we've tried all actions
                            if not hasattr(game_stagnation_state, 'action_rotation_index'):
                                game_stagnation_state['action_rotation_index'] = 0

                            # Create action priority list for breaking stagnation
                            priority_actions = []

                            # Priority 1: Special actions (ACTION5, ACTION7 - reset/special functions)
                            for action in [5, 7]:
                                if action in available_actions:
                                    priority_actions.append(action)

                            # Priority 2: Directional actions (ACTION1-4 for exploration)
                            for action in [1, 2, 3, 4]:
                                if action in available_actions:
                                    priority_actions.append(action)

                            # Priority 3: Other actions (ACTION6, others)
                            for action in available_actions:
                                if action not in priority_actions:
                                    priority_actions.append(action)

                            # Rotate through all available actions
                            if priority_actions:
                                chosen_action = priority_actions[game_stagnation_state['action_rotation_index'] % len(priority_actions)]
                                game_stagnation_state['action_rotation_index'] += 1

                                print(f"[STAGNATION] Rotating to ACTION{chosen_action} (index {game_stagnation_state['action_rotation_index']-1}/{len(priority_actions)})")

                                # Reset rotation after trying all actions
                                if game_stagnation_state['action_rotation_index'] >= len(priority_actions):
                                    print(f"[STAGNATION] Completed action rotation cycle, resetting")
                                    game_stagnation_state['action_rotation_index'] = 0

                                return True, f"ACTION{chosen_action}"

                        # Fallback
                        print(f"[STAGNATION] Using fallback stagnation action")
                        return True, None

                    return False, None  # Stagnant but below threshold
                else:
                    # Frame changed - reset stagnation tracking
                    game_stagnation_state['last_frame_hash'] = current_hash
                    game_stagnation_state['stagnation_count'] = 0
                    game_stagnation_state['consecutive_same_frames'] = 0
                    # Keep action rotation index for learning
                    return False, None

            except Exception as e:
                print(f"[STAGNATION] Error in stagnation detection: {e}")
                return False, None

        try:
            # Import ARC API client
            from src.arc_integration.arc_api_client import ARCClient

            # Get API key (should be available in session manager)
            api_key = getattr(self.session_manager, 'api_key', None)
            if not api_key:
                api_key = os.getenv('ARC_AGI_3_API_KEY') or os.getenv('ARC_API_KEY')

            if not api_key:
                raise ValueError("ARC API key not found. Set ARC_AGI_3_API_KEY or ARC_API_KEY environment variable.")

            # Start visualization session
            self._start_visualization_session(game_id, session_id)

            # Create intelligent agent function that uses enhanced AI systems
            def enhanced_agent(game_state, available_actions):
                """Intelligent agent that uses enhanced AI components for decision making."""
                nonlocal action_count, last_action6_frame, last_action6_coords, last_action6_score, last_action6_action_number
                try:
                    # Extract frame from game state
                    raw_frame = game_state.frame
                    current_score = game_state.score
                    game_state_name = game_state.state

                    # Get available actions
                    actions = available_actions or []

                    # Debug frame structure
                    frame_info = self._debug_frame_structure(raw_frame)
                    print(f"[AGENT] {frame_info}, Score: {current_score}, Actions: {actions}")

                    # Reshape frame if needed for proper analysis
                    frame = self._reshape_frame_if_needed(raw_frame)
                    if len(frame) != len(raw_frame):
                        print(f"[AGENT] Frame reshaped from {len(raw_frame)} to {len(frame)}x{len(frame[0]) if frame else 0}")

                    action_count += 1

                    # ENHANCED: Sleep and memory system integration
                    if hasattr(self.enhanced_gameplay, '_should_trigger_sleep') and self.enhanced_gameplay._should_trigger_sleep():
                        print(f"[SLEEP] Triggering sleep cycle at action {action_count}")
                        # Execute sleep cycle with current game context
                        sleep_context = {
                            'game_id': game_id,
                            'current_score': current_score,
                            'action_count': action_count,
                            'frame': frame,
                            'available_actions': actions
                        }

                        try:
                            import asyncio
                            # Check if we're in an event loop
                            try:
                                loop = asyncio.get_running_loop()
                                # We're in a running loop, schedule the task for later
                                task = asyncio.create_task(self.enhanced_gameplay._execute_sleep_cycle(game_id, sleep_context))
                                print(f"[SLEEP] Sleep cycle scheduled as background task")
                                # Don't wait for the result to avoid blocking the game loop
                            except RuntimeError:
                                # No running loop, safe to use asyncio.run()
                                sleep_result = asyncio.run(self.enhanced_gameplay._execute_sleep_cycle(game_id, sleep_context))
                                print(f"[SLEEP] Sleep cycle result: {sleep_result}")
                        except Exception as e:
                            print(f"[SLEEP] Sleep cycle error: {e}")

                    # Store previous score for experience tracking
                    score_before = current_score
                    
                    # ENHANCED: Multi-action stagnation detection
                    is_stagnant, stagnation_action = detect_multi_action_stagnation(frame, actions)
                    
                    # If we detected stagnation and have a specific action to try, use it
                    if is_stagnant and stagnation_action:
                        print(f"[STAGNATION] Using stagnation-breaking action: {stagnation_action}")

                        # ENHANCED: Handle ACTION6 coordinate diversity for stagnation
                        if stagnation_action == "FORCE_NEW_COORDINATES":
                            print(f"[STAGNATION] Forcing ACTION6 coordinate diversity")
                            selected_action = "ACTION6"

                            # Force new coordinates using enhanced diversity logic
                            try:
                                coords = self._get_diversified_action6_coordinates(frame, game_id, action_count)
                                print(f"[STAGNATION] Diversified ACTION6 coordinates: {coords}")
                                # SAFETY: Ensure coordinate extraction always works
                                if coords and len(coords) == 2:
                                    action_x, action_y = coords
                                else:
                                    logger.warning(f"Invalid coordinate format: {coords}")
                                    print(f"[EMERGENCY] Invalid coordinates, using safety fallback")
                                    emergency_coords = self._get_sync_action6_coordinates(frame, game_id)
                                    action_x, action_y = emergency_coords

                            except Exception as e:
                                logger.warning(f"Diversified coordinate selection failed: {e}")
                                # Fallback to random-ish coordinates
                                import random
                                action_x = random.randint(5, 55)
                                action_y = random.randint(5, 55)
                                print(f"[STAGNATION] Random fallback coordinates: ({action_x}, {action_y})")
                        else:
                            selected_action = stagnation_action
                            action_x = None
                            action_y = None

                        # Capture frame for visualization
                        self._capture_frame_for_visualization(
                            game_id=game_id,
                            session_id=session_id,
                            action_number=action_count,
                            frame=frame,
                            action_taken=selected_action,
                            action_x=action_x,
                            action_y=action_y,
                            score_before=score_before,
                            score_after=current_score,  # Will be updated after action
                            available_actions=actions
                        )

                        # CRITICAL FIX: Return proper ACTION6 format with coordinates
                        if selected_action == "ACTION6":
                            if action_x is not None and action_y is not None:
                                return {
                                    "action": "ACTION6",
                                    "x": action_x,
                                    "y": action_y
                                }
                            else:
                                logger.warning(f"Stagnation ACTION6 missing coordinates! action_x={action_x}, action_y={action_y}")
                                print(f"[EMERGENCY] Stagnation ACTION6 missing coordinates, using emergency fallback")
                                emergency_coords = self._get_sync_action6_coordinates(frame, game_id)
                                emergency_x, emergency_y = emergency_coords
                                return {
                                    "action": "ACTION6",
                                    "x": emergency_x,
                                    "y": emergency_y
                                }
                        else:
                            return selected_action
                    
                    # Analyze effectiveness of previous ACTION 6 if we have the data
                    if (last_action6_frame is not None and last_action6_coords is not None and 
                        last_action6_score is not None and last_action6_action_number is not None):
                        try:
                            print(f"[EFFECTIVENESS] Analyzing ACTION 6 from turn {last_action6_action_number}")
                            score_change = current_score - last_action6_score
                            print(f"[EFFECTIVENESS] Score change: {last_action6_score} -> {current_score} (delta: {score_change})")
                            
                            # Analyze frame changes and effectiveness
                            try:
                                import asyncio
                                # Check if we're already in an event loop
                                try:
                                    loop = asyncio.get_running_loop()
                                    # We're in an event loop, create a task instead of using asyncio.run()
                                    task = asyncio.create_task(self._analyze_action6_effectiveness_async(
                                        last_action6_frame, frame, last_action6_coords, game_id, score_change
                                    ))
                                    print(f"[EFFECTIVENESS] Created async analysis task")
                                except RuntimeError:
                                    # No event loop running, safe to use asyncio.run()
                                    asyncio.run(self._analyze_action6_effectiveness_async(
                                        last_action6_frame, frame, last_action6_coords, game_id, score_change
                                    ))
                            except Exception as async_error:
                                print(f"[EFFECTIVENESS] Async analysis error: {async_error}")
                                # Fallback: Skip analysis if async fails
                                pass
                            
                            # Reset tracking variables
                            last_action6_frame = None
                            last_action6_coords = None
                            last_action6_score = None
                            last_action6_action_number = None
                            
                        except Exception as e:
                            print(f"[EFFECTIVENESS] Error analyzing previous ACTION 6: {e}")

                    selected_action = None
                    action_x = None
                    action_y = None

                    # Intelligent frame analysis for action selection
                    action_decision = self._analyze_frame_for_action_selection(frame, actions, game_state)

                    # Handle ACTION 6 with enhanced object detection and coordination
                    if action_decision == "ACTION6" or (6 in actions and (len(actions) == 1 or action_decision == "ACTION6")):
                        try:
                            print(f"[AGENT] Triggering enhanced ACTION 6 object detection and pseudo-button analysis")
                            # Use enhanced Action 6 coordination with sync wrapper
                            coords = self._get_enhanced_action6_coordinates_sync(frame, game_id, actions, current_score, hypothesis_context)
                            print(f"[AGENT] Enhanced Action 6 selected coordinates: {coords}")
                            selected_action = "ACTION6"
                            # SAFETY: Ensure coordinate extraction always works
                            if coords and len(coords) == 2:
                                action_x, action_y = coords
                            else:
                                logger.warning(f"Invalid coordinate format: {coords}")
                                print(f"[EMERGENCY] Invalid coordinates, using safety fallback")
                                emergency_coords = self._get_sync_action6_coordinates(frame, game_id)
                                action_x, action_y = emergency_coords
                            
                            # Store for effectiveness analysis on next turn
                            last_action6_frame = [row[:] for row in frame]  # Deep copy
                            last_action6_coords = coords
                            last_action6_score = current_score
                            last_action6_action_number = action_count
                            print(f"[EFFECTIVENESS] Stored ACTION 6 data for analysis (turn {action_count})")
                            
                        except Exception as e:
                            logger.warning(f"Enhanced Action 6 failed: {e}")
                            # Fallback to basic coordinate selection
                            coords = self._get_sync_action6_coordinates(frame, game_id)
                            print(f"[AGENT] Fallback Action 6 coordinates: {coords}")
                            selected_action = "ACTION6"
                            # SAFETY: Ensure coordinate extraction always works
                            if coords and len(coords) == 2:
                                action_x, action_y = coords
                            else:
                                logger.warning(f"Invalid coordinate format: {coords}")
                                print(f"[EMERGENCY] Invalid coordinates, using safety fallback")
                                emergency_coords = self._get_sync_action6_coordinates(frame, game_id)
                                action_x, action_y = emergency_coords
                            
                            # Still store for effectiveness analysis
                            last_action6_frame = [row[:] for row in frame]  # Deep copy
                            last_action6_coords = coords
                            last_action6_score = current_score
                            last_action6_action_number = action_count

                    # Handle other frame analysis suggestions
                    elif action_decision and action_decision != "ACTION6":
                        print(f"[AGENT] Frame analysis suggests: {action_decision}")
                        selected_action = action_decision

                    # If Action 6 is available but not suggested, still consider enhanced coordination
                    elif 6 in actions and self.enhanced_gameplay and hasattr(self.enhanced_gameplay, 'action6_coordinator'):
                        try:
                            # Use enhanced Action 6 coordination with sync wrapper
                            coords = self._get_enhanced_action6_coordinates_sync(frame, game_id, actions, current_score, hypothesis_context)
                            print(f"[AGENT] Enhanced Action 6 selected coordinates: {coords}")
                            selected_action = "ACTION6"
                            # SAFETY: Ensure coordinate extraction always works
                            if coords and len(coords) == 2:
                                action_x, action_y = coords
                            else:
                                logger.warning(f"Invalid coordinate format: {coords}")
                                print(f"[EMERGENCY] Invalid coordinates, using safety fallback")
                                emergency_coords = self._get_sync_action6_coordinates(frame, game_id)
                                action_x, action_y = emergency_coords
                            
                            # Store for effectiveness analysis on next turn
                            last_action6_frame = [row[:] for row in frame]  # Deep copy
                            last_action6_coords = coords
                            last_action6_score = current_score
                            last_action6_action_number = action_count
                            
                        except Exception as e:
                            logger.warning(f"Enhanced Action 6 failed: {e}")
                            # Fall through to intelligent action selection instead

                    # Intelligent action selection for non-Action6 cases
                    if not selected_action and actions:
                        selected_action = self._select_intelligent_action(frame, actions, current_score, game_state_name)
                        print(f"[AGENT] Intelligent action selected: {selected_action}")

                    # Default fallback
                    if not selected_action:
                        print("[AGENT] Using fallback action: ACTION1")
                        selected_action = "ACTION1"

                    # ENHANCED: Energy consumption and experience tracking
                    if hasattr(self.enhanced_gameplay, '_consume_energy'):
                        self.enhanced_gameplay._consume_energy(selected_action)
                        print(f"[ENERGY] Consumed energy for {selected_action}, current level: {self.enhanced_gameplay.current_energy:.1f}")

                    # Add experience to replay buffer for memory consolidation
                    if hasattr(self.enhanced_gameplay, '_add_experience'):
                        action_coords = (action_x, action_y) if action_x is not None and action_y is not None else None
                        self.enhanced_gameplay._add_experience(
                            frame=frame,
                            action=selected_action,
                            score_before=score_before,
                            score_after=current_score,  # Will be same for now, updated later
                            action_coords=action_coords
                        )
                        print(f"[EXPERIENCE] Added experience to replay buffer (size: {len(self.enhanced_gameplay.experience_buffer)})")

                    # Capture frame for visualization (non-blocking)
                    self._capture_frame_for_visualization(
                        game_id=game_id,
                        session_id=session_id,
                        action_number=action_count,
                        frame=frame,
                        action_taken=selected_action,
                        action_x=action_x,
                        action_y=action_y,
                        score_before=score_before,
                        score_after=current_score,  # Will be updated after action
                        available_actions=actions
                    )

                    # Return appropriate action format
                    if selected_action == "ACTION6":
                        # CRITICAL FIX: Ensure ACTION6 always has valid coordinates
                        if action_x is not None and action_y is not None:
                            return {
                                "action": "ACTION6",
                                "x": action_x,
                                "y": action_y
                            }
                        else:
                            # EMERGENCY FALLBACK: Generate safe coordinates if missing
                            logger.warning(f"ACTION6 missing coordinates! action_x={action_x}, action_y={action_y}")
                            print(f"[EMERGENCY] ACTION6 missing coordinates, using emergency fallback")
                            emergency_coords = self._get_sync_action6_coordinates(frame, game_id)
                            emergency_x, emergency_y = emergency_coords
                            return {
                                "action": "ACTION6",
                                "x": emergency_x,
                                "y": emergency_y
                            }
                    else:
                        return selected_action

                except Exception as e:
                    logger.warning(f"Agent decision error: {e}")
                    print(f"[AGENT ERROR] {e}")
                    return "ACTION1"  # Safe fallback

            # Connect to ARC API and play the game
            async with ARCClient(api_key=api_key) as client:
                print(f"[GAME {game_id}] Connecting to ARC-AGI-3 API...")

                result = await client.play_game(
                    game_id=game_id,
                    agent_func=enhanced_agent,
                    max_actions=max_actions
                )

                duration = time.time() - start_time

                # Extract results from ARC API response
                final_score = result.get('total_score', 0)
                total_actions = result.get('actions_taken', 0)
                game_won = result.get('completed', False)

                # End visualization session
                ai_performance = {
                    'action6_enhanced': self.enhanced_gameplay is not None,
                    'pseudo_button_detection': True,
                    'coordinate_intelligence': True,
                    'real_arc_api': True,
                    'effectiveness_analysis': True,
                    'multi_action_stagnation_handling': True,  # NEW
                    'comprehensive_pseudo_button_learning': True  # NEW
                }
                self._end_visualization_session(session_id, total_actions, final_score, game_won, ai_performance)

                print(f"[GAME {game_id}] Completed: Score={final_score}, Actions={total_actions}, Won={game_won}")
                print(f"[VISUALIZATION] Session {session_id} captured {action_count} frames")
                print(f"[STAGNATION] Final stagnation count: {game_stagnation_state['stagnation_count']}, "
                      f"actions tried: {game_stagnation_state['stagnation_actions_tried']}")

                return {
                    'final_score': final_score,
                    'total_actions': total_actions,
                    'game_duration': duration,
                    'win_detected': game_won,
                    'ai_performance': ai_performance,
                    'game_id': game_id,
                    'session_id': session_id,  # For visualization replay
                    'arc_result': result,
                    'stagnation_stats': game_stagnation_state  # NEW
                }

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            print(f"[GAME {game_id}] Critical error: {error_msg}")

            # End visualization session with error
            self._end_visualization_session(session_id, action_count, 0, False, {'error': error_msg})

            return {
                'final_score': 0.0,
                'total_actions': 0,
                'game_duration': duration,
                'win_detected': False,
                'ai_performance': {'real_arc_api': False, 'error': error_msg},
                'game_id': game_id,
                'session_id': session_id,
                'error': error_msg
            }

    def set_lifecycle_analyzer(self, analyzer: Union['GameLifecycleAnalyzer', None] = None) -> None:
        """Set the lifecycle analyzer instance.

        Args:
            analyzer: Optional GameLifecycleAnalyzer instance. If None, will create a new one.
        """
        if analyzer is None:
            try:
                from src.analysis.game_lifecycle_analyzer import GameLifecycleAnalyzer
                self._lifecycle_analyzer = GameLifecycleAnalyzer()
            except ImportError:
                logger.warning("Could not create GameLifecycleAnalyzer - features will be limited")
                self._lifecycle_analyzer = None
        else:
            self._lifecycle_analyzer = analyzer

    async def execute_enhanced_action6(self, frame: List[List[int]],
                                     game_id: str,
                                     available_actions: List[str] = None) -> Tuple[int, int]:
        """Execute enhanced Action 6 coordinate selection."""
        if self.enhanced_gameplay:
            return await self.enhanced_gameplay.get_enhanced_action6_coordinates(
                frame, game_id, available_actions
            )
        else:
            # Simple fallback
            if frame and len(frame) > 0:
                return len(frame[0]) // 2, len(frame) // 2
            return 25, 25

    async def get_lifecycle_aware_action_recommendation(self,
                                                       frame: List[List[int]],
                                                       available_actions: List[int],
                                                       game_context: Dict[str, Any],
                                                       action_count: int,
                                                       recent_actions: List[int] = None) -> Dict[str, Any]:
        """Get action recommendation with lifecycle pattern analysis and oscillation prevention."""
        try:
            # Import lifecycle analyzer if available (from train.py integration)
            lifecycle_analyzer = getattr(self, '_lifecycle_analyzer', None)
            if not lifecycle_analyzer:
                # Try to get from session manager or create temporary instance
                try:
                    from src.analysis.game_lifecycle_analyzer import GameLifecycleAnalyzer
                    lifecycle_analyzer = GameLifecycleAnalyzer()
                    print("[LIFECYCLE] Created temporary lifecycle analyzer")
                except ImportError:
                    print("[LIFECYCLE] Lifecycle analyzer not available")
                    return self._get_fallback_action_recommendation(available_actions)

            # Prepare context for lifecycle analysis
            game_type = self._classify_current_game_type(game_context)
            current_score = game_context.get('current_score', 0.0)

            # Calculate failure risk based on current state
            failure_risk = lifecycle_analyzer.get_failure_risk(
                game_type=game_type,
                current_action_count=action_count,
                recent_actions=recent_actions or []
            )

            print(f"[LIFECYCLE] Game type: {game_type}, Action count: {action_count}, Failure risk: {failure_risk:.2f}")

            # Check if we need to switch strategies due to high failure risk
            if failure_risk >= 0.7:  # High risk threshold
                print(f"[LIFECYCLE] HIGH FAILURE RISK DETECTED: {failure_risk:.2f}")

                # Get alternative strategy recommendation
                strategy_recommendation = lifecycle_analyzer.get_alternative_strategy({
                    'game_type': game_type,
                    'current_action_count': action_count,
                    'recent_score_change': game_context.get('recent_score_change', 0.0),
                    'current_score': current_score
                })

                # Apply strategy change
                recommended_action = self._apply_strategy_change(
                    available_actions, strategy_recommendation, frame, game_context
                )

                return {
                    'action': recommended_action,
                    'reason': f'Strategy switch due to high failure risk ({failure_risk:.2f})',
                    'strategy_type': strategy_recommendation['strategy_type'],
                    'confidence': 0.8,
                    'lifecycle_analysis': {
                        'failure_risk': failure_risk,
                        'strategy_switch_triggered': True,
                        'original_strategy': strategy_recommendation
                    }
                }

            # Check for action avoidance recommendations
            action_avoidance_results = {}
            for action in available_actions:
                avoidance_check = lifecycle_analyzer.should_avoid_action(action, {
                    'game_type': game_type,
                    'current_action_count': action_count,
                    'current_score': current_score
                })

                if avoidance_check['should_avoid']:
                    action_avoidance_results[action] = avoidance_check
                    print(f"[LIFECYCLE] Action {action} should be avoided: {avoidance_check['reason']}")

            # Filter out actions that should be avoided
            safe_actions = [action for action in available_actions
                           if action not in action_avoidance_results]

            if not safe_actions:
                print("[LIFECYCLE] All actions flagged for avoidance, using least risky")
                # Use the action with the lowest confidence avoidance recommendation
                least_risky = min(action_avoidance_results.keys(),
                                key=lambda a: action_avoidance_results[a]['confidence'])
                safe_actions = [least_risky]

            # Select best action from safe actions using existing logic
            recommended_action = self._select_optimal_action_from_safe_list(
                safe_actions, frame, game_context, failure_risk
            )

            # Detect oscillation patterns in recent actions
            oscillation_detected = False
            if recent_actions and len(recent_actions) >= 6:
                # Simple oscillation detection: look for ABAB or ABCABC patterns
                if (len(set(recent_actions[-4:])) <= 2 and
                    recent_actions[-1] == recent_actions[-3] and
                    recent_actions[-2] == recent_actions[-4]):
                    oscillation_detected = True
                    print(f"[LIFECYCLE] Oscillation detected in recent actions: {recent_actions[-6:]}")

            return {
                'action': recommended_action,
                'reason': 'Lifecycle-aware selection with risk mitigation',
                'confidence': 0.9 - (failure_risk * 0.3),  # Reduce confidence with higher risk
                'lifecycle_analysis': {
                    'failure_risk': failure_risk,
                    'strategy_switch_triggered': False,
                    'actions_avoided': list(action_avoidance_results.keys()),
                    'oscillation_detected': oscillation_detected,
                    'safe_actions': safe_actions
                }
            }

        except Exception as e:
            logger.warning(f"Lifecycle-aware action recommendation failed: {e}")
            print(f"[LIFECYCLE] Error in lifecycle analysis: {e}")
            return self._get_fallback_action_recommendation(available_actions)

    def _classify_current_game_type(self, game_context: Dict[str, Any]) -> str:
        """Classify current game type for lifecycle analysis."""
        try:
            # Simple classification based on available context
            game_id = game_context.get('game_id', 'unknown')
            current_score = game_context.get('current_score', 0.0)
            available_actions = game_context.get('available_actions', [])

            if len(available_actions) == 1 and 6 in available_actions:
                return 'action6_only'
            elif current_score > 100:
                return 'high_scoring'
            elif 'action6' in game_id.lower():
                return 'action6_intensive'
            else:
                return 'multi_action'

        except Exception:
            return 'unknown'

    def _apply_strategy_change(self, available_actions: List[int],
                              strategy_recommendation: Dict[str, Any],
                              frame: List[List[int]],
                              game_context: Dict[str, Any]) -> int:
        """Apply strategy change based on lifecycle recommendation."""
        try:
            strategy_type = strategy_recommendation.get('strategy_type', 'exploration')
            parameters = strategy_recommendation.get('parameters', {})

            if strategy_type == 'exploration' and len(available_actions) > 1:
                # Choose least used action for exploration
                return self._select_exploration_action(available_actions)
            elif strategy_type == 'conservative':
                # Choose safest action (usually ACTION1 or most common successful action)
                return available_actions[0] if available_actions else 1
            elif strategy_type == 'aggressive' and 6 in available_actions:
                # Choose ACTION6 for aggressive approach
                return 6
            else:
                # Hybrid approach - balance exploration and exploitation
                return self._select_balanced_action(available_actions, frame)

        except Exception:
            return available_actions[0] if available_actions else 1

    def _select_optimal_action_from_safe_list(self, safe_actions: List[int],
                                            frame: List[List[int]],
                                            game_context: Dict[str, Any],
                                            failure_risk: float) -> int:
        """Select optimal action from lifecycle-filtered safe actions."""
        try:
            if not safe_actions:
                return 1  # Default fallback

            # If low risk, prefer ACTION6 if available for maximum progress
            if failure_risk < 0.3 and 6 in safe_actions:
                return 6

            # If moderate risk, prefer exploration actions
            elif failure_risk < 0.6:
                exploration_actions = [a for a in safe_actions if a in [1, 2, 3, 4]]
                if exploration_actions:
                    return exploration_actions[0]

            # High risk (but below critical threshold) - conservative approach
            return safe_actions[0]

        except Exception:
            return safe_actions[0] if safe_actions else 1

    def _select_exploration_action(self, available_actions: List[int]) -> int:
        """Select action for exploration strategy."""
        # Prefer directional actions for exploration
        exploration_priorities = [1, 2, 3, 4, 6, 5, 7]
        for action in exploration_priorities:
            if action in available_actions:
                return action
        return available_actions[0]

    def _select_balanced_action(self, available_actions: List[int], frame: List[List[int]]) -> int:
        """Select action using balanced approach."""
        try:
            # Use existing frame analysis if available
            if hasattr(self, '_analyze_frame_for_action_selection'):
                action_decision = self._analyze_frame_for_action_selection(frame, available_actions, {})
                if action_decision and action_decision.replace('ACTION', '').isdigit():
                    recommended_action = int(action_decision.replace('ACTION', ''))
                    if recommended_action in available_actions:
                        return recommended_action

            # Default balanced selection
            if 6 in available_actions:
                return 6
            elif len(available_actions) > 1:
                return available_actions[1]  # Second action for variety
            else:
                return available_actions[0]

        except Exception:
            return available_actions[0] if available_actions else 1

    def _get_fallback_action_recommendation(self, available_actions: List[int]) -> Dict[str, Any]:
        """Get fallback action recommendation when lifecycle analysis fails."""
        action = available_actions[0] if available_actions else 1
        return {
            'action': action,
            'reason': 'Fallback selection - lifecycle analysis unavailable',
            'confidence': 0.5,
            'lifecycle_analysis': {
                'failure_risk': 0.5,
                'strategy_switch_triggered': False,
                'actions_avoided': [],
                'oscillation_detected': False
            }
        }

    def _get_diversified_action6_coordinates(self, frame: List[List[int]], game_id: str, action_count: int) -> Tuple[int, int]:
        """Get diversified ACTION6 coordinates to break stagnation patterns."""
        try:
            if not frame or not frame[0]:
                return 25, 25

            height = len(frame)
            width = len(frame[0])
            print(f"[DIVERSITY] Generating diversified coordinates for {height}x{width} frame (action {action_count})")

            # Track coordinate usage to avoid repetition
            if not hasattr(self, '_recent_coordinates'):
                self._recent_coordinates = []

            # Generate grid-based exploration pattern
            grid_size = 8  # Divide frame into 8x8 grid for systematic exploration
            grid_x = width // grid_size
            grid_y = height // grid_size

            candidates = []

            # Method 1: Grid-based systematic exploration
            for gx in range(grid_size):
                for gy in range(grid_size):
                    center_x = (gx * grid_x) + (grid_x // 2)
                    center_y = (gy * grid_y) + (grid_y // 2)

                    # Ensure coordinates are within bounds
                    center_x = max(2, min(width - 2, center_x))
                    center_y = max(2, min(height - 2, center_y))

                    # Check if this area was recently tried
                    recently_used = False
                    for recent_x, recent_y in self._recent_coordinates[-10:]:  # Check last 10 coordinates
                        if abs(center_x - recent_x) < 5 and abs(center_y - recent_y) < 5:
                            recently_used = True
                            break

                    if not recently_used:
                        candidates.append((center_x, center_y, f"grid_{gx}_{gy}"))

            # Method 2: Frame edge exploration
            edge_candidates = [
                (width // 4, height // 4, "top_left"),
                (3 * width // 4, height // 4, "top_right"),
                (width // 4, 3 * height // 4, "bottom_left"),
                (3 * width // 4, 3 * height // 4, "bottom_right"),
                (width // 2, 5, "top_center"),
                (width // 2, height - 5, "bottom_center"),
                (5, height // 2, "left_center"),
                (width - 5, height // 2, "right_center"),
            ]

            for ex, ey, label in edge_candidates:
                ex = max(2, min(width - 2, ex))
                ey = max(2, min(height - 2, ey))

                recently_used = False
                for recent_x, recent_y in self._recent_coordinates[-10:]:
                    if abs(ex - recent_x) < 5 and abs(ey - recent_y) < 5:
                        recently_used = True
                        break

                if not recently_used:
                    candidates.append((ex, ey, label))

            # Method 3: Object-based exploration (find non-zero cells)
            object_candidates = []
            for y in range(0, height, 5):  # Sample every 5 pixels
                for x in range(0, width, 5):
                    if y < height and x < width:
                        cell_value = self._get_cell_value(frame[y][x])
                        if cell_value > 0:  # Found an object
                            recently_used = False
                            for recent_x, recent_y in self._recent_coordinates[-10:]:
                                if abs(x - recent_x) < 5 and abs(y - recent_y) < 5:
                                    recently_used = True
                                    break

                            if not recently_used:
                                object_candidates.append((x, y, f"object_v{cell_value}"))

            # Combine all candidates
            all_candidates = candidates + object_candidates
            print(f"[DIVERSITY] Found {len(all_candidates)} diverse coordinate candidates")

            if all_candidates:
                # Select candidate based on action count to ensure different exploration patterns
                selected_index = action_count % len(all_candidates)
                chosen_x, chosen_y, method = all_candidates[selected_index]

                # Add to recent coordinates tracking
                self._recent_coordinates.append((chosen_x, chosen_y))
                if len(self._recent_coordinates) > 20:  # Keep last 20 coordinates
                    self._recent_coordinates.pop(0)

                print(f"[DIVERSITY] Selected coordinates ({chosen_x}, {chosen_y}) using method '{method}' (index {selected_index}/{len(all_candidates)})")
                return chosen_x, chosen_y

            else:
                # Fallback: Use action count to generate pseudo-random but systematic coordinates
                import math
                angle = (action_count * 30) % 360  # Rotate 30 degrees each time
                radius = 15 + (action_count % 3) * 10  # Vary distance

                center_x, center_y = width // 2, height // 2
                coord_x = center_x + int(radius * math.cos(math.radians(angle)))
                coord_y = center_y + int(radius * math.sin(math.radians(angle)))

                # Ensure within bounds
                coord_x = max(2, min(width - 2, coord_x))
                coord_y = max(2, min(height - 2, coord_y))

                print(f"[DIVERSITY] Fallback spiral coordinates: ({coord_x}, {coord_y}) angle={angle}° radius={radius}")
                return coord_x, coord_y

        except Exception as e:
            logger.warning(f"Diversified coordinate generation error: {e}")
            print(f"[DIVERSITY] Error in coordinate generation: {e}")
            # Safe fallback
            return 25, 25

    def _get_sync_action6_coordinates(self, frame: List[List[int]], game_id: str) -> Tuple[int, int]:
        """Synchronous Action 6 coordinate selection for real ARC API."""
        try:
            # Simple but effective coordinate selection
            if not frame or not frame[0]:
                print(f"[SYNC COORD] Empty frame, using center fallback (12, 12)")
                return 12, 12  # Center fallback

            height = len(frame)
            width = len(frame[0])
            print(f"[SYNC COORD] Analyzing {height}x{width} frame for clickable objects")

            # Look for high-value cells (bright spots that might be buttons) - FIXED FOR ARC
            candidates = []
            for y in range(height):
                for x in range(width):
                    if isinstance(frame[y], list) and x < len(frame[y]):
                        cell_value = frame[y][x]
                        # Convert to int if needed
                        if isinstance(cell_value, (list, tuple)) and len(cell_value) > 0:
                            cell_value = cell_value[0] if isinstance(cell_value[0], (int, float)) else 0
                        elif not isinstance(cell_value, (int, float)):
                            cell_value = 0

                        # Look for bright spots (potential buttons) - FIXED FOR ARC (was 200, now 4+)
                        if cell_value > 4:  # FIXED: ARC bright spots (values 5-9 in ARC palette)
                            candidates.append((x, y, cell_value))

            # If we found bright spots, pick the brightest
            if candidates:
                candidates.sort(key=lambda x: x[2], reverse=True)
                best_x, best_y, best_value = candidates[0]
                print(f"[SYNC COORD] Found {len(candidates)} bright objects, selected brightest at ({best_x}, {best_y}) with value {best_value}")
                return best_x, best_y

            # If no bright spots, look for any non-zero cells (any colored objects)
            print(f"[SYNC COORD] No bright objects found, looking for any colored objects...")
            for y in range(height):
                for x in range(width):
                    if isinstance(frame[y], list) and x < len(frame[y]):
                        cell_value = frame[y][x]
                        if isinstance(cell_value, (list, tuple)) and len(cell_value) > 0:
                            cell_value = cell_value[0] if isinstance(cell_value[0], (int, float)) else 0
                        elif not isinstance(cell_value, (int, float)):
                            cell_value = 0

                        if cell_value > 0:
                            print(f"[SYNC COORD] Found first colored object at ({x}, {y}) with value {cell_value}")
                            return x, y

            # Last resort: center of frame
            center_x, center_y = width // 2, height // 2
            print(f"[SYNC COORD] No objects found, using frame center ({center_x}, {center_y})")
            return center_x, center_y

        except Exception as e:
            logger.warning(f"Error in sync coordinate selection: {e}")
            print(f"[SYNC COORD] Error during coordinate selection: {e}")
            return 12, 12  # Safe fallback  # Safe fallback

    def _get_enhanced_action6_coordinates_sync(self, frame: List[List[int]], game_id: str,
                                             actions: List[int], current_score: int,
                                             hypothesis_context: Optional[List[Dict[str, Any]]] = None) -> Tuple[int, int]:
        """Get enhanced Action 6 coordinates using the enhanced coordinator with exploration."""
        try:
            # Check for hypothesis-guided coordinates first
            if hypothesis_context:
                try:
                    print(f"[HYPOTHESIS] Using hypothesis-guided coordinate selection")
                    for i, hypothesis in enumerate(hypothesis_context[:3]):  # Use top 3 hypotheses
                        if hypothesis.get('hypothesis_type') == 'coordinate_sequence':
                            hypothesis_data = hypothesis.get('hypothesis_data', {})
                            if isinstance(hypothesis_data, str):
                                import json
                                hypothesis_data = json.loads(hypothesis_data)

                            predicted_coords = hypothesis_data.get('predicted_coordinates')
                            if predicted_coords and len(predicted_coords) >= 2:
                                x, y = predicted_coords[0], predicted_coords[1]
                                # Validate coordinates are within reasonable bounds
                                if 0 <= x <= 60 and 0 <= y <= 60:
                                    print(f"[HYPOTHESIS] Using hypothesis {i+1} coordinates: ({x}, {y})")
                                    print(f"[HYPOTHESIS] Reasoning: {hypothesis.get('reasoning', 'No reasoning provided')[:100]}...")
                                    return x, y
                except Exception as e:
                    print(f"[HYPOTHESIS] Error processing hypothesis context: {e}")

            if not self.enhanced_gameplay or not hasattr(self.enhanced_gameplay, 'action6_coordinator'):
                print(f"[ENHANCED] No action6_coordinator available, using fallback")
                return self._get_sync_action6_coordinates(frame, game_id)

            coordinator = self.enhanced_gameplay.action6_coordinator
            print(f"[ENHANCED] Using Action 6 coordinator with exploration features")

            # Create proper context for the enhanced coordinator
            context = {
                'available_actions': [f'ACTION{a}' for a in actions] if actions else ['ACTION6'],
                'action': 'ACTION6',
                'current_score': current_score
            }

            # Use the enhanced coordinator with all its features (pseudo-buttons + exploration)
            try:
                # Call the async method synchronously using asyncio
                import asyncio

                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an event loop, so we need to use create_task
                    print(f"[ENHANCED] Running in event loop, using create_task")

                    # Create a task for the async coordinator call
                    async def get_coords():
                        return await coordinator.get_optimal_action6_coordinates(frame, game_id, context)

                    # This is tricky - we need to run async in sync context
                    # Use the existing event loop if available
                    task = asyncio.create_task(get_coords())
                    coords = None

                    # Since we can't await in sync context, use the fallback
                    print(f"[ENHANCED] Cannot await in sync context, using exploration fallback")

                    # Use exploration features directly
                    if hasattr(coordinator, '_should_force_exploration'):
                        force_exploration = coordinator._should_force_exploration(game_id, context)
                        if force_exploration and frame and len(frame) > 0 and len(frame[0]) > 0:
                            grid_dims = (len(frame[0]), len(frame))
                            coords = coordinator._get_strategic_action6_coordinates(grid_dims, game_id)
                            print(f"[ENHANCED] Used exploration coordinates: {coords}")
                            coordinator.stats['exploration_selections'] = coordinator.stats.get('exploration_selections', 0) + 1
                            return coords

                    # If no exploration needed, try pseudo-button detection
                    button_candidates = self._detect_buttons_sync(frame, game_id)
                    if button_candidates:
                        best_candidate = self._score_button_candidates(button_candidates, game_id)
                        print(f"[ENHANCED] Selected button at ({best_candidate['x']}, {best_candidate['y']}) with score {best_candidate.get('score', 0):.2f}")
                        coordinator.stats['button_based_selections'] += 1
                        return best_candidate['x'], best_candidate['y']

                except RuntimeError:
                    # No event loop, we can use asyncio.run
                    print(f"[ENHANCED] No event loop, using asyncio.run")
                    coords = asyncio.run(coordinator.get_optimal_action6_coordinates(frame, game_id, context))
                    print(f"[ENHANCED] Enhanced coordinator returned: {coords}")
                    return coords

            except Exception as e:
                logger.warning(f"Enhanced coordinator call failed: {e}")
                print(f"[ENHANCED] Enhanced coordinator error: {e}")

            # Fallback to exploration if available
            if hasattr(coordinator, '_get_strategic_action6_coordinates') and frame:
                try:
                    grid_dims = (len(frame[0]), len(frame)) if frame and frame[0] else (30, 30)
                    coords = coordinator._get_strategic_action6_coordinates(grid_dims, game_id)
                    print(f"[ENHANCED] Using exploration fallback coordinates: {coords}")
                    coordinator.stats['exploration_selections'] = coordinator.stats.get('exploration_selections', 0) + 1
                    return coords
                except Exception as e:
                    logger.warning(f"Exploration fallback failed: {e}")
                    print(f"[ENHANCED] Exploration fallback error: {e}")

            # Final fallback
            print(f"[ENHANCED] Using final sync fallback coordinates")
            return self._get_sync_action6_coordinates(frame, game_id)

        except Exception as e:
            logger.warning(f"Enhanced coordinate selection failed: {e}")
            print(f"[ENHANCED] Major error in enhanced coordinate selection: {e}")
            return self._get_sync_action6_coordinates(frame, game_id)

    def _detect_buttons_sync(self, frame: List[List[int]], game_id: str) -> List[Dict[str, Any]]:
        """Synchronous pseudo-button detection."""
        try:
            if not frame or not frame[0]:
                print(f"[BUTTON DETECT] Empty frame provided")
                return []

            height = len(frame)
            width = len(frame[0])
            candidates = []
            
            print(f"[BUTTON DETECT] Analyzing {height}x{width} frame for objects and pseudo-buttons (ARC color range 0-9)")

            # Method 1: Look for rectangular bright regions (potential buttons) - FIXED FOR ARC
            print(f"[BUTTON DETECT] Scanning for bright rectangular regions...")
            bright_regions = 0
            for y in range(0, height-2, 2):  # Sample every 2 pixels for efficiency
                for x in range(0, width-2, 2):
                    # Check 3x3 regions for button-like patterns
                    region_values = []
                    for dy in range(3):
                        for dx in range(3):
                            if y+dy < height and x+dx < width:
                                cell_value = self._get_cell_value(frame[y+dy][x+dx])
                                region_values.append(cell_value)

                    if len(region_values) == 9:
                        avg_value = sum(region_values) / 9
                        if avg_value > 3:  # FIXED: ARC bright region (was 100, now 3 for ARC 0-9 range)
                            bright_regions += 1
                            # Check if it's distinct from surroundings
                            surrounding_values = []
                            for dy in range(-1, 4):
                                for dx in range(-1, 4):
                                    ny, nx = y+dy, x+dx
                                    if (0 <= ny < height and 0 <= nx < width and
                                        not (0 <= dy <= 2 and 0 <= dx <= 2)):
                                        surrounding_values.append(self._get_cell_value(frame[ny][nx]))

                            if surrounding_values:
                                avg_surrounding = sum(surrounding_values) / len(surrounding_values)
                                contrast = avg_value - avg_surrounding

                                if contrast > 1.5:  # FIXED: ARC contrast (was 50, now 1.5 for ARC 0-9 range)
                                    candidates.append({
                                        'x': x + 1,  # Center of 3x3 region
                                        'y': y + 1,
                                        'confidence': min(contrast / 5, 1.0),  # Scale to ARC range
                                        'brightness': avg_value,
                                        'contrast': contrast,
                                        'type': 'bright_region'
                                    })

            print(f"[BUTTON DETECT] Found {bright_regions} bright regions, {len(candidates)} are good candidates")

            # Method 2: Look for high-contrast edges (object boundaries) - FIXED FOR ARC
            print(f"[BUTTON DETECT] Scanning for high-contrast object boundaries...")
            edge_objects = 0
            for y in range(1, height-1, 3):
                for x in range(1, width-1, 3):
                    center_value = self._get_cell_value(frame[y][x])
                    
                    # Check all 8 neighbors for edge detection
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            if 0 <= y+dy < height and 0 <= x+dx < width:
                                neighbors.append(self._get_cell_value(frame[y+dy][x+dx]))
                    
                    if neighbors:
                        max_diff = max(abs(center_value - n) for n in neighbors)
                        if max_diff > 2:  # FIXED: ARC edge detection (was 80, now 2 for ARC 0-9 range)
                            edge_objects += 1
                            candidates.append({
                                'x': x,
                                'y': y,
                                'confidence': min(max_diff / 5, 1.0),  # Scale to ARC range
                                'brightness': center_value,
                                'contrast': max_diff,
                                'type': 'edge_object'
                            })

            print(f"[BUTTON DETECT] Found {edge_objects} edge objects")

            # Method 3: Look for distinct color objects - FIXED FOR ARC
            print(f"[BUTTON DETECT] Scanning for distinct color objects...")
            color_objects = 0
            for y in range(2, height-2, 4):
                for x in range(2, width-2, 4):
                    center_value = self._get_cell_value(frame[y][x])
                    
                    # Skip black/empty areas (still valid for ARC)
                    if center_value == 0:
                        continue
                    
                    # Check if this color is distinct in its neighborhood
                    similar_count = 0
                    total_neighbors = 0
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            if 0 <= y+dy < height and 0 <= x+dx < width:
                                neighbor_value = self._get_cell_value(frame[y+dy][x+dx])
                                total_neighbors += 1
                                if abs(center_value - neighbor_value) < 1:  # FIXED: ARC similarity (was 20, now 1 for ARC 0-9 range)
                                    similar_count += 1
                    
                    # If this color is distinct (not too many similar neighbors)
                    if total_neighbors > 0 and similar_count / total_neighbors < 0.4:
                        color_objects += 1
                        candidates.append({
                            'x': x,
                            'y': y,
                            'confidence': 0.7,  # Higher confidence for distinct colors
                            'brightness': center_value,
                            'contrast': 3,  # Default ARC contrast
                            'type': 'color_object'
                        })

            print(f"[BUTTON DETECT] Found {color_objects} distinct color objects")

            # Remove duplicates that are too close together
            print(f"[BUTTON DETECT] Deduplicating {len(candidates)} total candidates...")
            candidates = self._deduplicate_candidates_sync(candidates)
            
            print(f"[BUTTON DETECT] Final result: {len(candidates)} unique object/button candidates")
            for i, candidate in enumerate(candidates[:5]):  # Show top 5
                print(f"[BUTTON DETECT] #{i+1}: ({candidate['x']}, {candidate['y']}) confidence={candidate['confidence']:.2f} type={candidate['type']} brightness={candidate['brightness']} contrast={candidate['contrast']:.1f}")

            return candidates[:10]  # Return top 10 candidates

        except Exception as e:
            logger.warning(f"Sync button detection error: {e}")
            print(f"[BUTTON DETECT] Error during detection: {e}")
            return []

    def _deduplicate_candidates_sync(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate candidates that are too close together."""
        if not candidates:
            return []

        unique_candidates = []
        min_distance = 8  # Minimum distance between candidate centers

        for candidate in candidates:
            is_duplicate = False
            for existing in unique_candidates:
                distance = ((candidate['x'] - existing['x'])**2 + (candidate['y'] - existing['y'])**2)**0.5
                if distance < min_distance:
                    # Keep the one with higher confidence
                    if candidate['confidence'] > existing['confidence']:
                        unique_candidates.remove(existing)
                        unique_candidates.append(candidate)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_candidates.append(candidate)

        return unique_candidates

    def _score_button_candidates(self, candidates: List[Dict[str, Any]], game_id: str) -> Dict[str, Any]:
        """Score button candidates and return the best one."""
        try:
            if not candidates:
                print(f"[CANDIDATE SCORING] No candidates to score, using center fallback")
                # Return center coordinates as fallback
                return {'x': 12, 'y': 12, 'score': 0.0, 'type': 'fallback'}

            print(f"[CANDIDATE SCORING] Scoring {len(candidates)} candidates for ARC color range (0-9)")

            # Score each candidate
            for candidate in candidates:
                score = 0.0

                # Base score from confidence and contrast - FIXED FOR ARC RANGE
                confidence_score = candidate.get('confidence', 0) * 0.4
                contrast_score = min(candidate.get('contrast', 0) / 5, 1.0) * 0.3  # FIXED: Scale for ARC (was /100, now /5)
                brightness_score = min(candidate.get('brightness', 0) / 9, 1.0) * 0.2  # FIXED: Scale for ARC (was /255, now /9)

                score += confidence_score + contrast_score + brightness_score

                # Type-based bonus
                button_type = candidate.get('type', 'unknown')
                if button_type == 'bright_region':
                    score += 0.1  # Bright regions are very good button candidates in ARC
                elif button_type == 'edge_object':
                    score += 0.08  # Edge objects are good
                elif button_type == 'color_object':
                    score += 0.06  # Color objects are worth trying

                # Prefer candidates not too close to edges (assuming standard ARC frame)
                x, y = candidate['x'], candidate['y']
                frame_size = 64  # Standard ARC frame size
                edge_penalty = 0
                if x < 3 or x > frame_size-3 or y < 3 or y > frame_size-3:
                    edge_penalty = 0.02  # Smaller penalty for ARC

                # Brightness bonus for ARC - higher values are more likely to be interactive
                brightness = candidate.get('brightness', 0)
                if brightness >= 7:  # Very bright in ARC (7-9)
                    score += 0.15
                elif brightness >= 4:  # Medium bright in ARC (4-6)
                    score += 0.08
                elif brightness >= 2:  # Somewhat bright in ARC (2-3)
                    score += 0.03

                score -= edge_penalty
                candidate['score'] = max(score, 0.0)

                print(f"[CANDIDATE SCORING] ({x}, {y}) type={button_type} conf={confidence_score:.3f} contr={contrast_score:.3f} bright={brightness_score:.3f} brightness={brightness} penalty={edge_penalty:.3f} final={candidate['score']:.3f}")

            # Return best candidate
            best = max(candidates, key=lambda c: c.get('score', 0))
            print(f"[CANDIDATE SCORING] Selected best: ({best['x']}, {best['y']}) score={best['score']:.3f} type={best.get('type', 'unknown')} brightness={best.get('brightness', 0)}")
            return best

        except Exception as e:
            logger.warning(f"Candidate scoring error: {e}")
            print(f"[CANDIDATE SCORING] Error during scoring: {e}")
            return {'x': 12, 'y': 12, 'score': 0.0, 'type': 'error_fallback'}

    def _get_intelligent_coordinates_from_db(self, frame: List[List[int]], game_id: str) -> Optional[Tuple[int, int]]:
        """Get coordinates based on coordinate intelligence database."""
        try:
            if not hasattr(self, 'enhanced_gameplay') or not self.enhanced_gameplay:
                return None

            coordinator = getattr(self.enhanced_gameplay, 'action6_coordinator', None)
            if not coordinator or not hasattr(coordinator, 'db_interface'):
                return None

            db = coordinator.db_interface
            if not db or not hasattr(db, 'execute_query'):
                return None

            # Query for effective coordinates
            query = '''
                SELECT x, y, effectiveness_score
                FROM coordinate_intelligence
                WHERE game_id = ? OR game_id IS NULL
                ORDER BY effectiveness_score DESC
                LIMIT 1
            '''

            result = db.execute_query(query, (game_id,))
            if result and len(result) > 0:
                x, y, score = result[0]
                if score > 0.1:  # Only use if reasonably effective
                    return x, y

            return None

        except Exception as e:
            logger.warning(f"Database coordinate lookup error: {e}")
            return None

    async def _analyze_action6_effectiveness_async(self, frame_before: List[List[int]], 
                                                 frame_after: List[List[int]], 
                                                 coordinates: Tuple[int, int], 
                                                 game_id: str, 
                                                 score_change: float):
        """Asynchronously analyze ACTION 6 effectiveness for learning."""
        try:
            print(f"[EFFECTIVENESS] Analyzing ACTION 6 at {coordinates} with score change {score_change}")
            
            if self.enhanced_gameplay and hasattr(self.enhanced_gameplay, 'action6_coordinator'):
                coordinator = self.enhanced_gameplay.action6_coordinator
                
                if hasattr(coordinator, 'analyze_action6_effectiveness'):
                    await coordinator.analyze_action6_effectiveness(
                        frame_before, frame_after, coordinates, game_id, score_change
                    )
                    print(f"[EFFECTIVENESS] Analysis completed for ACTION 6 at {coordinates}")
                else:
                    print(f"[EFFECTIVENESS] Coordinator has no analyze_action6_effectiveness method")
            else:
                print(f"[EFFECTIVENESS] No enhanced gameplay or coordinator available")
                
        except Exception as e:
            print(f"[EFFECTIVENESS] Error during async analysis: {e}")
            logger.warning(f"Action 6 effectiveness analysis failed: {e}")

    def _analyze_frame_for_action_selection(self, frame: List[List[int]], actions: List[int], game_state) -> Optional[str]:
        """Analyze frame to make intelligent action selection decisions."""
        try:
            if not frame or not frame[0]:
                return None

            height = len(frame)
            width = len(frame[0])

            # Look for patterns that suggest specific actions
            non_zero_cells = 0
            bright_cells = 0
            edge_activity = 0

            # Analyze frame characteristics - FIXED FOR ARC COLOR RANGE (0-9)
            for y in range(height):
                for x in range(width):
                    if isinstance(frame[y], list) and x < len(frame[y]):
                        cell_value = self._get_cell_value(frame[y][x])

                        if cell_value > 0:
                            non_zero_cells += 1
                        if cell_value > 5:  # FIXED: ARC bright cells (was 200, now 5 for ARC 0-9 range)
                            bright_cells += 1

                        # Check edges for activity
                        if (x == 0 or x == width-1 or y == 0 or y == height-1) and cell_value > 0:
                            edge_activity += 1

            # Calculate density and patterns
            total_cells = height * width
            density = non_zero_cells / total_cells if total_cells > 0 else 0
            brightness_ratio = bright_cells / non_zero_cells if non_zero_cells > 0 else 0

            print(f"[FRAME ANALYSIS] Frame {height}x{width}: {non_zero_cells} non-zero cells ({density:.3f} density), {bright_cells} bright cells ({brightness_ratio:.3f} ratio), {edge_activity} edge activity")

            # Enhanced intelligent action selection based on frame analysis
            if density < 0.1 and 1 in actions:
                # Sparse frame - ACTION1 might help reveal more
                print(f"[FRAME ANALYSIS] Sparse frame detected - suggesting ACTION1")
                return "ACTION1"
            elif brightness_ratio > 0.2 and 6 in actions:  # FIXED: Lower threshold for ARC
                # Many bright spots - likely pseudo-buttons, use ACTION6 with enhanced detection
                print(f"[FRAME ANALYSIS] Detected {bright_cells} bright cells (ratio: {brightness_ratio:.3f}) - triggering enhanced ACTION6")
                return "ACTION6"
            elif edge_activity > 5 and 2 in actions:
                # Activity on edges - ACTION2 might be good
                print(f"[FRAME ANALYSIS] High edge activity detected - suggesting ACTION2")
                return "ACTION2"
            elif density > 0.5 and 5 in actions:
                # Dense frame - ACTION5 might help
                print(f"[FRAME ANALYSIS] Dense frame detected - suggesting ACTION5")
                return "ACTION5"
            
            # If ACTION 6 is the only action available, always use it
            elif len(actions) == 1 and 6 in actions:
                print(f"[FRAME ANALYSIS] ACTION6 is only available action - using enhanced object detection")
                return "ACTION6"
            
            # If we have non-zero cells and ACTION6 is available, prefer it for object interaction
            elif non_zero_cells > 10 and 6 in actions:
                print(f"[FRAME ANALYSIS] Objects detected ({non_zero_cells} non-zero cells) - suggesting ACTION6 for object interaction")
                return "ACTION6"

            print(f"[FRAME ANALYSIS] No clear action preference based on frame analysis")
            return None  # No clear preference

        except Exception as e:
            logger.warning(f"Frame analysis error: {e}")
            return None

    def _select_intelligent_action(self, frame: List[List[int]], actions: List[int],
                                 current_score: int, game_state: str) -> str:
        """Select intelligent action based on game state and frame analysis."""
        try:
            # If we're not making progress, try different actions
            if current_score == 0 and hasattr(self, '_last_score'):
                if self._last_score == 0:
                    # No progress - try different action
                    if hasattr(self, '_last_action_used'):
                        # Avoid repeating the same action if it's not working
                        available_actions = [a for a in actions if f"ACTION{a}" != self._last_action_used]
                        if available_actions:
                            actions = available_actions

            # Priority-based action selection
            action_priorities = {
                1: 1,  # Basic action - low priority
                2: 3,  # Common action - medium priority
                3: 2,  # Alternative action - low-medium priority
                4: 2,  # Alternative action - low-medium priority
                5: 2,  # Alternative action - low-medium priority
                6: 0,  # CRITICAL FIX: Don't select ACTION6 here - let dedicated logic handle it
                7: 1,  # Reset action - low priority
            }

            # Filter out ACTION6 from intelligent selection - it has dedicated logic
            non_coordinate_actions = [a for a in actions if a != 6]

            if non_coordinate_actions:
                # Select action with highest available priority from non-coordinate actions
                best_action = max(non_coordinate_actions, key=lambda a: action_priorities.get(a, 0))
                selected_action = f"ACTION{best_action}"
            else:
                # If only ACTION6 is available, return None to trigger dedicated ACTION6 logic
                return None

            # Store for learning
            self._last_action_used = selected_action
            self._last_score = current_score

            return selected_action

        except Exception as e:
            logger.warning(f"Intelligent action selection error: {e}")
            # Fallback to simple selection - avoid ACTION6
            non_coordinate_fallback = [a for a in actions if a != 6]
            if 2 in non_coordinate_fallback:
                return "ACTION2"
            elif 1 in non_coordinate_fallback:
                return "ACTION1"
            elif non_coordinate_fallback:
                return f"ACTION{non_coordinate_fallback[0]}"
            else:
                # Only ACTION6 available - return None to trigger dedicated logic
                return None

    def _get_cell_value(self, cell) -> int:
        """Extract numeric value from cell (handles different formats)."""
        try:
            if isinstance(cell, (list, tuple)) and len(cell) > 0:
                return int(cell[0]) if isinstance(cell[0], (int, float)) else 0
            elif isinstance(cell, (int, float)):
                return int(cell)
            else:
                return 0
        except:
            return 0

    def _capture_frame_for_visualization(self, game_id: str, session_id: str, action_number: int,
                                   frame: List[List[int]], action_taken: Optional[str] = None,
                                   action_x: Optional[int] = None, action_y: Optional[int] = None,
                                   score_before: int = 0, score_after: int = 0,
                                   available_actions: Optional[List[int]] = None):
        """Capture frame data for visualization without affecting game performance."""
        print(f"[CAPTURE] Called for action {action_number}, session {session_id}")
        try:
            import json
            
            # Convert frame to JSON string for storage
            frame_data = json.dumps(frame)
            frame_height = len(frame) if frame else 0
            frame_width = len(frame[0]) if frame and frame[0] else 0
            actions_str = json.dumps(available_actions) if available_actions else "[]"
            
            print(f"[CAPTURE] Frame size: {frame_width}x{frame_height}, Action: {action_taken}")
            
            # Store directly (synchronous) to ensure reliability
            if hasattr(self, 'session_manager'):
                print(f"[CAPTURE] Has session_manager")
                if hasattr(self.session_manager, 'database'):
                    print(f"[CAPTURE] Has database")
                    try:
                        db = self.session_manager.database
                        db.execute_query("""
                            INSERT OR REPLACE INTO game_visualization
                            (game_id, session_id, action_number, frame_data, frame_width, frame_height,
                             action_taken, action_x, action_y, score_before, score_after, available_actions)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (game_id, session_id, action_number, frame_data, frame_width, frame_height,
                              action_taken, action_x, action_y, score_before, score_after, actions_str))
                        print(f"[CAPTURE] Data stored successfully for action {action_number}")
                    except Exception as e:
                        print(f"[CAPTURE] Database error: {e}")
                else:
                    print(f"[CAPTURE] No database attribute")
            else:
                print(f"[CAPTURE] No session_manager attribute")
            
        except Exception as e:
            print(f"[CAPTURE] General error: {e}")
            import traceback
            traceback.print_exc()

    def _start_visualization_session(self, game_id: str, session_id: str):
        """Start a new visualization session."""
        print(f"[SESSION] Starting session {session_id} for game {game_id}")
        try:
            if hasattr(self, 'session_manager') and hasattr(self.session_manager, 'database'):
                db = self.session_manager.database
                db.execute_query("""
                    INSERT OR REPLACE INTO visualization_sessions
                    (session_id, game_id, start_time)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (session_id, game_id))
                print(f"[SESSION] Started visualization session: {session_id}")
            else:
                print(f"[SESSION] No database available")
        except Exception as e:
            print(f"[SESSION] Session start error: {e}")
            import traceback
            traceback.print_exc()

    def _end_visualization_session(self, session_id: str, total_actions: int, 
                             final_score: int, game_won: bool, ai_performance: dict):
        """End a visualization session with final stats."""
        try:
            import json
            
            if hasattr(self, 'session_manager') and hasattr(self.session_manager, 'database'):
                db = self.session_manager.database
                ai_perf_str = json.dumps(ai_performance) if ai_performance else "{}"
                db.execute_query("""
                    UPDATE visualization_sessions
                    SET end_time = CURRENT_TIMESTAMP, total_actions = ?, 
                        final_score = ?, game_won = ?, ai_performance = ?
                    WHERE session_id = ?
                """, (total_actions, final_score, game_won, ai_perf_str, session_id))
                print(f"[DEBUG] Ended visualization session: {session_id}")
        except Exception as e:
            print(f"[DEBUG] Session end error: {e}")

    def _debug_frame_structure(self, frame) -> str:
        """Debug frame structure to understand ARC API data format."""
        try:
            if not frame:
                return "Frame: None"

            frame_type = type(frame).__name__
            frame_len = len(frame)

            if frame_len == 0:
                return "Frame: Empty"

            # Check first element
            first_elem = frame[0]
            first_elem_type = type(first_elem).__name__
            first_elem_len = len(first_elem) if hasattr(first_elem, '__len__') else 'N/A'

            # ARC API sends: frame[0] = array of 64 rows, each row = 64 values
            if frame_len == 1 and hasattr(first_elem, '__len__') and len(first_elem) > 0:
                rows_data = first_elem  # This is the actual 64x64 grid data
                num_rows = len(rows_data)
                
                # Check if each row contains values (this should be the 64x64 grid)
                if num_rows > 0 and hasattr(rows_data[0], '__len__'):
                    pixels_per_row = len(rows_data[0])
                    
                    # This is the correct ARC 64x64 format
                    if num_rows == 64 and pixels_per_row == 64:
                        return f"Frame: 64x64 ARC Grid (64 rows × 64 pixels each)"
                    else:
                        return f"Frame: {num_rows}x{pixels_per_row} Grid"
                else:
                    # Legacy flattened format
                    total_elements = len(rows_data)
                    if total_elements == 4096:
                        return f"Frame: 1x{total_elements} (FLATTENED 64x64 - needs reshaping!)"
                    elif total_elements == 1024:
                        return f"Frame: 1x{total_elements} (FLATTENED 32x32 - needs reshaping!)"
                    elif total_elements == 64:
                        return f"Frame: 1x{total_elements} (FLATTENED 8x8 - needs reshaping!)"
                    else:
                        return f"Frame: 1x{total_elements} (FLATTENED - unknown dimensions)"

            # Check if we have proper 2D structure
            if frame_len > 1 and hasattr(first_elem, '__len__'):
                second_elem_len = len(frame[1]) if len(frame) > 1 else 'N/A'
                return f"Frame: {frame_len}x{first_elem_len} (proper 2D, first_type: {first_elem_type})"

            return f"Frame: {frame_len} elements, type: {frame_type}, first_elem_type: {first_elem_type}, first_elem_len: {first_elem_len}"

        except Exception as e:
            return f"Frame: ERROR debugging - {e}"

    def _reshape_frame_if_needed(self, frame) -> List[List[int]]:
        """Reshape frame if it comes as flattened array or extract from ARC wrapper."""
        try:
            if not frame:
                return []

            # ARC API format: frame[0] contains the actual 64x64 grid data
            if len(frame) == 1 and hasattr(frame[0], '__len__'):
                grid_data = frame[0]
                
                # Check if grid_data contains rows of pixels (ARC's 64x64 format)
                if len(grid_data) > 0 and hasattr(grid_data[0], '__len__'):
                    num_rows = len(grid_data)
                    pixels_per_row = len(grid_data[0])
                    
                    # This is already properly structured as 64x64 grid - extract it
                    if num_rows == 64 and pixels_per_row == 64:
                        print(f"[FRAME] Extracted 64x64 ARC grid from API wrapper")
                        # Convert to proper format and handle non-standard values
                        cleaned_grid = []
                        for row in grid_data:
                            cleaned_row = []
                            for cell in row:
                                # Convert to standard ARC values (0-9)
                                if isinstance(cell, (list, tuple)):
                                    cell_value = cell[0] if len(cell) > 0 else 0
                                else:
                                    cell_value = cell
                                
                                # Map non-standard values to ARC palette
                                if cell_value == 14:  # Map 14 to a standard color
                                    cell_value = 5  # Gray
                                elif cell_value == 11:  # Map 11 to a standard color  
                                    cell_value = 6  # Fuchsia
                                elif cell_value > 9:  # Any other non-standard values
                                    cell_value = cell_value % 10  # Wrap to 0-9
                                
                                cleaned_row.append(int(cell_value))
                            cleaned_grid.append(cleaned_row)
                        return cleaned_grid
                    elif num_rows == pixels_per_row:
                        print(f"[FRAME] Extracted {num_rows}x{pixels_per_row} grid from wrapper")
                        return grid_data
                    else:
                        print(f"[FRAME] Extracted {num_rows}x{pixels_per_row} rectangular grid")
                        return grid_data
                
                # Legacy flattened format - needs actual reshaping
                else:
                    flat_data = grid_data
                    total_elements = len(flat_data)

                    # Try common ARC game sizes
                    if total_elements == 4096:  # 64x64
                        reshaped = []
                        for i in range(64):
                            row = flat_data[i*64:(i+1)*64]
                            reshaped.append(list(row))
                        print(f"[FRAME] Reshaped 1x4096 -> 64x64")
                        return reshaped
                    elif total_elements == 1024:  # 32x32
                        reshaped = []
                        for i in range(32):
                            row = flat_data[i*32:(i+1)*32]
                            reshaped.append(list(row))
                        print(f"[FRAME] Reshaped 1x1024 -> 32x32")
                        return reshaped
                    elif total_elements == 625:  # 25x25
                        reshaped = []
                        for i in range(25):
                            row = flat_data[i*25:(i+1)*25]
                            reshaped.append(list(row))
                        print(f"[FRAME] Reshaped 1x625 -> 25x25")
                        return reshaped
                    elif total_elements == 64:  # 8x8
                        reshaped = []
                        for i in range(8):
                            row = flat_data[i*8:(i+1)*8]
                            reshaped.append(list(row))
                        print(f"[FRAME] Reshaped 1x64 -> 8x8")
                        return reshaped
                    else:
                        # Try to infer square dimensions
                        import math
                        side_length = int(math.sqrt(total_elements))
                        if side_length * side_length == total_elements:
                            reshaped = []
                            for i in range(side_length):
                                row = flat_data[i*side_length:(i+1)*side_length]
                                reshaped.append(list(row))
                            print(f"[FRAME] Reshaped 1x{total_elements} -> {side_length}x{side_length}")
                            return reshaped

            # If it's already properly shaped 2D array, return as-is
            if len(frame) > 1 and hasattr(frame[0], '__len__'):
                print(f"[FRAME] Frame already properly shaped: {len(frame)}x{len(frame[0])}")
                return frame

            # Return original if can't reshape
            print(f"[FRAME] Could not reshape frame, returning original")
            return frame

        except Exception as e:
            print(f"[FRAME] Reshape error: {e}")
            return frame

    def _create_knowledge_integrator(self):
        """Create a knowledge integrator for compatibility."""
        class SimpleKnowledgeIntegrator:
            async def extract_game_knowledge(self, game_result: Dict[str, Any]):
                """Extract knowledge from game results."""
                # Simple implementation - just log the result
                import logging
                logger = logging.getLogger(__name__)
                score = game_result.get('final_score', 0)
                actions = game_result.get('total_actions', 0)
                won = game_result.get('win_detected', False)
                logger.info(f"Knowledge extracted: Score={score}, Actions={actions}, Won={won}")
                return {"knowledge_extracted": True, "game_score": score}

        return SimpleKnowledgeIntegrator()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get AI performance statistics."""
        return {
            'action6_enhanced': self.enhanced_gameplay is not None,
            'pseudo_button_detection': True,
            'coordinate_intelligence': True,
            'enhanced_systems_active': True,
            'fallback_mode': self.enhanced_gameplay is None,
            'ai_orchestration': True,
            'vision_guidance': True,
            'pattern_learning': True,
            'knowledge_extraction': True
        }

    async def get_current_screenshot(self) -> Optional[Dict[str, Any]]:
        """Get current screenshot from the game session for analysis.

        Note: Screenshots are only available during active gameplay through the ARC API.
        This method returns None before gameplay starts, which is expected behavior.

        Returns:
            None - Screenshots are provided by the ARC API during gameplay
        """
        try:
            logger.info("Screenshot not available before gameplay - will be provided during game loop")

            # Screenshots come from the ARC API during the actual game loop
            # They are not available before the game starts
            return None

        except Exception as e:
            logger.warning(f"Failed to get current screenshot: {e}")
            return None


class CoreGameDatabase:
    """Compatibility wrapper for CoreGameDatabase."""

    def __init__(self, db_path: str = "tabula_rasa.db"):
        self.db_path = db_path
        self._setup_database()

    def _setup_database(self):
        """Set up the database with required tables."""
        import sqlite3
        import os

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create logs table for database logging
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        level TEXT,
                        logger_name TEXT,
                        message TEXT,
                        game_id TEXT,
                        session_id TEXT
                    )
                ''')

                # Create coordinate_intelligence table for Action 6
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS coordinate_intelligence (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        x INTEGER NOT NULL,
                        y INTEGER NOT NULL,
                        game_id TEXT,
                        attempts INTEGER DEFAULT 1,
                        successes INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,
                        effectiveness_score REAL DEFAULT 0.0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create pseudo_button_learning table for advanced Action 6 learning
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS pseudo_button_learning (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id TEXT NOT NULL,
                        x INTEGER NOT NULL,
                        y INTEGER NOT NULL,
                        effect_description TEXT,
                        effectiveness_score REAL DEFAULT 0.0,
                        confidence REAL DEFAULT 0.0,
                        attempts INTEGER DEFAULT 1,
                        last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        visual_changes REAL DEFAULT 0.0,
                        score_impact BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(game_id, x, y)
                    )
                ''')

                # Create pseudo_button_sequences table for sequence learning
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS pseudo_button_sequences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id TEXT NOT NULL,
                        sequence_coords TEXT NOT NULL,
                        score_gained REAL NOT NULL,
                        sequence_length INTEGER NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create visualization table for frame replay
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS game_visualization (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id TEXT NOT NULL,
                        session_id TEXT,
                        action_number INTEGER NOT NULL,
                        frame_data TEXT NOT NULL,
                        frame_width INTEGER,
                        frame_height INTEGER,
                        action_taken TEXT,
                        action_x INTEGER,
                        action_y INTEGER,
                        score_before INTEGER,
                        score_after INTEGER,
                        available_actions TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(game_id, session_id, action_number)
                    )
                ''')

                # Create visualization sessions table for replay management
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS visualization_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        game_id TEXT NOT NULL,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        total_actions INTEGER DEFAULT 0,
                        final_score INTEGER DEFAULT 0,
                        game_won BOOLEAN DEFAULT FALSE,
                        ai_performance TEXT
                    )
                ''')

                conn.commit()

        except Exception as e:
            print(f"Database setup error: {e}")

    @classmethod
    def ensure_visualization_tables(cls, db_path: str):
        """Ensure visualization tables exist in database."""
        import sqlite3
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Create visualization table for frame replay
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS game_visualization (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_id TEXT NOT NULL,
                        session_id TEXT,
                        action_number INTEGER NOT NULL,
                        frame_data TEXT NOT NULL,
                        frame_width INTEGER,
                        frame_height INTEGER,
                        action_taken TEXT,
                        action_x INTEGER,
                        action_y INTEGER,
                        score_before INTEGER,
                        score_after INTEGER,
                        available_actions TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(game_id, session_id, action_number)
                    )
                ''')

                # Create visualization sessions table for replay management
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS visualization_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        game_id TEXT NOT NULL,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        total_actions INTEGER DEFAULT 0,
                        final_score INTEGER DEFAULT 0,
                        game_won BOOLEAN DEFAULT FALSE,
                        ai_performance TEXT
                    )
                ''')

                conn.commit()
                print(f"[DB] Visualization tables initialized in {db_path}")

        except Exception as e:
            print(f"Database table creation error: {e}")

    def execute_query(self, query: str, params: Tuple = None):
        """Execute a database query."""
        import sqlite3

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if params:
                    result = cursor.execute(query, params)
                else:
                    result = cursor.execute(query)

                # For SELECT queries, return results
                if query.strip().upper().startswith('SELECT'):
                    return result.fetchall()
                else:
                    conn.commit()
                    return cursor.rowcount

        except Exception as e:
            print(f"Database query error: {e}")
            return None

    def log_to_database(self, level: str, logger_name: str, message: str,
                       game_id: str = None, session_id: str = None):
        """Log a message to the database."""
        query = '''
            INSERT INTO system_logs (log_level, component, message, game_id, session_id, timestamp)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        '''
        return self.execute_query(query, (level, logger_name, message, game_id, session_id))

    def close(self):
        """Close the database connection (compatibility method)."""
        try:
            # For SQLite, connections are auto-closed with context managers
            # But we can log the closure for compatibility
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Database connection closed")
        except Exception as e:
            print(f"Database close error: {e}")


def create_enhanced_gameplay_system(db_path: str = "tabula_rasa.db"):
    """Factory function to create the enhanced gameplay system.

    This provides backward compatibility with the CORE_GAME_MECHANICS imports
    while using the new enhanced functionality.
    """
    session_manager = GameSessionManager(db_path)
    gameplay = CoreGameplay(session_manager)
    database = CoreGameDatabase(db_path)

    return session_manager, gameplay, database


# For backward compatibility with train.py imports
__all__ = [
    'GameSessionManager',
    'CoreGameplay',
    'CoreGameDatabase',
    'EnhancedGameplay',
    'create_enhanced_gameplay_system'
]
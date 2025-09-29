#!/usr/bin/env python3
"""
Stagnation Intervention System

This system detects when the AI is stuck in repetitive patterns without progress
and triggers intervention from the Governor, Director, and advanced systems.

Key Features:
- Detects frame stagnation (no changes for N frames)
- Detects coordinate repetition without progress
- Triggers Governor intervention for strategy changes
- Uses Director for high-level decision making
- Integrates GAN and pattern predictors for new approaches
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict

from src.database.system_integration import get_system_integration
from src.database.api import LogLevel, Component
from src.database.director_commands import get_director_commands

logger = logging.getLogger(__name__)

class StagnationType(Enum):
    """Types of stagnation detected."""
    FRAME_STAGNATION = "frame_stagnation"  # No frame changes
    COORDINATE_REPETITION = "coordinate_repetition"  # Same coordinates repeatedly
    SCORE_STAGNATION = "score_stagnation"  # No score increases
    ACTION_REPETITION = "action_repetition"  # Same actions repeatedly
    PATTERN_LOCK = "pattern_lock"  # Stuck in same pattern

@dataclass
class StagnationEvent:
    """Event indicating stagnation detected."""
    type: StagnationType
    severity: float  # 0.0 to 1.0
    duration: int  # Frames stuck
    consecutive_count: int  # Number of consecutive occurrences
    game_id: str  # Game ID where stagnation occurred
    session_id: str  # Session ID where stagnation occurred
    coordinates: List[Tuple[int, int]]
    actions: List[int]
    frame_hash: str
    score: float
    timestamp: float
    intervention_required: bool

class StagnationInterventionSystem:
    """Detects stagnation and triggers multi-system intervention."""
    
    def __init__(self):
        self.integration = get_system_integration()
        self.director = get_director_commands()
        self.logger = logging.getLogger(__name__)
        
        # Dynamic stagnation detection parameters (will be adjusted intelligently)
        self.base_frame_stagnation_threshold = 8  # Base threshold for frame stagnation
        self.base_coordinate_repetition_threshold = 4  # Base threshold for coordinate repetition
        self.base_score_stagnation_threshold = 15  # Base threshold for score stagnation
        self.base_action_repetition_threshold = 6  # Base threshold for action repetition
        
        # Current dynamic thresholds (will be adjusted based on context)
        self.frame_stagnation_threshold = self.base_frame_stagnation_threshold
        self.coordinate_repetition_threshold = self.base_coordinate_repetition_threshold
        self.score_stagnation_threshold = self.base_score_stagnation_threshold
        self.action_repetition_threshold = self.base_action_repetition_threshold
        
        # Intelligence tracking
        self.frame_change_frequency = 0.0  # How often frames change
        self.coordinate_diversity = 0.0  # How diverse coordinates are
        self.action_diversity = 0.0  # How diverse actions are
        self.learning_rate = 0.1  # How fast to adapt thresholds
        
        # History tracking
        self.frame_history = deque(maxlen=50)
        self.coordinate_history = deque(maxlen=20)
        self.action_history = deque(maxlen=20)
        self.score_history = deque(maxlen=30)
        
        # Stagnation state
        self.current_stagnation = None
        self.intervention_active = False
        self.last_frame_hash = None
        self.last_score = 0.0
        
        # Advanced systems integration
        self.gan_system = None
        self.pattern_predictor = None
        self.governor = None
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for intervention."""
        try:
            from src.core.enhanced_space_time_governor import EnhancedSpaceTimeGovernor
            from src.core.pattern_discovery_curiosity import create_pattern_discovery_curiosity
            # GANActionGenerator doesn't exist, so we'll skip it
            # from src.core.gan_action_generator import GANActionGenerator

            # Use singleton pattern to avoid duplicate instances
            from .enhanced_space_time_governor import get_enhanced_space_time_governor, create_enhanced_space_time_governor
            self.governor = get_enhanced_space_time_governor() or create_enhanced_space_time_governor()
            self.pattern_predictor = create_pattern_discovery_curiosity()
            self.gan_system = None  # Set to None since module doesn't exist
            
            self.logger.info("[OK] Advanced systems initialized for stagnation intervention")
        except ImportError as e:
            self.logger.warning(f"Some advanced systems not available: {e}")
            self.governor = None
            self.pattern_predictor = None
            self.gan_system = None
    
    async def analyze_frame(self, frame_data: List[List[int]], game_state: Dict[str, Any]) -> Optional[StagnationEvent]:
        """Analyze current frame for stagnation patterns."""
        try:
            # Add debugging for frame_data type and content
            self.logger.info(f"[CHECK] STAGNATION DEBUG: analyze_frame called with frame_data type: {type(frame_data)}")
            if hasattr(frame_data, 'shape'):
                self.logger.info(f"[CHECK] STAGNATION DEBUG: frame_data shape: {frame_data.shape}")
            elif isinstance(frame_data, list):
                self.logger.info(f"[CHECK] STAGNATION DEBUG: frame_data length: {len(frame_data)}")
            
            current_score = game_state.get('score', 0)
            current_actions = game_state.get('available_actions', [])
            game_id = game_state.get('game_id', 'unknown')
            session_id = game_state.get('session_id', 'unknown')
            
            # Calculate frame hash for change detection
            frame_hash = self._calculate_frame_hash(frame_data)
        except Exception as e:
            self.logger.error(f"[CHECK] STAGNATION DEBUG: Error in analyze_frame: {e}")
            import traceback
            self.logger.error(f"[CHECK] STAGNATION DEBUG: Traceback: {traceback.format_exc()}")
            raise
        
        # Update history
        self.frame_history.append(frame_hash)
        self.score_history.append(current_score)
        
        # Debug logging
        if len(self.frame_history) % 10 == 0:  # Log every 10 frames
            self.logger.info(f" STAGNATION DEBUG - Frame {len(self.frame_history)}:")
            self.logger.info(f"   Frame hash: {frame_hash[:20]}...")
            self.logger.info(f"   Score: {current_score}")
            self.logger.info(f"   Actions: {current_actions}")
            self.logger.info(f"   Frame change freq: {self.frame_change_frequency:.2f}")
            self.logger.info(f"   Coordinate diversity: {self.coordinate_diversity:.2f}")
            self.logger.info(f"   Action diversity: {self.action_diversity:.2f}")
        
        # Detect stagnation types
        stagnation_events = []
        
        # 1. Frame stagnation detection
        try:
            self.logger.info(f"[CHECK] STAGNATION DEBUG: About to call _detect_frame_stagnation")
            if self._detect_frame_stagnation():
                self.logger.info(f"[CHECK] STAGNATION DEBUG: _detect_frame_stagnation returned True")
                self.logger.warning(f" FRAME STAGNATION DETECTED!")
                stagnation_events.append(StagnationEvent(
                    type=StagnationType.FRAME_STAGNATION,
                    severity=self._calculate_frame_stagnation_severity(),
                    duration=len(self.frame_history),
                    consecutive_count=len(self.frame_history),
                    game_id=game_id,
                    session_id=session_id,
                    coordinates=list(self.coordinate_history),
                    actions=list(self.action_history),
                    frame_hash=frame_hash,
                    score=current_score,
                    timestamp=time.time(),
                    intervention_required=True
                ))
        except Exception as e:
            self.logger.error(f"[CHECK] STAGNATION DEBUG: Error in _detect_frame_stagnation: {e}")
            import traceback
            self.logger.error(f"[CHECK] STAGNATION DEBUG: Traceback: {traceback.format_exc()}")
            raise
        
        # 2. Score stagnation detection
        if self._detect_score_stagnation():
            stagnation_events.append(StagnationEvent(
                type=StagnationType.SCORE_STAGNATION,
                severity=self._calculate_score_stagnation_severity(),
                duration=len(self.score_history),
                consecutive_count=len(self.score_history),
                game_id=game_id,
                session_id=session_id,
                coordinates=list(self.coordinate_history),
                actions=list(self.action_history),
                frame_hash=frame_hash,
                score=current_score,
                timestamp=time.time(),
                intervention_required=True
            ))
        
        # 3. Coordinate repetition detection
        if self._detect_coordinate_repetition():
            stagnation_events.append(StagnationEvent(
                type=StagnationType.COORDINATE_REPETITION,
                severity=self._calculate_coordinate_repetition_severity(),
                duration=len(self.coordinate_history),
                consecutive_count=len(self.coordinate_history),
                game_id=game_id,
                session_id=session_id,
                coordinates=list(self.coordinate_history),
                actions=list(self.action_history),
                frame_hash=frame_hash,
                score=current_score,
                timestamp=time.time(),
                intervention_required=True
            ))
        
        # 4. Action repetition detection (excluding Action 6)
        if self._detect_action_repetition():
            stagnation_events.append(StagnationEvent(
                type=StagnationType.ACTION_REPETITION,
                severity=self._calculate_action_repetition_severity(),
                duration=len(self.action_history),
                consecutive_count=len(self.action_history),
                game_id=game_id,
                session_id=session_id,
                coordinates=list(self.coordinate_history),
                actions=list(self.action_history),
                frame_hash=frame_hash,
                score=current_score,
                timestamp=time.time(),
                intervention_required=True
            ))
        
        # 5. Action 6 coordinate stagnation detection (special case)
        if self._detect_action6_coordinate_stagnation():
            stagnation_events.append(StagnationEvent(
                type=StagnationType.COORDINATE_REPETITION,  # Use coordinate repetition type
                severity=self._calculate_action6_coordinate_severity(),
                duration=len(self.coordinate_history),
                consecutive_count=len(self.coordinate_history),
                game_id=game_id,
                session_id=session_id,
                coordinates=list(self.coordinate_history),
                actions=[6] * len(self.coordinate_history),  # All Action 6
                frame_hash=frame_hash,
                score=current_score,
                timestamp=time.time(),
                intervention_required=True
            ))
        
        # Return most severe stagnation event
        if stagnation_events:
            most_severe = max(stagnation_events, key=lambda e: e.severity)
            self.current_stagnation = most_severe

            # Log stagnation detection to database
            self._log_stagnation_detection_to_database(most_severe, stagnation_events)

            return most_severe
        
        # Reset stagnation if no issues detected
        if self.current_stagnation and self._is_stagnation_resolved():
            self.logger.info(" STAGNATION RESOLVED - System back to normal operation")
            self.current_stagnation = None
            self.intervention_active = False
        
        return None
    
    def record_action(self, action: Dict[str, Any], coordinates: Optional[Tuple[int, int]] = None):
        """Record action for stagnation analysis with intelligent tracking."""
        action_id = action.get('id', 0)
        
        # Special handling for Action 6 - track coordinates, not the action itself
        if action_id == 6 and coordinates:
            # For Action 6, we track coordinates and treat it as coordinate-based
            self.coordinate_history.append(coordinates)
            # Don't add Action 6 to action_history since it's coordinate-based
            # Instead, we'll track it separately for coordinate analysis
        else:
            # For other actions, track normally
            self.action_history.append(action_id)
        
        # Update intelligence metrics
        self._update_intelligence_metrics()
        
        # Adapt thresholds based on current behavior
        self._adapt_thresholds()
    
    def _calculate_frame_hash(self, frame_data: List[List[int]]) -> str:
        """Calculate hash of frame for change detection."""
        try:
            self.logger.info(f"[CHECK] STAGNATION DEBUG: _calculate_frame_hash called with frame_data type: {type(frame_data)}")
            if hasattr(frame_data, 'shape'):
                self.logger.info(f"[CHECK] STAGNATION DEBUG: frame_data shape: {frame_data.shape}")
            elif isinstance(frame_data, list):
                self.logger.info(f"[CHECK] STAGNATION DEBUG: frame_data length: {len(frame_data)}")
            
            # Convert to numpy array and flatten for hashing
            frame_array = np.array(frame_data)
            self.logger.info(f"[CHECK] STAGNATION DEBUG: frame_array type: {type(frame_array)}, shape: {frame_array.shape}")
            
            # Use a more robust hashing method to avoid ambiguous truth value errors
            frame_bytes = frame_array.tobytes()
            return str(hash(frame_bytes))
        except Exception as e:
            self.logger.error(f"[CHECK] STAGNATION DEBUG: Error in _calculate_frame_hash: {e}")
            import traceback
            self.logger.error(f"[CHECK] STAGNATION DEBUG: Traceback: {traceback.format_exc()}")
            # Fallback to string representation
            try:
                return str(hash(str(frame_data)))
            except:
                # Ultimate fallback
                return str(time.time())
    
    def _detect_frame_stagnation(self) -> bool:
        """Detect if frames have been stagnant."""
        try:
            self.logger.info(f"[CHECK] STAGNATION DEBUG: _detect_frame_stagnation called")
            self.logger.info(f"[CHECK] STAGNATION DEBUG: frame_history length: {len(self.frame_history)}")
            self.logger.info(f"[CHECK] STAGNATION DEBUG: frame_stagnation_threshold: {self.frame_stagnation_threshold}")
            
            if len(self.frame_history) < self.frame_stagnation_threshold:
                self.logger.info(f"[CHECK] STAGNATION DEBUG: Not enough frames for stagnation detection")
                return False
            
            # Check if last N frames are identical
            recent_frames = list(self.frame_history)[-self.frame_stagnation_threshold:]
            self.logger.info(f"[CHECK] STAGNATION DEBUG: recent_frames length: {len(recent_frames)}")
            self.logger.info(f"[CHECK] STAGNATION DEBUG: recent_frames types: {[type(f) for f in recent_frames[:3]]}")
            
            # Use string comparison to avoid array ambiguity issues
            try:
                self.logger.info(f"[CHECK] STAGNATION DEBUG: About to convert frames to strings")
                frame_strings = [str(f) for f in recent_frames]
                self.logger.info(f"[CHECK] STAGNATION DEBUG: Converted to strings successfully")
                
                self.logger.info(f"[CHECK] STAGNATION DEBUG: About to create set")
                unique_frames = len(set(frame_strings))
                self.logger.info(f"[CHECK] STAGNATION DEBUG: Set created successfully, unique_frames: {unique_frames}")
                
                is_stagnant = unique_frames == 1
                self.logger.info(f" STAGNATION DEBUG: is_stagnant: {is_stagnant}")
                
            except Exception as e:
                self.logger.error(f" STAGNATION DEBUG: Error in string conversion: {e}")
                import traceback
                self.logger.error(f" STAGNATION DEBUG: Traceback: {traceback.format_exc()}")
                # Fallback to simple comparison
                is_stagnant = all(str(recent_frames[0]) == str(f) for f in recent_frames)
            
            if is_stagnant:
                self.logger.warning(f" FRAME STAGNATION DETECTED:")
                self.logger.warning(f"   Threshold: {self.frame_stagnation_threshold}")
                self.logger.warning(f"   Recent frames: {len(recent_frames)}")
                self.logger.warning(f"   Unique frames: {unique_frames}")
                self.logger.warning(f"   Frame hashes: {recent_frames[:3]}...")
            
            return is_stagnant
            
        except Exception as e:
            self.logger.error(f" STAGNATION DEBUG: Error in _detect_frame_stagnation: {e}")
            import traceback
            self.logger.error(f" STAGNATION DEBUG: Traceback: {traceback.format_exc()}")
            raise
    
    def _detect_score_stagnation(self) -> bool:
        """Detect if score has been stagnant."""
        if len(self.score_history) < self.score_stagnation_threshold:
            return False
        
        # Check if score hasn't increased
        recent_scores = list(self.score_history)[-self.score_stagnation_threshold:]
        return all(score == recent_scores[0] for score in recent_scores)
    
    def _detect_coordinate_repetition(self) -> bool:
        """Detect if coordinates are being repeated excessively (intelligent detection)."""
        if len(self.coordinate_history) < self.coordinate_repetition_threshold:
            return False
        
        # Check if same coordinate appears too frequently
        recent_coords = list(self.coordinate_history)[-self.coordinate_repetition_threshold:]
        coord_counts = defaultdict(int)
        for coord in recent_coords:
            coord_counts[coord] += 1
        
        # Intelligent detection: consider both frequency and diversity
        max_repetition = max(coord_counts.values()) if coord_counts else 0
        unique_coords = len(coord_counts)
        
        # If we have very low diversity AND high repetition, it's stagnation
        diversity_ratio = unique_coords / len(recent_coords)
        repetition_ratio = max_repetition / len(recent_coords)
        
        # Stagnation if: high repetition AND low diversity
        is_stagnant = (repetition_ratio > 0.6) and (diversity_ratio < 0.4)
        
        if is_stagnant:
            self.logger.warning(f" COORDINATE STAGNATION DETECTED:")
            self.logger.warning(f"   Max repetition: {max_repetition}/{len(recent_coords)}")
            self.logger.warning(f"   Diversity ratio: {diversity_ratio:.2f}")
            self.logger.warning(f"   Repetition ratio: {repetition_ratio:.2f}")
        
        return is_stagnant
    
    def _detect_action_repetition(self) -> bool:
        """Detect if actions are being repeated excessively (excluding Action 6)."""
        if len(self.action_history) < self.action_repetition_threshold:
            return False
        
        # Check if same action appears too frequently (Action 6 is handled separately)
        recent_actions = list(self.action_history)[-self.action_repetition_threshold:]
        action_counts = defaultdict(int)
        for action in recent_actions:
            action_counts[action] += 1
        
        # Intelligent detection: consider both frequency and diversity
        max_repetition = max(action_counts.values()) if action_counts else 0
        unique_actions = len(action_counts)
        
        # If we have very low diversity AND high repetition, it's stagnation
        diversity_ratio = unique_actions / len(recent_actions)
        repetition_ratio = max_repetition / len(recent_actions)
        
        # Stagnation if: high repetition AND low diversity
        is_stagnant = (repetition_ratio > 0.7) and (diversity_ratio < 0.5)
        
        if is_stagnant:
            self.logger.warning(f" ACTION STAGNATION DETECTED:")
            self.logger.warning(f"   Max repetition: {max_repetition}/{len(recent_actions)}")
            self.logger.warning(f"   Diversity ratio: {diversity_ratio:.2f}")
            self.logger.warning(f"   Repetition ratio: {repetition_ratio:.2f}")
        
        return is_stagnant
    
    def _detect_action6_coordinate_stagnation(self) -> bool:
        """Detect if Action 6 coordinates are being repeated excessively."""
        if len(self.coordinate_history) < 3:  # Need at least 3 coordinates to detect stagnation
            return False
        
        # Check if same coordinates are being repeated without progress
        recent_coords = list(self.coordinate_history)[-6:]  # Last 6 coordinates
        coord_counts = defaultdict(int)
        for coord in recent_coords:
            coord_counts[coord] += 1
        
        # Intelligent detection for Action 6 coordinates
        max_repetition = max(coord_counts.values()) if coord_counts else 0
        unique_coords = len(coord_counts)
        
        # Action 6 stagnation if: high repetition AND low diversity AND no frame changes
        diversity_ratio = unique_coords / len(recent_coords)
        repetition_ratio = max_repetition / len(recent_coords)
        
        # More sensitive for Action 6 since it's coordinate-based
        is_stagnant = (repetition_ratio > 0.5) and (diversity_ratio < 0.5)
        
        if is_stagnant:
            self.logger.warning(f" ACTION 6 COORDINATE STAGNATION DETECTED:")
            self.logger.warning(f"   Max repetition: {max_repetition}/{len(recent_coords)}")
            self.logger.warning(f"   Diversity ratio: {diversity_ratio:.2f}")
            self.logger.warning(f"   Repetition ratio: {repetition_ratio:.2f}")
            self.logger.warning(f"   Recent coordinates: {recent_coords}")
        
        return is_stagnant
    
    def _calculate_action6_coordinate_severity(self) -> float:
        """Calculate severity of Action 6 coordinate stagnation."""
        if len(self.coordinate_history) < 3:
            return 0.0
        
        recent_coords = list(self.coordinate_history)[-6:]
        coord_counts = defaultdict(int)
        for coord in recent_coords:
            coord_counts[coord] += 1
        
        max_repetition = max(coord_counts.values()) if coord_counts else 0
        repetition_ratio = max_repetition / len(recent_coords)
        
        # Higher severity for Action 6 coordinate stagnation
        return min(1.0, repetition_ratio * 1.5)
    
    def _calculate_frame_stagnation_severity(self) -> float:
        """Calculate severity of frame stagnation."""
        stagnant_frames = 0
        for i in range(len(self.frame_history) - 1, 0, -1):
            # Use string comparison to avoid array ambiguity issues
            if str(self.frame_history[i]) == str(self.frame_history[i-1]):
                stagnant_frames += 1
            else:
                break
        
        return min(1.0, stagnant_frames / self.frame_stagnation_threshold)
    
    def _calculate_score_stagnation_severity(self) -> float:
        """Calculate severity of score stagnation."""
        stagnant_frames = 0
        initial_score = self.score_history[-self.score_stagnation_threshold]
        
        for score in reversed(list(self.score_history)[-self.score_stagnation_threshold:]):
            if score == initial_score:
                stagnant_frames += 1
            else:
                break
        
        return min(1.0, stagnant_frames / self.score_stagnation_threshold)
    
    def _calculate_coordinate_repetition_severity(self) -> float:
        """Calculate severity of coordinate repetition."""
        recent_coords = list(self.coordinate_history)[-self.coordinate_repetition_threshold:]
        coord_counts = defaultdict(int)
        for coord in recent_coords:
            coord_counts[coord] += 1
        
        max_repetition = max(coord_counts.values()) if coord_counts else 0
        return min(1.0, max_repetition / self.coordinate_repetition_threshold)
    
    def _calculate_action_repetition_severity(self) -> float:
        """Calculate severity of action repetition."""
        recent_actions = list(self.action_history)[-self.action_repetition_threshold:]
        action_counts = defaultdict(int)
        for action in recent_actions:
            action_counts[action] += 1
        
        max_repetition = max(action_counts.values()) if action_counts else 0
        return min(1.0, max_repetition / self.action_repetition_threshold)
    
    def _is_stagnation_resolved(self) -> bool:
        """Check if stagnation has been resolved."""
        # Check if recent frames show changes
        if len(self.frame_history) >= 3:
            recent_frames = list(self.frame_history)[-3:]
            # Use string conversion to avoid array ambiguity issues
            if len(set(str(f) for f in recent_frames)) > 1:
                return True
        
        # Check if score has increased
        if len(self.score_history) >= 3:
            recent_scores = list(self.score_history)[-3:]
            if recent_scores[-1] > recent_scores[0]:
                return True
        
        return False
    
    def _update_intelligence_metrics(self):
        """Update intelligence metrics for dynamic threshold adaptation."""
        try:
            self.logger.info(f" STAGNATION DEBUG: About to call _update_intelligence_metrics")
            # Calculate frame change frequency
            if len(self.frame_history) >= 10:
                recent_frames = list(self.frame_history)[-10:]
                # Use string conversion to avoid array ambiguity issues
                unique_frames = len(set(str(f) for f in recent_frames))
                self.frame_change_frequency = unique_frames / len(recent_frames)
        except Exception as e:
            self.logger.error(f" STAGNATION DEBUG: Error in _update_intelligence_metrics: {e}")
            import traceback
            self.logger.error(f" STAGNATION DEBUG: Traceback: {traceback.format_exc()}")
            raise
        
        # Calculate coordinate diversity
        if len(self.coordinate_history) >= 5:
            recent_coords = list(self.coordinate_history)[-5:]
            unique_coords = len(set(recent_coords))
            self.coordinate_diversity = unique_coords / len(recent_coords)
        
        # Calculate action diversity (excluding Action 6)
        if len(self.action_history) >= 5:
            recent_actions = list(self.action_history)[-5:]
            unique_actions = len(set(recent_actions))
            self.action_diversity = unique_actions / len(recent_actions)
    
    def _adapt_thresholds(self):
        """Dynamically adapt thresholds based on current behavior patterns."""
        try:
            self.logger.info(f" STAGNATION DEBUG: About to call _adapt_thresholds")
            # Adapt frame stagnation threshold based on frame change frequency
            if self.frame_change_frequency > 0.5:  # High frame change frequency
                # System is active, can tolerate more frames without changes
                self.frame_stagnation_threshold = min(
                    self.base_frame_stagnation_threshold * 1.5,
                    self.frame_stagnation_threshold + self.learning_rate
                )
            elif self.frame_change_frequency < 0.2:  # Low frame change frequency
                # System is stagnant, be more sensitive
                self.frame_stagnation_threshold = max(
                    self.base_frame_stagnation_threshold * 0.7,
                    self.frame_stagnation_threshold - self.learning_rate
                )
        
            # Adapt coordinate repetition threshold based on coordinate diversity
            if self.coordinate_diversity > 0.6:  # High coordinate diversity
                # System is exploring well, can tolerate some repetition
                self.coordinate_repetition_threshold = min(
                    self.base_coordinate_repetition_threshold * 1.3,
                    self.coordinate_repetition_threshold + self.learning_rate
                )
            elif self.coordinate_diversity < 0.3:  # Low coordinate diversity
                # System is stuck in same coordinates, be more sensitive
                self.coordinate_repetition_threshold = max(
                    self.base_coordinate_repetition_threshold * 0.6,
                    self.coordinate_repetition_threshold - self.learning_rate
                )
        
            # Adapt action repetition threshold based on action diversity
            if self.action_diversity > 0.7:  # High action diversity
                # System is using diverse actions, can tolerate some repetition
                self.action_repetition_threshold = min(
                    self.base_action_repetition_threshold * 1.2,
                    self.action_repetition_threshold + self.learning_rate
                )
            elif self.action_diversity < 0.4:  # Low action diversity
                # System is stuck in same actions, be more sensitive
                self.action_repetition_threshold = max(
                    self.base_action_repetition_threshold * 0.7,
                    self.action_repetition_threshold - self.learning_rate
                )
        
            # Log threshold changes for debugging
            if hasattr(self, '_last_logged_thresholds'):
                if (abs(self.frame_stagnation_threshold - self._last_logged_thresholds[0]) > 0.1 or
                    abs(self.coordinate_repetition_threshold - self._last_logged_thresholds[1]) > 0.1 or
                    abs(self.action_repetition_threshold - self._last_logged_thresholds[2]) > 0.1):
                    
                    self.logger.info(f" INTELLIGENT THRESHOLDS ADAPTED:")
                    self.logger.info(f"   Frame stagnation: {self.frame_stagnation_threshold:.1f}")
                    self.logger.info(f"   Coordinate repetition: {self.coordinate_repetition_threshold:.1f}")
                    self.logger.info(f"   Action repetition: {self.action_repetition_threshold:.1f}")
                    self.logger.info(f"   Frame change freq: {self.frame_change_frequency:.2f}")
                    self.logger.info(f"   Coordinate diversity: {self.coordinate_diversity:.2f}")
                    self.logger.info(f"   Action diversity: {self.action_diversity:.2f}")
        except Exception as e:
            self.logger.error(f" STAGNATION DEBUG: Error in _adapt_thresholds: {e}")
            import traceback
            self.logger.error(f" STAGNATION DEBUG: Traceback: {traceback.format_exc()}")
            raise
        
        self._last_logged_thresholds = (
            self.frame_stagnation_threshold,
            self.coordinate_repetition_threshold,
            self.action_repetition_threshold
        )
    
    async def trigger_intervention(self, stagnation_event: StagnationEvent) -> Dict[str, Any]:
        """Trigger multi-system intervention for stagnation."""
        self.logger.warning(f" STAGNATION INTERVENTION TRIGGERED: {stagnation_event.type.value}")
        self.logger.warning(f"   Severity: {stagnation_event.severity:.2f}")
        self.logger.warning(f"   Duration: {stagnation_event.duration} frames")
        
        self.intervention_active = True
        
        # 1. Governor intervention - Strategy change
        governor_recommendation = await self._get_governor_recommendation(stagnation_event)
        
        # 2. Director intervention - High-level strategy
        director_strategy = await self._get_director_strategy(stagnation_event)
        
        # 3. GAN intervention - Generate new approaches
        gan_suggestions = await self._get_gan_suggestions(stagnation_event)
        
        # 4. Pattern predictor intervention - Find new patterns
        pattern_suggestions = await self._get_pattern_suggestions(stagnation_event)
        
        # 5. Emergency override - Force different actions
        emergency_actions = self._get_emergency_actions(stagnation_event)
        
        intervention = {
            'type': 'stagnation_intervention',
            'stagnation_event': stagnation_event,
            'governor_recommendation': governor_recommendation,
            'director_strategy': director_strategy,
            'gan_suggestions': gan_suggestions,
            'pattern_suggestions': pattern_suggestions,
            'emergency_actions': emergency_actions,
            'timestamp': time.time()
        }
        
        # Log intervention
        await self._log_intervention(intervention)

        # Log intervention to database
        await self._log_intervention_to_database(intervention, stagnation_event)

        return intervention
    
    async def _get_governor_recommendation(self, stagnation_event: StagnationEvent) -> Dict[str, Any]:
        """Get Governor recommendation for stagnation."""
        if not self.governor:
            return {'error': 'Governor not available'}
        
        try:
            # Analyze current state
            context = {
                'stagnation_type': stagnation_event.type.value,
                'severity': stagnation_event.severity,
                'duration': stagnation_event.duration,
                'coordinates': stagnation_event.coordinates,
                'actions': stagnation_event.actions,
                'score': stagnation_event.score
            }
            
            # Get Governor recommendation
            try:
                recommendation = await self.governor.analyze_and_recommend(context)
            except AttributeError:
                # Fallback if method doesn't exist
                recommendation = type('Recommendation', (), {
                    'type': type('Type', (), {'value': 'exploration'})()
                })()
            
            return {
                'recommendation_type': recommendation.type.value if hasattr(recommendation, 'type') else 'unknown',
                'configuration_changes': recommendation.configuration_changes if hasattr(recommendation, 'configuration_changes') else {},
                'confidence': recommendation.confidence if hasattr(recommendation, 'confidence') else 0.0,
                'reasoning': recommendation.reasoning if hasattr(recommendation, 'reasoning') else 'No reasoning provided'
            }
        except Exception as e:
            self.logger.error(f"Error getting Governor recommendation: {e}")
            return {'error': str(e)}
    
    async def _get_director_strategy(self, stagnation_event: StagnationEvent) -> Dict[str, Any]:
        """Get Director strategy for stagnation."""
        try:
            # Get system overview
            system_overview = await self.director.get_system_overview()
            
            # Get learning analysis
            learning_analysis = await self.director.get_learning_analysis()
            
            # Generate strategy based on stagnation type
            # Defensive: ensure actions/coordinates are lists (avoid numpy ambiguous truth values)
            actions_list = list(stagnation_event.actions) if (stagnation_event.actions is not None and getattr(stagnation_event.actions, '__len__', None) is not None) else []
            coords_list = list(stagnation_event.coordinates) if (stagnation_event.coordinates is not None and getattr(stagnation_event.coordinates, '__len__', None) is not None) else []

            strategy = {
                'strategy_type': 'stagnation_break',
                'focus': self._get_strategy_focus(stagnation_event),
                'actions_to_avoid': actions_list[-5:],
                'coordinates_to_avoid': coords_list[-3:],
                'new_approach': self._get_new_approach(stagnation_event),
                'system_health': system_overview.get('health_score', 0.0),
                'learning_insights': learning_analysis.get('key_insights', [])
            }
            
            return strategy
        except Exception as e:
            self.logger.error(f"Error getting Director strategy: {e}")
            return {'error': str(e)}
    
    async def _get_gan_suggestions(self, stagnation_event: StagnationEvent) -> List[Dict[str, Any]]:
        """Get GAN-generated action suggestions."""
        if not self.gan_system:
            return []
        
        try:
            # Generate new action suggestions using GAN
            suggestions = await self.gan_system.generate_actions(
                context={
                    'stagnation_type': stagnation_event.type.value,
                    'avoid_actions': stagnation_event.actions,
                    'avoid_coordinates': stagnation_event.coordinates,
                    'current_score': stagnation_event.score
                },
                num_suggestions=5
            )
            
            return suggestions
        except Exception as e:
            self.logger.error(f"Error getting GAN suggestions: {e}")
            return []
    
    async def _get_pattern_suggestions(self, stagnation_event: StagnationEvent) -> List[Dict[str, Any]]:
        """Get pattern predictor suggestions."""
        if not self.pattern_predictor:
            return []
        
        try:
            # Generate pattern-based suggestions
            try:
                suggestions = await self.pattern_predictor.discover_patterns(
                    context={
                        'stagnation_type': stagnation_event.type.value,
                        'avoid_patterns': stagnation_event.actions,
                        'current_score': stagnation_event.score
                    }
                )
                return suggestions
            except AttributeError:
                # Fallback if method doesn't exist
                return []
        except Exception as e:
            self.logger.error(f"Error getting pattern suggestions: {e}")
            return []
    
    def _get_emergency_actions(self, stagnation_event: StagnationEvent) -> List[Dict[str, Any]]:
        """Get emergency actions to break stagnation."""
        emergency_actions = []
        
        # Force different action types
        # Defensive conversion to list to avoid numpy ambiguous truth-tests or membership issues
        actions_list = list(stagnation_event.actions) if (stagnation_event.actions is not None and getattr(stagnation_event.actions, '__len__', None) is not None) else []

        if stagnation_event.type == StagnationType.ACTION_REPETITION:
            # Suggest different actions
            for action_id in [1, 2, 3, 4, 5, 7, 8, 9, 10]:
                if action_id not in actions_list[-5:]:
                    emergency_actions.append({
                        'action': f'ACTION{action_id}',
                        'id': action_id,
                        'confidence': 0.8,
                        'reason': f'Emergency action to break repetition - Action {action_id}',
                        'source': 'emergency_override'
                    })
        
        # Force different coordinates
        if stagnation_event.type == StagnationType.COORDINATE_REPETITION:
            # Suggest random coordinates
            coords_list = list(stagnation_event.coordinates) if (stagnation_event.coordinates is not None and getattr(stagnation_event.coordinates, '__len__', None) is not None) else []
            recent_coords = coords_list[-3:]
            for _ in range(3):
                x = np.random.randint(0, 50)
                y = np.random.randint(0, 50)
                if (x, y) not in recent_coords:
                    emergency_actions.append({
                        'action': 'ACTION6',
                        'id': 6,
                        'x': x,
                        'y': y,
                        'confidence': 0.7,
                        'reason': f'Emergency coordinate to break repetition - ({x}, {y})',
                        'source': 'emergency_override'
                    })
        
        return emergency_actions
    
    def _get_strategy_focus(self, stagnation_event: StagnationEvent) -> str:
        """Get strategy focus based on stagnation type."""
        if stagnation_event.type == StagnationType.FRAME_STAGNATION:
            return "Force frame changes through different actions"
        elif stagnation_event.type == StagnationType.SCORE_STAGNATION:
            return "Focus on score-increasing actions and coordinates"
        elif stagnation_event.type == StagnationType.COORDINATE_REPETITION:
            return "Explore new coordinates and avoid repeated ones"
        elif stagnation_event.type == StagnationType.ACTION_REPETITION:
            return "Use different action types and sequences"
        else:
            return "General stagnation break strategy"
    
    def _get_new_approach(self, stagnation_event: StagnationEvent) -> str:
        """Get new approach description."""
        if stagnation_event.type == StagnationType.FRAME_STAGNATION:
            return "Try movement actions (1-4) to change frame state"
        elif stagnation_event.type == StagnationType.SCORE_STAGNATION:
            return "Focus on Action 6 with unexplored coordinates"
        elif stagnation_event.type == StagnationType.COORDINATE_REPETITION:
            return "Systematically explore new coordinate areas"
        elif stagnation_event.type == StagnationType.ACTION_REPETITION:
            return "Use action sequence variation and exploration"
        else:
            return "Multi-system intervention approach"
    
    def _log_stagnation_detection_to_database(self, stagnation_event: StagnationEvent, all_events: List[StagnationEvent]):
        """Log stagnation detection to database."""
        try:
            import asyncio
            from src.database.system_integration import get_system_integration

            async def log_stagnation():
                integration = get_system_integration()

                stagnation_data = {
                    'stagnation_type': stagnation_event.type.value,
                    'severity': stagnation_event.severity,
                    'duration': stagnation_event.duration,
                    'coordinates': list(stagnation_event.coordinates) if stagnation_event.coordinates else None,
                    'total_events_detected': len(all_events),
                    'intervention_required': stagnation_event.intervention_required,
                    'frame_count': len(self.frame_history),
                    'component': 'stagnation_detector'
                }

                await integration.log_system_event(
                    level="WARNING",
                    component="stagnation_intervention_system",
                    message=f"Stagnation detected: {stagnation_event.type.value}",
                    data=stagnation_data,
                    session_id='unknown'  # TODO: Add session tracking
                )

            # Schedule the database logging
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(log_stagnation())

        except Exception as e:
            self.logger.debug(f"Non-fatal: Failed to log stagnation detection to database: {e}")

    async def _log_intervention_to_database(self, intervention: Dict[str, Any], stagnation_event: StagnationEvent):
        """Log intervention details to database."""
        try:
            from src.database.system_integration import get_system_integration

            integration = get_system_integration()

            intervention_data = {
                'intervention_type': 'stagnation_intervention',
                'stagnation_type': stagnation_event.type.value,
                'severity': stagnation_event.severity,
                'governor_recommendation': intervention.get('governor_recommendation', {}),
                'director_strategy': intervention.get('director_strategy', {}),
                'gan_suggestions': intervention.get('gan_suggestions', {}),
                'pattern_suggestions': intervention.get('pattern_suggestions', {}),
                'emergency_actions': intervention.get('emergency_actions', {}),
                'component': 'intervention_system'
            }

            await integration.log_system_event(
                level="ERROR",  # Use ERROR level for interventions as they indicate problems
                component="stagnation_intervention_system",
                message=f"Intervention triggered for {stagnation_event.type.value}",
                data=intervention_data,
                session_id='unknown'  # TODO: Add session tracking
            )

        except Exception as e:
            self.logger.debug(f"Non-fatal: Failed to log intervention to database: {e}")

    async def _log_intervention(self, intervention: Dict[str, Any]):
        """Log intervention to database."""
        try:
            # Convert StagnationEvent to serializable format
            serializable_intervention = {
                'stagnation_event': {
                    'type': intervention['stagnation_event'].type.value,
                    'severity': intervention['stagnation_event'].severity,
                    'consecutive_count': intervention['stagnation_event'].consecutive_count,
                    'game_id': intervention['stagnation_event'].game_id,
                    'session_id': intervention['stagnation_event'].session_id
                },
                'intervention_type': intervention.get('intervention_type', 'unknown'),
                'actions': intervention.get('actions', []),
                'confidence': intervention.get('confidence', 0.0)
            }
            
            await self.integration.log_system_event(
                level=LogLevel.WARNING,
                component=Component.SUBSYSTEM_MONITOR,
                message=f"Stagnation intervention triggered: {intervention['stagnation_event'].type.value}",
                data=serializable_intervention
            )
        except Exception as e:
            self.logger.error(f"Error logging intervention: {e}")
    
    def is_intervention_active(self) -> bool:
        """Check if intervention is currently active."""
        return self.intervention_active
    
    def get_intervention_status(self) -> Dict[str, Any]:
        """Get current intervention status."""
        return {
            'intervention_active': self.intervention_active,
            'current_stagnation': self.current_stagnation.type.value if self.current_stagnation else None,
            'stagnation_severity': self.current_stagnation.severity if self.current_stagnation else 0.0,
            'frame_history_length': len(self.frame_history),
            'coordinate_history_length': len(self.coordinate_history),
            'action_history_length': len(self.action_history),
            'score_history_length': len(self.score_history)
        }

# Global singleton instance
_stagnation_intervention_instance: Optional[StagnationInterventionSystem] = None

def create_stagnation_intervention_system() -> StagnationInterventionSystem:
    """Create or get the singleton StagnationInterventionSystem instance."""
    global _stagnation_intervention_instance
    if _stagnation_intervention_instance is None:
        logger.debug("Creating singleton StagnationInterventionSystem instance")
        _stagnation_intervention_instance = StagnationInterventionSystem()
    return _stagnation_intervention_instance

def get_stagnation_intervention_system() -> Optional[StagnationInterventionSystem]:
    """Get the existing StagnationInterventionSystem singleton instance."""
    return _stagnation_intervention_instance

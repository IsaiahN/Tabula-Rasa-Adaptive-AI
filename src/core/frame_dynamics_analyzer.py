"""
Frame Dynamics Analyzer for ARC-AGI-3
Implements advanced frame sequence analysis to understand game physics and dynamics.
Generates strategic inferences for the Architect and Governor systems.
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import time
from datetime import datetime

class PhysicsType(Enum):
    """Types of physics behaviors detected in the game."""
    GRAVITY = "gravity"
    MOMENTUM = "momentum"
    COLLISION = "collision"
    ROTATION = "rotation"
    SCALING = "scaling"
    TRANSLATION = "translation"
    COLOR_TRANSITION = "color_transition"
    SHAPE_MORPHING = "shape_morphing"
    PARTICLE_SYSTEM = "particle_system"
    WAVE_PROPAGATION = "wave_propagation"

class GameMechanic(Enum):
    """Game mechanics inferred from frame analysis."""
    PUZZLE_SOLVING = "puzzle_solving"
    PATTERN_MATCHING = "pattern_matching"
    OBJECT_MANIPULATION = "object_manipulation"
    SPATIAL_REASONING = "spatial_reasoning"
    TEMPORAL_SEQUENCING = "temporal_sequencing"
    CAUSAL_CHAINING = "causal_chaining"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"

@dataclass
class PhysicsInference:
    """Represents a physics-based inference about game behavior."""
    physics_type: PhysicsType
    confidence: float
    evidence: List[str]
    spatial_region: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    temporal_span: int  # Number of frames
    symbolic_reference: str  # Human-readable description
    strategic_implication: str  # How this affects gameplay

@dataclass
class GameMechanicInference:
    """Represents a game mechanic inference."""
    mechanic: GameMechanic
    confidence: float
    evidence: List[str]
    spatial_region: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    temporal_span: int  # Number of frames
    symbolic_reference: str  # Human-readable description
    required_actions: List[int]  # Action types needed
    success_criteria: List[str]  # What constitutes success
    strategic_guidance: str  # How to approach this mechanic

@dataclass
class FrameSequence:
    """Represents a sequence of frames for analysis."""
    frames: List[np.ndarray]
    timestamps: List[float]
    game_id: str
    action_history: List[Tuple[int, Tuple[int, int]]]  # (action, coordinates)

class FrameDynamicsAnalyzer:
    """
    Advanced frame dynamics analyzer that understands game physics and mechanics.
    Generates strategic inferences for the Architect and Governor systems.
    """
    
    def __init__(self, max_sequence_length: int = 20):
        self.max_sequence_length = max_sequence_length
        self.frame_sequences: Dict[str, FrameSequence] = {}
        self.physics_inferences: Dict[str, List[PhysicsInference]] = defaultdict(list)
        self.mechanic_inferences: Dict[str, List[GameMechanicInference]] = defaultdict(list)
        self.symbolic_references: Dict[str, str] = {}  # game_id -> symbolic description
        self.logger = logging.getLogger(__name__)
        
        # Physics detection parameters
        self.movement_threshold = 0.1
        self.color_change_threshold = 0.2
        self.shape_change_threshold = 0.15
        
        # External knowledge base for physics inference
        self.physics_knowledge = self._initialize_physics_knowledge()
        
    def _initialize_physics_knowledge(self) -> Dict[str, Any]:
        """Initialize external physics knowledge for inference."""
        return {
            "gravity_patterns": {
                "downward_acceleration": "Objects fall with increasing speed",
                "terminal_velocity": "Objects reach maximum falling speed",
                "bounce_behavior": "Objects bounce off surfaces"
            },
            "momentum_patterns": {
                "conservation": "Objects maintain velocity unless acted upon",
                "collision_transfer": "Momentum transfers between colliding objects",
                "friction_effects": "Objects slow down over time"
            },
            "color_transitions": {
                "gradient_changes": "Colors change smoothly over time",
                "discrete_switches": "Colors change instantly",
                "conditional_changes": "Colors change based on conditions"
            },
            "shape_transformations": {
                "scaling": "Objects grow or shrink",
                "rotation": "Objects rotate around center",
                "morphing": "Objects change shape completely"
            }
        }
    
    def analyze_frame_sequence(self, frames: List[np.ndarray], 
                             timestamps: List[float],
                             game_id: str,
                             action_history: List[Tuple[int, Tuple[int, int]]]) -> Dict[str, Any]:
        """
        Analyze a sequence of frames to understand game dynamics.
        
        Args:
            frames: List of frame arrays
            timestamps: Corresponding timestamps
            game_id: Game identifier
            action_history: History of actions taken
            
        Returns:
            Dictionary containing physics inferences and strategic guidance
        """
        if len(frames) < 2:
            return {"error": "Need at least 2 frames for sequence analysis"}
        
        # Store frame sequence
        sequence = FrameSequence(frames, timestamps, game_id, action_history)
        self.frame_sequences[game_id] = sequence
        
        # Analyze physics patterns
        physics_inferences = self._analyze_physics_patterns(sequence)
        self.physics_inferences[game_id].extend(physics_inferences)
        
        # Analyze game mechanics
        mechanic_inferences = self._analyze_game_mechanics(sequence)
        self.mechanic_inferences[game_id].extend(mechanic_inferences)
        
        # Generate symbolic references
        symbolic_refs = self._generate_symbolic_references(sequence, physics_inferences, mechanic_inferences)
        self.symbolic_references[game_id] = symbolic_refs
        
        # Generate strategic guidance
        strategic_guidance = self._generate_strategic_guidance(physics_inferences, mechanic_inferences)
        
        return {
            "physics_inferences": [self._inference_to_dict(inf) for inf in physics_inferences],
            "mechanic_inferences": [self._inference_to_dict(inf) for inf in mechanic_inferences],
            "symbolic_references": symbolic_refs,
            "strategic_guidance": strategic_guidance,
            "confidence_score": self._calculate_confidence_score(physics_inferences, mechanic_inferences)
        }
    
    def _analyze_physics_patterns(self, sequence: FrameSequence) -> List[PhysicsInference]:
        """Analyze frame sequence for physics patterns with enhanced OpenCV analysis."""
        inferences = []
        frames = sequence.frames
        
        # Enhanced motion analysis using optical flow
        for i in range(len(frames) - 1):
            prev_frame = frames[i]
            curr_frame = frames[i + 1]
            
            # Analyze optical flow for motion patterns
            flow_analysis = self._analyze_optical_flow(prev_frame, curr_frame)
            
            if flow_analysis['motion_detected']:
                # Create motion inference
                motion_inference = PhysicsInference(
                    inference_type="motion_detected",
                    confidence=min(flow_analysis['motion_magnitude'] / 10.0, 1.0),
                    description=f"Motion detected: magnitude={flow_analysis['motion_magnitude']:.2f}, direction={flow_analysis['motion_direction']:.2f}",
                    spatial_region=(0, 0, frames[0].shape[1], frames[0].shape[0]),
                    temporal_span=2,
                    symbolic_reference="Motion Analysis"
                )
                inferences.append(motion_inference)
                
                # Analyze motion consistency for physics patterns
                if flow_analysis['motion_consistency'] < 0.5:
                    consistent_motion = PhysicsInference(
                        inference_type="consistent_motion",
                        confidence=0.8,
                        description="Consistent directional motion detected",
                        spatial_region=(0, 0, frames[0].shape[1], frames[0].shape[0]),
                        temporal_span=2,
                        symbolic_reference="Linear Motion"
                    )
                    inferences.append(consistent_motion)
                else:
                    chaotic_motion = PhysicsInference(
                        inference_type="chaotic_motion",
                        confidence=0.6,
                        description="Chaotic or complex motion pattern detected",
                        spatial_region=(0, 0, frames[0].shape[1], frames[0].shape[0]),
                        temporal_span=2,
                        symbolic_reference="Complex Motion"
                    )
                    inferences.append(chaotic_motion)
        
        # Detect gravity patterns
        gravity_inf = self._detect_gravity_patterns(frames)
        if gravity_inf:
            inferences.append(gravity_inf)
        
        # Detect momentum patterns
        momentum_inf = self._detect_momentum_patterns(frames)
        if momentum_inf:
            inferences.append(momentum_inf)
        
        # Detect collision patterns
        collision_inf = self._detect_collision_patterns(frames)
        if collision_inf:
            inferences.append(collision_inf)
        
        # Detect color transition patterns
        color_inf = self._detect_color_transition_patterns(frames)
        if color_inf:
            inferences.append(color_inf)
        
        # Detect shape morphing patterns
        shape_inf = self._detect_shape_morphing_patterns(frames)
        if shape_inf:
            inferences.append(shape_inf)
        
        return inferences
    
    def _detect_gravity_patterns(self, frames: List[np.ndarray]) -> Optional[PhysicsInference]:
        """Detect gravity-like behavior in frame sequence."""
        if len(frames) < 3:
            return None
        
        # Look for objects moving downward with increasing speed
        downward_movements = []
        for i in range(1, len(frames)):
            diff = frames[i] - frames[i-1]
            # Find regions with downward movement
            try:
                # Check if array is large enough for gradient calculation
                if diff.shape[0] < 2:
                    continue
                vertical_gradients = np.gradient(diff, axis=0)
                downward_regions = np.where(vertical_gradients > self.movement_threshold)
            except ValueError as e:
                # Skip if gradient calculation fails due to array size
                continue
            
            if len(downward_regions[0]) > 0:
                downward_movements.append({
                    'frame': i,
                    'regions': list(zip(downward_regions[0], downward_regions[1]))
                })
        
        if len(downward_movements) >= 2:
            # Check if speed is increasing (acceleration)
            speeds = []
            for movement in downward_movements:
                speeds.append(len(movement['regions']))
            
            if len(speeds) >= 2 and speeds[-1] > speeds[0]:
                return PhysicsInference(
                    physics_type=PhysicsType.GRAVITY,
                    confidence=0.8,
                    evidence=[f"Downward movement detected in {len(downward_movements)} frames",
                             f"Acceleration pattern: {speeds}"],
                    spatial_region=self._get_movement_region(downward_movements),
                    temporal_span=len(downward_movements),
                    symbolic_reference="Objects fall with gravity-like acceleration",
                    strategic_implication="Use gravity to manipulate objects downward"
                )
        
        return None
    
    def _detect_momentum_patterns(self, frames: List[np.ndarray]) -> Optional[PhysicsInference]:
        """Detect momentum conservation patterns."""
        if len(frames) < 4:
            return None
        
        # Look for objects maintaining velocity
        velocity_vectors = []
        for i in range(1, len(frames)):
            try:
                diff = frames[i] - frames[i-1]
                # Calculate velocity vector
                velocity = np.sum(diff, axis=(0, 1))
                velocity_vectors.append(velocity)
            except (ValueError, IndexError) as e:
                # Skip if calculation fails due to array size or shape issues
                continue
        
        # Check for consistent velocity direction
        if len(velocity_vectors) >= 3:
            directions = [np.sign(v) for v in velocity_vectors]
            if len(set(tuple(d) for d in directions)) == 1:  # All same direction
                return PhysicsInference(
                    physics_type=PhysicsType.MOMENTUM,
                    confidence=0.7,
                    evidence=[f"Consistent velocity direction: {directions[0]}",
                             f"Velocity maintained over {len(velocity_vectors)} frames"],
                    spatial_region=(0, 0, frames[0].shape[1], frames[0].shape[0]),
                    temporal_span=len(velocity_vectors),
                    symbolic_reference="Objects maintain momentum in consistent direction",
                    strategic_implication="Momentum can be used to predict object movement"
                )
        
        return None
    
    def _detect_collision_patterns(self, frames: List[np.ndarray]) -> Optional[PhysicsInference]:
        """Detect collision and interaction patterns."""
        if len(frames) < 3:
            return None
        
        # Look for sudden changes in movement patterns
        collision_events = []
        for i in range(1, len(frames)):
            try:
                prev_diff = frames[i-1] - frames[i-2] if i > 1 else np.zeros_like(frames[i])
                curr_diff = frames[i] - frames[i-1]
                
                # Detect sudden direction changes
                prev_direction = np.sign(np.sum(prev_diff, axis=(0, 1)))
                curr_direction = np.sign(np.sum(curr_diff, axis=(0, 1)))
            except (ValueError, IndexError) as e:
                # Skip if calculation fails due to array size or shape issues
                continue
            
            if not np.array_equal(prev_direction, curr_direction):
                collision_events.append(i)
        
        if len(collision_events) >= 2:
            return PhysicsInference(
                physics_type=PhysicsType.COLLISION,
                confidence=0.75,
                evidence=[f"Direction changes detected at frames: {collision_events}",
                         f"Total collision events: {len(collision_events)}"],
                spatial_region=(0, 0, frames[0].shape[1], frames[0].shape[0]),
                temporal_span=len(collision_events),
                symbolic_reference="Objects collide and change direction",
                strategic_implication="Use collisions to redirect object movement"
            )
        
        return None
    
    def _detect_color_transition_patterns(self, frames: List[np.ndarray]) -> Optional[PhysicsInference]:
        """Detect color transition patterns."""
        if len(frames) < 3:
            return None
        
        # Track color changes over time
        color_changes = []
        for i in range(1, len(frames)):
            prev_colors = set(frames[i-1].flatten())
            curr_colors = set(frames[i].flatten())
            
            new_colors = curr_colors - prev_colors
            lost_colors = prev_colors - curr_colors
            
            if new_colors or lost_colors:
                color_changes.append({
                    'frame': i,
                    'new_colors': new_colors,
                    'lost_colors': lost_colors
                })
        
        if len(color_changes) >= 2:
            # Analyze pattern
            pattern_type = self._classify_color_pattern(color_changes)
            
            return PhysicsInference(
                physics_type=PhysicsType.COLOR_TRANSITION,
                confidence=0.6,
                evidence=[f"Color changes detected in {len(color_changes)} frames",
                         f"Pattern type: {pattern_type}"],
                spatial_region=(0, 0, frames[0].shape[1], frames[0].shape[0]),
                temporal_span=len(color_changes),
                symbolic_reference=f"Colors change with {pattern_type} pattern",
                strategic_implication="Color changes may indicate state transitions"
            )
        
        return None
    
    def _detect_shape_morphing_patterns(self, frames: List[np.ndarray]) -> Optional[PhysicsInference]:
        """Detect shape morphing and transformation patterns."""
        if len(frames) < 3:
            return None
        
        # Analyze shape changes
        shape_changes = []
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            # Detect structural changes
            prev_contours = self._find_contours(prev_frame)
            curr_contours = self._find_contours(curr_frame)
            
            if len(prev_contours) != len(curr_contours):
                shape_changes.append({
                    'frame': i,
                    'contour_count_change': len(curr_contours) - len(prev_contours),
                    'prev_contours': len(prev_contours),
                    'curr_contours': len(curr_contours)
                })
        
        if len(shape_changes) >= 2:
            return PhysicsInference(
                physics_type=PhysicsType.SHAPE_MORPHING,
                confidence=0.65,
                evidence=[f"Shape changes detected in {len(shape_changes)} frames",
                         f"Contour count variations: {[c['contour_count_change'] for c in shape_changes]}"],
                spatial_region=(0, 0, frames[0].shape[1], frames[0].shape[0]),
                temporal_span=len(shape_changes),
                symbolic_reference="Objects change shape and structure",
                strategic_implication="Shape changes may indicate object state or function"
            )
        
        return None
    
    def _analyze_game_mechanics(self, sequence: FrameSequence) -> List[GameMechanicInference]:
        """Analyze frame sequence for game mechanics."""
        inferences = []
        frames = sequence.frames
        actions = sequence.action_history
        
        # Detect puzzle-solving patterns
        puzzle_inf = self._detect_puzzle_solving_patterns(frames, actions)
        if puzzle_inf:
            inferences.append(puzzle_inf)
        
        # Detect pattern matching requirements
        pattern_inf = self._detect_pattern_matching_requirements(frames)
        if pattern_inf:
            inferences.append(pattern_inf)
        
        # Detect object manipulation mechanics
        manipulation_inf = self._detect_object_manipulation_mechanics(frames, actions)
        if manipulation_inf:
            inferences.append(manipulation_inf)
        
        return inferences
    
    def _detect_puzzle_solving_patterns(self, frames: List[np.ndarray], 
                                      actions: List[Tuple[int, Tuple[int, int]]]) -> Optional[GameMechanicInference]:
        """Detect puzzle-solving mechanics."""
        if len(frames) < 5:
            return None
        
        # Look for goal-oriented progression
        progression_indicators = []
        for i in range(1, len(frames)):
            # Check for increasing complexity or goal achievement
            complexity = self._calculate_frame_complexity(frames[i])
            prev_complexity = self._calculate_frame_complexity(frames[i-1])
            
            if complexity > prev_complexity * 1.1:  # 10% increase
                progression_indicators.append(i)
        
        if len(progression_indicators) >= 2:
            return GameMechanicInference(
                mechanic=GameMechanic.PUZZLE_SOLVING,
                confidence=0.7,
                evidence=[f"Progression detected in {len(progression_indicators)} frames",
                         f"Complexity increases: {progression_indicators}"],
                spatial_region=(0, 0, frames[0].shape[1], frames[0].shape[0]),
                temporal_span=len(frames),
                symbolic_reference="Systematic progression toward goal state",
                required_actions=[6],  # ACTION6 for interaction
                success_criteria=["Frame complexity increases", "Goal state achieved"],
                strategic_guidance="Focus on systematic progression toward goal state"
            )
        
        return None
    
    def _detect_pattern_matching_requirements(self, frames: List[np.ndarray]) -> Optional[GameMechanicInference]:
        """Detect pattern matching requirements."""
        if len(frames) < 3:
            return None
        
        # Look for repeating patterns
        patterns = self._find_repeating_patterns(frames)
        
        if len(patterns) >= 2:
            return GameMechanicInference(
                mechanic=GameMechanic.PATTERN_MATCHING,
                confidence=0.8,
                evidence=[f"Found {len(patterns)} repeating patterns",
                         f"Pattern types: {[p['type'] for p in patterns]}"],
                spatial_region=(0, 0, frames[0].shape[1], frames[0].shape[0]),
                temporal_span=len(frames),
                symbolic_reference="Repeating patterns require identification and matching",
                required_actions=[6],  # ACTION6 for pattern recognition
                success_criteria=["Patterns identified", "Patterns matched"],
                strategic_guidance="Identify and match repeating patterns"
            )
        
        return None
    
    def _detect_object_manipulation_mechanics(self, frames: List[np.ndarray], 
                                            actions: List[Tuple[int, Tuple[int, int]]]) -> Optional[GameMechanicInference]:
        """Detect object manipulation mechanics."""
        if len(actions) < 2:
            return None
        
        # Analyze action effects on frames
        manipulation_effects = []
        for i, (action, coords) in enumerate(actions):
            if i < len(frames) - 1:
                # Check if action caused frame changes
                diff = frames[i+1] - frames[i]
                change_magnitude = np.sum(np.abs(diff))
                
                if change_magnitude > self.movement_threshold:
                    manipulation_effects.append({
                        'action': action,
                        'coords': coords,
                        'effect_magnitude': change_magnitude
                    })
        
        if len(manipulation_effects) >= 2:
            return GameMechanicInference(
                mechanic=GameMechanic.OBJECT_MANIPULATION,
                confidence=0.75,
                evidence=[f"Manipulation effects from {len(manipulation_effects)} actions",
                         f"Actions: {[e['action'] for e in manipulation_effects]}"],
                spatial_region=(0, 0, frames[0].shape[1], frames[0].shape[0]),
                temporal_span=len(frames),
                symbolic_reference="Objects respond to direct manipulation actions",
                required_actions=list(set([e['action'] for e in manipulation_effects])),
                success_criteria=["Objects respond to actions", "Desired changes achieved"],
                strategic_guidance="Use actions to manipulate objects toward goal"
            )
        
        return None
    
    def _generate_symbolic_references(self, sequence: FrameSequence, 
                                    physics_inferences: List[PhysicsInference],
                                    mechanic_inferences: List[GameMechanicInference]) -> str:
        """Generate symbolic references for the game."""
        references = []
        
        # Physics-based references
        for inf in physics_inferences:
            references.append(f"Physics: {inf.symbolic_reference}")
        
        # Mechanic-based references
        for inf in mechanic_inferences:
            references.append(f"Mechanic: {inf.mechanic.value}")
        
        # Action-based references
        if sequence.action_history:
            action_types = set([action for action, _ in sequence.action_history])
            references.append(f"Actions: {sorted(action_types)}")
        
        return "; ".join(references)
    
    def _generate_strategic_guidance(self, physics_inferences: List[PhysicsInference],
                                   mechanic_inferences: List[GameMechanicInference]) -> Dict[str, Any]:
        """Generate strategic guidance for Architect and Governor."""
        guidance = {
            "architect_guidance": [],
            "governor_guidance": [],
            "action_priorities": [],
            "success_strategies": []
        }
        
        # Physics-based guidance
        for inf in physics_inferences:
            guidance["architect_guidance"].append({
                "type": "physics_understanding",
                "physics_type": inf.physics_type.value,
                "confidence": inf.confidence,
                "implication": inf.strategic_implication
            })
        
        # Mechanic-based guidance
        for inf in mechanic_inferences:
            guidance["governor_guidance"].append({
                "type": "mechanic_understanding",
                "mechanic": inf.mechanic.value,
                "confidence": inf.confidence,
                "guidance": inf.strategic_guidance,
                "required_actions": inf.required_actions
            })
            
            guidance["action_priorities"].extend(inf.required_actions)
        
        # Generate success strategies
        if physics_inferences or mechanic_inferences:
            guidance["success_strategies"].append("Leverage detected physics patterns")
            guidance["success_strategies"].append("Focus on identified game mechanics")
            guidance["success_strategies"].append("Use symbolic references for decision making")
        
        return guidance
    
    def _calculate_confidence_score(self, physics_inferences: List[PhysicsInference],
                                  mechanic_inferences: List[GameMechanicInference]) -> float:
        """Calculate overall confidence score for the analysis."""
        if not physics_inferences and not mechanic_inferences:
            return 0.0
        
        physics_conf = np.mean([inf.confidence for inf in physics_inferences]) if physics_inferences else 0.0
        mechanic_conf = np.mean([inf.confidence for inf in mechanic_inferences]) if mechanic_inferences else 0.0
        
        return (physics_conf + mechanic_conf) / 2.0
    
    def _inference_to_dict(self, inference) -> Dict[str, Any]:
        """Convert inference object to dictionary."""
        return {
            "type": inference.__class__.__name__,
            "confidence": inference.confidence,
            "evidence": inference.evidence,
            "spatial_region": inference.spatial_region,
            "temporal_span": inference.temporal_span,
            "symbolic_reference": inference.symbolic_reference,
            "strategic_implication": getattr(inference, 'strategic_implication', None),
            "strategic_guidance": getattr(inference, 'strategic_guidance', None),
            "required_actions": getattr(inference, 'required_actions', None),
            "success_criteria": getattr(inference, 'success_criteria', None)
        }
    
    # Helper methods
    def _get_movement_region(self, movements: List[Dict]) -> Tuple[int, int, int, int]:
        """Get bounding box of movement regions."""
        if not movements:
            return (0, 0, 64, 64)
        
        all_coords = []
        for movement in movements:
            all_coords.extend(movement['regions'])
        
        if not all_coords:
            return (0, 0, 64, 64)
        
        x_coords = [coord[1] for coord in all_coords]
        y_coords = [coord[0] for coord in all_coords]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def _classify_color_pattern(self, color_changes: List[Dict]) -> str:
        """Classify the type of color change pattern."""
        if len(color_changes) < 2:
            return "unknown"
        
        # Simple classification based on change frequency
        change_frequency = len(color_changes) / len(color_changes)
        
        if change_frequency > 0.8:
            return "rapid_changes"
        elif change_frequency > 0.4:
            return "moderate_changes"
        else:
            return "gradual_changes"
    
    def _find_contours(self, frame: np.ndarray) -> List:
        """Find contours in frame with enhanced OpenCV analysis."""
        try:
            # Use proper format validation from OpenCV processor
            from src.core.opencv_utils import opencv_processor
            frame_processed = opencv_processor._ensure_proper_format(frame)
            
            # Ensure we have a 2D grayscale image
            if len(frame_processed.shape) > 2:
                frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(frame_processed, (5, 5), 0)
            
            # Use adaptive thresholding for better contour detection
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find contours with hierarchy for better object detection
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and complexity
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10:  # Minimum area threshold
                    # Calculate contour complexity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.1:  # Filter out very irregular shapes
                            filtered_contours.append(contour)
            
            return filtered_contours
        except Exception as e:
            self.logger.warning(f"Contour detection failed: {e}")
            return []
    
    def _calculate_frame_complexity(self, frame: np.ndarray) -> float:
        """Calculate complexity score for a frame using OpenCV."""
        try:
            # Use proper format validation from OpenCV processor
            from src.core.opencv_utils import opencv_processor
            frame_processed = opencv_processor._ensure_proper_format(frame)
            
            # Calculate edge density using Canny edge detection
            edges = cv2.Canny(frame_processed, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Calculate texture complexity using Laplacian variance
            laplacian_var = cv2.Laplacian(frame_processed, cv2.CV_64F).var()
            
            # Calculate color complexity
            unique_colors = len(set(frame.flatten()))
            
            # Combine metrics for comprehensive complexity score
            complexity = (edge_density * 1000) + (laplacian_var * 0.1) + (unique_colors * 0.01)
            return complexity
        except Exception as e:
            self.logger.warning(f"Frame complexity calculation failed: {e}")
            return 0.0
    
    def _analyze_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Dict[str, Any]:
        """Analyze optical flow between consecutive frames using OpenCV."""
        try:
            # Use proper format validation
            from src.core.opencv_utils import opencv_processor
            prev_gray = opencv_processor._ensure_proper_format(prev_frame)
            curr_gray = opencv_processor._ensure_proper_format(curr_frame)
            
            # Ensure we have 2D grayscale images
            if len(prev_gray.shape) > 2:
                prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
            if len(curr_gray.shape) > 2:
                curr_gray = cv2.cvtColor(curr_gray, cv2.COLOR_BGR2GRAY)
            
            # Detect feature points using corner detection
            corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            if corners is not None and len(corners) > 0:
                # Calculate optical flow using Lucas-Kanade method
                flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)
                
                # Calculate motion magnitude and direction
                if flow[0] is not None and flow[1] is not None:
                    good_old = flow[0]
                    good_new = flow[1]
                    
                    if len(good_old) > 0 and len(good_new) > 0:
                        # Calculate displacement vectors
                        displacement = good_new - good_old
                        motion_magnitude = np.linalg.norm(displacement, axis=1)
                        avg_motion = np.mean(motion_magnitude)
                        
                        # Calculate motion direction
                        motion_direction = np.arctan2(displacement[:, 1], displacement[:, 0])
                        dominant_direction = np.median(motion_direction)
                        
                        return {
                            'motion_detected': avg_motion > 1.0,
                            'motion_magnitude': float(avg_motion),
                            'motion_direction': float(dominant_direction),
                            'motion_consistency': float(np.std(motion_direction)),
                            'feature_points': len(good_old)
                        }
            
            # Fallback: Use dense optical flow if no corners detected
            try:
                dense_flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None)
                if dense_flow[0] is not None and dense_flow[1] is not None:
                    good_old = dense_flow[0]
                    good_new = dense_flow[1]
                    
                    if len(good_old) > 0 and len(good_new) > 0:
                        displacement = good_new - good_old
                        motion_magnitude = np.linalg.norm(displacement, axis=1)
                        avg_motion = np.mean(motion_magnitude)
                        
                        return {
                            'motion_detected': avg_motion > 1.0,
                            'motion_magnitude': float(avg_motion),
                            'motion_direction': 0.0,
                            'motion_consistency': 0.0,
                            'feature_points': len(good_old)
                        }
            except:
                pass
            
            return {
                'motion_detected': False,
                'motion_magnitude': 0.0,
                'motion_direction': 0.0,
                'motion_consistency': 0.0,
                'feature_points': 0
            }
        except Exception as e:
            self.logger.warning(f"Optical flow analysis failed: {e}")
            return {
                'motion_detected': False,
                'motion_magnitude': 0.0,
                'motion_direction': 0.0,
                'motion_consistency': 0.0,
                'feature_points': 0
            }
    
    def _find_repeating_patterns(self, frames: List[np.ndarray]) -> List[Dict]:
        """Find repeating patterns in frame sequence using OpenCV template matching."""
        patterns = []
        
        # Look for color pattern repetition
        for i in range(len(frames) - 2):
            for j in range(i + 1, len(frames) - 1):
                if np.array_equal(frames[i], frames[j]):
                    patterns.append({
                        'type': 'exact_match',
                        'frames': [i, j],
                        'confidence': 1.0
                    })
        
        return patterns

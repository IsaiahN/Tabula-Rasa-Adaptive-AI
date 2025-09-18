"""
Frame Analysis System for ARC-AGI-3 Training
Implements computer vision for agent position tracking and movement detection.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import json
from datetime import datetime

# Import the meta-learner
try:
    from ..core.reward_cap_meta_learner import RewardCapMetaLearner
except ImportError:
    try:
        from core.reward_cap_meta_learner import RewardCapMetaLearner
    except ImportError:
        # Fallback for when meta-learner is not available
        class RewardCapMetaLearner:
            def __init__(self, *args, **kwargs):
                pass
            def update_performance(self, *args, **kwargs):
                pass
            def get_current_caps(self):
                return type('Caps', (), {
                    'productivity_multiplier': 25.0,
                    'productivity_max': 100.0,
                    'recent_gains_multiplier': 15.0,
                    'recent_gains_max': 75.0,
                    'recent_losses_multiplier': 10.0,
                    'recent_losses_max': 50.0,
                    'exploration_bonus': 15.0,
                    'movement_bonus': 20.0
                })()


class FrameAnalyzer:
    """
    Lightweight computer vision system for tracking agent position and movement
    in ARC-AGI-3 game frames.
    """
    
    def __init__(self, base_path: str = "."):
        self.previous_frame = None
        self.agent_position = None  # (x, y) coordinates
        self.frame_history = []  # Store recent frames for pattern detection
        self.max_history = 10  # Increased for better tracking
        self.grid_size = 64  # ARC-AGI-3 uses 64x64 grids
        
        # Color detection parameters
        self.agent_colors = []  # Learn colors that represent agent
        self.background_color = 0  # Assume 0 is background initially
        
        # Movement detection parameters
        self.movement_threshold = 0.1  # Minimum change to detect movement
        self.position_confidence = 0.0  # How confident we are about position
        
        # ENHANCED FULL-FRAME TRACKING SYSTEM
        self.tried_coordinates = set()  # Track all previously tried coordinates
        self.coordinate_results = {}  # Map coordinates to their results
        self.color_object_tracker = {}  # Track all color objects and their movements
        self.frame_scan_history = []  # Complete frame analysis history
        self.exploration_spiral = 0  # For systematic exploration
        
        # Meta-learning system for dynamic caps
        self.meta_learner = RewardCapMetaLearner(base_path=base_path)
        self.current_caps = self.meta_learner.get_current_caps()
        self.last_successful_coords = []  # Track successful interactions
        
        # ENHANCED INTERACTION LOGGING SYSTEM
        self.interaction_log = []  # Detailed log of all ACTION6 interactions
        self.object_interaction_history = {}  # object_id -> list of interaction results
        self.color_behavior_patterns = {}  # color -> behavior patterns observed
        self.shape_behavior_patterns = {}  # shape_type -> behavior patterns observed
        self.spatial_interaction_map = {}  # (x, y) -> interaction results and context
        self.hypothesis_database = []  # Generated hypotheses about object behaviors
        
        # INITIAL EXPLORATION PHASE - User insight implementation
        self.exploration_phase = True
        self.color_objects_to_explore = set()  # Colors we haven't tried yet
        self.explored_color_objects = set()   # Colors we've clicked at least once
        self.exploration_complete = False
        self.current_game_colors = set()      # Colors detected in current game
        self.game_state_memory = {}           # Remember game states and transitions

    def _normalize_local_frame(self, frame: Any) -> np.ndarray:
        """Small helper to convert various frame wrappers into a 2D numpy array.

        Accepts:
        - frame as [[row1, row2, ...]] (wrapped)
        - frame as [row1, row2, ...] (2D list)
        - frame as flat list (will attempt reasonable reshape)
        - numpy arrays
        Returns a numpy 2D array; raises if conversion fails.
        """
        if frame is None:
            return np.zeros((64, 64), dtype=int)

        try:
            # If it's already a numpy array
            if isinstance(frame, np.ndarray):
                arr = frame.copy()
            else:
                # If wrapped in an extra list: [[row1,row2,...]]
                # Handle wrapped frames by extracting the inner list
                if isinstance(frame, list) and len(frame) > 0 and isinstance(frame[0], list):
                    # Unwrap: [[row1, row2, ...]] -> [row1, row2, ...]
                    arr = np.array(frame[0])
                else:
                    arr = np.array(frame)

            # If 1D try reshape to square or keep as single-row
            if arr.ndim == 1:
                n = arr.size
                side = int(np.sqrt(n))
                if side * side == n:
                    arr = arr.reshape((side, side))
                else:
                    arr = arr.reshape((1, n))

            # Ensure 2D
            if arr.ndim >= 2:
                return arr

        except Exception:
            pass

        # Fallback
        return np.zeros((64, 64), dtype=int)
        
    def reset_for_new_game(self, game_id: str = None):
        """Reset exploration phase for a new game while preserving learned patterns."""
        print(f"🎯 Reset frame analyzer tracking for new game")
        
        # Reset exploration phase
        self.exploration_phase = True
        self.exploration_complete = False
        self.current_game_colors = set()
        self.color_objects_to_explore = set()
        
        # Keep learned patterns but reset game-specific data
        self.previous_frame = None
        self.frame_history = []
        self.tried_coordinates = set()  # Reset for new game
        
        # Keep coordinate results and behavior patterns for cross-game learning
        # Only reset if explicitly requested or if patterns are very poor
        poor_performance = len(self.coordinate_results) > 50 and sum(
            result.get('total_score_change', 0) for result in self.coordinate_results.values()
        ) <= 0
        
        if poor_performance:
            print("🔄 Poor performance detected - resetting coordinate intelligence")
            self.coordinate_results = {}
            self.color_behavior_patterns = {}
        
        print(f"   📊 Retaining {len(self.coordinate_results)} coordinate results")
        print(f"   🎨 Retaining {len(self.color_behavior_patterns)} color behavior patterns")
        
    def analyze_frame_for_action6_targets(self, frame: List[List[List[int]]], game_id: str = None) -> Dict[str, Any]:
        """
        ADVANCED VISUAL TARGETING SYSTEM for ACTION6.
        
        Implements the new paradigm: ACTION6(x,y) = "touch/interact with object at (x,y)"
        Not movement, but interaction with visual elements.
        
        Args:
            frame: 64x64 grid of color indices (0-15)
            game_id: Current game identifier
            
        Returns:
            Dictionary with interactive targets, ranked by priority
        """
        # Convert frame to numpy array using local normalizer
        frame_array = self._normalize_local_frame(frame)
        targets = []
        analysis = {
            'interactive_targets': [],
            'recommended_action6_coord': None,
            'targeting_reason': '',
            'confidence': 0.0,
            'visual_features': {},
            'exploration_zones': []
        }

        try:
            # 1. COLOR CLUSTERING - Find unique/standout colors
            color_targets = self._find_color_anomalies(frame_array)
            targets.extend(color_targets)
            
            # 2. BRIGHTNESS/CONTRAST - Find bright/dark extremes  
            contrast_targets = self._find_contrast_extremes(frame_array)
            targets.extend(contrast_targets)
            
            # 3. PATTERN/SHAPE DETECTION - Simple geometric shapes
            shape_targets = self._find_geometric_shapes(frame_array)
            targets.extend(shape_targets)
            
            # 3.5. EDGE DETECTION - Find edge features that might be interactive
            edge_targets = self._find_edge_features(frame_array)
            targets.extend(edge_targets)
            
            # 3.6. PATTERN BREAK DETECTION - Find breaks in patterns
            pattern_break_targets = self._find_pattern_breaks(frame_array)
            targets.extend(pattern_break_targets)
            
            # 3.7. SYMMETRY DETECTION - Find symmetry points
            symmetry_targets = self._find_symmetry_points(frame_array)
            targets.extend(symmetry_targets)
            
            # 4. FRAME DIFFERENCING - Objects that changed
            if self.previous_frame is not None:
                change_targets = self._find_frame_changes(frame_array, self.previous_frame)
                targets.extend(change_targets)
            
            # 5. RANK TARGETS by interaction potential
            ranked_targets = self._rank_interaction_targets(targets)
            
            # 5.5. INITIAL EXPLORATION PHASE - User insight implementation
            exploration_target = self._check_exploration_phase(frame_array, ranked_targets)
            
            # 6. ENHANCED TARGET SELECTION with coordinate avoidance
            selected_target = None
            emergency_mode = False
            
            # EXPLORATION PHASE takes priority over normal selection
            if exploration_target:
                selected_target = exploration_target
                print(f"🔍 EXPLORATION MODE: Targeting {exploration_target['reason']}")
            else:
                # Check if we're stuck (more than 200 attempts at stuck coordinates)
                total_stuck_attempts = sum(
                    result['try_count'] for result in self.coordinate_results.values() 
                    if result.get('is_stuck_coordinate', False)
                )
                
                # Only trigger emergency mode if we have a significant number of stuck attempts
                # and we're not making progress
                if total_stuck_attempts > 500:
                    emergency_mode = True
                    print(f"🚨 EMERGENCY MODE: {total_stuck_attempts} attempts at stuck coordinates")
                    
                    # Force diversification - use random coordinates avoiding stuck areas
                    emergency_coord = self._get_emergency_diversification_target(
                        frame, (frame_array.shape[1], frame_array.shape[0])
                    )
                    selected_target = {
                        'x': emergency_coord[0], 
                        'y': emergency_coord[1],
                        'reason': f'emergency_diversification_from_{total_stuck_attempts}_stuck_attempts',
                        'confidence': 0.8  # High confidence in diversification
                    }
                else:
                    # Normal target selection with avoidance
                    for target in ranked_targets:
                        if not self.should_avoid_coordinate(target['x'], target['y']):
                            selected_target = target
                            break
                        else:
                            avoidance_score = self.get_coordinate_avoidance_score(target['x'], target['y'])
                            print(f"🚫 AVOIDING ({target['x']},{target['y']}): avoidance score {avoidance_score:.2f}")
            
            if selected_target:
                analysis['interactive_targets'] = ranked_targets[:5]  # Top 5 for reference
                analysis['recommended_action6_coord'] = (selected_target['x'], selected_target['y'])
                analysis['targeting_reason'] = selected_target['reason']
                analysis['confidence'] = selected_target['confidence']
                
                if emergency_mode:
                    analysis['targeting_reason'] += "_EMERGENCY_MODE"
                    
            else:
                # 7. ENHANCED FALLBACK: Smart exploration avoiding stuck coordinates  
                exploration_coord = self._generate_smart_exploration_coordinate(frame_array)
                if exploration_coord:
                    analysis['recommended_action6_coord'] = exploration_coord
                    analysis['targeting_reason'] = 'smart_exploration_avoiding_stuck_coordinates'
                    analysis['confidence'] = 0.4  # Medium confidence for smart exploration
            
            # Store visual feature summary
            if frame_array.size > 0:
                analysis['visual_features'] = {
                    'unique_colors': len(np.unique(frame_array)),
                    'max_brightness': int(np.max(frame_array)),
                    'min_brightness': int(np.min(frame_array)),
                    'color_variance': float(np.var(frame_array)),
                    'frame_size': frame_array.shape
                }
            else:
                analysis['visual_features'] = {
                    'unique_colors': 0,
                    'max_brightness': 0,
                    'min_brightness': 0,
                    'color_variance': 0.0,
                    'frame_size': (0, 0)
                }
            
            # Update frame history
            self.previous_frame = frame_array.copy()
            
        except Exception as e:
            # Safe fallback
            analysis['recommended_action6_coord'] = (32, 32)  # Center
            analysis['targeting_reason'] = f'analysis_error_fallback: {str(e)[:50]}'
            analysis['confidence'] = 0.1
            
        return analysis
    
    def _find_color_anomalies(self, frame_array: np.ndarray) -> List[Dict]:
        """ENHANCED: Find ALL color objects across ENTIRE frame and track their movements."""
        targets = []
        
        # Get COMPLETE color frequency distribution across entire frame
        unique_colors, counts = np.unique(frame_array, return_counts=True)
        total_pixels = frame_array.size
        
        # Track ALL color objects, not just rare ones
        for color, count in zip(unique_colors, counts):
            frequency = count / total_pixels
            
            # EXPANDED CRITERIA: Track more colors (from very rare to moderately common)
            if 0.001 < frequency < 0.2:  # Much broader range than before
                # Find ALL positions with this color (not just centroid)
                positions = np.where(frame_array == color)
                
                if len(positions[0]) > 0:
                    # Group nearby positions into SEPARATE OBJECTS
                    color_objects = self._cluster_color_positions(positions[0], positions[1], color)
                    
                    for obj_id, obj_data in color_objects.items():
                        center_x, center_y = obj_data['centroid']
                        object_size = obj_data['size']
                        
                        # Skip if we've already tried this exact coordinate repeatedly
                        coord_key = (int(center_x), int(center_y))
                        if coord_key in self.tried_coordinates:
                            # Check if this coordinate has been tried too many times recently
                            recent_tries = self.coordinate_results.get(coord_key, {}).get('try_count', 0)
                            if recent_tries > 3:  # Skip if tried more than 3 times
                                continue
                        
                        # Calculate confidence based on rarity and size
                        confidence = min(1.0, (1.0 - frequency) * 2 + object_size / 50.0)
                        
                        # Track this color object for movement analysis
                        object_id = f"color_{color}_{int(center_x)}_{int(center_y)}"
                        self._update_color_object_tracking(object_id, center_x, center_y, color, object_size)
                        
                        targets.append({
                            'x': int(center_x),
                            'y': int(center_y),
                            'reason': f'color_{color}_freq_{frequency:.3f}_size_{object_size}',
                            'confidence': confidence,
                            'type': 'color_object',
                            'object_id': object_id,
                            'color': color,
                            'size': object_size
                        })
        
        return targets
    
    def _cluster_color_positions(self, y_positions: np.ndarray, x_positions: np.ndarray, color: int) -> Dict[str, Dict]:
        """Group nearby color positions into separate objects."""
        objects = {}
        positions = list(zip(x_positions, y_positions))
        
        if not positions:
            return objects
        
        # Simple clustering: group positions within distance threshold
        distance_threshold = 5  # pixels
        object_counter = 0
        
        for x, y in positions:
            # Find which existing object this position belongs to
            assigned = False
            for obj_id, obj_data in objects.items():
                obj_x, obj_y = obj_data['centroid']
                distance = np.sqrt((x - obj_x)**2 + (y - obj_y)**2)
                
                if distance <= distance_threshold:
                    # Add to existing object
                    obj_data['positions'].append((x, y))
                    # Update centroid
                    obj_data['centroid'] = (
                        np.mean([p[0] for p in obj_data['positions']]),
                        np.mean([p[1] for p in obj_data['positions']])
                    )
                    obj_data['size'] = len(obj_data['positions'])
                    assigned = True
                    break
            
            if not assigned:
                # Create new object
                obj_id = f"obj_{object_counter}"
                objects[obj_id] = {
                    'centroid': (x, y),
                    'positions': [(x, y)],
                    'size': 1,
                    'color': color
                }
                object_counter += 1
        
        return objects
    
    def _get_clustered_colors(self, frame_array: np.ndarray) -> List[np.ndarray]:
        """Enhanced color clustering with noise reduction and better grouping."""
        try:
            # Reshape frame to 2D array of pixels
            pixels = frame_array.reshape(-1, frame_array.shape[-1])
            
            # Use K-means clustering for color quantization
            from sklearn.cluster import KMeans
            
            # Determine optimal number of clusters based on frame complexity
            n_colors = min(20, max(5, len(np.unique(pixels.view(np.void, dtype=pixels.dtype)))))
            
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (representative colors)
            clustered_colors = kmeans.cluster_centers_.astype(int)
            
            # Filter out very similar colors
            unique_colors = []
            for color in clustered_colors:
                is_unique = True
                for existing_color in unique_colors:
                    if np.linalg.norm(color - existing_color) < 10:  # Similarity threshold
                        is_unique = False
                        break
                if is_unique:
                    unique_colors.append(color)
            
            return unique_colors
            
        except Exception as e:
            logger.warning(f"Color clustering failed, using simple unique colors: {e}")
            # Fallback to simple unique colors
            unique_colors = np.unique(frame_array.reshape(-1, frame_array.shape[-1]), axis=0)
            return unique_colors
    
    def _create_color_mask_with_tolerance(self, frame_array: np.ndarray, target_color: np.ndarray, tolerance: int = 5) -> np.ndarray:
        """Create a mask for a color with tolerance for slight variations."""
        try:
            # Calculate color distance for each pixel
            color_diff = np.linalg.norm(frame_array - target_color, axis=2)
            return color_diff <= tolerance
        except Exception as e:
            logger.warning(f"Color mask creation failed: {e}")
            return np.zeros(frame_array.shape[:2], dtype=bool)
    
    def _classify_color_object_enhanced(self, color: np.ndarray, width: int, height: int, area: int, mask: np.ndarray) -> str:
        """Enhanced object classification based on color, size, and shape."""
        try:
            # Color-based classification
            if len(color) >= 3:
                r, g, b = color[:3]
                
                # Red objects (often interactive)
                if r > 200 and g < 100 and b < 100:
                    return 'red_interactive'
                
                # Green objects (often goals or positive elements)
                if g > 200 and r < 100 and b < 100:
                    return 'green_goal'
                
                # Blue objects (often structural)
                if b > 200 and r < 100 and g < 100:
                    return 'blue_structural'
                
                # White objects (often boundaries or highlights)
                if r > 200 and g > 200 and b > 200:
                    return 'white_boundary'
                
                # Dark objects (often obstacles)
                if r < 50 and g < 50 and b < 50:
                    return 'dark_obstacle'
            
            # Size-based classification
            if area < 5:
                return 'small_object'
            elif area < 20:
                return 'medium_object'
            else:
                return 'large_object'
                
        except Exception as e:
            logger.warning(f"Object classification failed: {e}")
            return 'unknown_object'
    
    def _calculate_object_movement(self, x: int, y: int, color: np.ndarray, area: int) -> Optional[Tuple[float, float]]:
        """Calculate movement vector for an object based on its characteristics."""
        try:
            # Create object signature for tracking
            obj_signature = f"{color.tolist()}_{area}"
            
            if obj_signature in self.color_object_tracker:
                tracker = self.color_object_tracker[obj_signature]
                if len(tracker['positions']) > 1:
                    # Calculate movement from last position
                    last_pos = tracker['positions'][-1]
                    current_pos = (x, y)
                    
                    movement_x = current_pos[0] - last_pos[0]
                    movement_y = current_pos[1] - last_pos[1]
                    
                    return (movement_x, movement_y)
            
            return None
            
        except Exception as e:
            logger.warning(f"Movement calculation failed: {e}")
            return None
    
    def _calculate_object_stability(self, x: int, y: int, color: np.ndarray, area: int) -> float:
        """Calculate how stable an object is (higher = more stable)."""
        try:
            obj_signature = f"{color.tolist()}_{area}"
            
            if obj_signature in self.color_object_tracker:
                tracker = self.color_object_tracker[obj_signature]
                positions = tracker['positions']
                
                if len(positions) < 2:
                    return 0.5  # Neutral stability for new objects
                
                # Calculate position variance
                positions_array = np.array(positions)
                variance = np.var(positions_array, axis=0)
                total_variance = np.sum(variance)
                
                # Convert to stability score (0-1, higher = more stable)
                stability = max(0, 1 - (total_variance / 100))  # Normalize variance
                return min(1.0, stability)
            
            return 0.5  # Neutral for new objects
            
        except Exception as e:
            logger.warning(f"Stability calculation failed: {e}")
            return 0.5
    
    def _update_color_object_tracking(self, object_id: str, x: float, y: float, color: int, size: int):
        """Track color object movements over time."""
        current_time = len(self.frame_history)
        
        if object_id not in self.color_object_tracker:
            self.color_object_tracker[object_id] = {
                'positions': [],
                'color': color,
                'first_seen': current_time,
                'size_history': [],
                'movement_vector': None
            }
        
        tracker = self.color_object_tracker[object_id]
        tracker['positions'].append((x, y, current_time))
        tracker['size_history'].append(size)
        
        # Calculate movement if we have multiple positions
        if len(tracker['positions']) > 1:
            prev_x, prev_y, _ = tracker['positions'][-2]
            movement = (x - prev_x, y - prev_y)
            tracker['movement_vector'] = movement
            
        # Keep only recent history
        if len(tracker['positions']) > 10:
            tracker['positions'] = tracker['positions'][-10:]
            tracker['size_history'] = tracker['size_history'][-10:]
    
    def _find_contrast_extremes(self, frame_array: np.ndarray) -> List[Dict]:
        """Find brightest/darkest points that might be interactive."""
        targets = []
        
        # Find brightest points
        max_val = np.max(frame_array)
        bright_positions = np.where(frame_array == max_val)
        if len(bright_positions[0]) > 0:
            # Get first bright point
            bright_y, bright_x = bright_positions[0][0], bright_positions[1][0]
            targets.append({
                'x': int(bright_x),
                'y': int(bright_y),
                'reason': f'brightest_point_value_{max_val}',
                'confidence': 0.7,
                'type': 'brightness_extreme'
            })
        
        # Find darkest points (but not background)
        min_val = np.min(frame_array)
        if min_val > 0:  # Skip if it's just background
            dark_positions = np.where(frame_array == min_val)
            if len(dark_positions[0]) > 0:
                dark_y, dark_x = dark_positions[0][0], dark_positions[1][0]
                targets.append({
                    'x': int(dark_x),
                    'y': int(dark_y),
                    'reason': f'darkest_point_value_{min_val}',
                    'confidence': 0.6,
                    'type': 'darkness_extreme'
                })
        
        return targets
    
    def _find_geometric_shapes(self, frame_array: np.ndarray) -> List[Dict]:
        """Find simple geometric shapes that might be buttons/objects."""
        targets = []
        
        try:
            # Convert to binary for edge detection
            binary = (frame_array > np.mean(frame_array)).astype(np.uint8) * 255
            
            # Find edges
            edges = cv2.Canny(binary, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Look for shapes with reasonable area (not too small/large)
                if 10 < area < 200:  # Adjust based on 64x64 grid
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Analyze shape
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    num_vertices = len(approx)
                    
                    shape_type = 'unknown'
                    confidence = 0.5
                    
                    if num_vertices == 4:
                        shape_type = 'rectangle'
                        confidence = 0.8  # Rectangles often buttons
                    elif num_vertices > 6:
                        shape_type = 'circle'
                        confidence = 0.7  # Circles often ports/buttons
                    
                    targets.append({
                        'x': int(center_x),
                        'y': int(center_y),
                        'reason': f'{shape_type}_shape_area_{int(area)}',
                        'confidence': confidence,
                        'type': 'geometric_shape'
                    })
                    
        except Exception as e:
            # OpenCV operations can fail, continue gracefully
            pass
            
        return targets
    
    def _find_edge_features(self, frame_array: np.ndarray) -> List[Dict]:
        """Find edge features that might indicate interactive elements."""
        targets = []
        
        try:
            # Simple edge detection using gradient magnitude
            if len(frame_array.shape) == 3:
                gray = np.mean(frame_array, axis=2)
            else:
                gray = frame_array
            
            # Calculate gradients
            grad_x = np.gradient(gray.astype(float), axis=1)
            grad_y = np.gradient(gray.astype(float), axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Find high gradient points
            threshold = np.percentile(gradient_magnitude, 85)  # Top 15% of gradients
            edge_points = np.where(gradient_magnitude > threshold)
            
            for y, x in zip(edge_points[0], edge_points[1]):
                if 5 <= x < 59 and 5 <= y < 59:  # Avoid edges
                    targets.append({
                        'x': int(x),
                        'y': int(y),
                        'type': 'edge_feature',
                        'confidence': min(0.9, gradient_magnitude[y, x] / np.max(gradient_magnitude)),
                        'reason': f'Edge feature with gradient {gradient_magnitude[y, x]:.2f}'
                    })
                    
        except Exception as e:
            pass
            
        return targets
    
    def _find_pattern_breaks(self, frame_array: np.ndarray) -> List[Dict]:
        """Find breaks in patterns that might indicate interactive boundaries."""
        targets = []
        
        try:
            if len(frame_array.shape) == 3:
                gray = np.mean(frame_array, axis=2)
            else:
                gray = frame_array
            
            # Look for horizontal and vertical pattern breaks
            for direction in ['horizontal', 'vertical']:
                if direction == 'horizontal':
                    # Check for horizontal pattern breaks
                    for y in range(2, len(gray) - 2):
                        for x in range(2, len(gray[0]) - 2):
                            # Check if this row breaks a pattern
                            row = gray[y, x-2:x+3]
                            if len(set(row)) > 2:  # Multiple different values
                                # Check if it's different from surrounding rows
                                above = gray[y-1, x-2:x+3]
                                below = gray[y+1, x-2:x+3]
                                # Fix array ambiguity by ensuring arrays have same shape before comparison
                                if (row.shape == above.shape and row.shape == below.shape and 
                                    not np.array_equal(row, above) and not np.array_equal(row, below)):
                                    targets.append({
                                        'x': int(x),
                                        'y': int(y),
                                        'type': 'pattern_break',
                                        'confidence': 0.7,
                                        'reason': f'Horizontal pattern break at row {y}'
                                    })
                else:
                    # Check for vertical pattern breaks
                    for x in range(2, len(gray[0]) - 2):
                        for y in range(2, len(gray) - 2):
                            # Check if this column breaks a pattern
                            col = gray[y-2:y+3, x]
                            if len(set(col)) > 2:  # Multiple different values
                                # Check if it's different from surrounding columns
                                left = gray[y-2:y+3, x-1]
                                right = gray[y-2:y+3, x+1]
                                # Fix array ambiguity by ensuring arrays have same shape before comparison
                                if (col.shape == left.shape and col.shape == right.shape and 
                                    not np.array_equal(col, left) and not np.array_equal(col, right)):
                                    targets.append({
                                        'x': int(x),
                                        'y': int(y),
                                        'type': 'pattern_break',
                                        'confidence': 0.7,
                                        'reason': f'Vertical pattern break at column {x}'
                                    })
                    
        except Exception as e:
            pass
            
        return targets
    
    def _find_symmetry_points(self, frame_array: np.ndarray) -> List[Dict]:
        """Find symmetry points that might indicate important elements."""
        targets = []
        
        try:
            if len(frame_array.shape) == 3:
                gray = np.mean(frame_array, axis=2)
            else:
                gray = frame_array
            
            height, width = gray.shape
            center_x, center_y = width // 2, height // 2
            
            # Check for horizontal symmetry
            for y in range(height):
                for x in range(width // 2):
                    mirror_x = width - 1 - x
                    if abs(gray[y, x] - gray[y, mirror_x]) < 1:  # Similar values
                        # Check if this is part of a larger symmetric pattern
                        symmetry_score = 0
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                ny, nx = y + dy, x + dx
                                nmy, nmx = y + dy, mirror_x - dx
                                if (0 <= ny < height and 0 <= nx < width and 
                                    0 <= nmy < height and 0 <= nmx < width):
                                    if abs(gray[ny, nx] - gray[nmy, nmx]) < 1:
                                        symmetry_score += 1
                        
                        if symmetry_score > 5:  # Significant symmetry
                            targets.append({
                                'x': int(x),
                                'y': int(y),
                                'type': 'symmetry_point',
                                'confidence': min(0.8, symmetry_score / 25.0),
                                'reason': f'Symmetry point with score {symmetry_score}'
                            })
                            
        except Exception as e:
            pass
            
        return targets
    
    def _find_frame_changes(self, current_frame: np.ndarray, previous_frame: np.ndarray) -> List[Dict]:
        """Find areas that changed between frames - likely interactive."""
        targets = []
        
        try:
            # Calculate frame difference
            diff = np.abs(current_frame.astype(int) - previous_frame.astype(int))
            
            # Find significant changes
            change_threshold = np.std(diff) + 1  # Adaptive threshold
            significant_changes = np.where(diff > change_threshold)
            
            if len(significant_changes[0]) > 0:
                # Group changes into clusters
                change_points = list(zip(significant_changes[1], significant_changes[0]))  # (x,y) format
                
                if change_points:
                    # Find center of mass of changes
                    center_x = int(np.mean([p[0] for p in change_points]))
                    center_y = int(np.mean([p[1] for p in change_points]))
                    
                    confidence = min(1.0, len(change_points) / 50.0)  # More changes = higher confidence
                    
                    targets.append({
                        'x': center_x,
                        'y': center_y,
                        'reason': f'frame_change_cluster_{len(change_points)}_points',
                        'confidence': confidence,
                        'type': 'dynamic_change'
                    })
                    
        except Exception as e:
            pass
            
        return targets
    
    def _rank_interaction_targets(self, targets: List[Dict]) -> List[Dict]:
        """ENHANCED: Rank targets prioritizing productive coordinates and avoiding stuck ones."""
        # Sort by confidence, then by type priority
        type_priority = {
            'dynamic_change': 6,      # Moving objects = highest priority (increased from 5)
            'color_object': 5,        # Enhanced color objects = high priority (increased from 4)
            'geometric_shape': 4,     # Shapes = possible buttons (increased from 3)
            'brightness_extreme': 3,  # Brightness = might be indicators (increased from 2)
            'color_anomaly': 2,       # Basic color anomaly = lowest priority (increased from 1)
            'edge_feature': 4,        # New: Edge features often indicate interactive elements
            'pattern_break': 5,       # New: Pattern breaks suggest interactive boundaries
            'symmetry_point': 3       # New: Symmetry points often indicate important elements
        }
        
        # Collect productivity data for consolidated logging
        productivity_bonuses = []
        productivity_penalties = []
        recent_gains_bonuses = []
        recent_losses_penalties = []
        
        for target in targets:
            base_score = target['confidence'] * 100
            type_score = type_priority.get(target['type'], 0) * 10
            
            # PRODUCTIVITY BONUS - Key user insight implementation!
            coord_key = (target['x'], target['y'])
            productivity_bonus = 0
            avoidance_penalty = 0
            
            if coord_key in self.coordinate_results:
                result = self.coordinate_results[coord_key]
                
                # IMPROVED PRODUCTIVITY SYSTEM - Handles both positive and negative scores
                total_score_change = result.get('total_score_change', 0)
                
                # Calculate net productivity (positive = good, negative = bad)
                if total_score_change > 0:
                    # Net positive - give bonus (DYNAMIC MULTIPLIERS)
                    productivity_bonus = min(total_score_change * self.current_caps.productivity_multiplier, self.current_caps.productivity_max)
                    productivity_bonuses.append((target['x'], target['y'], productivity_bonus, total_score_change))
                elif total_score_change < 0:
                    # Net negative - give penalty (DYNAMIC MULTIPLIERS)
                    productivity_penalty = min(abs(total_score_change) * self.current_caps.recent_losses_multiplier, self.current_caps.recent_losses_max)
                    productivity_bonus = -productivity_penalty
                    productivity_penalties.append((target['x'], target['y'], productivity_penalty, total_score_change))
                
                # Check recent attempts for ongoing productivity (both positive and negative)
                recent_attempts = result.get('recent_attempts', [])
                recent_positive = sum(attempt.get('score_change', 0) 
                                    for attempt in recent_attempts 
                                    if attempt.get('score_change', 0) > 0)
                recent_negative = sum(attempt.get('score_change', 0) 
                                    for attempt in recent_attempts 
                                    if attempt.get('score_change', 0) < 0)
                
                # Recent gains bonus (only if net positive) (DYNAMIC MULTIPLIERS)
                if recent_positive > 0:
                    recent_net = recent_positive + recent_negative  # recent_negative is already negative
                    if recent_net > 0:
                        recent_bonus = min(recent_net * self.current_caps.recent_gains_multiplier, self.current_caps.recent_gains_max)
                        productivity_bonus += recent_bonus
                        recent_gains_bonuses.append((target['x'], target['y'], recent_bonus, recent_net))
                    elif recent_net < 0:
                        recent_penalty = min(abs(recent_net) * self.current_caps.recent_losses_multiplier, self.current_caps.recent_losses_max)
                        productivity_bonus -= recent_penalty
                        recent_losses_penalties.append((target['x'], target['y'], recent_penalty, recent_net))
                
                # PENALTY for stuck coordinates (but reduced if they had past gains)
                if result.get('is_stuck_coordinate', False):
                    if total_score_change > 0:
                        avoidance_penalty = 30  # Reduced penalty for previously productive stuck coords
                    else:
                        avoidance_penalty = 80  # Heavy penalty for unproductive stuck coords
                elif result.get('try_count', 0) > 0:
                    success_count = result.get('success_count', 0)
                    try_count = result['try_count']
                    
                    if success_count == 0 and try_count > 3:
                        avoidance_penalty = 40  # Penalty for repeatedly unsuccessful
                    elif success_count > 0 and total_score_change == 0:
                        avoidance_penalty = 20  # Light penalty for "successful" but no score gain
            
            # MOVEMENT BONUS for objects that have moved (DYNAMIC)
            movement_bonus = 0
            if 'object_id' in target and target['object_id'] in self.color_object_tracker:
                tracker = self.color_object_tracker[target['object_id']]
                if tracker.get('movement_vector') and tracker['movement_vector'] != (0, 0):
                    movement_bonus = self.current_caps.movement_bonus  # Dynamic bonus for moving objects
            
            # EXPLORATION BONUS for unexplored regions (DYNAMIC)
            exploration_bonus = 0
            if coord_key not in self.tried_coordinates:
                exploration_bonus = self.current_caps.exploration_bonus  # Dynamic bonus for new coordinates
            
            target['priority_score'] = (
                base_score + 
                type_score + 
                productivity_bonus +  # NEW: Major boost for productive coordinates
                movement_bonus + 
                exploration_bonus - 
                avoidance_penalty
            )
            
            target['productivity_bonus'] = productivity_bonus
            target['avoidance_penalty'] = avoidance_penalty
            target['movement_bonus'] = movement_bonus
            target['exploration_bonus'] = exploration_bonus
        
        # Consolidated logging - only show if there are bonuses/penalties
        if productivity_bonuses:
            coords_str = ",".join([f"({x},{y})" for x, y, bonus, score in productivity_bonuses])
            avg_bonus = sum(bonus for _, _, bonus, _ in productivity_bonuses) // len(productivity_bonuses)
            avg_score = sum(score for _, _, _, score in productivity_bonuses) / len(productivity_bonuses)
            print(f"🎯 PRODUCTIVITY BONUS: +{avg_bonus} {avg_score:.1f} score net gain {coords_str}")
        
        if productivity_penalties:
            coords_str = ",".join([f"({x},{y})" for x, y, penalty, score in productivity_penalties])
            avg_penalty = sum(penalty for _, _, penalty, _ in productivity_penalties) // len(productivity_penalties)
            avg_score = sum(score for _, _, _, score in productivity_penalties) / len(productivity_penalties)
            print(f"⚠️ PRODUCTIVITY PENALTY: -{avg_penalty} {avg_score:.1f} score loss {coords_str}")
        
        if recent_gains_bonuses:
            coords_str = ",".join([f"({x},{y})" for x, y, bonus, score in recent_gains_bonuses])
            avg_bonus = sum(bonus for _, _, bonus, _ in recent_gains_bonuses) // len(recent_gains_bonuses)
            avg_score = sum(score for _, _, _, score in recent_gains_bonuses) / len(recent_gains_bonuses)
            print(f"🔥 RECENT GAINS BONUS: +{avg_bonus} {avg_score:.1f} recent net gain {coords_str}")
        
        if recent_losses_penalties:
            coords_str = ",".join([f"({x},{y})" for x, y, penalty, score in recent_losses_penalties])
            avg_penalty = sum(penalty for _, _, penalty, _ in recent_losses_penalties) // len(recent_losses_penalties)
            avg_score = sum(score for _, _, _, score in recent_losses_penalties) / len(recent_losses_penalties)
            print(f"❄️ RECENT LOSSES PENALTY: -{avg_penalty} {avg_score:.1f} recent net loss {coords_str}")
        
        return sorted(targets, key=lambda t: t['priority_score'], reverse=True)
    
    def _generate_exploration_coordinate(self, frame_array: np.ndarray) -> Optional[Tuple[int, int]]:
        """ENHANCED: Generate systematic exploration avoiding previously tried coordinates."""
        # Use spiral exploration pattern from center outward
        center_x, center_y = frame_array.shape[1] // 2, frame_array.shape[0] // 2
        
        # Generate spiral coordinates
        spiral_coords = self._generate_spiral_coordinates(center_x, center_y, max_radius=25)
        
        # Find first coordinate not yet tried
        for x, y in spiral_coords:
            # Ensure within bounds
            if 0 <= x < frame_array.shape[1] and 0 <= y < frame_array.shape[0]:
                coord_key = (x, y)
                if coord_key not in self.tried_coordinates:
                    return (x, y)
                    
                # If coordinate was tried but unsuccessful, consider retrying after many attempts
                if coord_key in self.tried_coordinates:
                    try_count = self.coordinate_results.get(coord_key, {}).get('try_count', 0)
                    if try_count < 2:  # Allow up to 2 retries for exploration
                        return (x, y)
        
        # If all spiral coordinates tried, use frame-based exploration
        frame_hash = hash(frame_array.tobytes()) % 1000
        attempts = 0
        
        while attempts < 50:  # Max 50 attempts to find untried coordinate
            explore_x = (frame_hash + attempts * 7) % frame_array.shape[1]
            explore_y = ((frame_hash + attempts * 11) // frame_array.shape[1]) % frame_array.shape[0]
            
            coord_key = (explore_x, explore_y)
            if coord_key not in self.tried_coordinates:
                return (explore_x, explore_y)
            
            attempts += 1
        
        # Last resort: return center with offset
        offset = len(self.tried_coordinates) % 20
        return (center_x + offset - 10, center_y + offset - 10)
    
    def _generate_spiral_coordinates(self, center_x: int, center_y: int, max_radius: int) -> List[Tuple[int, int]]:
        """Generate coordinates in a spiral pattern from center outward."""
        coords = [(center_x, center_y)]
        
        for radius in range(1, max_radius):
            # Generate coordinates in a square spiral at this radius
            for i in range(radius * 8):  # 8 points per radius level
                angle = (i / (radius * 8)) * 2 * np.pi
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                coords.append((x, y))
        
        return coords

    def record_coordinate_attempt(self, x: int, y: int, was_successful: bool, score_change: float = 0):
        """Record the result of trying a specific coordinate for learning."""
        coord_key = (x, y)
        self.tried_coordinates.add(coord_key)
        
        if coord_key not in self.coordinate_results:
            self.coordinate_results[coord_key] = {
                'try_count': 0,
                'success_count': 0,
                'total_score_change': 0.0,
                'last_tried_frame': len(self.frame_history)
            }
        
        result = self.coordinate_results[coord_key]
        result['try_count'] += 1
        result['last_tried_frame'] = len(self.frame_history)
        result['total_score_change'] += score_change
        
        if was_successful or score_change > 0:
            result['success_count'] += 1
            # Track successful coordinates for pattern learning
            if len(self.last_successful_coords) >= 10:
                self.last_successful_coords.pop(0)
            self.last_successful_coords.append((x, y, len(self.frame_history)))
    
    def get_movement_analysis(self) -> Dict[str, Any]:
        """Get comprehensive movement analysis of all tracked objects."""
        movement_summary = {
            'tracked_objects': len(self.color_object_tracker),
            'moving_objects': 0,
            'static_objects': 0,
            'object_details': {}
        }
        
        for object_id, tracker in self.color_object_tracker.items():
            movement_vector = tracker.get('movement_vector', (0, 0))
            if movement_vector is None:
                movement_vector = (0, 0)
            
            is_moving = movement_vector != (0, 0) and (abs(movement_vector[0]) > 0.5 or abs(movement_vector[1]) > 0.5)
            
            if is_moving:
                movement_summary['moving_objects'] += 1
            else:
                movement_summary['static_objects'] += 1
            
            if len(tracker['positions']) > 1:
                last_pos = tracker['positions'][-1]
                first_pos = tracker['positions'][0]
                total_movement = np.sqrt((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)
                
                movement_summary['object_details'][object_id] = {
                    'color': tracker['color'],
                    'current_position': (last_pos[0], last_pos[1]),
                    'is_moving': is_moving,
                    'movement_vector': movement_vector,
                    'total_distance_moved': total_movement,
                    'size': tracker['size_history'][-1] if tracker['size_history'] else 0
                }
        
        return movement_summary
    
    def reset_coordinate_tracking(self):
        """Reset coordinate tracking for new game or when needed."""
        self.tried_coordinates.clear()
        self.coordinate_results.clear()
        self.color_object_tracker.clear()
        self.last_successful_coords.clear()
        self.exploration_spiral = 0
        
        # Reset interaction logging for new game
        self.interaction_log.clear()
        self.object_interaction_history.clear()
    
    def reset_avoidance_scores(self, reason: str = "game_state_change"):
        """Reset avoidance scores for all coordinates to allow re-exploration."""
        reset_count = 0
        for coord_key, result in self.coordinate_results.items():
            # Reset stuck status and zero progress streak
            if result.get('is_stuck_coordinate', False):
                result['is_stuck_coordinate'] = False
                reset_count += 1
            
            # Reduce zero progress streak by half (partial forgiveness)
            if 'zero_progress_streak' in result:
                result['zero_progress_streak'] = max(0, result['zero_progress_streak'] // 2)
        
        if reset_count > 0:
            print(f"🔄 Reset avoidance scores for {reset_count} coordinates due to {reason}")
    
    def decay_avoidance_scores(self):
        """Apply time-based decay to all avoidance scores."""
        current_frame = len(self.frame_history)
        decayed_count = 0
        
        for coord_key, result in self.coordinate_results.items():
            frames_since_last_try = current_frame - result.get('last_tried_frame', 0)
            
            # Only decay if it's been a while since last try
            if frames_since_last_try > 10:
                # Apply decay to zero progress streak
                if 'zero_progress_streak' in result and result['zero_progress_streak'] > 0:
                    decay_factor = 0.99  # 1% decay per frame
                    result['zero_progress_streak'] = max(0, int(result['zero_progress_streak'] * (decay_factor ** frames_since_last_try)))
                    decayed_count += 1
                
                # Remove stuck status if it's been a long time
                if result.get('is_stuck_coordinate', False) and frames_since_last_try > 50:
                    result['is_stuck_coordinate'] = False
                    decayed_count += 1
        
        if decayed_count > 0:
            print(f"⏰ Applied decay to {decayed_count} coordinate avoidance scores")
    
    def get_meta_learning_summary(self) -> Dict[str, Any]:
        """Get meta-learning performance summary."""
        return self.meta_learner.get_performance_summary()
    
    def get_current_caps(self) -> Dict[str, Any]:
        """Get current dynamic cap configuration."""
        return self.current_caps.to_dict()

    def log_action6_interaction(self, x: int, y: int, target_info: Dict[str, Any], 
                               before_state: Dict[str, Any], after_state: Dict[str, Any],
                               score_change: float = 0, game_id: str = None) -> str:
        """
        Log detailed information about an ACTION6 interaction with a visual object.
        
        Args:
            x, y: Coordinates clicked
            target_info: Information about the visual target (from frame analysis)
            before_state: Game state before the action
            after_state: Game state after the action  
            score_change: Change in game score
            game_id: Current game identifier
            
        Returns:
            Unique interaction ID for reference
        """
        import uuid
        from datetime import datetime

        interaction_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # Extract visual characteristics
        visual_data = {
            'coordinate': (x, y),
            'color': target_info.get('color', 'unknown'),
            'object_type': target_info.get('type', 'unknown'),
            'object_size': target_info.get('size', 0),
            'confidence': target_info.get('confidence', 0.0),
            'targeting_reason': target_info.get('reason', 'unknown'),
            'shape_detected': self._detect_shape_at_coordinate(x, y, before_state.get('frame')),
            'local_context': self._analyze_local_context(x, y, before_state.get('frame'))
        }
        
        # Analyze what changed
        interaction_result = {
            'score_change': score_change,
            'state_change': before_state.get('state') != after_state.get('state'),
            'new_actions_available': after_state.get('available_actions', []),
            'actions_removed': list(set(before_state.get('available_actions', [])) - 
                                  set(after_state.get('available_actions', []))),
            'actions_added': list(set(after_state.get('available_actions', [])) - 
                                set(before_state.get('available_actions', []))),
            'frame_changes': self._analyze_frame_changes(before_state.get('frame'), 
                                                        after_state.get('frame')),
            'success_indicators': self._detect_success_indicators(before_state, after_state, score_change)
        }
        
        # Create comprehensive interaction record
        interaction_record = {
            'id': interaction_id,
            'timestamp': timestamp,
            'game_id': game_id or 'unknown',
            'visual_data': visual_data,
            'interaction_result': interaction_result,
            'context': {
                'frame_number': len(self.frame_history),
                'total_interactions': len(self.interaction_log),
                'previous_attempts_at_coord': self.coordinate_results.get((x, y), {}).get('try_count', 0),
                'nearby_interactions': self._find_nearby_interactions(x, y, radius=5)
            }
        }
        
        # Store in various tracking systems
        self.interaction_log.append(interaction_record)
        
        # Update object-specific history
        object_id = target_info.get('object_id', f'obj_{x}_{y}')
        if object_id not in self.object_interaction_history:
            self.object_interaction_history[object_id] = []
        self.object_interaction_history[object_id].append(interaction_record)
        
        # Update color behavior patterns
        color = visual_data['color']
        if color != 'unknown':
            if color not in self.color_behavior_patterns:
                self.color_behavior_patterns[color] = {
                    'interactions': [],
                    'success_rate': 0.0,
                    'common_effects': [],
                    'best_contexts': []
                }
            self.color_behavior_patterns[color]['interactions'].append(interaction_record)
            self._update_color_behavior_analysis(color)
        
        # Update spatial interaction map
        coord_key = (x, y)
        if coord_key not in self.spatial_interaction_map:
            self.spatial_interaction_map[coord_key] = []
        self.spatial_interaction_map[coord_key].append(interaction_record)
        
        print(f"📝 INTERACTION LOGGED: {interaction_id}")
        print(f"   🎯 Target: {visual_data['object_type']} color_{color} at ({x},{y})")
        print(f"   📊 Result: Score {score_change:+.1f}, Actions {len(interaction_result['actions_added'])} added, {len(interaction_result['actions_removed'])} removed")
        print(f"   🔍 Success indicators: {len(interaction_result['success_indicators'])}")
        
        # Record coordinate effectiveness for the avoidance system
        api_success = len(interaction_result['success_indicators']) > 0 or interaction_result['state_change']
        context_info = f"interaction_{interaction_id}_indicators_{len(interaction_result['success_indicators'])}"
        self._record_coordinate_effectiveness(x, y, api_success, score_change, context_info)

        # Append an action trace for this interaction (if logging available)
        try:
            from arc_integration.action_trace_logger import log_action_trace
            import time as _time
            trace = {
                'ts': _time.time(),
                'game_id': game_id or interaction_record.get('game_id', 'unknown'),
                'interaction_id': interaction_id,
                'x': x,
                'y': y,
                'score_change': score_change,
                'success': bool(api_success)
            }
            log_action_trace(trace)
        except Exception:
            pass
        
        return interaction_id
    
    def _detect_shape_at_coordinate(self, x: int, y: int, frame: List[List[int]]) -> Dict[str, Any]:
        """Analyze the shape/pattern at the clicked coordinate."""
        if not frame:
            return {'shape_type': 'unknown', 'details': {}}
            
        try:
            frame_array = self._normalize_local_frame(frame)
            
            # Extract local region around click point
            region_size = 7
            y_start = max(0, y - region_size//2)
            y_end = min(frame_array.shape[0], y + region_size//2 + 1)
            x_start = max(0, x - region_size//2) 
            x_end = min(frame_array.shape[1], x + region_size//2 + 1)
            
            local_region = frame_array[y_start:y_end, x_start:x_end]
            
            # Analyze shape characteristics
            unique_colors = len(np.unique(local_region))
            center_color = frame_array[y, x] if 0 <= y < frame_array.shape[0] and 0 <= x < frame_array.shape[1] else -1
            
            # Check for geometric patterns
            shape_analysis = {
                'shape_type': 'pixel_cluster',
                'details': {
                    'region_size': local_region.shape,
                    'unique_colors': unique_colors,
                    'center_color': int(center_color),
                    'dominant_color': int(np.bincount(local_region.flatten()).argmax()),
                    'is_uniform': unique_colors == 1,
                    'has_pattern': unique_colors > 2
                }
            }
            
            return shape_analysis
            
        except Exception as e:
            return {'shape_type': 'analysis_error', 'details': {'error': str(e)}}
    
    def _analyze_local_context(self, x: int, y: int, frame: List[List[int]]) -> Dict[str, Any]:
        """Analyze the local context around a clicked coordinate."""
        if not frame:
            return {}
            
        try:
            frame_array = self._normalize_local_frame(frame)
            
            # Analyze surrounding area
            context_size = 10
            y_start = max(0, y - context_size)
            y_end = min(frame_array.shape[0], y + context_size + 1)
            x_start = max(0, x - context_size)
            x_end = min(frame_array.shape[1], x + context_size + 1)
            
            context_region = frame_array[y_start:y_end, x_start:x_end]
            
            return {
                'surrounding_colors': list(np.unique(context_region).astype(int)),
                'context_diversity': len(np.unique(context_region)),
                'distance_to_edge': min(x, y, frame_array.shape[1] - x - 1, frame_array.shape[0] - y - 1),
                'is_near_boundary': min(x, y, frame_array.shape[1] - x - 1, frame_array.shape[0] - y - 1) < 3,
                'local_density': float(np.count_nonzero(context_region)) / context_region.size
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_frame_changes(self, before_frame: List[List[int]], after_frame: List[List[int]]) -> Dict[str, Any]:
        """Analyze what changed in the frame after the interaction."""
        if not before_frame or not after_frame:
            return {'changes_detected': False}
            
        try:
            # Normalize both frames
            before_array = self._normalize_local_frame(before_frame)
            after_array = self._normalize_local_frame(after_frame)
            
            # Calculate differences
            if not np.array_equal(before_array.shape, after_array.shape):
                return {
                    'changes_detected': True,
                    'change_type': 'frame_size_change',
                    'before_size': before_array.shape,
                    'after_size': after_array.shape
                }
            
            diff = before_array != after_array
            num_changes = np.sum(diff)
            
            if num_changes == 0:
                return {'changes_detected': False, 'change_type': 'no_visual_changes'}
            
            # Find areas of change
            changed_coords = np.where(diff)
            change_locations = list(zip(changed_coords[1], changed_coords[0]))  # (x, y) format
            
            return {
                'changes_detected': True,
                'change_type': 'visual_changes',
                'num_pixels_changed': int(num_changes),
                'change_percentage': float(num_changes) / before_array.size,
                'change_locations': change_locations[:20],  # Limit to first 20 for storage
                'change_center': (float(np.mean(changed_coords[1])), float(np.mean(changed_coords[0]))) if len(changed_coords[0]) > 0 else None
            }
            
        except Exception as e:
            return {'changes_detected': False, 'error': str(e)}
    
    def _detect_success_indicators(self, before_state: Dict, after_state: Dict, score_change: float) -> List[str]:
        """Detect various indicators of successful interaction."""
        indicators = []
        
        # Score-based indicators
        if score_change > 0:
            indicators.append(f'score_increase_{score_change}')
        
        # State change indicators
        if before_state.get('state') != after_state.get('state'):
            indicators.append('game_state_changed')
        
        # Action availability indicators
        before_actions = set(before_state.get('available_actions', []))
        after_actions = set(after_state.get('available_actions', []))
        
        if len(after_actions) > len(before_actions):
            indicators.append('new_actions_unlocked')
        
        if len(after_actions) < len(before_actions):
            indicators.append('actions_consumed')
        
        # Frame-based indicators (if visual changes occurred)
        frame_changes = self._analyze_frame_changes(before_state.get('frame'), after_state.get('frame'))
        if frame_changes.get('changes_detected'):
            indicators.append('visual_feedback')
            if frame_changes.get('num_pixels_changed', 0) > 10:
                indicators.append('significant_visual_change')
        
        return indicators
    
    def _find_nearby_interactions(self, x: int, y: int, radius: int = 5) -> List[Dict]:
        """Find previous interactions near this coordinate."""
        nearby = []
        
        for record in self.interaction_log:
            other_x, other_y = record['visual_data']['coordinate']
            distance = np.sqrt((x - other_x)**2 + (y - other_y)**2)
            
            if distance <= radius:
                nearby.append({
                    'id': record['id'],
                    'distance': distance,
                    'color': record['visual_data']['color'],
                    'success_indicators': len(record['interaction_result']['success_indicators']),
                    'score_change': record['interaction_result']['score_change']
                })
        
        return sorted(nearby, key=lambda x: x['distance'])[:5]  # Return closest 5
    
    def _update_color_behavior_analysis(self, color: int):
        """Update behavior pattern analysis for a specific color."""
        if color not in self.color_behavior_patterns:
            return
            
        pattern_data = self.color_behavior_patterns[color]
        interactions = pattern_data['interactions']
        
        if not interactions:
            return
        
        # Calculate success rate
        successful_interactions = sum(1 for record in interactions 
                                    if record['interaction_result']['score_change'] > 0 
                                    or len(record['interaction_result']['success_indicators']) > 0)
        pattern_data['success_rate'] = successful_interactions / len(interactions)
        
        # Identify common effects
        all_effects = []
        for record in interactions:
            all_effects.extend(record['interaction_result']['success_indicators'])
        
        from collections import Counter
        common_effects = Counter(all_effects).most_common(3)
        pattern_data['common_effects'] = [effect for effect, count in common_effects]
        
        # Find best contexts (simplified)
        successful_records = [r for r in interactions 
                            if r['interaction_result']['score_change'] > 0 
                            or len(r['interaction_result']['success_indicators']) > 0]
        
        pattern_data['best_contexts'] = [
            {
                'context': record['visual_data']['local_context'],
                'score_change': record['interaction_result']['score_change']
            }
            for record in successful_records[-3:]  # Last 3 successful contexts
        ]

    def generate_interaction_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Generate hypotheses about object behaviors based on interaction history.
        This method is designed to be called during sleep/consolidation phases.
        """
        if len(self.interaction_log) < 3:  # Need some data to form hypotheses
            return []
        
        new_hypotheses = []
        
        # HYPOTHESIS 1: Color-based behavior patterns
        color_hypotheses = self._generate_color_hypotheses()
        new_hypotheses.extend(color_hypotheses)
        
        # HYPOTHESIS 2: Spatial pattern hypotheses  
        spatial_hypotheses = self._generate_spatial_hypotheses()
        new_hypotheses.extend(spatial_hypotheses)
        
        # HYPOTHESIS 3: Context-based hypotheses
        context_hypotheses = self._generate_context_hypotheses()
        new_hypotheses.extend(context_hypotheses)
        
        # HYPOTHESIS 4: Sequence-based hypotheses
        sequence_hypotheses = self._generate_sequence_hypotheses()
        new_hypotheses.extend(sequence_hypotheses)
        
        # Add to database and remove duplicates
        for hypothesis in new_hypotheses:
            if not self._is_duplicate_hypothesis(hypothesis):
                hypothesis['generated_at'] = len(self.frame_history)
                hypothesis['confidence_score'] = self._calculate_hypothesis_confidence(hypothesis)
                self.hypothesis_database.append(hypothesis)
        
        # Sort by confidence and keep top hypotheses
        self.hypothesis_database.sort(key=lambda h: h['confidence_score'], reverse=True)
        self.hypothesis_database = self.hypothesis_database[:50]  # Keep top 50 hypotheses
        
        print(f"💡 HYPOTHESIS GENERATION: Generated {len(new_hypotheses)} new hypotheses")
        print(f"   📊 Total hypotheses in database: {len(self.hypothesis_database)}")
        print(f"   🎯 Top hypothesis: {self.hypothesis_database[0]['description'] if self.hypothesis_database else 'None'}")
        
        return new_hypotheses
    
    def _generate_color_hypotheses(self) -> List[Dict[str, Any]]:
        """Generate hypotheses about specific color behaviors."""
        hypotheses = []
        
        for color, pattern_data in self.color_behavior_patterns.items():
            interactions = pattern_data['interactions']
            if len(interactions) < 2:
                continue
                
            success_rate = pattern_data['success_rate']
            common_effects = pattern_data['common_effects']
            
            # High success rate hypothesis
            if success_rate > 0.7:
                hypotheses.append({
                    'type': 'color_behavior',
                    'description': f'Color {color} objects are highly effective (success rate: {success_rate:.1%})',
                    'prediction': f'Clicking color {color} objects will likely produce positive results',
                    'evidence': {
                        'interaction_count': len(interactions),
                        'success_rate': success_rate,
                        'common_effects': common_effects
                    },
                    'actionable_advice': f'Prioritize color {color} objects when available',
                    'color': color
                })
            
            # Specific effect hypothesis
            if common_effects:
                most_common_effect = common_effects[0]
                hypotheses.append({
                    'type': 'color_effect',
                    'description': f'Color {color} objects commonly cause: {most_common_effect}',
                    'prediction': f'Clicking color {color} objects will likely result in {most_common_effect}',
                    'evidence': {
                        'interaction_count': len(interactions),
                        'effect_frequency': common_effects
                    },
                    'actionable_advice': f'Use color {color} objects when {most_common_effect} is desired',
                    'color': color,
                    'expected_effect': most_common_effect
                })
        
        return hypotheses
    
    def _generate_spatial_hypotheses(self) -> List[Dict[str, Any]]:
        """Generate hypotheses about spatial patterns and locations."""
        hypotheses = []
        
        # Analyze successful coordinate clusters
        successful_coords = []
        for record in self.interaction_log:
            if (record['interaction_result']['score_change'] > 0 or 
                len(record['interaction_result']['success_indicators']) > 0):
                successful_coords.append(record['visual_data']['coordinate'])
        
        if len(successful_coords) >= 3:
            # Find spatial clusters of success
            clusters = self._find_coordinate_clusters(successful_coords)
            
            for cluster in clusters:
                if len(cluster['members']) >= 2:
                    hypotheses.append({
                        'type': 'spatial_cluster',
                        'description': f'Area around {cluster["center"]} has high success rate',
                        'prediction': f'Coordinates near {cluster["center"]} are likely to be effective',
                        'evidence': {
                            'successful_interactions': len(cluster['members']),
                            'cluster_center': cluster['center'],
                            'cluster_radius': cluster['radius']
                        },
                        'actionable_advice': f'Explore coordinates within {cluster["radius"]} pixels of {cluster["center"]}',
                        'target_area': cluster
                    })
        
        # Edge vs center hypothesis
        edge_successes = sum(1 for record in self.interaction_log 
                           if (record['interaction_result']['score_change'] > 0 or 
                               len(record['interaction_result']['success_indicators']) > 0) and
                               record['visual_data']['local_context'].get('is_near_boundary', False))
        
        total_edge_attempts = sum(1 for record in self.interaction_log 
                                if record['visual_data']['local_context'].get('is_near_boundary', False))
        
        if total_edge_attempts > 0:
            edge_success_rate = edge_successes / total_edge_attempts
            if edge_success_rate > 0.6:
                hypotheses.append({
                    'type': 'spatial_preference',
                    'description': 'Boundary/edge locations have higher success rates',
                    'prediction': 'Objects near frame boundaries are more likely to be interactive',
                    'evidence': {
                        'edge_success_rate': edge_success_rate,
                        'edge_successes': edge_successes,
                        'total_edge_attempts': total_edge_attempts
                    },
                    'actionable_advice': 'Prioritize objects near frame boundaries',
                    'spatial_preference': 'boundary'
                })
        
        return hypotheses
    
    def _generate_context_hypotheses(self) -> List[Dict[str, Any]]:
        """Generate hypotheses about contextual factors that influence success."""
        hypotheses = []
        
        # Context diversity hypothesis
        high_diversity_successes = []
        low_diversity_successes = []
        
        for record in self.interaction_log:
            context = record['visual_data']['local_context']
            diversity = context.get('context_diversity', 0)
            is_successful = (record['interaction_result']['score_change'] > 0 or 
                           len(record['interaction_result']['success_indicators']) > 0)
            
            if is_successful:
                if diversity > 3:
                    high_diversity_successes.append(record)
                else:
                    low_diversity_successes.append(record)
        
        if len(high_diversity_successes) > len(low_diversity_successes) and len(high_diversity_successes) > 2:
            hypotheses.append({
                'type': 'context_complexity',
                'description': 'Objects in visually complex areas are more interactive',
                'prediction': 'Target objects in areas with high color diversity',
                'evidence': {
                    'high_diversity_successes': len(high_diversity_successes),
                    'low_diversity_successes': len(low_diversity_successes)
                },
                'actionable_advice': 'Prioritize objects in areas with multiple different colors',
                'context_preference': 'high_diversity'
            })
        
        return hypotheses
    
    def _generate_sequence_hypotheses(self) -> List[Dict[str, Any]]:
        """Generate hypotheses about action sequences and timing."""
        hypotheses = []
        
        # Recent vs distant success patterns
        recent_interactions = self.interaction_log[-10:] if len(self.interaction_log) >= 10 else self.interaction_log
        recent_successes = sum(1 for record in recent_interactions 
                             if (record['interaction_result']['score_change'] > 0 or 
                                 len(record['interaction_result']['success_indicators']) > 0))
        
        if len(recent_interactions) > 0:
            recent_success_rate = recent_successes / len(recent_interactions)
            overall_success_rate = len([r for r in self.interaction_log 
                                      if (r['interaction_result']['score_change'] > 0 or 
                                          len(r['interaction_result']['success_indicators']) > 0)]) / len(self.interaction_log)
            
            if recent_success_rate > overall_success_rate * 1.5:  # Recent performance is much better
                hypotheses.append({
                    'type': 'learning_trend',
                    'description': 'Recent interactions show improved success patterns',
                    'prediction': 'Current strategy is working better than earlier approaches',
                    'evidence': {
                        'recent_success_rate': recent_success_rate,
                        'overall_success_rate': overall_success_rate,
                        'improvement_factor': recent_success_rate / overall_success_rate
                    },
                    'actionable_advice': 'Continue with current targeting approach',
                    'trend': 'improving'
                })
        
        return hypotheses
    
    def _find_coordinate_clusters(self, coordinates: List[Tuple[int, int]], max_distance: int = 8) -> List[Dict]:
        """Find clusters of coordinates that are close together."""
        if not coordinates:
            return []
        
        clusters = []
        used_coords = set()
        
        for coord in coordinates:
            if coord in used_coords:
                continue
                
            # Find all coordinates within max_distance
            cluster_members = [coord]
            used_coords.add(coord)
            
            for other_coord in coordinates:
                if other_coord in used_coords:
                    continue
                    
                distance = np.sqrt((coord[0] - other_coord[0])**2 + (coord[1] - other_coord[1])**2)
                if distance <= max_distance:
                    cluster_members.append(other_coord)
                    used_coords.add(other_coord)
            
            if len(cluster_members) >= 2:  # Only clusters with multiple points
                center_x = sum(c[0] for c in cluster_members) / len(cluster_members)
                center_y = sum(c[1] for c in cluster_members) / len(cluster_members)
                
                clusters.append({
                    'center': (center_x, center_y),
                    'members': cluster_members,
                    'radius': max_distance
                })
        
        return clusters
    
    def _is_duplicate_hypothesis(self, new_hypothesis: Dict) -> bool:
        """Check if a similar hypothesis already exists."""
        for existing in self.hypothesis_database:
            if (existing['type'] == new_hypothesis['type'] and 
                existing.get('color') == new_hypothesis.get('color') and
                abs(existing.get('confidence_score', 0) - self._calculate_hypothesis_confidence(new_hypothesis)) < 0.1):
                return True
        return False
    
    def _calculate_hypothesis_confidence(self, hypothesis: Dict) -> float:
        """Calculate confidence score for a hypothesis based on evidence strength."""
        base_confidence = 0.5
        evidence = hypothesis.get('evidence', {})
        
        # Adjust based on sample size
        interaction_count = evidence.get('interaction_count', 0)
        if interaction_count > 10:
            base_confidence += 0.3
        elif interaction_count > 5:
            base_confidence += 0.2
        elif interaction_count > 2:
            base_confidence += 0.1
        
        # Adjust based on success rate
        success_rate = evidence.get('success_rate', 0)
        base_confidence += success_rate * 0.3
        
        # Adjust based on hypothesis type
        if hypothesis['type'] in ['color_behavior', 'spatial_cluster']:
            base_confidence += 0.1  # These tend to be more reliable
        
        return min(1.0, base_confidence)
    
    def get_actionable_recommendations(self, current_frame: List[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Get actionable recommendations based on current hypotheses and frame analysis.
        This method provides specific guidance for ACTION6 targeting.
        """
        if not self.hypothesis_database:
            return []
        
        recommendations = []
        
        # Sort hypotheses by confidence
        top_hypotheses = sorted(self.hypothesis_database, key=lambda h: h['confidence_score'], reverse=True)[:10]
        
        for hypothesis in top_hypotheses:
            if hypothesis['confidence_score'] < 0.3:  # Skip low-confidence hypotheses
                continue
            
            rec = {
                'hypothesis_id': hypothesis.get('generated_at', 0),
                'type': hypothesis['type'],
                'recommendation': hypothesis.get('actionable_advice', ''),
                'confidence': hypothesis['confidence_score'],
                'reasoning': hypothesis['description'],
                'priority': self._calculate_recommendation_priority(hypothesis)
            }
            
            # Add specific targeting advice
            if hypothesis['type'] == 'color_behavior' and 'color' in hypothesis:
                rec['target_criteria'] = {'preferred_color': hypothesis['color']}
            elif hypothesis['type'] == 'spatial_cluster' and 'target_area' in hypothesis:
                rec['target_criteria'] = {'preferred_area': hypothesis['target_area']}
            elif hypothesis['type'] == 'context_complexity':
                rec['target_criteria'] = {'context_preference': hypothesis.get('context_preference')}
            
            recommendations.append(rec)
        
        return sorted(recommendations, key=lambda r: r['priority'], reverse=True)
    
    def _calculate_recommendation_priority(self, hypothesis: Dict) -> float:
        """Calculate priority score for a recommendation."""
        base_priority = hypothesis['confidence_score']
        
        # Boost priority for actionable hypotheses
        if hypothesis.get('actionable_advice'):
            base_priority += 0.2
        
        # Boost priority for recent hypotheses
        generation_time = hypothesis.get('generated_at', 0)
        current_time = len(self.frame_history)
        recency_factor = max(0, 1 - (current_time - generation_time) / 100.0)
        base_priority += recency_factor * 0.1
        
        return base_priority

    def consolidate_learning_during_sleep(self) -> Dict[str, Any]:
        """
        Perform comprehensive learning consolidation during sleep phases.
        This is the main method called by the sleep system.
        """
        print(f"🛏️ SLEEP CONSOLIDATION: Starting learning consolidation...")
        
        consolidation_results = {
            'new_hypotheses_generated': 0,
            'patterns_discovered': 0,
            'recommendations_updated': 0,
            'memory_optimization': {},
            'learning_insights': []
        }
        
        # Generate new hypotheses
        new_hypotheses = self.generate_interaction_hypotheses()
        consolidation_results['new_hypotheses_generated'] = len(new_hypotheses)
        
        # Discover new patterns
        new_patterns = self._discover_interaction_patterns()
        consolidation_results['patterns_discovered'] = len(new_patterns)
        
        # Update color behavior analysis
        self._consolidate_color_behaviors()
        
        # Optimize memory usage
        memory_stats = self._optimize_memory_usage()
        consolidation_results['memory_optimization'] = memory_stats
        
        # Generate learning insights
        insights = self._generate_learning_insights()
        consolidation_results['learning_insights'] = insights
        
        # Update recommendations
        recommendations = self.get_actionable_recommendations()
        consolidation_results['recommendations_updated'] = len(recommendations)
        
        print(f"✅ SLEEP CONSOLIDATION COMPLETE:")
        print(f"   📝 New hypotheses: {consolidation_results['new_hypotheses_generated']}")
        print(f"   🔍 Patterns found: {consolidation_results['patterns_discovered']}")
        print(f"   💡 Insights generated: {len(consolidation_results['learning_insights'])}")
        print(f"   🎯 Active recommendations: {consolidation_results['recommendations_updated']}")
        
        return consolidation_results
    
    def _discover_interaction_patterns(self) -> List[Dict[str, Any]]:
        """Discover new interaction patterns from recent data."""
        if len(self.interaction_log) < 5:
            return []
        
        patterns = []
        
        # Pattern 1: Sequential success patterns
        recent_log = self.interaction_log[-20:]  # Last 20 interactions
        success_sequences = []
        current_sequence = []
        
        for record in recent_log:
            is_successful = (record['interaction_result']['score_change'] > 0 or 
                           len(record['interaction_result']['success_indicators']) > 0)
            
            if is_successful:
                current_sequence.append(record)
            else:
                if len(current_sequence) >= 2:
                    success_sequences.append(current_sequence)
                current_sequence = []
        
        if len(current_sequence) >= 2:
            success_sequences.append(current_sequence)
        
        for sequence in success_sequences:
            if len(sequence) >= 3:  # Meaningful sequence
                patterns.append({
                    'type': 'success_sequence',
                    'description': f'Found successful sequence of {len(sequence)} interactions',
                    'sequence_data': {
                        'length': len(sequence),
                        'colors_involved': [r['visual_data']['dominant_color'] for r in sequence if 'dominant_color' in r.get('visual_data', {})],
                        'coordinates': [r['visual_data']['coordinate'] for r in sequence if 'coordinate' in r.get('visual_data', {})]
                    }
                })
        
        return patterns
    
    def _consolidate_color_behaviors(self):
        """Consolidate and refine color behavior patterns."""
        for color in list(self.color_behavior_patterns.keys()):
            pattern_data = self.color_behavior_patterns[color]
            interactions = pattern_data['interactions']
            
            if len(interactions) < 2:
                continue  # Need more data
            
            # Recalculate success rate
            successful = sum(1 for i in interactions if i['was_successful'])
            pattern_data['success_rate'] = successful / len(interactions)
            
            # Update common effects
            effects = [i['effects'] for i in interactions if i['effects']]
            all_effects = []
            for effect_list in effects:
                all_effects.extend(effect_list)
            
            if all_effects:
                from collections import Counter
                effect_counts = Counter(all_effects)
                pattern_data['common_effects'] = [effect for effect, count in effect_counts.most_common(3)]
    
    def _optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory by removing old/irrelevant data."""
        initial_counts = {
            'frame_history': len(self.frame_history),
            'interaction_log': len(self.interaction_log),
            'hypothesis_database': len(self.hypothesis_database)
        }
        
        # Keep only last 100 frames
        if len(self.frame_history) > 100:
            self.frame_history = self.frame_history[-100:]
        
        # Keep only last 200 interactions
        if len(self.interaction_log) > 200:
            self.interaction_log = self.interaction_log[-200:]
        
        # Remove low-confidence old hypotheses
        current_time = len(self.frame_history)
        self.hypothesis_database = [
            h for h in self.hypothesis_database
            if h['confidence_score'] > 0.3 or (current_time - h.get('generated_at', 0)) < 50
        ]
        
        final_counts = {
            'frame_history': len(self.frame_history),
            'interaction_log': len(self.interaction_log),
            'hypothesis_database': len(self.hypothesis_database)
        }
        
        return {
            'initial_counts': initial_counts,
            'final_counts': final_counts,
            'memory_saved': {
                'frame_history': initial_counts['frame_history'] - final_counts['frame_history'],
                'interaction_log': initial_counts['interaction_log'] - final_counts['interaction_log'],
                'hypothesis_database': initial_counts['hypothesis_database'] - final_counts['hypothesis_database']
            }
        }
    
    def _generate_learning_insights(self) -> List[str]:
        """Generate high-level insights about learning progress."""
        insights = []
        
        if len(self.interaction_log) < 5:
            insights.append("Need more interaction data to generate meaningful insights")
            return insights
        
        # Overall success trend
        recent_success_rate = self._calculate_recent_success_rate()
        overall_success_rate = self._calculate_overall_success_rate()
        
        if recent_success_rate > overall_success_rate * 1.2:
            insights.append("🔥 Learning trend: Performance improving over time")
        elif recent_success_rate < overall_success_rate * 0.8:
            insights.append("⚠️ Learning trend: Recent performance decline - need strategy adjustment")
        else:
            insights.append("➡️ Learning trend: Performance stable")
        
        # Color strategy insights
        best_colors = self._get_best_performing_colors()
        if best_colors:
            insights.append(f"🎨 Best performing colors: {', '.join(best_colors[:3])}")
        
        # Spatial strategy insights
        if len(self.tried_coordinates) > 20:
            coverage = len(self.tried_coordinates) / (80 * 80) * 100  # Assuming 80x80 grid
            insights.append(f"🗺️ Frame exploration: {coverage:.1f}% coverage achieved")
        
        # Hypothesis quality
        high_confidence_hypotheses = len([h for h in self.hypothesis_database if h['confidence_score'] > 0.7])
        insights.append(f"💡 Knowledge base: {high_confidence_hypotheses} high-confidence hypotheses")
        
        return insights
    
    def _calculate_recent_success_rate(self, window_size: int = 10) -> float:
        """Calculate success rate for recent interactions."""
        if len(self.interaction_log) == 0:
            return 0.0
        
        recent_log = self.interaction_log[-window_size:]
        successes = sum(1 for record in recent_log 
                       if (record['interaction_result']['score_change'] > 0 or 
                           len(record['interaction_result']['success_indicators']) > 0))
        
        return successes / len(recent_log)
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate."""
        if len(self.interaction_log) == 0:
            return 0.0
        
        successes = sum(1 for record in self.interaction_log 
                       if (record['interaction_result']['score_change'] > 0 or 
                           len(record['interaction_result']['success_indicators']) > 0))
        
        return successes / len(self.interaction_log)
    
    def _get_best_performing_colors(self, min_interactions: int = 3) -> List[str]:
        """Get list of best performing colors."""
        color_performance = []
        
        for color, pattern_data in self.color_behavior_patterns.items():
            if len(pattern_data['interactions']) >= min_interactions:
                color_performance.append((color, pattern_data['success_rate']))
        
        color_performance.sort(key=lambda x: x[1], reverse=True)
        return [color for color, rate in color_performance if rate > 0.5]
    
    def get_current_learning_state(self) -> Dict[str, Any]:
        """Get comprehensive snapshot of current learning state."""
        return {
            'interaction_count': len(self.interaction_log),
            'hypothesis_count': len(self.hypothesis_database),
            'color_patterns_tracked': len(self.color_behavior_patterns),
            'coordinates_explored': len(self.tried_coordinates),
            'recent_success_rate': self._calculate_recent_success_rate(),
            'overall_success_rate': self._calculate_overall_success_rate(),
            'best_colors': self._get_best_performing_colors(),
            'top_hypotheses': [h['description'] for h in sorted(self.hypothesis_database, 
                                                               key=lambda x: x['confidence_score'], 
                                                               reverse=True)[:3]],
            'current_recommendations': len(self.get_actionable_recommendations()),
            'learning_insights': self._generate_learning_insights()
        }

    def _record_coordinate_effectiveness(self, x: int, y: int, success: bool, 
                                       score_change: float = 0, context: str = ""):
        """Enhanced coordinate effectiveness tracking with detailed analysis."""
        coord_key = (x, y)
        
        if coord_key not in self.coordinate_results:
            self.coordinate_results[coord_key] = {
                'try_count': 0,
                'success_count': 0,
                'total_score_change': 0,
                'last_attempt': 0,
                'effectiveness_score': 0,
                'contexts': [],
                'recent_attempts': deque(maxlen=5),  # Track last 5 attempts
                'zero_progress_streak': 0,  # Track consecutive zero-progress attempts
                'is_stuck_coordinate': False,  # Flag for coordinates that cause loops
                'penalty_score': 0.0  # Penalty for ineffective coordinates
            }
        
        result = self.coordinate_results[coord_key]
        
        # DEFENSIVE: Ensure all required keys exist (for backward compatibility)
        if 'zero_progress_streak' not in result:
            result['zero_progress_streak'] = 0
        if 'is_stuck_coordinate' not in result:
            result['is_stuck_coordinate'] = False
        if 'penalty_score' not in result:
            result['penalty_score'] = 0.0
        if 'recent_attempts' not in result:
            result['recent_attempts'] = deque(maxlen=5)
        result['try_count'] += 1
        result['last_attempt'] = len(self.frame_history)
        if 'contexts' not in result:
            result['contexts'] = []
        result['contexts'].append(context)
        
        # ENHANCED SUCCESS DETECTION - Zero score change = ineffective
        is_truly_effective = success and score_change > 0
        made_progress = score_change != 0  # Any score change (positive or negative) is progress
        
        # SCORE INCREASE OVERRIDES STUCK STATUS - Key insight from user!
        if score_change > 0:
            # Reset all negative tracking when score increases
            result['zero_progress_streak'] = 0
            result['success_count'] += 1
            result['total_score_change'] += score_change
            
            # CRITICAL: Remove stuck flag if coordinate produces score increase
            if result['is_stuck_coordinate']:
                result['is_stuck_coordinate'] = False
                result['penalty_score'] = max(result['penalty_score'] - 0.5, 0.0)  # Bigger penalty reduction
                print(f"🎯 COORDINATE ({x},{y}) PRODUCTIVE: Score +{score_change} - removing stuck flag!")
        elif score_change == 0:
            # Only increment zero progress if no score gained
            result['zero_progress_streak'] += 1
        else:
            # Negative score change still counts as some form of progress
            result['zero_progress_streak'] = 0
        
        # Mark as stuck coordinate ONLY if too many consecutive zero-progress attempts
        # AND no recent positive score changes
        if result['zero_progress_streak'] >= 8:  # Increased threshold to be less aggressive
            # Check recent attempts for any positive score changes
            recent_positive_scores = [attempt for attempt in result['recent_attempts'] 
                                    if attempt.get('score_change', 0) > 0]
            
            # Only mark as stuck if no positive scores in recent attempts
            if not recent_positive_scores:
                result['is_stuck_coordinate'] = True
                result['penalty_score'] = min(result['penalty_score'] + 0.2, 1.0)
                print(f"⚠️ COORDINATE ({x},{y}) MARKED AS STUCK: {result['zero_progress_streak']} zero-progress attempts, no recent gains")
        
        if is_truly_effective and score_change == 0:
            # Handle case where action was "successful" but no score change
            result['success_count'] += 1
        
        # Track recent attempt results
        attempt_result = {
            'success': is_truly_effective,
            'score_change': score_change,
            'frame_num': len(self.frame_history),
            'made_progress': made_progress
        }
        result['recent_attempts'].append(attempt_result)
        
        # Update meta-learner with performance data
        bonus_type = "productivity" if score_change > 0 else "recent_losses" if score_change < 0 else "exploration"
        is_exploration = coord_key not in self.tried_coordinates
        self.meta_learner.update_performance(score_change, bonus_type, is_exploration)
        
        # Update current caps from meta-learner
        self.current_caps = self.meta_learner.get_current_caps()
        
        # ENHANCED EFFECTIVENESS SCORING with penalties for stuck coordinates
        if result['try_count'] > 0:
            base_success_rate = result['success_count'] / result['try_count']
            score_bonus = min(result['total_score_change'] / max(result['try_count'], 1), 0.5)
            
            # Apply penalties for stuck coordinates
            stuck_penalty = result['penalty_score'] * 0.8  # Heavy penalty for stuck coordinates
            zero_streak_penalty = min(result['zero_progress_streak'] * 0.1, 0.7)  # Progressive penalty
            
            result['effectiveness_score'] = max(0.0, base_success_rate + score_bonus - stuck_penalty - zero_streak_penalty)
            
            # Extra penalty for coordinates with high try count but no real progress
            if result['try_count'] > 10 and result['total_score_change'] <= 0:
                result['effectiveness_score'] = max(0.0, result['effectiveness_score'] - 0.5)
        
        # Add coordinate to tried set for avoidance
        self.tried_coordinates.add(coord_key)

    def get_coordinate_avoidance_score(self, x: int, y: int) -> float:
        """Enhanced avoidance score with decay and forgetting mechanisms."""
        coord_key = (x, y)
        
        if coord_key not in self.coordinate_results:
            return 0.0  # No penalty for untried coordinates
        
        result = self.coordinate_results[coord_key]
        current_frame = len(self.frame_history)
        frames_since_last_try = current_frame - result.get('last_tried_frame', 0)
        
        # 1. IMPROVED PRODUCTIVITY ASSESSMENT - Handles both positive and negative scores
        recent_attempts = result.get('recent_attempts', [])
        recent_positive = sum(attempt.get('score_change', 0) 
                            for attempt in recent_attempts 
                            if attempt.get('score_change', 0) > 0)
        recent_negative = sum(attempt.get('score_change', 0) 
                            for attempt in recent_attempts 
                            if attempt.get('score_change', 0) < 0)
        recent_net = recent_positive + recent_negative  # recent_negative is already negative
        
        # Check total score change for overall productivity
        total_score_change = result.get('total_score_change', 0)
        
        if total_score_change > 0:
            # Net positive coordinate - actively seek it
            productivity_bonus = min(total_score_change * 0.1, 0.5)
            return -productivity_bonus  # Negative = attractive, not avoided
        elif total_score_change < 0:
            # Net negative coordinate - avoid it more strongly
            productivity_penalty = min(abs(total_score_change) * 0.15, 0.8)
            return productivity_penalty  # Positive = avoid more strongly
        
        # For zero net change, check recent trends
        if recent_net > 0:
            # Recent positive trend - slight attraction
            return -min(recent_net * 0.05, 0.2)
        elif recent_net < 0:
            # Recent negative trend - slight avoidance
            return min(abs(recent_net) * 0.1, 0.4)
        
        # 2. TIME-BASED DECAY: Avoidance scores decay over time
        base_avoidance = 0.0
        
        if result.get('is_stuck_coordinate', False):
            # Stuck coordinates are permanently avoided - no decay
            return 1.0  # Maximum avoidance score
        else:
            # Progressive avoidance based on zero progress streak
            streak_penalty = min(result['zero_progress_streak'] * 0.1, 0.6)
            base_avoidance = streak_penalty
            
            # High try count with no progress gets additional penalty
            if result['try_count'] > 8 and result['total_score_change'] <= 0:
                base_avoidance += 0.2
        
        # 3. DECAY CALCULATION
        if frames_since_last_try > 0:
            # Exponential decay: avoidance decreases over time
            decay_rate = 0.95  # 5% decay per frame
            decay_factor = decay_rate ** frames_since_last_try
            decayed_avoidance = base_avoidance * decay_factor
            
            # 4. ADAPTIVE DECAY: Faster decay for coordinates with mixed history
            if result.get('success_count', 0) > 0:
                # Coordinates that have succeeded before decay faster
                adaptive_decay = 0.98  # 2% decay per frame
                decay_factor = adaptive_decay ** frames_since_last_try
                decayed_avoidance = base_avoidance * decay_factor
            
            # 5. EXPLORATION INCENTIVE: Very old coordinates get exploration bonus
            if frames_since_last_try > 100:
                exploration_bonus = -0.1  # Slight attraction for very old coordinates
                decayed_avoidance = max(0.0, decayed_avoidance + exploration_bonus)
            
            # 6. MINIMUM THRESHOLD: Never completely forget, but make very old memories weak
            min_avoidance = 0.1 if base_avoidance > 0.5 else 0.0
            final_avoidance = max(decayed_avoidance, min_avoidance)
            
            return final_avoidance
        
        return base_avoidance

    def _check_exploration_phase(self, frame_array: np.ndarray, ranked_targets: List[Dict]) -> Optional[Dict]:
        """
        EXPLORATION PHASE - User insight implementation.
        Systematically click every color object at least once to discover what's clickable.
        """
        if self.exploration_complete:
            return None
        
        # 1. DETECT ALL COLORS in current frame
        unique_colors = set(np.unique(frame_array))
        self.current_game_colors.update(unique_colors)
        
        # Remove background color (assume 0 is background)
        unique_colors.discard(0)
        
        # 2. IDENTIFY UNEXPLORED COLORS
        unexplored_colors = unique_colors - self.explored_color_objects
        
        if not unexplored_colors:
            # All colors explored - switch to normal targeting
            self.exploration_phase = False
            self.exploration_complete = True
            print(f"✅ EXPLORATION COMPLETE: All {len(self.explored_color_objects)} colors tried at least once")
            return None
        
        print(f"🔍 EXPLORATION STATUS: {len(self.explored_color_objects)} explored, {len(unexplored_colors)} remaining: {sorted(list(unexplored_colors))}")
        
        # 3. FIND TARGET FOR NEXT UNEXPLORED COLOR
        target_color = next(iter(unexplored_colors))
        
        # Find coordinates where this color appears
        color_locations = np.argwhere(frame_array == target_color)
        
        if len(color_locations) == 0:
            print(f"⚠️ No locations found for color {target_color}, marking as explored anyway")
            # Mark as explored to prevent infinite loops
            self.explored_color_objects.add(target_color)
            return None
        
        # Choose a representative location for this color (center of mass) 
        # Note: argwhere returns (row, col) which is (y, x)
        center_x, center_y = np.mean(color_locations, axis=0).astype(int)
        
        # Ensure coordinates are within bounds
        center_x = max(0, min(center_x, frame_array.shape[1] - 1))
        center_y = max(0, min(center_y, frame_array.shape[0] - 1))
        
        # Double-check we're targeting the right color
        actual_color_at_target = frame_array[center_y, center_x]
        if actual_color_at_target != target_color:
            # If center of mass doesn't hit the target color, pick first occurrence
            center_x, center_y = color_locations[0]
            print(f"🔧 Center of mass mismatch, using first occurrence: ({center_x},{center_y})")
        
        # PRODUCTIVITY OVERRIDE - Key user insight implementation!
        # If current exploration target coordinate has been tried many times without progress,
        # let productivity system override exploration phase
        coord_key = (int(center_x), int(center_y))
        if coord_key in self.coordinate_results:
            result = self.coordinate_results[coord_key]
            
            # Check if this coordinate is stuck and should be skipped in exploration
            if (result.get('zero_progress_streak', 0) >= 5 and 
                result.get('total_score_change', 0) <= 0):
                
                print(f"🎯 PRODUCTIVITY OVERRIDE: Color {target_color} at ({center_x},{center_y}) has {result['zero_progress_streak']} failed attempts")
                print(f"   Marking color {target_color} as explored and moving to next color")
                
                # Mark this color as explored to move on
                self.explored_color_objects.add(target_color)
                
                # Try to find next unexplored color immediately
                new_unexplored = unique_colors - self.explored_color_objects
                if new_unexplored:
                    print(f"   Moving to next unexplored color from {sorted(list(new_unexplored))}")
                    # Return None to let normal targeting take over this frame
                    return None
                else:
                    # No more colors to explore
                    self.exploration_phase = False
                    self.exploration_complete = True
                    print(f"✅ EXPLORATION COMPLETE (via productivity override)")
                    return None
        
        # Create exploration target
        exploration_target = {
            'x': int(center_x),
            'y': int(center_y),
            'color': int(target_color),
            'reason': f'exploration_color_{target_color}_of_{len(unexplored_colors)}_remaining',
            'confidence': 0.9,  # High confidence in exploration
            'type': 'color_exploration',
            'exploration_phase': True
        }
        
        print(f"🔍 EXPLORATION TARGET: Color {target_color} at ({center_x},{center_y}) - {len(unexplored_colors)} colors remaining")
        
        return exploration_target

    def mark_color_explored(self, color: int, coordinate: Tuple[int, int], 
                          success: bool, score_change: float, frame_changes: Dict):
        """
        Mark a color as explored and record what happened when we clicked it.
        This builds the knowledge base for future targeting.
        """
        print(f"🔍 MARKING COLOR {color} AS EXPLORED at {coordinate}")
        print(f"   Before: {len(self.explored_color_objects)} colors explored: {sorted(list(self.explored_color_objects))}")
        
        self.explored_color_objects.add(color)
        
        print(f"   After: {len(self.explored_color_objects)} colors explored: {sorted(list(self.explored_color_objects))}")
        
        # Record detailed exploration result
        exploration_record = {
            'color': color,
            'coordinate': coordinate,
            'success': success,
            'score_change': score_change,
            'frame_changes': frame_changes,
            'timestamp': len(self.frame_history),
            'clickable': success or score_change != 0 or frame_changes.get('changes_detected', False),
            'effects_observed': []
        }
        
        # Detect specific effects
        if score_change > 0:
            exploration_record['effects_observed'].append('score_increase')
        if score_change < 0:
            exploration_record['effects_observed'].append('score_decrease')
        if frame_changes.get('changes_detected', False):
            exploration_record['effects_observed'].append('visual_change')
            if frame_changes.get('num_pixels_changed', 0) > 50:
                exploration_record['effects_observed'].append('significant_visual_change')
        
        # Initialize color behavior pattern if not exists
        if color not in self.color_behavior_patterns:
            self.color_behavior_patterns[color] = {
                'interactions': [],
                'success_rate': 0.0,
                'avg_score_change': 0.0,
                'common_effects': [],
                'clickable': exploration_record['clickable'],
                'exploration_result': exploration_record
            }
        
        pattern = self.color_behavior_patterns[color]
        pattern['exploration_result'] = exploration_record
        pattern['clickable'] = exploration_record['clickable']
        
        print(f"📝 COLOR {color} EXPLORATION RESULT:")
        print(f"   Clickable: {exploration_record['clickable']}")
        print(f"   Score change: {score_change:+}")
        print(f"   Effects: {exploration_record['effects_observed']}")
        
        # Check if exploration phase is complete
        remaining_colors = self.current_game_colors - self.explored_color_objects - {0}  # Exclude background
        print(f"🔍 EXPLORATION CHECK: current_colors={len(self.current_game_colors)}, explored={len(self.explored_color_objects)}, remaining={len(remaining_colors)}")
        
        if not remaining_colors:
            self.exploration_phase = False
            self.exploration_complete = True
            
            # Summarize exploration results
            clickable_colors = [color for color, pattern in self.color_behavior_patterns.items() 
                              if pattern.get('clickable', False)]
            
            print(f"🎯 EXPLORATION PHASE COMPLETE!")
            print(f"   Total colors tested: {len(self.explored_color_objects)}")
            print(f"   Clickable colors found: {len(clickable_colors)} {clickable_colors}")
            print(f"   Now switching to intelligent targeting based on exploration results")
        else:
            print(f"🔍 EXPLORATION CONTINUES: {len(remaining_colors)} colors still need testing: {sorted(list(remaining_colors))}")

    def should_avoid_coordinate(self, x: int, y: int) -> bool:
        """Check if a coordinate should be avoided due to ineffectiveness."""
        avoidance_score = self.get_coordinate_avoidance_score(x, y)
        
        # Dynamic threshold based on exploration progress
        # Lower threshold when we have many avoided coordinates (encourage re-exploration)
        total_avoided = sum(1 for coord in self.coordinate_results.values() 
                           if self.get_coordinate_avoidance_score(coord.get('x', 0), coord.get('y', 0)) > 0.5)
        
        # Adaptive threshold: higher when few coordinates avoided, lower when many avoided
        base_threshold = 0.5
        if total_avoided > 20:  # Many coordinates avoided
            threshold = 0.3  # Lower threshold to encourage re-exploration
        elif total_avoided > 10:
            threshold = 0.4
        else:
            threshold = base_threshold
            
        return avoidance_score > threshold

    def get_emergency_diversification_target(self, current_frame: List[List[int]], 
                                           grid_bounds: Tuple[int, int]) -> Tuple[int, int]:
        """Get an emergency diversification target when stuck in loops."""
        width, height = grid_bounds
        
        # Find all coordinates NOT in stuck coordinates
        available_coords = []
        
        for y in range(0, height, max(1, height // 10)):  # Sample every 10% of height
            for x in range(0, width, max(1, width // 10)):  # Sample every 10% of width
                if not self.should_avoid_coordinate(x, y):
                    # Check if there's actually something interesting at this coordinate
                    # Use safe normalized indexing
                    cf = self._normalize_local_frame(current_frame)
                    if cf is not None and y < cf.shape[0] and x < cf.shape[1]:
                        pixel_value = current_frame[y][x] if isinstance(current_frame[y], list) else current_frame[y][x] if hasattr(current_frame[y], '__getitem__') else 0
                        if pixel_value != 0:  # Non-zero pixel = potentially interesting
                            available_coords.append((x, y))
        
        if available_coords:
            # Prefer coordinates far from previously stuck coordinates
            best_coord = None
            best_distance = 0
            
            for coord in available_coords:
                min_distance = float('inf')
                for stuck_coord in self.coordinate_results:
                    if self.coordinate_results[stuck_coord]['is_stuck_coordinate']:
                        distance = np.sqrt((coord[0] - stuck_coord[0])**2 + (coord[1] - stuck_coord[1])**2)
                        min_distance = min(min_distance, distance)
                
                if min_distance > best_distance:
                    best_distance = min_distance
                    best_coord = coord
            
            if best_coord:
                print(f"🚀 EMERGENCY DIVERSIFICATION: Selected {best_coord} (distance {best_distance:.1f} from stuck coordinates)")
                return best_coord
        
        # Fallback: Random coordinate avoiding known stuck areas
        import random
        attempts = 0
        while attempts < 20:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if not self.should_avoid_coordinate(x, y):
                print(f"🎲 RANDOM DIVERSIFICATION: Selected ({x},{y}) after {attempts + 1} attempts")
                return (x, y)
            attempts += 1
        
        # Last resort: Pick center if everything else fails
        center_x, center_y = width // 2, height // 2
        print(f"🎯 FALLBACK DIVERSIFICATION: Using center ({center_x},{center_y})")
        return (center_x, center_y)

    def _generate_smart_exploration_coordinate(self, frame_array: np.ndarray) -> Tuple[int, int]:
        """Generate exploration coordinate that avoids known stuck coordinates."""
        height, width = frame_array.shape
        
        # Try to find an interesting coordinate that's not stuck
        candidates = []
        
        # Look for non-zero pixels (potentially interactive) that aren't stuck
        for y in range(0, height, max(1, height // 16)):  # Sample grid
            for x in range(0, width, max(1, width // 16)):
                if frame_array[y][x] != 0 and not self.should_avoid_coordinate(x, y):
                    # Score by distance from stuck coordinates
                    min_distance_from_stuck = float('inf')
                    for coord_key, result in self.coordinate_results.items():
                        if result.get('is_stuck_coordinate', False):
                            distance = np.sqrt((x - coord_key[0])**2 + (y - coord_key[1])**2)
                            min_distance_from_stuck = min(min_distance_from_stuck, distance)
                    
                    candidates.append((x, y, min_distance_from_stuck, frame_array[y][x]))
        
        if candidates:
            # Sort by distance from stuck coordinates (prefer farther)
            candidates.sort(key=lambda c: c[2], reverse=True)
            chosen = candidates[0]
            print(f"🔍 SMART EXPLORATION: ({chosen[0]},{chosen[1]}) - distance {chosen[2]:.1f} from stuck coords")
            return (chosen[0], chosen[1])
        
        # Fallback: Emergency diversification
        return self.get_emergency_diversification_target(frame_array.tolist(), (width, height))

    def analyze_frame(self, frame: List[List[List[int]]], game_id: str = None) -> Dict[str, Any]:
        """
        Analyze a single frame to detect agent position and changes.
        
        Args:
            frame: 64x64 grid of color indices (0-15)
            game_id: Current game identifier
            
        Returns:
            Dictionary with position info, movement detected, confidence
        """
        # Convert to numpy array for easier processing
        frame_array = self._normalize_local_frame(frame)

        result = {
            'agent_position': None,
            'movement_detected': False,
            'position_confidence': 0.0,
            'frame_changes': [],
            'colors_detected': set(),
            'timestamp': datetime.now().isoformat()
        }

        # Track unique colors in frame
        try:
            unique_colors = set(frame_array.flatten())
        except Exception:
            unique_colors = set()
        result['colors_detected'] = unique_colors

        # Detect movement if we have previous frame
        if self.previous_frame is not None:
            movement_info = self._detect_movement(self.previous_frame, frame_array)
            result.update(movement_info)

        # Attempt to detect agent position
        position_info = self._detect_agent_position(frame_array, unique_colors)
        result.update(position_info)

        # Update history
        self._update_frame_history(frame_array)

        self.previous_frame = frame_array.copy()

        return result
    
    def _detect_movement(self, old_frame: np.ndarray, new_frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect movement between two frames using difference analysis.
        """
        # Calculate frame difference
        diff = np.abs(new_frame.astype(int) - old_frame.astype(int))
        
        # Find areas of significant change
        change_mask = diff > 0
        change_percentage = np.sum(change_mask) / (self.grid_size * self.grid_size)
        
        # Find coordinates of changes
        changed_coords = np.where(change_mask)
        change_locations = list(zip(changed_coords[1], changed_coords[0]))  # (x, y) format
        
        # Detect if changes suggest agent movement
        movement_detected = change_percentage > self.movement_threshold
        
        # Try to identify from->to movement pattern
        movement_vector = None
        if movement_detected and len(change_locations) >= 2:
            movement_vector = self._calculate_movement_vector(change_locations, old_frame, new_frame)
        
        return {
            'movement_detected': movement_detected,
            'change_percentage': change_percentage,
            'changed_coordinates': change_locations,
            'movement_vector': movement_vector,
            'frame_difference_sum': np.sum(diff)
        }
    
    def _calculate_movement_vector(self, change_locations: List[Tuple[int, int]], 
                                 old_frame: np.ndarray, new_frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Calculate probable movement vector from frame changes.
        """
        if len(change_locations) < 2:
            return None
            
        # Separate locations into "disappeared" and "appeared" based on color changes
        appeared = []
        disappeared = []
        
        for x, y in change_locations:
            old_color = old_frame[y, x]
            new_color = new_frame[y, x]
            
            if old_color == self.background_color and new_color != self.background_color:
                appeared.append((x, y))
            elif old_color != self.background_color and new_color == self.background_color:
                disappeared.append((x, y))
        
        # If we have both appeared and disappeared locations, calculate movement
        if appeared and disappeared:
            # Calculate centroids
            disappeared_center = np.mean(disappeared, axis=0)
            appeared_center = np.mean(appeared, axis=0)
            
            vector = appeared_center - disappeared_center
            distance = np.linalg.norm(vector)
            
            return {
                'from_position': tuple(disappeared_center.astype(int)),
                'to_position': tuple(appeared_center.astype(int)),
                'vector': tuple(vector),
                'distance': distance
            }
        
        return None
    
    def _detect_agent_position(self, frame: np.ndarray, unique_colors: set) -> Dict[str, Any]:
        """
        Attempt to detect current agent position using color analysis and patterns.
        """
        # Update background color detection
        self._update_background_color(frame, unique_colors)
        
        # Find non-background pixels
        non_bg_mask = frame != self.background_color
        non_bg_coords = np.where(non_bg_mask)
        
        if len(non_bg_coords[0]) == 0:
            return {
                'agent_position': None,
                'position_confidence': 0.0,
                'detection_method': 'no_non_background_found'
            }
        
        # Try different detection methods
        position_candidates = []
        
        # Method 1: Look for isolated pixels (potential agent)
        isolated_position = self._find_isolated_pixels(frame, non_bg_mask)
        if isolated_position:
            position_candidates.append({
                'position': isolated_position,
                'confidence': 0.7,
                'method': 'isolated_pixel'
            })
        
        # Method 2: Look for specific color patterns learned from previous frames
        pattern_position = self._find_learned_patterns(frame)
        if pattern_position:
            position_candidates.append({
                'position': pattern_position,
                'confidence': 0.8,
                'method': 'learned_pattern'
            })
        
        # Method 3: Use movement history if available
        if hasattr(self, 'recent_positions') and self.recent_positions:
            predicted_position = self._predict_position_from_history()
            if predicted_position:
                position_candidates.append({
                    'position': predicted_position,
                    'confidence': 0.6,
                    'method': 'movement_prediction'
                })
        
        # Select best candidate
        if position_candidates:
            best_candidate = max(position_candidates, key=lambda x: x['confidence'])
            self.agent_position = best_candidate['position']
            self.position_confidence = best_candidate['confidence']
            
            return {
                'agent_position': best_candidate['position'],
                'position_confidence': best_candidate['confidence'],
                'detection_method': best_candidate['method'],
                'all_candidates': position_candidates
            }
        
        # Fallback: Use centroid of all non-background pixels
        centroid_x = int(np.mean(non_bg_coords[1]))
        centroid_y = int(np.mean(non_bg_coords[0]))
        
        return {
            'agent_position': (centroid_x, centroid_y),
            'position_confidence': 0.3,
            'detection_method': 'centroid_fallback'
        }
    
    def _update_background_color(self, frame: np.ndarray, unique_colors: set):
        """
        Intelligently determine background color from frame analysis.
        """
        # Count frequency of each color
        color_counts = {}
        for color in unique_colors:
            color_counts[color] = np.sum(frame == color)
        
        # Background is likely the most frequent color
        most_frequent_color = max(color_counts, key=color_counts.get)
        
        # Update background color if this is significantly more frequent
        total_pixels = self.grid_size * self.grid_size
        if color_counts[most_frequent_color] > total_pixels * 0.5:
            self.background_color = most_frequent_color
    
    def _find_isolated_pixels(self, frame: np.ndarray, non_bg_mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find isolated non-background pixels that could represent the agent.
        """
        # Use morphological operations to find isolated regions
        kernel = np.ones((3, 3), np.uint8)
        
        # Find regions with few neighbors (potential agent)
        isolated_mask = cv2.morphologyEx(non_bg_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        isolated_coords = np.where(isolated_mask > 0)
        
        if len(isolated_coords[0]) > 0:
            # Return centroid of isolated region
            center_x = int(np.mean(isolated_coords[1]))
            center_y = int(np.mean(isolated_coords[0]))
            return (center_x, center_y)
        
        return None
    
    def _find_learned_patterns(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Look for previously learned agent color patterns.
        """
        if not hasattr(self, 'agent_colors') or not self.agent_colors:
            return None
        
        # Look for pixels matching learned agent colors
        agent_mask = np.zeros_like(frame, dtype=bool)
        for color in self.agent_colors:
            agent_mask |= (frame == color)
        
        agent_coords = np.where(agent_mask)
        if len(agent_coords[0]) > 0:
            center_x = int(np.mean(agent_coords[1]))
            center_y = int(np.mean(agent_coords[0]))
            return (center_x, center_y)
        
        return None
    
    def _predict_position_from_history(self) -> Optional[Tuple[int, int]]:
        """
        Predict likely position based on recent movement patterns.
        """
        # This would use movement history to predict next position
        # Implementation depends on building up movement patterns
        return None
    
    def _update_frame_history(self, frame: np.ndarray):
        """
        Update frame history for pattern learning.
        """
        self.frame_history.append(frame.copy())
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
    
    def learn_agent_colors(self, successful_actions: List[Dict[str, Any]]):
        """
        Learn which colors represent the agent based on successful actions.
        """
        # Extract colors from frames where agent actions were successful
        for action_info in successful_actions:
            if 'frame_before' in action_info and 'frame_after' in action_info:
                # Analyze color changes between successful actions
                self._extract_agent_colors_from_action(action_info)
    
    def _extract_agent_colors_from_action(self, action_info: Dict[str, Any]):
        """
        Extract likely agent colors from successful action sequences.
        """
        # This would analyze frame differences from successful actions
        # to learn which colors represent the agent
        pass
    
    def get_strategic_coordinates(self, available_actions: List[int], 
                                current_position: Optional[Tuple[int, int]] = None) -> Dict[int, Tuple[int, int]]:
        """
        Generate strategic coordinates for ACTION6 based on current analysis.
        
        Returns:
            Dictionary mapping actions to coordinate suggestions
        """
        coords = {}
        
        if 6 in available_actions:
            if current_position is None:
                # Center start strategy as requested
                coords[6] = (32, 32)
            else:
                # Generate strategic movement options
                x, y = current_position
                
                # Boundary-aware coordinate suggestions
                strategic_options = [
                    (min(63, x + 10), y),      # Move right
                    (max(0, x - 10), y),       # Move left  
                    (x, min(63, y + 10)),      # Move down
                    (x, max(0, y - 10)),       # Move up
                    (32, 32),                  # Center
                    (0, 0),                    # Corner exploration
                    (63, 63),                  # Opposite corner
                ]
                
                # Select based on frame analysis
                coords[6] = self._select_best_strategic_coordinate(strategic_options)
        
        return coords
    
    def _select_best_strategic_coordinate(self, options: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select the most strategic coordinate from available options.
        """
        # For now, return center strategy, but this could be enhanced
        # with analysis of frame patterns, unexplored areas, etc.
        return (32, 32)
    
    def clamp_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """
        Ensure coordinates stay within valid 0-63 range.
        """
        clamped_x = max(0, min(63, x))
        clamped_y = max(0, min(63, y))
        return (clamped_x, clamped_y)
    
    def _get_emergency_diversification_target(self, frame: Any, grid_dimensions: Tuple[int, int]) -> Tuple[int, int]:
        """
        Generate emergency diversification coordinates when stuck.
        Uses random selection avoiding known stuck coordinates.
        """
        import random
        
        grid_width, grid_height = grid_dimensions
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Generate random coordinates
            x = random.randint(1, grid_width - 2)
            y = random.randint(1, grid_height - 2)
            
            # Check if this coordinate should be avoided
            if not self.should_avoid_coordinate(x, y):
                print(f"🎲 RANDOM DIVERSIFICATION: Selected ({x},{y}) after {attempt + 1} attempts")
                return (x, y)
        
        # If all attempts failed, return a random coordinate anyway
        x = random.randint(1, grid_width - 2)
        y = random.randint(1, grid_height - 2)
        print(f"🎲 RANDOM DIVERSIFICATION: Selected ({x},{y}) after {max_attempts} attempts (forced)")
        return (x, y)

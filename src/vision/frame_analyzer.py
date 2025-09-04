"""
Frame Analysis System for ARC-AGI-3 Training
Implements computer vision for agent position tracking and movement detection.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime


class FrameAnalyzer:
    """
    Lightweight computer vision system for tracking agent position and movement
    in ARC-AGI-3 game frames.
    """
    
    def __init__(self):
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
        self.last_successful_coords = []  # Track successful interactions
        
        # ENHANCED INTERACTION LOGGING SYSTEM
        self.interaction_log = []  # Detailed log of all ACTION6 interactions
        self.object_interaction_history = {}  # object_id -> list of interaction results
        self.color_behavior_patterns = {}  # color -> behavior patterns observed
        self.shape_behavior_patterns = {}  # shape_type -> behavior patterns observed
        self.spatial_interaction_map = {}  # (x, y) -> interaction results and context
        self.hypothesis_database = []  # Generated hypotheses about object behaviors
        
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
        # Convert frame to numpy array
        frame_array = np.array(frame[0]) if isinstance(frame[0][0], list) else np.array(frame)
        
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
            
            # 4. FRAME DIFFERENCING - Objects that changed
            if self.previous_frame is not None:
                change_targets = self._find_frame_changes(frame_array, self.previous_frame)
                targets.extend(change_targets)
            
            # 5. RANK TARGETS by interaction potential
            ranked_targets = self._rank_interaction_targets(targets)
            
            # 6. SELECT BEST TARGET
            if ranked_targets:
                best_target = ranked_targets[0]
                analysis['interactive_targets'] = ranked_targets[:5]  # Top 5
                analysis['recommended_action6_coord'] = (best_target['x'], best_target['y'])
                analysis['targeting_reason'] = best_target['reason']
                analysis['confidence'] = best_target['confidence']
            else:
                # 7. FALLBACK: EXPLORATORY TAPPING in systematic pattern
                exploration_coord = self._generate_exploration_coordinate(frame_array)
                if exploration_coord:
                    analysis['recommended_action6_coord'] = exploration_coord
                    analysis['targeting_reason'] = 'exploratory_systematic_tapping'
                    analysis['confidence'] = 0.3  # Lower confidence for exploration
            
            # Store visual feature summary
            analysis['visual_features'] = {
                'unique_colors': len(np.unique(frame_array)),
                'max_brightness': int(np.max(frame_array)),
                'min_brightness': int(np.min(frame_array)),
                'color_variance': float(np.var(frame_array)),
                'frame_size': frame_array.shape
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
                        center_y, center_x = obj_data['centroid']
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
        """ENHANCED: Rank targets avoiding repetitive coordinates and prioritizing unexplored areas."""
        # Sort by confidence, then by type priority
        type_priority = {
            'dynamic_change': 5,      # Moving objects = highest priority
            'color_object': 4,        # Enhanced color objects = high priority  
            'geometric_shape': 3,     # Shapes = possible buttons
            'brightness_extreme': 2,  # Brightness = might be indicators
            'color_anomaly': 1        # Basic color anomaly = lowest priority
        }
        
        for target in targets:
            base_score = target['confidence'] * 100
            type_score = type_priority.get(target['type'], 0) * 10
            
            # COORDINATE AVOIDANCE PENALTY
            coord_key = (target['x'], target['y'])
            avoidance_penalty = 0
            
            if coord_key in self.tried_coordinates:
                try_count = self.coordinate_results.get(coord_key, {}).get('try_count', 0)
                success_count = self.coordinate_results.get(coord_key, {}).get('success_count', 0)
                
                if try_count > 0:
                    # Heavy penalty for repeatedly tried coordinates with no success
                    if success_count == 0 and try_count > 2:
                        avoidance_penalty = 80  # Heavy penalty
                    elif success_count == 0 and try_count > 0:
                        avoidance_penalty = 40  # Moderate penalty
                    elif success_count > 0:
                        avoidance_penalty = 10  # Light penalty for previously successful
            
            # MOVEMENT BONUS for objects that have moved
            movement_bonus = 0
            if 'object_id' in target and target['object_id'] in self.color_object_tracker:
                tracker = self.color_object_tracker[target['object_id']]
                if tracker.get('movement_vector') and tracker['movement_vector'] != (0, 0):
                    movement_bonus = 20  # Bonus for moving objects
            
            # EXPLORATION BONUS for unexplored regions
            exploration_bonus = 0
            if coord_key not in self.tried_coordinates:
                exploration_bonus = 15  # Bonus for completely new coordinates
            
            target['priority_score'] = (
                base_score + 
                type_score + 
                movement_bonus + 
                exploration_bonus - 
                avoidance_penalty
            )
            
            target['avoidance_penalty'] = avoidance_penalty
            target['movement_bonus'] = movement_bonus
            target['exploration_bonus'] = exploration_bonus
        
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
        self.color_behavior_patterns.clear() 
        self.shape_behavior_patterns.clear()
        self.spatial_interaction_map.clear()
        # Keep hypothesis database across games for pattern learning

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
        
        return interaction_id
    
    def _detect_shape_at_coordinate(self, x: int, y: int, frame: List[List[int]]) -> Dict[str, Any]:
        """Analyze the shape/pattern at the clicked coordinate."""
        if not frame:
            return {'shape_type': 'unknown', 'details': {}}
            
        try:
            frame_array = np.array(frame[0]) if isinstance(frame[0][0], list) else np.array(frame)
            
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
            frame_array = np.array(frame[0]) if isinstance(frame[0][0], list) else np.array(frame)
            
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
            before_array = np.array(before_frame[0]) if isinstance(before_frame[0][0], list) else np.array(before_frame)
            after_array = np.array(after_frame[0]) if isinstance(after_frame[0][0], list) else np.array(after_frame)
            
            # Calculate differences
            if before_array.shape != after_array.shape:
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
        frame_array = np.array(frame[0])  # Frame is wrapped in extra list
        
        result = {
            'agent_position': None,
            'movement_detected': False,
            'position_confidence': 0.0,
            'frame_changes': [],
            'colors_detected': set(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Track unique colors in frame
        unique_colors = set(frame_array.flatten())
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

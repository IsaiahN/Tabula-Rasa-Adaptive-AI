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
        self.max_history = 5
        self.grid_size = 64  # ARC-AGI-3 uses 64x64 grids
        
        # Color detection parameters
        self.agent_colors = []  # Learn colors that represent agent
        self.background_color = 0  # Assume 0 is background initially
        
        # Movement detection parameters
        self.movement_threshold = 0.1  # Minimum change to detect movement
        self.position_confidence = 0.0  # How confident we are about position
        
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

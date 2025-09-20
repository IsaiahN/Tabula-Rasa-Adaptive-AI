#!/usr/bin/env python3
"""
Movement Detector - Detects and analyzes movement patterns in frames.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import cv2

class MovementDetector:
    """Detects and analyzes movement patterns in ARC-AGI-3 frames."""
    
    def __init__(self):
        self.movement_history = deque(maxlen=20)
        self.movement_threshold = 0.1  # Minimum change to consider as movement
        self.stability_threshold = 0.05  # Threshold for considering position stable
        
    def detect_movement(self, current_frame: np.ndarray, previous_frame: np.ndarray) -> Dict[str, Any]:
        """Detect movement between two frames."""
        if previous_frame is None:
            return {
                'has_movement': False,
                'movement_magnitude': 0.0,
                'movement_centroid': None,
                'changed_regions': []
            }
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(current_frame, previous_frame)
        movement_magnitude = np.mean(frame_diff)
        
        has_movement = movement_magnitude > self.movement_threshold
        
        if not has_movement:
            return {
                'has_movement': False,
                'movement_magnitude': float(movement_magnitude),
                'movement_centroid': None,
                'changed_regions': []
            }
        
        # Find changed regions
        changed_regions = self._find_changed_regions(frame_diff)
        
        # Calculate movement centroid
        movement_centroid = self._calculate_movement_centroid(frame_diff)
        
        # Store movement data
        movement_data = {
            'timestamp': np.datetime64('now'),
            'magnitude': float(movement_magnitude),
            'centroid': movement_centroid,
            'regions': changed_regions
        }
        self.movement_history.append(movement_data)
        
        return {
            'has_movement': True,
            'movement_magnitude': float(movement_magnitude),
            'movement_centroid': movement_centroid,
            'changed_regions': changed_regions
        }
    
    def _find_changed_regions(self, frame_diff: np.ndarray) -> List[Dict[str, Any]]:
        """Find regions where significant changes occurred."""
        # Threshold the difference to find significant changes
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours of changed regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Filter out very small regions
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': float(area),
                    'center': (int(x + w/2), int(y + h/2))
                })
        
        return regions
    
    def _calculate_movement_centroid(self, frame_diff: np.ndarray) -> Optional[Tuple[int, int]]:
        """Calculate the centroid of movement in the frame."""
        # Find non-zero pixels (where movement occurred)
        y_coords, x_coords = np.where(frame_diff > 0)
        
        if len(x_coords) == 0:
            return None
        
        # Calculate centroid
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        
        return (centroid_x, centroid_y)
    
    def analyze_movement_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in recent movement history."""
        if len(self.movement_history) < 3:
            return {
                'pattern_type': 'insufficient_data',
                'movement_frequency': 0.0,
                'average_magnitude': 0.0,
                'movement_trend': 'stable'
            }
        
        # Extract movement data
        magnitudes = [m['magnitude'] for m in self.movement_history]
        centroids = [m['centroid'] for m in self.movement_history if m['centroid'] is not None]
        
        # Calculate statistics
        avg_magnitude = np.mean(magnitudes)
        movement_frequency = len(self.movement_history) / 20.0  # Assuming 20 frame history
        
        # Analyze movement trend
        if len(magnitudes) >= 5:
            recent_magnitudes = magnitudes[-5:]
            trend = np.polyfit(range(len(recent_magnitudes)), recent_magnitudes, 1)[0]
            if trend > 0.01:
                movement_trend = 'increasing'
            elif trend < -0.01:
                movement_trend = 'decreasing'
            else:
                movement_trend = 'stable'
        else:
            movement_trend = 'stable'
        
        # Determine pattern type
        if movement_frequency > 0.7:
            pattern_type = 'high_activity'
        elif movement_frequency > 0.3:
            pattern_type = 'moderate_activity'
        elif movement_frequency > 0.1:
            pattern_type = 'low_activity'
        else:
            pattern_type = 'minimal_activity'
        
        # Analyze centroid movement
        centroid_movement = self._analyze_centroid_movement(centroids)
        
        return {
            'pattern_type': pattern_type,
            'movement_frequency': movement_frequency,
            'average_magnitude': avg_magnitude,
            'movement_trend': movement_trend,
            'centroid_movement': centroid_movement,
            'total_movements': len(self.movement_history)
        }
    
    def _analyze_centroid_movement(self, centroids: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze how the movement centroid is moving."""
        if len(centroids) < 2:
            return {
                'direction': 'unknown',
                'distance_traveled': 0.0,
                'is_stable': True
            }
        
        # Calculate distances between consecutive centroids
        distances = []
        for i in range(1, len(centroids)):
            prev_centroid = centroids[i-1]
            curr_centroid = centroids[i]
            distance = np.sqrt((curr_centroid[0] - prev_centroid[0])**2 + 
                             (curr_centroid[1] - prev_centroid[1])**2)
            distances.append(distance)
        
        total_distance = sum(distances)
        avg_distance = np.mean(distances) if distances else 0.0
        
        # Determine if movement is stable (centroid not moving much)
        is_stable = avg_distance < 5.0  # Threshold for stability
        
        # Determine primary direction
        if len(centroids) >= 3:
            recent_centroids = centroids[-3:]
            x_coords = [c[0] for c in recent_centroids]
            y_coords = [c[1] for c in recent_centroids]
            
            x_trend = np.polyfit(range(len(x_coords)), x_coords, 1)[0]
            y_trend = np.polyfit(range(len(y_coords)), y_coords, 1)[0]
            
            if abs(x_trend) > abs(y_trend):
                direction = 'horizontal'
            elif abs(y_trend) > abs(x_trend):
                direction = 'vertical'
            else:
                direction = 'diagonal'
        else:
            direction = 'unknown'
        
        return {
            'direction': direction,
            'distance_traveled': total_distance,
            'average_distance': avg_distance,
            'is_stable': is_stable
        }
    
    def get_movement_summary(self) -> Dict[str, Any]:
        """Get a summary of recent movement activity."""
        if not self.movement_history:
            return {
                'total_movements': 0,
                'recent_activity': 'none',
                'movement_consistency': 0.0
            }
        
        # Calculate movement consistency
        magnitudes = [m['magnitude'] for m in self.movement_history]
        if len(magnitudes) > 1:
            consistency = 1.0 - (np.std(magnitudes) / (np.mean(magnitudes) + 1e-6))
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 1.0
        
        # Determine recent activity level
        recent_movements = list(self.movement_history)[-5:]
        recent_avg_magnitude = np.mean([m['magnitude'] for m in recent_movements])
        
        if recent_avg_magnitude > 0.5:
            recent_activity = 'high'
        elif recent_avg_magnitude > 0.2:
            recent_activity = 'moderate'
        elif recent_avg_magnitude > 0.05:
            recent_activity = 'low'
        else:
            recent_activity = 'minimal'
        
        return {
            'total_movements': len(self.movement_history),
            'recent_activity': recent_activity,
            'movement_consistency': consistency,
            'average_magnitude': np.mean(magnitudes),
            'last_movement': self.movement_history[-1] if self.movement_history else None
        }
    
    def reset_movement_tracking(self):
        """Reset movement tracking data."""
        self.movement_history.clear()
    
    def set_movement_threshold(self, threshold: float):
        """Set the movement detection threshold."""
        self.movement_threshold = max(0.0, min(1.0, threshold))
    
    def set_stability_threshold(self, threshold: float):
        """Set the stability detection threshold."""
        self.stability_threshold = max(0.0, min(1.0, threshold))
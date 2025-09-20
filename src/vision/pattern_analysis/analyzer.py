#!/usr/bin/env python3
"""
Pattern Analyzer - Analyzes patterns in ARC-AGI-3 frames.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

class PatternAnalyzer:
    """Analyzes patterns in ARC-AGI-3 frames."""
    
    def __init__(self):
        self.pattern_history = deque(maxlen=50)
        self.pattern_templates = {}
        self.symmetry_threshold = 0.8
        self.edge_threshold = 50
        
    def find_color_anomalies(self, frame_array: np.ndarray) -> List[Dict]:
        """Find color anomalies in the frame."""
        anomalies = []
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(frame_array, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(frame_array, cv2.COLOR_RGB2LAB)
        
        # Find unique colors
        unique_colors = np.unique(frame_array.reshape(-1, frame_array.shape[-1]), axis=0)
        
        for color in unique_colors:
            # Create mask for this color
            mask = np.all(frame_array == color, axis=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5:  # Filter small areas
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate color properties
                    color_props = self._analyze_color_properties(color, mask[y:y+h, x:x+w])
                    
                    anomalies.append({
                        'type': 'color_anomaly',
                        'color': color.tolist(),
                        'position': (int(x), int(y)),
                        'size': (int(w), int(h)),
                        'area': float(area),
                        'properties': color_props,
                        'center': (int(x + w/2), int(y + h/2))
                    })
        
        return anomalies
    
    def _analyze_color_properties(self, color: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Analyze properties of a color region."""
        # Calculate color statistics
        color_mean = np.mean(color)
        color_std = np.std(color)
        
        # Calculate shape properties
        area = np.sum(mask)
        perimeter = cv2.arcLength(mask.astype(np.uint8), True)
        
        # Calculate compactness
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        
        # Calculate aspect ratio
        y_coords, x_coords = np.where(mask)
        if len(x_coords) > 0 and len(y_coords) > 0:
            width = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)
            aspect_ratio = width / height if height > 0 else 1.0
        else:
            aspect_ratio = 1.0
        
        return {
            'mean_intensity': float(color_mean),
            'std_intensity': float(color_std),
            'area': float(area),
            'perimeter': float(perimeter),
            'compactness': float(compactness),
            'aspect_ratio': float(aspect_ratio)
        }
    
    def find_geometric_shapes(self, frame_array: np.ndarray) -> List[Dict]:
        """Find geometric shapes in the frame."""
        shapes = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Filter small areas
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Classify shape
                shape_type = self._classify_shape(len(approx), area)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate shape properties
                properties = self._calculate_shape_properties(contour, area)
                
                shapes.append({
                    'type': shape_type,
                    'position': (int(x), int(y)),
                    'size': (int(w), int(h)),
                    'area': float(area),
                    'vertices': len(approx),
                    'properties': properties,
                    'center': (int(x + w/2), int(y + h/2))
                })
        
        return shapes
    
    def _classify_shape(self, vertices: int, area: float) -> str:
        """Classify a shape based on its vertices and area."""
        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            return 'rectangle'
        elif vertices > 8:
            return 'circle'
        elif vertices == 5:
            return 'pentagon'
        elif vertices == 6:
            return 'hexagon'
        else:
            return 'polygon'
    
    def _calculate_shape_properties(self, contour: np.ndarray, area: float) -> Dict[str, Any]:
        """Calculate properties of a shape."""
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate compactness
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        
        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Calculate extent (area / bounding rectangle area)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        return {
            'perimeter': float(perimeter),
            'compactness': float(compactness),
            'solidity': float(solidity),
            'extent': float(extent)
        }
    
    def analyze_pattern_consistency(self) -> Dict[str, Any]:
        """Analyze consistency of patterns over time."""
        if len(self.pattern_history) < 3:
            return {
                'consistency_score': 0.0,
                'pattern_stability': 'unknown',
                'dominant_patterns': []
            }
        
        # Analyze recent patterns
        recent_patterns = list(self.pattern_history)[-10:]
        
        # Calculate consistency metrics
        pattern_types = [p.get('type', 'unknown') for p in recent_patterns]
        type_counts = {}
        for pattern_type in pattern_types:
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
        
        # Calculate consistency score
        total_patterns = len(recent_patterns)
        max_count = max(type_counts.values()) if type_counts else 0
        consistency_score = max_count / total_patterns if total_patterns > 0 else 0.0
        
        # Determine pattern stability
        if consistency_score > 0.8:
            stability = 'high'
        elif consistency_score > 0.5:
            stability = 'medium'
        else:
            stability = 'low'
        
        # Get dominant patterns
        dominant_patterns = [pt for pt, count in type_counts.items() if count > total_patterns * 0.3]
        
        return {
            'consistency_score': consistency_score,
            'pattern_stability': stability,
            'dominant_patterns': dominant_patterns,
            'total_patterns': total_patterns,
            'pattern_distribution': type_counts
        }
    
    def add_pattern_to_history(self, pattern: Dict[str, Any]):
        """Add a pattern to the history for analysis."""
        self.pattern_history.append(pattern)
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get a summary of all detected patterns."""
        if not self.pattern_history:
            return {
                'total_patterns': 0,
                'pattern_types': [],
                'recent_activity': 'none'
            }
        
        # Count pattern types
        pattern_types = [p.get('type', 'unknown') for p in self.pattern_history]
        type_counts = {}
        for pattern_type in pattern_types:
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
        
        # Determine recent activity
        recent_patterns = list(self.pattern_history)[-5:]
        if len(recent_patterns) >= 4:
            recent_activity = 'high'
        elif len(recent_patterns) >= 2:
            recent_activity = 'medium'
        else:
            recent_activity = 'low'
        
        return {
            'total_patterns': len(self.pattern_history),
            'pattern_types': list(type_counts.keys()),
            'pattern_counts': type_counts,
            'recent_activity': recent_activity,
            'consistency_analysis': self.analyze_pattern_consistency()
        }
    
    def reset_pattern_tracking(self):
        """Reset pattern tracking data."""
        self.pattern_history.clear()
        self.pattern_templates.clear()

"""
Frame Analysis System for ARC-AGI-3 Training
Implements computer vision for agent position tracking and movement detection.

This file has been modularized. The main functionality is now in src/vision/ sub-packages.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Import from the new modular structure
from .position_tracking import PositionTracker
from .movement_detection import MovementDetector
from .pattern_analysis import PatternAnalyzer
from .frame_processing import FrameProcessor

class FrameAnalyzer:
    """
    Lightweight computer vision system for tracking agent position and movement
    in ARC-AGI-3 game frames.
    
    This class now orchestrates the modular components instead of containing
    all functionality in one massive file.
    """
    
    def __init__(self, base_path: str = "."):
        self.base_path = base_path
        
        # Initialize modular components
        self.position_tracker = PositionTracker(base_path)
        self.movement_detector = MovementDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.frame_processor = FrameProcessor()
        
        # Legacy attributes for backward compatibility
        self.previous_frame = None
        self.agent_position = None
        self.frame_history = []
        self.max_history = 10
        self.grid_size = 64
        
        # Coordinate tracking for learning
        self.coordinate_attempts = {}
        self.avoidance_scores = {}
        self.success_patterns = {}
    
    def _normalize_local_frame(self, frame: Any) -> np.ndarray:
        """Normalize a frame to a standard format."""
        return self.frame_processor.normalize_frame(frame)
    
    def reset_for_new_game(self, game_id: str = None):
        """Reset tracking data for a new game."""
        self.previous_frame = None
        self.agent_position = None
        self.frame_history.clear()
        
        # Reset modular components
        self.position_tracker.reset_for_new_game(game_id)
        self.movement_detector.reset_movement_tracking()
        self.pattern_analyzer.reset_pattern_tracking()
        self.frame_processor.clear_history()
        
        # Reset coordinate tracking
        self.coordinate_attempts.clear()
        self.avoidance_scores.clear()
        self.success_patterns.clear()
        
        if game_id:
            print(f"Frame analyzer reset for game {game_id}")
    
    def analyze_frame_for_action6_targets(self, frame: List[List[List[int]]], game_id: str = None) -> Dict[str, Any]:
        """Analyze frame for action 6 targets using modular components."""
        try:
            # Process the frame
            processed = self.frame_processor.preprocess_frame(frame)
            frame_array = processed['frame']
            
            # Update frame history
            self.frame_history.append(frame_array)
            if len(self.frame_history) > self.max_history:
                self.frame_history.pop(0)
            
            # Detect movement
            movement_data = self.movement_detector.detect_movement(frame_array, self.previous_frame)
            
            # Find color anomalies
            color_anomalies = self.pattern_analyzer.find_color_anomalies(frame_array)
            
            # Find geometric shapes
            geometric_shapes = self.pattern_analyzer.find_geometric_shapes(frame_array)
            
            # Update position tracking if we have movement
            if movement_data['has_movement'] and movement_data['movement_centroid']:
                centroid = movement_data['movement_centroid']
                if isinstance(centroid, (tuple, list)) and len(centroid) >= 2:
                    self.position_tracker.update_position(centroid[0], centroid[1])
                    self.agent_position = (centroid[0], centroid[1])
            
            # Update previous frame
            self.previous_frame = frame_array
            
            # Generate targets based on analysis
            targets = self._generate_targets_from_analysis(
                color_anomalies, geometric_shapes, movement_data, processed
            )
            
            # Add patterns to history
            for anomaly in color_anomalies:
                self.pattern_analyzer.add_pattern_to_history(anomaly)
            for shape in geometric_shapes:
                self.pattern_analyzer.add_pattern_to_history(shape)
            
            return {
                'targets': targets,
                'movement_data': movement_data,
                'color_anomalies': color_anomalies,
                'geometric_shapes': geometric_shapes,
                'frame_properties': processed['properties'],
                'agent_position': self.agent_position,
                'analysis_metadata': {
                    'game_id': game_id,
                    'frame_processed': True,
                    'modular_analysis': True
                }
            }
            
        except Exception as e:
            print(f"Error in frame analysis: {e}")
            return {
                'targets': [],
                'error': str(e),
                'analysis_metadata': {
                    'game_id': game_id,
                    'frame_processed': False,
                    'error_occurred': True
                }
            }
    
    def _generate_targets_from_analysis(self, color_anomalies: List[Dict], 
                                      geometric_shapes: List[Dict], 
                                      movement_data: Dict, 
                                      processed: Dict) -> List[Dict]:
        """Generate interaction targets from analysis results."""
        targets = []
        
        # Add color anomaly targets
        for anomaly in color_anomalies:
            if anomaly['area'] > 10:  # Filter small anomalies
                targets.append({
                    'type': 'color_anomaly',
                    'x': anomaly['center'][0],
                    'y': anomaly['center'][1],
                    'confidence': min(1.0, anomaly['area'] / 100.0),
                    'properties': anomaly['properties'],
                    'source': 'color_analysis'
                })
        
        # Add geometric shape targets
        for shape in geometric_shapes:
            if shape['area'] > 20:  # Filter small shapes
                targets.append({
                    'type': 'geometric_shape',
                    'x': shape['center'][0],
                    'y': shape['center'][1],
                    'confidence': min(1.0, shape['area'] / 200.0),
                    'shape_type': shape['type'],
                    'properties': shape['properties'],
                    'source': 'shape_analysis'
                })
        
        # Add movement-based targets
        if movement_data['has_movement'] and movement_data['changed_regions']:
            for region in movement_data['changed_regions']:
                targets.append({
                    'type': 'movement_region',
                    'x': region['center'][0],
                    'y': region['center'][1],
                    'confidence': min(1.0, region['area'] / 150.0),
                    'area': region['area'],
                    'source': 'movement_analysis'
                })
        
        # Rank targets by confidence
        targets.sort(key=lambda t: t['confidence'], reverse=True)
        
        return targets[:10]  # Return top 10 targets
    
    def record_coordinate_attempt(self, x: int, y: int, was_successful: bool, score_change: float = 0):
        """Record a coordinate attempt for learning."""
        # Use the position tracker's method
        self.position_tracker.record_coordinate_attempt(x, y, was_successful, score_change)
        
        # Also update local tracking for backward compatibility
        coord_key = f"{x},{y}"
        if coord_key not in self.coordinate_attempts:
            self.coordinate_attempts[coord_key] = {
                'attempts': 0,
                'successes': 0,
                'total_score_change': 0.0
            }
        
        record = self.coordinate_attempts[coord_key]
        record['attempts'] += 1
        record['total_score_change'] += score_change
        
        if was_successful:
            record['successes'] += 1
    
    def get_movement_analysis(self) -> Dict[str, Any]:
        """Get analysis of recent movement patterns."""
        return self.position_tracker.get_movement_analysis()
    
    def get_coordinate_effectiveness(self, x: int, y: int) -> Dict[str, Any]:
        """Get effectiveness data for a specific coordinate."""
        return self.position_tracker.get_coordinate_effectiveness(x, y)
    
    def reset_coordinate_tracking(self):
        """Reset coordinate tracking data."""
        self.position_tracker.reset_coordinate_tracking()
        self.coordinate_attempts.clear()
        self.avoidance_scores.clear()
        self.success_patterns.clear()
    
    def get_meta_learning_summary(self) -> Dict[str, Any]:
        """Get summary of meta-learning data."""
        return self.position_tracker.get_meta_learning_summary()
    
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Get pattern analysis summary."""
        return self.pattern_analyzer.get_pattern_summary()
    
    def get_movement_summary(self) -> Dict[str, Any]:
        """Get movement detection summary."""
        return self.movement_detector.get_movement_summary()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get frame processing statistics."""
        return self.frame_processor.get_processing_stats()

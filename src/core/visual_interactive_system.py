#!/usr/bin/env python3
"""
Visual-Interactive Action6 Targeting System

This module implements the touchscreen paradigm for Action6, treating it as a universal
targeting system for touching visual elements rather than movement.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.database.system_integration import get_system_integration
from src.learning.game_type_classifier import get_game_type_classifier

logger = logging.getLogger(__name__)

@dataclass
class VisualTarget:
    """Represents a visual target for Action6 interaction."""
    x: int
    y: int
    target_type: str  # 'button', 'object', 'anomaly', 'interactive_element'
    confidence: float
    detection_method: str  # 'opencv', 'frame_analysis', 'pattern_matching'
    frame_changes_detected: bool = False
    score_impact: float = 0.0
    interaction_successful: bool = False

class VisualInteractiveSystem:
    """
    Visual-Interactive Action6 Targeting System
    
    Treats Action6 as a touchscreen interface for interacting with visual elements.
    Analyzes frames to detect buttons, objects, anomalies, and interactive elements.
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        self.game_type_classifier = get_game_type_classifier()
        self.detected_targets: Dict[str, List[VisualTarget]] = {}
        self.target_interaction_history: Dict[str, Dict[Tuple[int, int], Dict]] = {}
        
    async def analyze_frame_for_action6_targets(self, 
                                              frame: Any, 
                                              game_id: str,
                                              available_actions: List[int]) -> Dict[str, Any]:
        """
        Analyze frame for Action6 visual targets using touchscreen paradigm.
        
        Args:
            frame: Game frame data
            game_id: Current game identifier
            available_actions: List of available actions
            
        Returns:
            Dictionary with targeting analysis and recommendations
        """
        try:
            # Only analyze if Action6 is available
            if 6 not in available_actions:
                return {
                    'recommended_action6_coord': None,
                    'targeting_reason': 'Action6 not available',
                    'confidence': 0.0,
                    'interactive_targets': []
                }
            
            # Get game type for context
            game_type = self.game_type_classifier.extract_game_type(game_id)
            
            # Load historical targets for this game type
            historical_targets = await self._load_historical_targets(game_type)
            
            # Detect visual targets using multiple methods
            targets = await self._detect_visual_targets(frame, game_id, historical_targets)
            
            # Filter and prioritize targets
            prioritized_targets = await self._prioritize_targets(targets, game_id, game_type)
            
            # Select best target
            best_target = await self._select_best_target(prioritized_targets, game_id)
            
            # Store targets for this game
            self.detected_targets[game_id] = targets
            
            if best_target:
                return {
                    'recommended_action6_coord': (best_target.x, best_target.y),
                    'targeting_reason': f"Visual target detected: {best_target.target_type}",
                    'confidence': best_target.confidence,
                    'interactive_targets': [
                        {
                            'x': t.x, 'y': t.y, 'type': t.target_type,
                            'confidence': t.confidence, 'method': t.detection_method
                        } for t in prioritized_targets[:5]  # Top 5 targets
                    ]
                }
            else:
                return {
                    'recommended_action6_coord': None,
                    'targeting_reason': 'No clear visual targets detected',
                    'confidence': 0.0,
                    'interactive_targets': []
                }
                
        except Exception as e:
            logger.error(f"Error analyzing frame for Action6 targets: {e}")
            return {
                'recommended_action6_coord': None,
                'targeting_reason': f'Analysis failed: {str(e)}',
                'confidence': 0.0,
                'interactive_targets': []
            }
    
    async def _detect_visual_targets(self, 
                                   frame: Any, 
                                   game_id: str,
                                   historical_targets: List[Dict]) -> List[VisualTarget]:
        """Detect visual targets using multiple detection methods."""
        targets = []
        
        try:
            # Method 1: OpenCV-based detection
            opencv_targets = await self._detect_opencv_targets(frame, game_id)
            targets.extend(opencv_targets)
            
            # Method 2: Frame analysis-based detection
            frame_analysis_targets = await self._detect_frame_analysis_targets(frame, game_id)
            targets.extend(frame_analysis_targets)
            
            # Method 3: Pattern matching with historical data
            pattern_targets = await self._detect_pattern_targets(frame, game_id, historical_targets)
            targets.extend(pattern_targets)
            
            # Method 4: Anomaly detection
            anomaly_targets = await self._detect_anomaly_targets(frame, game_id)
            targets.extend(anomaly_targets)
            
            # Remove duplicates and merge similar targets
            targets = await self._merge_similar_targets(targets)
            
            return targets
            
        except Exception as e:
            logger.error(f"Error detecting visual targets: {e}")
            return []
    
    async def _detect_opencv_targets(self, frame: Any, game_id: str) -> List[VisualTarget]:
        """Detect targets using OpenCV-based analysis."""
        targets = []
        
        try:
            # Convert frame to numpy array if needed
            if isinstance(frame, list):
                frame_array = np.array(frame)
            else:
                frame_array = frame
            
            if frame_array is None or frame_array.size == 0:
                return targets
            
            # Ensure 2D array
            if frame_array.ndim > 2:
                frame_array = frame_array.reshape(-1, frame_array.shape[-1])
            
            # Simple edge detection for button-like objects
            if len(frame_array.shape) == 2:
                # Grayscale image
                edges = self._simple_edge_detection(frame_array)
                button_candidates = self._find_button_candidates(edges)
                
                for x, y, confidence in button_candidates:
                    targets.append(VisualTarget(
                        x=x, y=y,
                        target_type='button',
                        confidence=confidence,
                        detection_method='opencv'
                    ))
            
            # Color-based detection for interactive elements
            color_targets = self._detect_color_based_targets(frame_array)
            for x, y, confidence, target_type in color_targets:
                targets.append(VisualTarget(
                    x=x, y=y,
                    target_type=target_type,
                    confidence=confidence,
                    detection_method='opencv'
                ))
            
            return targets
            
        except Exception as e:
            logger.error(f"Error in OpenCV target detection: {e}")
            return []
    
    async def _detect_frame_analysis_targets(self, frame: Any, game_id: str) -> List[VisualTarget]:
        """Detect targets using frame analysis methods."""
        targets = []
        
        try:
            if isinstance(frame, list):
                frame_array = np.array(frame)
            else:
                frame_array = frame
            
            if frame_array is None or frame_array.size == 0:
                return targets
            
            # Detect high-contrast areas (potential buttons)
            contrast_targets = self._detect_high_contrast_areas(frame_array)
            for x, y, confidence in contrast_targets:
                targets.append(VisualTarget(
                    x=x, y=y,
                    target_type='interactive_element',
                    confidence=confidence,
                    detection_method='frame_analysis'
                ))
            
            # Detect symmetric patterns (potential UI elements)
            symmetry_targets = self._detect_symmetric_patterns(frame_array)
            for x, y, confidence in symmetry_targets:
                targets.append(VisualTarget(
                    x=x, y=y,
                    target_type='object',
                    confidence=confidence,
                    detection_method='frame_analysis'
                ))
            
            return targets
            
        except Exception as e:
            logger.error(f"Error in frame analysis target detection: {e}")
            return []
    
    async def _detect_pattern_targets(self, 
                                   frame: Any, 
                                   game_id: str,
                                   historical_targets: List[Dict]) -> List[VisualTarget]:
        """Detect targets using pattern matching with historical data."""
        targets = []
        
        try:
            if not historical_targets:
                return targets
            
            if isinstance(frame, list):
                frame_array = np.array(frame)
            else:
                frame_array = frame
            
            if frame_array is None or frame_array.size == 0:
                return targets
            
            # Match patterns from historical successful targets
            for hist_target in historical_targets:
                if hist_target.get('interaction_successful', False):
                    # Look for similar patterns in current frame
                    pattern_match = await self._match_historical_pattern(
                        frame_array, hist_target, game_id
                    )
                    if pattern_match:
                        targets.append(VisualTarget(
                            x=pattern_match['x'],
                            y=pattern_match['y'],
                            target_type=hist_target.get('target_type', 'object'),
                            confidence=pattern_match['confidence'],
                            detection_method='pattern_matching'
                        ))
            
            return targets
            
        except Exception as e:
            logger.error(f"Error in pattern target detection: {e}")
            return []
    
    async def _detect_anomaly_targets(self, frame: Any, game_id: str) -> List[VisualTarget]:
        """Detect targets using anomaly detection methods."""
        targets = []
        
        try:
            if isinstance(frame, list):
                frame_array = np.array(frame)
            else:
                frame_array = frame
            
            if frame_array is None or frame_array.size == 0:
                return targets
            
            # Detect unusual patterns or outliers
            anomalies = self._detect_frame_anomalies(frame_array)
            for x, y, confidence in anomalies:
                targets.append(VisualTarget(
                    x=x, y=y,
                    target_type='anomaly',
                    confidence=confidence,
                    detection_method='anomaly_detection'
                ))
            
            return targets
            
        except Exception as e:
            logger.error(f"Error in anomaly target detection: {e}")
            return []
    
    def _simple_edge_detection(self, frame: np.ndarray) -> np.ndarray:
        """Simple edge detection for button-like objects."""
        try:
            # Convert to grayscale if needed
            if len(frame.shape) > 2:
                frame = np.mean(frame, axis=2)
            
            # Simple gradient-based edge detection
            grad_x = np.abs(np.diff(frame, axis=1))
            grad_y = np.abs(np.diff(frame, axis=0))
            
            # Pad to match original size
            grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
            grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
            
            edges = np.sqrt(grad_x**2 + grad_y**2)
            return edges
            
        except Exception as e:
            logger.error(f"Error in edge detection: {e}")
            return np.zeros_like(frame)
    
    def _find_button_candidates(self, edges: np.ndarray) -> List[Tuple[int, int, float]]:
        """Find button-like candidates from edge detection."""
        candidates = []
        
        try:
            # Find local maxima in edge strength
            threshold = np.percentile(edges, 80)  # Top 20% of edge strengths
            high_edges = edges > threshold
            
            # Find connected components
            from scipy import ndimage
            labeled, num_features = ndimage.label(high_edges)
            
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) > 4:  # Minimum size for button
                    # Find center of component
                    y_coords, x_coords = np.where(component)
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    
                    # Calculate confidence based on edge strength and size
                    edge_strength = np.mean(edges[component])
                    size_score = min(1.0, np.sum(component) / 50.0)  # Normalize size
                    confidence = min(1.0, (edge_strength / threshold) * size_score)
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        candidates.append((center_x, center_y, confidence))
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error finding button candidates: {e}")
            return []
    
    def _detect_color_based_targets(self, frame: np.ndarray) -> List[Tuple[int, int, float, str]]:
        """Detect targets based on color patterns."""
        targets = []
        
        try:
            if len(frame.shape) < 3:
                return targets  # Need color information
            
            # Detect bright/colored areas that might be buttons
            brightness = np.mean(frame, axis=2)
            bright_threshold = np.percentile(brightness, 85)
            bright_areas = brightness > bright_threshold
            
            # Find connected bright areas
            from scipy import ndimage
            labeled, num_features = ndimage.label(bright_areas)
            
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) > 2:  # Minimum size
                    y_coords, x_coords = np.where(component)
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    
                    # Calculate confidence based on brightness and size
                    brightness_score = np.mean(brightness[component]) / 255.0
                    size_score = min(1.0, np.sum(component) / 30.0)
                    confidence = min(1.0, brightness_score * size_score)
                    
                    if confidence > 0.4:
                        targets.append((center_x, center_y, confidence, 'interactive_element'))
            
            return targets
            
        except Exception as e:
            logger.error(f"Error in color-based target detection: {e}")
            return []
    
    def _detect_high_contrast_areas(self, frame: np.ndarray) -> List[Tuple[int, int, float]]:
        """Detect high-contrast areas that might be interactive elements."""
        targets = []
        
        try:
            if len(frame.shape) > 2:
                frame = np.mean(frame, axis=2)  # Convert to grayscale
            
            # Calculate local contrast
            from scipy import ndimage
            kernel = np.ones((3, 3)) / 9
            local_mean = ndimage.convolve(frame, kernel, mode='constant')
            local_contrast = np.abs(frame - local_mean)
            
            # Find high contrast areas
            threshold = np.percentile(local_contrast, 80)
            high_contrast = local_contrast > threshold
            
            # Find connected components
            labeled, num_features = ndimage.label(high_contrast)
            
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) > 3:
                    y_coords, x_coords = np.where(component)
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    
                    contrast_score = np.mean(local_contrast[component])
                    confidence = min(1.0, contrast_score / threshold)
                    
                    if confidence > 0.3:
                        targets.append((center_x, center_y, confidence))
            
            return targets
            
        except Exception as e:
            logger.error(f"Error detecting high contrast areas: {e}")
            return []
    
    def _detect_symmetric_patterns(self, frame: np.ndarray) -> List[Tuple[int, int, float]]:
        """Detect symmetric patterns that might be UI elements."""
        targets = []
        
        try:
            if len(frame.shape) > 2:
                frame = np.mean(frame, axis=2)
            
            # Look for horizontal and vertical symmetry
            h, w = frame.shape
            
            # Check horizontal symmetry
            for y in range(2, h - 2):
                for x in range(2, w - 2):
                    # Check if area around (x,y) is horizontally symmetric
                    window_size = 3
                    half_window = window_size // 2
                    
                    if (x - half_window >= 0 and x + half_window < w and
                        y - half_window >= 0 and y + half_window < h):
                        
                        window = frame[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
                        horizontal_symmetry = np.mean(np.abs(window - np.fliplr(window)))
                        
                        if horizontal_symmetry < 10:  # Low difference = high symmetry
                            confidence = min(1.0, (20 - horizontal_symmetry) / 20)
                            if confidence > 0.5:
                                targets.append((x, y, confidence))
            
            return targets
            
        except Exception as e:
            logger.error(f"Error detecting symmetric patterns: {e}")
            return []
    
    async def _match_historical_pattern(self, 
                                      frame: np.ndarray, 
                                      hist_target: Dict, 
                                      game_id: str) -> Optional[Dict]:
        """Match patterns from historical successful targets."""
        try:
            # Simple pattern matching - look for similar pixel patterns
            hist_x = hist_target.get('target_x', 0)
            hist_y = hist_target.get('target_y', 0)
            
            # Get a small region around the historical target location
            if (hist_x < frame.shape[1] and hist_y < frame.shape[0] and
                hist_x >= 0 and hist_y >= 0):
                
                # Simple confidence based on proximity to historical successful location
                confidence = 0.6  # Base confidence for historical patterns
                
                return {
                    'x': hist_x,
                    'y': hist_y,
                    'confidence': confidence
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error matching historical pattern: {e}")
            return None
    
    def _detect_frame_anomalies(self, frame: np.ndarray) -> List[Tuple[int, int, float]]:
        """Detect anomalies in the frame that might be interactive elements."""
        targets = []
        
        try:
            if len(frame.shape) > 2:
                frame = np.mean(frame, axis=2)
            
            # Detect statistical anomalies
            mean_val = np.mean(frame)
            std_val = np.std(frame)
            threshold = mean_val + 2 * std_val  # 2 standard deviations above mean
            
            anomalies = frame > threshold
            
            # Find connected anomaly regions
            from scipy import ndimage
            labeled, num_features = ndimage.label(anomalies)
            
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) > 1:
                    y_coords, x_coords = np.where(component)
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    
                    anomaly_strength = np.mean(frame[component])
                    confidence = min(1.0, (anomaly_strength - mean_val) / (2 * std_val))
                    
                    if confidence > 0.3:
                        targets.append((center_x, center_y, confidence))
            
            return targets
            
        except Exception as e:
            logger.error(f"Error detecting frame anomalies: {e}")
            return []
    
    async def _merge_similar_targets(self, targets: List[VisualTarget]) -> List[VisualTarget]:
        """Merge similar targets that are close to each other."""
        if not targets:
            return []
        
        merged = []
        used = set()
        
        for i, target in enumerate(targets):
            if i in used:
                continue
            
            # Find similar targets
            similar_targets = [target]
            for j, other_target in enumerate(targets[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if targets are close (within 5 pixels)
                distance = np.sqrt((target.x - other_target.x)**2 + (target.y - other_target.y)**2)
                if distance < 5:
                    similar_targets.append(other_target)
                    used.add(j)
            
            # Merge similar targets
            if len(similar_targets) > 1:
                # Use the target with highest confidence
                best_target = max(similar_targets, key=lambda t: t.confidence)
                # Boost confidence for multiple detections
                best_target.confidence = min(1.0, best_target.confidence * 1.2)
                merged.append(best_target)
            else:
                merged.append(target)
            
            used.add(i)
        
        return merged
    
    async def _prioritize_targets(self, 
                                targets: List[VisualTarget], 
                                game_id: str,
                                game_type: str) -> List[VisualTarget]:
        """Prioritize targets based on type, confidence, and historical success."""
        if not targets:
            return []
        
        # Load historical success data
        historical_success = await self._load_target_success_data(game_type)
        
        # Calculate priority scores
        for target in targets:
            priority_score = target.confidence
            
            # Boost priority for historically successful target types
            if target.target_type in historical_success:
                success_rate = historical_success[target.target_type]
                priority_score *= (1.0 + success_rate)
            
            # Boost priority for button types
            if target.target_type == 'button':
                priority_score *= 1.3
            
            # Store priority score
            target.priority_score = priority_score
        
        # Sort by priority score
        return sorted(targets, key=lambda t: t.priority_score, reverse=True)
    
    async def _select_best_target(self, 
                                prioritized_targets: List[VisualTarget], 
                                game_id: str) -> Optional[VisualTarget]:
        """Select the best target from prioritized list."""
        if not prioritized_targets:
            return None
        
        # Return the highest priority target
        return prioritized_targets[0]
    
    async def _load_historical_targets(self, game_type: str) -> List[Dict]:
        """Load historical targets for the game type."""
        try:
            query = """
                SELECT target_x, target_y, target_type, confidence, interaction_successful
                FROM visual_targets vt
                JOIN game_results gr ON vt.game_id = gr.game_id
                WHERE gr.game_id LIKE ? AND vt.interaction_successful = 1
                ORDER BY vt.detection_timestamp DESC
                LIMIT 50
            """
            
            results = await self.integration.db.fetch_all(query, (f"%{game_type}%",))
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error loading historical targets: {e}")
            return []
    
    async def _load_target_success_data(self, game_type: str) -> Dict[str, float]:
        """Load target success rates by type for the game type."""
        try:
            query = """
                SELECT target_type, 
                       AVG(CASE WHEN interaction_successful = 1 THEN 1.0 ELSE 0.0 END) as success_rate
                FROM visual_targets vt
                JOIN game_results gr ON vt.game_id = gr.game_id
                WHERE gr.game_id LIKE ?
                GROUP BY target_type
            """
            
            results = await self.integration.db.fetch_all(query, (f"%{game_type}%",))
            return {row['target_type']: row['success_rate'] for row in results}
            
        except Exception as e:
            logger.error(f"Error loading target success data: {e}")
            return {}
    
    async def record_target_interaction(self, 
                                      game_id: str,
                                      target: VisualTarget,
                                      interaction_successful: bool,
                                      frame_changes_detected: bool,
                                      score_impact: float = 0.0):
        """Record the result of a target interaction."""
        try:
            # Store in database
            await self.integration.db.execute("""
                INSERT INTO visual_targets 
                (game_id, target_x, target_y, target_type, confidence, detection_method,
                 interaction_successful, frame_changes_detected, score_impact, detection_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, target.x, target.y, target.target_type, target.confidence,
                target.detection_method, interaction_successful, frame_changes_detected,
                score_impact, time.time()
            ))
            
            # Update local tracking
            if game_id not in self.target_interaction_history:
                self.target_interaction_history[game_id] = {}
            
            coord_key = (target.x, target.y)
            self.target_interaction_history[game_id][coord_key] = {
                'target': target,
                'interaction_successful': interaction_successful,
                'frame_changes_detected': frame_changes_detected,
                'score_impact': score_impact,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error recording target interaction: {e}")
    
    async def get_target_interaction_history(self, game_id: str) -> Dict[Tuple[int, int], Dict]:
        """Get interaction history for a game."""
        return self.target_interaction_history.get(game_id, {})
    
    async def should_avoid_coordinate(self, x: int, y: int, game_id: str) -> bool:
        """Check if a coordinate should be avoided based on interaction history."""
        try:
            if game_id not in self.target_interaction_history:
                return False
            
            coord_key = (x, y)
            if coord_key not in self.target_interaction_history[game_id]:
                return False
            
            interaction_data = self.target_interaction_history[game_id][coord_key]
            
            # Avoid coordinates that have failed multiple times
            if not interaction_data.get('interaction_successful', False):
                # Check if this coordinate has failed recently
                failure_time = interaction_data.get('timestamp', 0)
                if time.time() - failure_time < 300:  # 5 minutes
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking coordinate avoidance: {e}")
            return False

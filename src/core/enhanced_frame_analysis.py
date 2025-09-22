#!/usr/bin/env python3
"""
Enhanced Frame Change Analysis System

This module implements advanced frame change analysis with movement pattern
detection and change classification.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of frame changes that can be detected."""
    MAJOR_MOVEMENT = "major_movement"
    OBJECT_MOVEMENT = "object_movement"
    SMALL_MOVEMENT = "small_movement"
    VISUAL_CHANGE = "visual_change"
    MINOR_CHANGE = "minor_change"
    FRAME_RESIZE = "frame_resize"

@dataclass
class FrameChangeAnalysis:
    """Represents a frame change analysis result."""
    game_id: str
    action_number: int
    coordinates: Optional[Tuple[int, int]]
    change_type: ChangeType
    num_pixels_changed: int
    change_percentage: float
    movement_detected: bool
    change_locations: List[Tuple[int, int]]
    classification_confidence: float
    analysis_timestamp: float

class EnhancedFrameAnalysisSystem:
    """
    Enhanced Frame Change Analysis System
    
    Implements sophisticated frame change detection, movement pattern analysis,
    and change classification for better action effectiveness tracking.
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        self.frame_history: Dict[str, List[np.ndarray]] = {}
        self.change_analysis_cache: Dict[str, List[FrameChangeAnalysis]] = {}
        
    async def analyze_frame_changes(self, 
                                  before_frame: Any,
                                  after_frame: Any,
                                  game_id: str,
                                  action_number: int,
                                  coordinates: Optional[Tuple[int, int]] = None) -> Optional[FrameChangeAnalysis]:
        """
        Analyze changes between two frames with advanced classification.
        
        Args:
            before_frame: Frame before action
            after_frame: Frame after action
            game_id: Game identifier
            action_number: Action that was performed
            coordinates: Coordinates used (if applicable)
            
        Returns:
            FrameChangeAnalysis if changes detected, None otherwise
        """
        try:
            # Convert frames to numpy arrays
            before_array = self._normalize_frame(before_frame)
            after_array = self._normalize_frame(after_frame)
            
            if before_array is None or after_array is None:
                return None
            
            # Ensure frames have same dimensions
            if before_array.shape != after_array.shape:
                return self._analyze_frame_resize(before_array, after_array, game_id, action_number, coordinates)
            
            # Calculate basic change metrics
            pixel_diff = np.abs(after_array.astype(float) - before_array.astype(float))
            changed_pixels = pixel_diff > 0
            num_pixels_changed = np.sum(changed_pixels)
            change_percentage = (num_pixels_changed / before_array.size) * 100
            
            if num_pixels_changed == 0:
                return None
            
            # Get change locations
            change_locations = self._get_change_locations(changed_pixels)
            
            # Detect movement patterns
            movement_detected = self._detect_movement_pattern(change_locations, num_pixels_changed)
            
            # Classify change type
            change_type = self._classify_change_type(
                change_locations, num_pixels_changed, change_percentage, movement_detected
            )
            
            # Calculate classification confidence
            confidence = self._calculate_classification_confidence(
                change_type, num_pixels_changed, change_percentage, movement_detected
            )
            
            # Create analysis result
            analysis = FrameChangeAnalysis(
                game_id=game_id,
                action_number=action_number,
                coordinates=coordinates,
                change_type=change_type,
                num_pixels_changed=num_pixels_changed,
                change_percentage=change_percentage,
                movement_detected=movement_detected,
                change_locations=change_locations[:20],  # Limit for storage
                classification_confidence=confidence,
                analysis_timestamp=time.time()
            )
            
            # Store in database
            await self._store_frame_change_analysis(analysis)
            
            # Update local cache
            if game_id not in self.change_analysis_cache:
                self.change_analysis_cache[game_id] = []
            self.change_analysis_cache[game_id].append(analysis)
            
            # Keep only recent analyses
            if len(self.change_analysis_cache[game_id]) > 50:
                self.change_analysis_cache[game_id] = self.change_analysis_cache[game_id][-50:]
            
            logger.debug(f"Frame change analysis: {change_type.value} - "
                        f"{num_pixels_changed} pixels ({change_percentage:.1f}%), "
                        f"movement: {movement_detected}, confidence: {confidence:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing frame changes: {e}")
            return None
    
    def _normalize_frame(self, frame: Any) -> Optional[np.ndarray]:
        """Normalize frame to numpy array for analysis."""
        try:
            if frame is None:
                return None
            
            if isinstance(frame, np.ndarray):
                return frame
            
            if isinstance(frame, list):
                frame_array = np.array(frame)
            else:
                frame_array = np.array(frame)
            
            # Ensure 2D array
            if frame_array.ndim > 2:
                frame_array = frame_array.reshape(-1, frame_array.shape[-1])
            
            return frame_array
            
        except Exception as e:
            logger.error(f"Error normalizing frame: {e}")
            return None
    
    def _analyze_frame_resize(self, 
                            before_array: np.ndarray,
                            after_array: np.ndarray,
                            game_id: str,
                            action_number: int,
                            coordinates: Optional[Tuple[int, int]]) -> FrameChangeAnalysis:
        """Analyze frame resize changes."""
        try:
            before_size = before_array.shape
            after_size = after_array.shape
            
            # Calculate resize metrics
            size_change = abs(after_size[0] * after_size[1] - before_size[0] * before_size[1])
            resize_percentage = (size_change / (before_size[0] * before_size[1])) * 100
            
            analysis = FrameChangeAnalysis(
                game_id=game_id,
                action_number=action_number,
                coordinates=coordinates,
                change_type=ChangeType.FRAME_RESIZE,
                num_pixels_changed=size_change,
                change_percentage=resize_percentage,
                movement_detected=False,
                change_locations=[],
                classification_confidence=1.0,  # High confidence for resize
                analysis_timestamp=time.time()
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing frame resize: {e}")
            return None
    
    def _get_change_locations(self, changed_pixels: np.ndarray) -> List[Tuple[int, int]]:
        """Get coordinates of changed pixels."""
        try:
            y_coords, x_coords = np.where(changed_pixels)
            return list(zip(x_coords, y_coords))
            
        except Exception as e:
            logger.error(f"Error getting change locations: {e}")
            return []
    
    def _detect_movement_pattern(self, 
                               change_locations: List[Tuple[int, int]], 
                               num_pixels_changed: int) -> bool:
        """Detect if pixel changes represent object movement."""
        try:
            if not change_locations or num_pixels_changed < 3:
                return False
            
            # Calculate spatial distribution of changes
            if len(change_locations) >= 2:
                x_coords = [loc[0] for loc in change_locations]
                y_coords = [loc[1] for loc in change_locations]
                
                # Check if changes are clustered (indicating object movement)
                x_variance = np.var(x_coords) if len(x_coords) > 1 else 0
                y_variance = np.var(y_coords) if len(y_coords) > 1 else 0
                
                # Movement typically shows moderate clustering (not too spread out, not too tight)
                total_variance = x_variance + y_variance
                is_clustered = 5 < total_variance < 100  # Reasonable clustering range
                
                # Check for directional patterns (movement often has direction)
                if len(change_locations) >= 3:
                    # Calculate if changes form a line or have directional bias
                    x_trend = np.polyfit(range(len(x_coords)), x_coords, 1)[0] if len(x_coords) > 1 else 0
                    y_trend = np.polyfit(range(len(y_coords)), y_coords, 1)[0] if len(y_coords) > 1 else 0
                    
                    has_direction = abs(x_trend) > 0.5 or abs(y_trend) > 0.5
                else:
                    has_direction = False
                
                return is_clustered and (has_direction or num_pixels_changed > 10)
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting movement pattern: {e}")
            return False
    
    def _classify_change_type(self, 
                            change_locations: List[Tuple[int, int]],
                            num_pixels_changed: int,
                            change_percentage: float,
                            movement_detected: bool) -> ChangeType:
        """Classify the type of frame change."""
        try:
            if movement_detected:
                if num_pixels_changed > 50:
                    return ChangeType.MAJOR_MOVEMENT
                elif num_pixels_changed > 10:
                    return ChangeType.OBJECT_MOVEMENT
                else:
                    return ChangeType.SMALL_MOVEMENT
            elif change_percentage > 10:  # More than 10% of frame changed
                return ChangeType.VISUAL_CHANGE
            else:
                return ChangeType.MINOR_CHANGE
                
        except Exception as e:
            logger.error(f"Error classifying change type: {e}")
            return ChangeType.MINOR_CHANGE
    
    def _calculate_classification_confidence(self,
                                          change_type: ChangeType,
                                          num_pixels_changed: int,
                                          change_percentage: float,
                                          movement_detected: bool) -> float:
        """Calculate confidence in the change type classification."""
        try:
            confidence = 0.5  # Base confidence
            
            # Boost confidence based on change characteristics
            if change_type == ChangeType.MAJOR_MOVEMENT:
                if num_pixels_changed > 100 and movement_detected:
                    confidence = 0.9
                elif num_pixels_changed > 50:
                    confidence = 0.7
            elif change_type == ChangeType.OBJECT_MOVEMENT:
                if movement_detected and 10 < num_pixels_changed < 50:
                    confidence = 0.8
                elif movement_detected:
                    confidence = 0.6
            elif change_type == ChangeType.VISUAL_CHANGE:
                if change_percentage > 20:
                    confidence = 0.8
                elif change_percentage > 10:
                    confidence = 0.6
            elif change_type == ChangeType.MINOR_CHANGE:
                if num_pixels_changed < 5:
                    confidence = 0.7
                else:
                    confidence = 0.5
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating classification confidence: {e}")
            return 0.5
    
    async def _store_frame_change_analysis(self, analysis: FrameChangeAnalysis):
        """Store frame change analysis in database."""
        try:
            await self.integration.db.execute("""
                INSERT INTO frame_change_analysis
                (game_id, action_number, coordinates_x, coordinates_y, change_type,
                 num_pixels_changed, change_percentage, movement_detected,
                 change_locations, classification_confidence, analysis_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.game_id, analysis.action_number,
                analysis.coordinates[0] if analysis.coordinates else None,
                analysis.coordinates[1] if analysis.coordinates else None,
                analysis.change_type.value, analysis.num_pixels_changed,
                analysis.change_percentage, analysis.movement_detected,
                json.dumps(analysis.change_locations, default=str), analysis.classification_confidence,
                analysis.analysis_timestamp
            ))
            
        except Exception as e:
            logger.error(f"Error storing frame change analysis: {e}")
    
    async def get_action_effectiveness_analysis(self, 
                                              game_id: str,
                                              action_number: int) -> Dict[str, Any]:
        """Get effectiveness analysis for a specific action."""
        try:
            query = """
                SELECT change_type, COUNT(*) as count,
                       AVG(num_pixels_changed) as avg_pixels_changed,
                       AVG(change_percentage) as avg_change_percentage,
                       AVG(CASE WHEN movement_detected = 1 THEN 1.0 ELSE 0.0 END) as movement_rate,
                       AVG(classification_confidence) as avg_confidence
                FROM frame_change_analysis
                WHERE game_id = ? AND action_number = ?
                GROUP BY change_type
            """
            
            results = await self.integration.db.fetch_all(query, (game_id, action_number))
            
            effectiveness = {
                'total_attempts': sum(row['count'] for row in results),
                'change_types': {},
                'overall_movement_rate': 0.0,
                'overall_confidence': 0.0
            }
            
            total_attempts = effectiveness['total_attempts']
            if total_attempts > 0:
                for row in results:
                    change_type = row['change_type']
                    effectiveness['change_types'][change_type] = {
                        'count': row['count'],
                        'percentage': (row['count'] / total_attempts) * 100,
                        'avg_pixels_changed': row['avg_pixels_changed'],
                        'avg_change_percentage': row['avg_change_percentage'],
                        'movement_rate': row['movement_rate'],
                        'avg_confidence': row['avg_confidence']
                    }
                
                # Calculate overall metrics
                effectiveness['overall_movement_rate'] = sum(
                    row['movement_rate'] * row['count'] for row in results
                ) / total_attempts
                
                effectiveness['overall_confidence'] = sum(
                    row['avg_confidence'] * row['count'] for row in results
                ) / total_attempts
            
            return effectiveness
            
        except Exception as e:
            logger.error(f"Error getting action effectiveness analysis: {e}")
            return {}
    
    async def get_movement_pattern_analysis(self, game_id: str) -> Dict[str, Any]:
        """Get movement pattern analysis for a game."""
        try:
            query = """
                SELECT action_number, coordinates_x, coordinates_y,
                       COUNT(*) as attempts,
                       AVG(CASE WHEN movement_detected = 1 THEN 1.0 ELSE 0.0 END) as movement_rate,
                       AVG(num_pixels_changed) as avg_pixels_changed
                FROM frame_change_analysis
                WHERE game_id = ? AND movement_detected = 1
                GROUP BY action_number, coordinates_x, coordinates_y
                ORDER BY movement_rate DESC, attempts DESC
            """
            
            results = await self.integration.db.fetch_all(query, (game_id,))
            
            movement_patterns = {
                'total_movement_events': len(results),
                'high_movement_actions': [],
                'movement_coordinates': []
            }
            
            for row in results:
                if row['movement_rate'] > 0.5:  # High movement rate
                    movement_patterns['high_movement_actions'].append({
                        'action': row['action_number'],
                        'coordinates': (row['coordinates_x'], row['coordinates_y']),
                        'movement_rate': row['movement_rate'],
                        'attempts': row['attempts'],
                        'avg_pixels_changed': row['avg_pixels_changed']
                    })
                
                if row['coordinates_x'] is not None and row['coordinates_y'] is not None:
                    movement_patterns['movement_coordinates'].append({
                        'x': row['coordinates_x'],
                        'y': row['coordinates_y'],
                        'movement_rate': row['movement_rate'],
                        'attempts': row['attempts']
                    })
            
            return movement_patterns
            
        except Exception as e:
            logger.error(f"Error getting movement pattern analysis: {e}")
            return {}
    
    async def get_change_trend_analysis(self, game_id: str) -> Dict[str, Any]:
        """Get change trend analysis over time for a game."""
        try:
            query = """
                SELECT analysis_timestamp, change_type, num_pixels_changed,
                       change_percentage, movement_detected
                FROM frame_change_analysis
                WHERE game_id = ?
                ORDER BY analysis_timestamp ASC
            """
            
            results = await self.integration.db.fetch_all(query, (game_id,))
            
            if not results:
                return {}
            
            timestamps = [row['analysis_timestamp'] for row in results]
            pixels_changed = [row['num_pixels_changed'] for row in results]
            change_percentages = [row['change_percentage'] for row in results]
            movement_events = [row['movement_detected'] for row in results]
            
            # Calculate trends
            if len(timestamps) > 1:
                time_span = timestamps[-1] - timestamps[0]
                pixels_trend = np.polyfit(range(len(pixels_changed)), pixels_changed, 1)[0]
                percentage_trend = np.polyfit(range(len(change_percentages)), change_percentages, 1)[0]
            else:
                time_span = 0
                pixels_trend = 0
                percentage_trend = 0
            
            # Calculate movement frequency
            movement_frequency = sum(movement_events) / len(movement_events) if movement_events else 0
            
            return {
                'time_span': time_span,
                'total_events': len(results),
                'pixels_trend': pixels_trend,
                'percentage_trend': percentage_trend,
                'movement_frequency': movement_frequency,
                'avg_pixels_changed': np.mean(pixels_changed),
                'avg_change_percentage': np.mean(change_percentages),
                'recent_events': [
                    {
                        'timestamp': row['analysis_timestamp'],
                        'change_type': row['change_type'],
                        'pixels_changed': row['num_pixels_changed'],
                        'movement': bool(row['movement_detected'])
                    } for row in results[-10:]  # Last 10 events
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting change trend analysis: {e}")
            return {}
    
    async def get_frame_analysis_summary(self, game_id: str) -> Dict[str, Any]:
        """Get comprehensive frame analysis summary for a game."""
        try:
            # Get basic statistics
            query = """
                SELECT COUNT(*) as total_events,
                       AVG(num_pixels_changed) as avg_pixels_changed,
                       AVG(change_percentage) as avg_change_percentage,
                       AVG(CASE WHEN movement_detected = 1 THEN 1.0 ELSE 0.0 END) as movement_rate,
                       AVG(classification_confidence) as avg_confidence
                FROM frame_change_analysis
                WHERE game_id = ?
            """
            
            result = await self.integration.db.fetch_one(query, (game_id,))
            
            if not result or result['total_events'] == 0:
                return {'total_events': 0}
            
            # Get change type distribution
            type_query = """
                SELECT change_type, COUNT(*) as count
                FROM frame_change_analysis
                WHERE game_id = ?
                GROUP BY change_type
                ORDER BY count DESC
            """
            
            type_results = await self.integration.db.fetch_all(type_query, (game_id,))
            
            # Get action effectiveness
            action_query = """
                SELECT action_number, COUNT(*) as attempts,
                       AVG(CASE WHEN movement_detected = 1 THEN 1.0 ELSE 0.0 END) as movement_rate,
                       AVG(num_pixels_changed) as avg_pixels_changed
                FROM frame_change_analysis
                WHERE game_id = ?
                GROUP BY action_number
                ORDER BY movement_rate DESC
            """
            
            action_results = await self.integration.db.fetch_all(action_query, (game_id,))
            
            return {
                'total_events': result['total_events'],
                'avg_pixels_changed': result['avg_pixels_changed'],
                'avg_change_percentage': result['avg_change_percentage'],
                'movement_rate': result['movement_rate'],
                'avg_confidence': result['avg_confidence'],
                'change_type_distribution': {
                    row['change_type']: row['count'] for row in type_results
                },
                'action_effectiveness': [
                    {
                        'action': row['action_number'],
                        'attempts': row['attempts'],
                        'movement_rate': row['movement_rate'],
                        'avg_pixels_changed': row['avg_pixels_changed']
                    } for row in action_results
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting frame analysis summary: {e}")
            return {}

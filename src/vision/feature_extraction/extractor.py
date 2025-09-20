"""
Feature Extractor

Comprehensive feature extraction from ARC puzzle grids.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import Counter

from ..object_detection import ObjectDetector, DetectedObject
from ..spatial_analysis import SpatialAnalyzer, SpatialRelationship
from ..pattern_recognition import PatternRecognizer, PatternInfo
from ..change_detection import ChangeDetector, ChangeInfo

logger = logging.getLogger(__name__)

@dataclass
class ActionableTarget:
    """Represents an actionable target in the ARC puzzle grid."""
    object_id: int
    action_type: str  # 'click', 'drag', 'select', 'move'
    priority: float  # 0.0 to 1.0
    location: Tuple[int, int]  # Grid coordinates
    confidence: float  # 0.0 to 1.0
    description: str  # Human-readable description

class FeatureExtractor:
    """Comprehensive feature extractor for ARC puzzle analysis."""
    
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.spatial_analyzer = SpatialAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.change_detector = ChangeDetector()
    
    def extract_features(self, grid: List[List[int]], game_id: str = "unknown") -> Dict[str, Any]:
        """Extract comprehensive features from an ARC puzzle grid."""
        try:
            # Detect objects
            objects = self.object_detector.detect_objects(grid)
            
            # Analyze spatial relationships
            relationships = self.spatial_analyzer.analyze_relationships(objects)
            
            # Detect patterns
            patterns = self.pattern_recognizer.detect_patterns(grid)
            
            # Analyze colors
            colors = self._analyze_colors(grid)
            
            # Find actionable targets
            actionable_targets = self._find_actionable_targets(objects)
            
            # Create structured output
            feature_description = {
                'game_id': game_id,
                'grid_dimensions': (len(grid[0]) if grid and grid[0] else 0, len(grid)),
                'objects': [self._object_to_dict(obj) for obj in objects],
                'relationships': [self._relationship_to_dict(rel) for rel in relationships],
                'patterns': [self._pattern_to_dict(pattern) for pattern in patterns],
                'dominant_colors': colors,
                'actionable_targets': [self._target_to_dict(target) for target in actionable_targets],
                'timestamp': self._get_timestamp()
            }
            
            logger.info(f"Extracted {len(objects)} objects, {len(relationships)} relationships, "
                       f"{len(patterns)} patterns, {len(actionable_targets)} actionable targets for {game_id}")
            return feature_description
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {game_id}: {e}")
            return {'error': str(e), 'game_id': game_id}
    
    def detect_changes(self, input_grid: List[List[int]], output_grid: List[List[int]], 
                      game_id: str = "unknown") -> Dict[str, Any]:
        """Detect changes between input and output grids."""
        try:
            # Detect changes
            changes = self.change_detector.detect_changes(input_grid, output_grid)
            
            # Analyze change patterns
            change_analysis = self._analyze_changes(changes)
            
            # Create structured output
            change_description = {
                'game_id': game_id,
                'total_changes': len(changes),
                'changes': [self._change_to_dict(change) for change in changes],
                'change_analysis': change_analysis,
                'timestamp': self._get_timestamp()
            }
            
            logger.info(f"Detected {len(changes)} changes for {game_id}")
            return change_description
            
        except Exception as e:
            logger.error(f"Change detection failed for {game_id}: {e}")
            return {'error': str(e), 'game_id': game_id}
    
    def _analyze_colors(self, grid: List[List[int]]) -> Dict[str, Any]:
        """Analyze color distribution in the grid."""
        try:
            # Flatten grid and count colors
            flat_grid = [pixel for row in grid for pixel in row]
            color_counts = Counter(flat_grid)
            
            # Get dominant colors
            total_pixels = len(flat_grid)
            dominant_colors = []
            
            for color, count in color_counts.most_common(5):
                percentage = (count / total_pixels) * 100
                dominant_colors.append({
                    'color': color,
                    'count': count,
                    'percentage': percentage
                })
            
            return {
                'total_colors': len(color_counts),
                'dominant_colors': dominant_colors,
                'color_diversity': len(color_counts) / max(total_pixels, 1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing colors: {e}")
            return {'total_colors': 0, 'dominant_colors': [], 'color_diversity': 0.0}
    
    def _find_actionable_targets(self, objects: List[DetectedObject]) -> List[ActionableTarget]:
        """Find actionable targets among detected objects."""
        targets = []
        
        try:
            for obj in objects:
                if obj.is_actionable:
                    targets.append(ActionableTarget(
                        object_id=obj.id,
                        action_type=obj.action_type,
                        priority=obj.priority,
                        location=obj.centroid,
                        confidence=obj.priority,
                        description=f"Actionable {obj.type} at {obj.centroid}"
                    ))
            
            # Sort by priority
            targets.sort(key=lambda x: x.priority, reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding actionable targets: {e}")
        
        return targets
    
    def _analyze_changes(self, changes: List[ChangeInfo]) -> Dict[str, Any]:
        """Analyze patterns in detected changes."""
        try:
            if not changes:
                return {'total_changes': 0, 'change_types': {}, 'analysis': 'No changes detected'}
            
            # Count change types
            change_types = Counter(change.change_type for change in changes)
            
            # Analyze change distribution
            locations = [change.location for change in changes]
            if locations:
                x_coords = [loc[0] for loc in locations]
                y_coords = [loc[1] for loc in locations]
                
                analysis = {
                    'total_changes': len(changes),
                    'change_types': dict(change_types),
                    'change_center': (np.mean(x_coords), np.mean(y_coords)),
                    'change_spread': (np.std(x_coords), np.std(y_coords)),
                    'most_common_change': change_types.most_common(1)[0][0] if change_types else 'none'
                }
            else:
                analysis = {
                    'total_changes': len(changes),
                    'change_types': dict(change_types),
                    'analysis': 'Changes detected but no location analysis available'
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing changes: {e}")
            return {'total_changes': len(changes), 'error': str(e)}
    
    def _object_to_dict(self, obj: DetectedObject) -> Dict[str, Any]:
        """Convert DetectedObject to dictionary."""
        return {
            'id': obj.id,
            'type': obj.type,
            'color': obj.color,
            'area': obj.area,
            'centroid': obj.centroid,
            'bounding_box': obj.bounding_box,
            'is_hollow': obj.is_hollow,
            'is_actionable': obj.is_actionable,
            'action_type': obj.action_type,
            'priority': obj.priority
        }
    
    def _relationship_to_dict(self, rel: SpatialRelationship) -> Dict[str, Any]:
        """Convert SpatialRelationship to dictionary."""
        return {
            'object1_id': rel.object1_id,
            'object2_id': rel.object2_id,
            'relationship_type': rel.relationship_type,
            'confidence': rel.confidence,
            'distance': rel.distance,
            'angle': rel.angle,
            'description': rel.description
        }
    
    def _pattern_to_dict(self, pattern: PatternInfo) -> Dict[str, Any]:
        """Convert PatternInfo to dictionary."""
        return {
            'pattern_type': pattern.pattern_type,
            'description': pattern.description,
            'confidence': pattern.confidence,
            'locations': pattern.locations,
            'properties': pattern.properties,
            'size': pattern.size
        }
    
    def _change_to_dict(self, change: ChangeInfo) -> Dict[str, Any]:
        """Convert ChangeInfo to dictionary."""
        return {
            'change_type': change.change_type,
            'location': change.location,
            'old_value': change.old_value,
            'new_value': change.new_value,
            'confidence': change.confidence,
            'description': change.description
        }
    
    def _target_to_dict(self, target: ActionableTarget) -> Dict[str, Any]:
        """Convert ActionableTarget to dictionary."""
        return {
            'object_id': target.object_id,
            'action_type': target.action_type,
            'priority': target.priority,
            'location': target.location,
            'confidence': target.confidence,
            'description': target.description
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        import time
        return str(time.time())

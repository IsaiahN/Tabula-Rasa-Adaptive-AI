"""
Spatial Analyzer

Analyzes spatial relationships between objects in ARC puzzle grids.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from math import sqrt, atan2, degrees

logger = logging.getLogger(__name__)

@dataclass
class SpatialRelationship:
    """Represents spatial relationship between two objects."""
    object1_id: int
    object2_id: int
    relationship_type: str  # 'inside', 'left_of', 'above', 'touching', 'aligned', 'symmetric'
    confidence: float  # 0.0 to 1.0
    distance: float  # Distance between objects
    angle: float  # Angle between objects (degrees)
    description: str  # Human-readable description

class SpatialAnalyzer:
    """Analyzes spatial relationships between objects."""
    
    def __init__(self, proximity_threshold: float = 10.0, alignment_threshold: float = 5.0):
        self.proximity_threshold = proximity_threshold
        self.alignment_threshold = alignment_threshold
    
    def analyze_relationships(self, objects: List[Any]) -> List[SpatialRelationship]:
        """Analyze spatial relationships between all objects."""
        try:
            relationships = []
            
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects[i+1:], i+1):
                    relationship = self._analyze_object_pair(obj1, obj2)
                    if relationship:
                        relationships.append(relationship)
            
            logger.debug(f"Found {len(relationships)} spatial relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Spatial analysis failed: {e}")
            return []
    
    def _analyze_object_pair(self, obj1: Any, obj2: Any) -> Optional[SpatialRelationship]:
        """Analyze spatial relationship between two objects."""
        try:
            # Calculate basic spatial properties
            distance = self._calculate_distance(obj1, obj2)
            angle = self._calculate_angle(obj1, obj2)
            
            # Determine relationship type
            relationship_type, confidence = self._classify_relationship(obj1, obj2, distance, angle)
            
            if confidence > 0.3:  # Only include relationships with sufficient confidence
                description = self._generate_description(obj1, obj2, relationship_type, distance, angle)
                
                return SpatialRelationship(
                    object1_id=obj1.id,
                    object2_id=obj2.id,
                    relationship_type=relationship_type,
                    confidence=confidence,
                    distance=distance,
                    angle=angle,
                    description=description
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing object pair: {e}")
            return None
    
    def _calculate_distance(self, obj1: Any, obj2: Any) -> float:
        """Calculate distance between two objects."""
        try:
            x1, y1 = obj1.centroid
            x2, y2 = obj2.centroid
            return sqrt((x2 - x1)**2 + (y2 - y1)**2)
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0
    
    def _calculate_angle(self, obj1: Any, obj2: Any) -> float:
        """Calculate angle between two objects."""
        try:
            x1, y1 = obj1.centroid
            x2, y2 = obj2.centroid
            angle_rad = atan2(y2 - y1, x2 - x1)
            return degrees(angle_rad)
        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            return 0.0
    
    def _classify_relationship(self, obj1: Any, obj2: Any, distance: float, angle: float) -> Tuple[str, float]:
        """Classify the type of spatial relationship."""
        try:
            # Check for touching
            if self._are_touching(obj1, obj2):
                return 'touching', 0.9
            
            # Check for inside relationship
            if self._is_inside(obj1, obj2):
                return 'inside', 0.8
            
            # Check for alignment
            if self._are_aligned(obj1, obj2):
                return 'aligned', 0.7
            
            # Check for directional relationships
            if distance > self.proximity_threshold:
                if self._is_left_of(obj1, obj2):
                    return 'left_of', 0.6
                elif self._is_above(obj1, obj2):
                    return 'above', 0.6
                elif self._is_symmetric(obj1, obj2):
                    return 'symmetric', 0.5
            
            # Default to proximity
            if distance < self.proximity_threshold:
                return 'near', 0.4
            
            return 'distant', 0.1
            
        except Exception as e:
            logger.error(f"Error classifying relationship: {e}")
            return 'unknown', 0.0
    
    def _are_touching(self, obj1: Any, obj2: Any) -> bool:
        """Check if two objects are touching."""
        try:
            # Get bounding boxes
            x1, y1, w1, h1 = obj1.bounding_box
            x2, y2, w2, h2 = obj2.bounding_box
            
            # Check for overlap or adjacency
            return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
        except Exception as e:
            logger.error(f"Error checking if touching: {e}")
            return False
    
    def _is_inside(self, obj1: Any, obj2: Any) -> bool:
        """Check if obj1 is inside obj2."""
        try:
            x1, y1, w1, h1 = obj1.bounding_box
            x2, y2, w2, h2 = obj2.bounding_box
            
            # Check if obj1's bounding box is completely inside obj2's
            return (x2 <= x1 and y2 <= y1 and 
                    x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2)
        except Exception as e:
            logger.error(f"Error checking if inside: {e}")
            return False
    
    def _are_aligned(self, obj1: Any, obj2: Any) -> bool:
        """Check if two objects are aligned."""
        try:
            x1, y1, w1, h1 = obj1.bounding_box
            x2, y2, w2, h2 = obj2.bounding_box
            
            # Check horizontal alignment
            if abs(y1 + h1/2 - (y2 + h2/2)) < self.alignment_threshold:
                return True
            
            # Check vertical alignment
            if abs(x1 + w1/2 - (x2 + w2/2)) < self.alignment_threshold:
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking alignment: {e}")
            return False
    
    def _is_left_of(self, obj1: Any, obj2: Any) -> bool:
        """Check if obj1 is to the left of obj2."""
        try:
            x1, y1, w1, h1 = obj1.bounding_box
            x2, y2, w2, h2 = obj2.bounding_box
            
            return x1 + w1 < x2
        except Exception as e:
            logger.error(f"Error checking left_of: {e}")
            return False
    
    def _is_above(self, obj1: Any, obj2: Any) -> bool:
        """Check if obj1 is above obj2."""
        try:
            x1, y1, w1, h1 = obj1.bounding_box
            x2, y2, w2, h2 = obj2.bounding_box
            
            return y1 + h1 < y2
        except Exception as e:
            logger.error(f"Error checking above: {e}")
            return False
    
    def _is_symmetric(self, obj1: Any, obj2: Any) -> bool:
        """Check if two objects are symmetric."""
        try:
            # Simple symmetry check based on shape similarity
            if obj1.type != obj2.type:
                return False
            
            # Check if areas are similar
            area_ratio = min(obj1.area, obj2.area) / max(obj1.area, obj2.area)
            if area_ratio < 0.8:
                return False
            
            # Check if they have similar aspect ratios
            x1, y1, w1, h1 = obj1.bounding_box
            x2, y2, w2, h2 = obj2.bounding_box
            
            aspect1 = w1 / h1 if h1 > 0 else 1
            aspect2 = w2 / h2 if h2 > 0 else 1
            
            aspect_ratio = min(aspect1, aspect2) / max(aspect1, aspect2)
            return aspect_ratio > 0.9
            
        except Exception as e:
            logger.error(f"Error checking symmetry: {e}")
            return False
    
    def _generate_description(self, obj1: Any, obj2: Any, relationship_type: str, 
                           distance: float, angle: float) -> str:
        """Generate human-readable description of the relationship."""
        try:
            descriptions = {
                'touching': f"Object {obj1.id} is touching object {obj2.id}",
                'inside': f"Object {obj1.id} is inside object {obj2.id}",
                'left_of': f"Object {obj1.id} is to the left of object {obj2.id}",
                'above': f"Object {obj1.id} is above object {obj2.id}",
                'aligned': f"Object {obj1.id} is aligned with object {obj2.id}",
                'symmetric': f"Object {obj1.id} is symmetric to object {obj2.id}",
                'near': f"Object {obj1.id} is near object {obj2.id} (distance: {distance:.1f})",
                'distant': f"Object {obj1.id} is distant from object {obj2.id} (distance: {distance:.1f})"
            }
            
            return descriptions.get(relationship_type, f"Object {obj1.id} has unknown relationship with object {obj2.id}")
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return f"Relationship between object {obj1.id} and object {obj2.id}"

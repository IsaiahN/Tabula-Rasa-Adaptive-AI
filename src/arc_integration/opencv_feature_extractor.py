"""
OpenCV-Powered Feature Extraction for ARC Puzzle Analysis

This module implements computer vision-based feature extraction to transform
raw ARC puzzle grids into high-level symbolic descriptions for intelligent reasoning.

Key Features:
1. Object Detection & Segmentation
2. Spatial Relationship Analysis  
3. Pattern & Texture Recognition
4. Change Detection (Input/Output pairs)
5. Color and Shape Quantization
6. Feature Vector Extraction
"""

import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

@dataclass
class DetectedObject:
    """Represents a detected object in the ARC puzzle grid."""
    id: int
    type: str  # 'rectangle', 'circle', 'line', 'blob', 'unknown'
    points: List[Tuple[int, int]]  # Contour points
    color: int  # Dominant color value
    area: int  # Area in pixels
    centroid: Tuple[float, float]  # Center point
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    hu_moments: List[float]  # Shape descriptors
    is_hollow: bool  # Whether object is hollow/filled

@dataclass
class SpatialRelationship:
    """Represents spatial relationship between two objects."""
    object1_id: int
    object2_id: int
    relationship_type: str  # 'inside', 'left_of', 'above', 'touching', 'aligned', 'symmetric'
    confidence: float  # 0.0 to 1.0
    distance: float  # Distance between objects
    angle: float  # Angle of relationship

@dataclass
class PatternInfo:
    """Represents detected patterns in the grid."""
    pattern_type: str  # 'checkerboard', 'stripes', 'gradient', 'uniform', 'repeating'
    confidence: float
    region: Tuple[int, int, int, int]  # (x, y, width, height)
    parameters: Dict[str, Any]  # Pattern-specific parameters

@dataclass
class ChangeInfo:
    """Represents changes between input and output grids."""
    change_type: str  # 'translation', 'recolor', 'appear', 'disappear', 'transform'
    objects_affected: List[int]  # IDs of affected objects
    vector: Optional[Tuple[int, int]]  # For translations
    confidence: float
    description: str  # Human-readable description

class OpenCVFeatureExtractor:
    """
    OpenCV-powered feature extractor for ARC puzzle analysis.
    
    Transforms raw pixel grids into structured symbolic descriptions
    that can be used for intelligent reasoning and action selection.
    """
    
    def __init__(self):
        """Initialize the feature extractor with OpenCV parameters."""
        self.color_clusters = 4  # Number of dominant colors to extract
        self.min_contour_area = 5  # Minimum area for object detection
        self.max_contour_area = 10000  # Maximum area for object detection
        self.shape_templates = self._initialize_shape_templates()
        
    def _initialize_shape_templates(self) -> Dict[str, np.ndarray]:
        """Initialize shape templates for pattern matching."""
        templates = {}
        
        # Create basic shape templates
        # Square template
        square = np.ones((10, 10), dtype=np.uint8)
        templates['square'] = square
        
        # Circle template
        circle = np.zeros((10, 10), dtype=np.uint8)
        cv2.circle(circle, (5, 5), 4, 1, -1)
        templates['circle'] = circle
        
        # Line templates (horizontal, vertical, diagonal)
        line_h = np.zeros((3, 10), dtype=np.uint8)
        line_h[1, :] = 1
        templates['line_horizontal'] = line_h
        
        line_v = np.zeros((10, 3), dtype=np.uint8)
        line_v[:, 1] = 1
        templates['line_vertical'] = line_v
        
        return templates
    
    def extract_features(self, grid: List[List[int]], game_id: str = "unknown") -> Dict[str, Any]:
        """
        Extract comprehensive features from an ARC puzzle grid.
        
        Args:
            grid: 2D list representing the puzzle grid
            game_id: Identifier for the game
            
        Returns:
            Dictionary containing structured feature description
        """
        try:
            # Convert grid to OpenCV format
            cv_image = self._grid_to_opencv(grid)
            
            # Extract all features
            objects = self._detect_objects(cv_image)
            relationships = self._analyze_spatial_relationships(objects)
            patterns = self._detect_patterns(cv_image)
            colors = self._quantize_colors(cv_image)
            features = self._extract_feature_vectors(objects, cv_image)
            
            # Create structured output
            feature_description = {
                'game_id': game_id,
                'grid_dimensions': (len(grid[0]), len(grid)),
                'objects': [self._object_to_dict(obj) for obj in objects],
                'relationships': [self._relationship_to_dict(rel) for rel in relationships],
                'patterns': [self._pattern_to_dict(pattern) for pattern in patterns],
                'dominant_colors': colors,
                'feature_vectors': features,
                'summary': self._generate_summary(objects, relationships, patterns)
            }
            
            logger.info(f"Extracted {len(objects)} objects, {len(relationships)} relationships, {len(patterns)} patterns for {game_id}")
            return feature_description
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {game_id}: {e}")
            return {'error': str(e), 'game_id': game_id}
    
    def detect_changes(self, input_grid: List[List[int]], output_grid: List[List[int]], 
                      game_id: str = "unknown") -> Dict[str, Any]:
        """
        Detect changes between input and output grids.
        
        Args:
            input_grid: Input puzzle grid
            output_grid: Output puzzle grid  
            game_id: Identifier for the game
            
        Returns:
            Dictionary containing change analysis
        """
        try:
            # Convert grids to OpenCV format
            input_cv = self._grid_to_opencv(input_grid)
            output_cv = self._grid_to_opencv(output_grid)
            
            # Detect objects in both grids
            input_objects = self._detect_objects(input_cv)
            output_objects = self._detect_objects(output_cv)
            
            # Analyze changes
            changes = self._analyze_changes(input_objects, output_objects, input_cv, output_cv)
            
            change_description = {
                'game_id': game_id,
                'input_objects': len(input_objects),
                'output_objects': len(output_objects),
                'changes': [self._change_to_dict(change) for change in changes],
                'change_summary': self._generate_change_summary(changes)
            }
            
            logger.info(f"Detected {len(changes)} changes between input/output for {game_id}")
            return change_description
            
        except Exception as e:
            logger.error(f"Change detection failed for {game_id}: {e}")
            return {'error': str(e), 'game_id': game_id}
    
    def _grid_to_opencv(self, grid: List[List[int]]) -> np.ndarray:
        """Convert ARC grid format to OpenCV image format."""
        # Convert to numpy array and scale to 0-255 range
        grid_array = np.array(grid, dtype=np.uint8)
        
        # Scale colors to 0-255 range (ARC uses 0-9, scale to 0-255)
        if grid_array.max() <= 9:
            grid_array = (grid_array * 28).astype(np.uint8)  # Scale 0-9 to 0-252
        
        return grid_array
    
    def _detect_objects(self, cv_image: np.ndarray) -> List[DetectedObject]:
        """Detect and segment objects in the image."""
        objects = []
        
        # Find contours
        contours, _ = cv2.findContours(cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            
            # Calculate object properties
            points = [(int(point[0][0]), int(point[0][1])) for point in contour]
            centroid = self._calculate_centroid(contour)
            bounding_box = cv2.boundingRect(contour)
            
            # Determine object type
            object_type = self._classify_object_type(contour, area)
            
            # Calculate Hu moments for shape description
            hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
            
            # Determine if object is hollow
            is_hollow = self._is_hollow_object(contour, cv_image)
            
            # Get dominant color
            color = self._get_dominant_color(contour, cv_image)
            
            obj = DetectedObject(
                id=i,
                type=object_type,
                points=points,
                color=color,
                area=int(area),
                centroid=centroid,
                bounding_box=bounding_box,
                hu_moments=hu_moments.tolist(),
                is_hollow=is_hollow
            )
            
            objects.append(obj)
        
        return objects
    
    def _classify_object_type(self, contour: np.ndarray, area: int) -> str:
        """Classify the type of object based on contour analysis."""
        # Calculate shape properties
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 'unknown'
        
        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Classify based on properties
        if circularity > 0.8:
            return 'circle'
        elif 0.7 < aspect_ratio < 1.3 and circularity > 0.6:
            return 'square'
        elif aspect_ratio > 3 or aspect_ratio < 0.33:
            return 'line'
        else:
            return 'blob'
    
    def _is_hollow_object(self, contour: np.ndarray, image: np.ndarray) -> bool:
        """Determine if an object is hollow (has internal structure)."""
        # Create mask for the contour
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Check if there are holes inside
        # This is a simplified check - could be improved
        return False  # For now, assume all objects are solid
    
    def _get_dominant_color(self, contour: np.ndarray, image: np.ndarray) -> int:
        """Get the dominant color value within the contour."""
        # Create mask for the contour
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Get pixels within the contour
        pixels = image[mask > 0]
        
        if len(pixels) == 0:
            return 0
        
        # Return the most common color
        unique, counts = np.unique(pixels, return_counts=True)
        return int(unique[np.argmax(counts)])
    
    def _calculate_centroid(self, contour: np.ndarray) -> Tuple[float, float]:
        """Calculate the centroid of a contour."""
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return (0.0, 0.0)
        
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        return (float(cx), float(cy))
    
    def _analyze_spatial_relationships(self, objects: List[DetectedObject]) -> List[SpatialRelationship]:
        """Analyze spatial relationships between objects."""
        relationships = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # Calculate distance
                distance = np.sqrt((obj1.centroid[0] - obj2.centroid[0])**2 + 
                                 (obj1.centroid[1] - obj2.centroid[1])**2)
                
                # Check various relationships
                rel_type, confidence = self._determine_relationship(obj1, obj2, distance)
                
                if rel_type != 'none':
                    angle = np.arctan2(obj2.centroid[1] - obj1.centroid[1],
                                     obj2.centroid[0] - obj1.centroid[0])
                    
                    rel = SpatialRelationship(
                        object1_id=obj1.id,
                        object2_id=obj2.id,
                        relationship_type=rel_type,
                        confidence=confidence,
                        distance=distance,
                        angle=float(angle)
                    )
                    relationships.append(rel)
        
        return relationships
    
    def _determine_relationship(self, obj1: DetectedObject, obj2: DetectedObject, distance: float) -> Tuple[str, float]:
        """Determine the spatial relationship between two objects."""
        # Check if objects are touching
        if distance < 5:  # Threshold for touching
            return 'touching', 0.9
        
        # Check if one object is inside another
        if self._is_inside(obj1, obj2):
            return 'inside', 0.8
        elif self._is_inside(obj2, obj1):
            return 'inside', 0.8
        
        # Check alignment
        if self._is_aligned(obj1, obj2):
            return 'aligned', 0.7
        
        # Check relative positions
        dx = obj2.centroid[0] - obj1.centroid[0]
        dy = obj2.centroid[1] - obj1.centroid[1]
        
        if abs(dx) > abs(dy):
            if dx > 0:
                return 'left_of', 0.6
            else:
                return 'right_of', 0.6
        else:
            if dy > 0:
                return 'above', 0.6
            else:
                return 'below', 0.6
        
        return 'none', 0.0
    
    def _is_inside(self, obj1: DetectedObject, obj2: DetectedObject) -> bool:
        """Check if obj1 is inside obj2."""
        # Simple bounding box check
        x1, y1, w1, h1 = obj1.bounding_box
        x2, y2, w2, h2 = obj2.bounding_box
        
        return (x2 <= x1 and y2 <= y1 and 
                x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1)
    
    def _is_aligned(self, obj1: DetectedObject, obj2: DetectedObject) -> bool:
        """Check if objects are aligned (same row or column)."""
        # Check horizontal alignment
        if abs(obj1.centroid[1] - obj2.centroid[1]) < 5:
            return True
        
        # Check vertical alignment
        if abs(obj1.centroid[0] - obj2.centroid[0]) < 5:
            return True
        
        return False
    
    def _detect_patterns(self, cv_image: np.ndarray) -> List[PatternInfo]:
        """Detect patterns and textures in the image."""
        patterns = []
        
        # Check for checkerboard pattern
        checkerboard = self._detect_checkerboard(cv_image)
        if checkerboard:
            patterns.append(checkerboard)
        
        # Check for stripes
        stripes = self._detect_stripes(cv_image)
        if stripes:
            patterns.append(stripes)
        
        # Check for uniform regions
        uniform = self._detect_uniform_regions(cv_image)
        if uniform:
            patterns.append(uniform)
        
        return patterns
    
    def _detect_checkerboard(self, cv_image: np.ndarray) -> Optional[PatternInfo]:
        """Detect checkerboard patterns."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated pattern matching
        return None
    
    def _detect_stripes(self, cv_image: np.ndarray) -> Optional[PatternInfo]:
        """Detect stripe patterns."""
        # This is a simplified implementation
        return None
    
    def _detect_uniform_regions(self, cv_image: np.ndarray) -> Optional[PatternInfo]:
        """Detect uniform color regions."""
        # Calculate color variance
        variance = np.var(cv_image)
        
        if variance < 10:  # Low variance indicates uniform region
            return PatternInfo(
                pattern_type='uniform',
                confidence=0.8,
                region=(0, 0, cv_image.shape[1], cv_image.shape[0]),
                parameters={'variance': variance}
            )
        
        return None
    
    def _quantize_colors(self, cv_image: np.ndarray) -> List[int]:
        """Quantize colors to dominant color palette."""
        # Reshape image to 1D array of pixels
        pixels = cv_image.reshape(-1, 1)
        
        # Use K-means clustering to find dominant colors
        kmeans = KMeans(n_clusters=self.color_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Return dominant colors
        return [int(color[0]) for color in kmeans.cluster_centers_]
    
    def _extract_feature_vectors(self, objects: List[DetectedObject], cv_image: np.ndarray) -> Dict[str, Any]:
        """Extract numerical feature vectors for the entire grid."""
        features = {
            'object_count': len(objects),
            'total_area': sum(obj.area for obj in objects),
            'color_diversity': len(set(obj.color for obj in objects)),
            'shape_diversity': len(set(obj.type for obj in objects)),
            'grid_complexity': self._calculate_grid_complexity(cv_image)
        }
        
        return features
    
    def _calculate_grid_complexity(self, cv_image: np.ndarray) -> float:
        """Calculate a complexity score for the grid."""
        # Use edge detection to measure complexity
        edges = cv2.Canny(cv_image, 50, 150)
        edge_density = np.sum(edges > 0) / (cv_image.shape[0] * cv_image.shape[1])
        return float(edge_density)
    
    def _analyze_changes(self, input_objects: List[DetectedObject], output_objects: List[DetectedObject],
                        input_image: np.ndarray, output_image: np.ndarray) -> List[ChangeInfo]:
        """Analyze changes between input and output grids."""
        changes = []
        
        # Simple change detection based on object count
        if len(output_objects) > len(input_objects):
            changes.append(ChangeInfo(
                change_type='appear',
                objects_affected=list(range(len(input_objects), len(output_objects))),
                vector=None,
                confidence=0.8,
                description=f"{len(output_objects) - len(input_objects)} new objects appeared"
            ))
        elif len(output_objects) < len(input_objects):
            changes.append(ChangeInfo(
                change_type='disappear',
                objects_affected=list(range(len(output_objects), len(input_objects))),
                vector=None,
                confidence=0.8,
                description=f"{len(input_objects) - len(output_objects)} objects disappeared"
            ))
        
        return changes
    
    def _generate_summary(self, objects: List[DetectedObject], relationships: List[SpatialRelationship], 
                         patterns: List[PatternInfo]) -> str:
        """Generate a human-readable summary of the grid."""
        summary_parts = []
        
        # Object summary
        if objects:
            object_types = {}
            for obj in objects:
                object_types[obj.type] = object_types.get(obj.type, 0) + 1
            
            type_str = ", ".join([f"{count} {obj_type}" for obj_type, count in object_types.items()])
            summary_parts.append(f"Objects: {type_str}")
        
        # Relationship summary
        if relationships:
            rel_types = {}
            for rel in relationships:
                rel_types[rel.relationship_type] = rel_types.get(rel.relationship_type, 0) + 1
            
            rel_str = ", ".join([f"{count} {rel_type}" for rel_type, count in rel_types.items()])
            summary_parts.append(f"Relationships: {rel_str}")
        
        # Pattern summary
        if patterns:
            pattern_str = ", ".join([p.pattern_type for p in patterns])
            summary_parts.append(f"Patterns: {pattern_str}")
        
        return "; ".join(summary_parts) if summary_parts else "Empty grid"
    
    def _generate_change_summary(self, changes: List[ChangeInfo]) -> str:
        """Generate a human-readable summary of changes."""
        if not changes:
            return "No changes detected"
        
        change_descriptions = [change.description for change in changes]
        return "; ".join(change_descriptions)
    
    # Helper methods for serialization
    def _object_to_dict(self, obj: DetectedObject) -> Dict[str, Any]:
        """Convert DetectedObject to dictionary."""
        return {
            'id': obj.id,
            'type': obj.type,
            'points': obj.points,
            'color': obj.color,
            'area': obj.area,
            'centroid': obj.centroid,
            'bounding_box': obj.bounding_box,
            'hu_moments': obj.hu_moments,
            'is_hollow': obj.is_hollow
        }
    
    def _relationship_to_dict(self, rel: SpatialRelationship) -> Dict[str, Any]:
        """Convert SpatialRelationship to dictionary."""
        return {
            'object1_id': rel.object1_id,
            'object2_id': rel.object2_id,
            'relationship_type': rel.relationship_type,
            'confidence': rel.confidence,
            'distance': rel.distance,
            'angle': rel.angle
        }
    
    def _pattern_to_dict(self, pattern: PatternInfo) -> Dict[str, Any]:
        """Convert PatternInfo to dictionary."""
        return {
            'pattern_type': pattern.pattern_type,
            'confidence': pattern.confidence,
            'region': pattern.region,
            'parameters': pattern.parameters
        }
    
    def _change_to_dict(self, change: ChangeInfo) -> Dict[str, Any]:
        """Convert ChangeInfo to dictionary."""
        return {
            'change_type': change.change_type,
            'objects_affected': change.objects_affected,
            'vector': change.vector,
            'confidence': change.confidence,
            'description': change.description
        }

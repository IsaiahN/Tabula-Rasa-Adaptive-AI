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
import time
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
    is_actionable: bool = False  # Whether this object can be interacted with
    action_type: str = "none"  # Type of action (click, move_to, etc.)
    priority: float = 0.0  # Priority for action selection (0.0 to 1.0)

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

@dataclass
class ActionableTarget:
    """Represents an actionable target for ACTION6 commands."""
    id: int
    coordinates: Tuple[int, int]  # Target coordinates for ACTION6
    object_type: str  # Type of actionable element
    action_type: str  # Type of action (click, move_to, interact)
    priority: float  # Priority for action selection (0.0 to 1.0)
    confidence: float  # Confidence in target identification
    description: str  # Human-readable description
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    color: int  # Dominant color
    area: int  # Area in pixels

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
            # Use a simplified approach that doesn't rely on OpenCV's problematic functions
            objects = self._detect_objects_simple(grid)
            relationships = self._analyze_spatial_relationships_simple(objects)
            patterns = self._detect_patterns_simple(grid)
            colors = self._analyze_colors_simple(grid)
            
            # Create structured output
            feature_description = {
                'game_id': game_id,
                'grid_dimensions': (len(grid[0]) if grid and grid[0] else 0, len(grid)),
                'objects': objects,
                'relationships': relationships,
                'patterns': patterns,
                'dominant_colors': colors,
                'timestamp': time.time()
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
            input_objects = self._detect_objects_simple(input_grid)
            output_objects = self._detect_objects_simple(output_grid)
            
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
        # Debug: Log grid properties
        logger.debug(f"Grid type: {type(grid)}, length: {len(grid) if isinstance(grid, list) else 'N/A'}")
        if isinstance(grid, list) and len(grid) > 0:
            logger.debug(f"Grid[0] type: {type(grid[0])}, length: {len(grid[0]) if isinstance(grid[0], list) else 'N/A'}")
        
        # Convert to numpy array and scale to 0-255 range
        grid_array = np.array(grid, dtype=np.uint8)
        logger.debug(f"Grid array shape: {grid_array.shape}, dtype: {grid_array.dtype}, min: {grid_array.min()}, max: {grid_array.max()}")
        
        # Scale colors to 0-255 range (ARC uses 0-9, scale to 0-255)
        if grid_array.max() <= 9:
            grid_array = (grid_array * 28).astype(np.uint8)  # Scale 0-9 to 0-252
        
        # Ensure the image is in the correct format for OpenCV
        # Convert to 3-channel if needed, or ensure it's single channel
        if len(grid_array.shape) == 2:
            # Single channel image - ensure it's the right type
            grid_array = grid_array.astype(np.uint8)
        elif len(grid_array.shape) == 3:
            # Multi-channel image - convert to grayscale
            grid_array = cv2.cvtColor(grid_array, cv2.COLOR_BGR2GRAY)
        
        # Ensure minimum size for OpenCV operations
        if grid_array.shape[0] < 3 or grid_array.shape[1] < 3:
            # Pad the image to minimum size
            padded = np.zeros((max(3, grid_array.shape[0]), max(3, grid_array.shape[1])), dtype=np.uint8)
            padded[:grid_array.shape[0], :grid_array.shape[1]] = grid_array
            grid_array = padded
        
        return grid_array
    
    def _detect_objects(self, cv_image: np.ndarray) -> List[DetectedObject]:
        """Detect and segment objects in the image."""
        objects = []
        
        try:
            # Debug: Log image properties
            logger.debug(f"Image shape: {cv_image.shape}, dtype: {cv_image.dtype}, min: {cv_image.min()}, max: {cv_image.max()}")
            
            # Ensure image is in correct format - single channel grayscale
            if len(cv_image.shape) == 3:
                # Multi-channel image - convert to grayscale
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            elif len(cv_image.shape) == 2:
                # Already single channel - ensure it's uint8
                cv_image = cv_image.astype(np.uint8)
            else:
                logger.warning(f"Unexpected image shape: {cv_image.shape}")
                return objects
            
            # Ensure image is not empty
            if cv_image.size == 0:
                return objects
            
            # Use a simpler approach - detect connected components instead of contours
            # This avoids the OpenCV contour detection bug
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv_image, connectivity=8)
            
            # Convert connected components to objects
            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if area < self.min_contour_area or area > self.max_contour_area:
                    continue
                
                # Get bounding box
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Get centroid
                cx, cy = centroids[i]
                
                # Create points for the bounding box
                points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                
                # Determine object type based on aspect ratio and area
                aspect_ratio = w / h if h > 0 else 1
                if 0.8 < aspect_ratio < 1.2 and area > 10:
                    object_type = 'square'
                elif area < 20:
                    object_type = 'blob'
                else:
                    object_type = 'rectangle'
                
                # Get dominant color
                mask = (labels == i).astype(np.uint8)
                color = int(np.median(cv_image[mask > 0])) if np.any(mask) else 0
                
                obj = DetectedObject(
                    id=i-1,
                    type=object_type,
                    points=points,
                    color=color,
                    area=int(area),
                    centroid=(float(cx), float(cy)),
                    bounding_box=(x, y, w, h),
                    hu_moments=[0.0] * 7,  # Simplified - no Hu moments
                    is_hollow=False
                )
                
                objects.append(obj)
                
        except Exception as e:
            logger.warning(f"Object detection failed: {e}")
            return objects
        
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
    
    def _detect_objects_simple(self, grid: List[List[int]]) -> List[Dict[str, Any]]:
        """Simple object detection without OpenCV."""
        objects = []
        
        try:
            if not grid or not grid[0]:
                return objects
            
            # Debug: Log grid structure
            logger.debug(f"Grid structure: {type(grid)}, length: {len(grid) if isinstance(grid, list) else 'N/A'}")
            if isinstance(grid, list) and len(grid) > 0:
                logger.debug(f"Grid[0] structure: {type(grid[0])}, length: {len(grid[0]) if isinstance(grid[0], list) else 'N/A'}")
                if isinstance(grid[0], list) and len(grid[0]) > 0:
                    logger.debug(f"Grid[0][0] structure: {type(grid[0][0])}, value: {grid[0][0]}")
            
            # Ensure grid is properly formatted
            if isinstance(grid[0], list) and len(grid[0]) > 0 and isinstance(grid[0][0], list):
                # Grid contains nested lists - flatten them
                grid = [[int(item) if isinstance(item, (int, float)) else 0 for item in row] for row in grid]
                logger.debug(f"Flattened grid sample: {grid[0][:5] if grid and grid[0] else 'empty'}")
            
            height, width = len(grid), len(grid[0])
            logger.debug(f"Grid dimensions: {width}x{height}")
            
            # Find connected components using simple flood fill
            visited = [[False for _ in range(width)] for _ in range(height)]
            
            for y in range(height):
                for x in range(width):
                    if not visited[y][x] and grid[y][x] != 0:  # Non-background pixel
                        # Start flood fill
                        color = grid[y][x]
                        points = []
                        stack = [(x, y)]
                        
                        while stack:
                            cx, cy = stack.pop()
                            if (0 <= cx < width and 0 <= cy < height and 
                                not visited[cy][cx] and grid[cy][cx] == color):
                                visited[cy][cx] = True
                                points.append((cx, cy))
                                
                                # Add neighbors
                                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    stack.append((cx + dx, cy + dy))
                        
                        if len(points) >= 1:  # Minimum object size - lowered threshold
                            # Calculate bounding box
                            min_x = min(p[0] for p in points)
                            max_x = max(p[0] for p in points)
                            min_y = min(p[1] for p in points)
                            max_y = max(p[1] for p in points)
                            
                            # Calculate centroid
                            cx = sum(p[0] for p in points) / len(points)
                            cy = sum(p[1] for p in points) / len(points)
                            
                            # Determine object type
                            w, h = max_x - min_x + 1, max_y - min_y + 1
                            aspect_ratio = w / h if h > 0 else 1
                            
                            if 0.8 < aspect_ratio < 1.2 and w * h < 50:
                                obj_type = 'square'
                            elif w * h < 10:
                                obj_type = 'blob'
                            else:
                                obj_type = 'rectangle'
                            
                            obj = {
                                'id': len(objects),
                                'type': obj_type,
                                'points': points,
                                'color': int(color),
                                'area': len(points),
                                'centroid': (float(cx), float(cy)),
                                'bounding_box': (min_x, min_y, w, h)
                            }
                            objects.append(obj)
            
            return objects
            
        except Exception as e:
            logger.warning(f"Simple object detection failed: {e}")
            return []
    
    def _analyze_spatial_relationships_simple(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple spatial relationship analysis."""
        relationships = []
        
        try:
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects[i+1:], i+1):
                    # Calculate distance between centroids
                    x1, y1 = obj1['centroid']
                    x2, y2 = obj2['centroid']
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    
                    # Check if objects are touching (bounding boxes overlap)
                    bbox1 = obj1['bounding_box']
                    bbox2 = obj2['bounding_box']
                    
                    touching = not (bbox1[0] + bbox1[2] < bbox2[0] or 
                                  bbox2[0] + bbox2[2] < bbox1[0] or
                                  bbox1[1] + bbox1[3] < bbox2[1] or 
                                  bbox2[1] + bbox2[3] < bbox1[1])
                    
                    # Determine relationship type
                    if touching:
                        rel_type = 'touching'
                        confidence = 0.9
                    elif distance < 10:
                        rel_type = 'near'
                        confidence = 0.7
                    else:
                        rel_type = 'far'
                        confidence = 0.5
                    
                    rel = {
                        'object1_id': obj1['id'],
                        'object2_id': obj2['id'],
                        'relationship_type': rel_type,
                        'confidence': confidence,
                        'distance': distance
                    }
                    relationships.append(rel)
            
            return relationships
            
        except Exception as e:
            logger.warning(f"Simple relationship analysis failed: {e}")
            return []
    
    def _detect_patterns_simple(self, grid: List[List[int]]) -> List[Dict[str, Any]]:
        """Simple pattern detection."""
        patterns = []
        
        try:
            if not grid or not grid[0]:
                return patterns
            
            # Ensure grid is properly formatted
            if isinstance(grid[0], list) and len(grid[0]) > 0 and isinstance(grid[0][0], list):
                # Grid contains nested lists - flatten them
                grid = [[int(item) if isinstance(item, (int, float)) else 0 for item in row] for row in grid]
            
            height, width = len(grid), len(grid[0])
            
            # Check for uniform regions
            for y in range(height - 5):
                for x in range(width - 5):
                    # Check 5x5 region
                    region_colors = set()
                    for dy in range(5):
                        for dx in range(5):
                            region_colors.add(grid[y + dy][x + dx])
                    
                    if len(region_colors) == 1:  # Uniform region
                        patterns.append({
                            'pattern_type': 'uniform',
                            'confidence': 0.8,
                            'region': (x, y, 5, 5),
                            'color': list(region_colors)[0]
                        })
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Simple pattern detection failed: {e}")
            return []
    
    def _analyze_colors_simple(self, grid: List[List[int]]) -> Dict[str, Any]:
        """Simple color analysis."""
        try:
            if not grid or not grid[0]:
                return {'dominant_colors': [], 'color_distribution': {}}
            
            # Ensure grid is properly formatted
            if isinstance(grid[0], list) and len(grid[0]) > 0 and isinstance(grid[0][0], list):
                # Grid contains nested lists - flatten them
                grid = [[int(item) if isinstance(item, (int, float)) else 0 for item in row] for row in grid]
            
            # Count color frequencies
            color_counts = {}
            for row in grid:
                for color in row:
                    # Ensure color is an integer
                    color_val = int(color) if isinstance(color, (int, float)) else 0
                    color_counts[color_val] = color_counts.get(color_val, 0) + 1
            
            # Get dominant colors
            total_pixels = sum(color_counts.values())
            dominant_colors = []
            
            for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_pixels
                if percentage > 0.05:  # At least 5% of pixels
                    dominant_colors.append({
                        'color': color,
                        'percentage': percentage,
                        'count': count
                    })
            
            return {
                'dominant_colors': dominant_colors,
                'color_distribution': color_counts,
                'total_colors': len(color_counts)
            }
            
        except Exception as e:
            logger.warning(f"Simple color analysis failed: {e}")
            return {'dominant_colors': [], 'color_distribution': {}}
    
    def identify_actionable_targets(self, grid: List[List[int]], game_id: str = "unknown") -> List[ActionableTarget]:
        """
        Identify actionable targets (buttons, portals, interactive elements) for ACTION6 commands.
        
        This method analyzes the grid to find elements that can be interacted with
        using ACTION6 coordinate-based actions.
        
        Args:
            grid: 2D list representing the puzzle grid
            game_id: Identifier for the game
            
        Returns:
            List of ActionableTarget objects representing clickable/interactive elements
        """
        try:
            if not grid or not grid[0]:
                return []
            
            # Ensure grid is properly formatted
            if isinstance(grid[0], list) and len(grid[0]) > 0 and isinstance(grid[0][0], list):
                # Grid contains nested lists - flatten them
                grid = [[int(item) if isinstance(item, (int, float)) else 0 for item in row] for row in grid]
            
            height, width = len(grid), len(grid[0])
            actionable_targets = []
            
            # Detect objects first
            objects = self._detect_objects_simple(grid)
            
            # Classify objects as actionable targets
            for obj in objects:
                target = self._classify_as_actionable_target(obj, grid)
                if target:
                    actionable_targets.append(target)
            
            # Look for specific actionable patterns
            pattern_targets = self._detect_actionable_patterns(grid)
            actionable_targets.extend(pattern_targets)
            
            # Sort by priority (highest first)
            actionable_targets.sort(key=lambda t: t.priority, reverse=True)
            
            logger.info(f"Identified {len(actionable_targets)} actionable targets for {game_id}")
            return actionable_targets
            
        except Exception as e:
            logger.error(f"Actionable target identification failed for {game_id}: {e}")
            return []
    
    def _classify_as_actionable_target(self, obj: Dict[str, Any], grid: List[List[int]]) -> Optional[ActionableTarget]:
        """Classify a detected object as an actionable target."""
        try:
            # Extract object properties
            obj_type = obj.get('type', 'unknown')
            area = obj.get('area', 0)
            color = obj.get('color', 0)
            centroid = obj.get('centroid', (0, 0))
            bbox = obj.get('bounding_box', (0, 0, 0, 0))
            
            # Determine if object is actionable based on properties
            is_actionable = False
            action_type = "none"
            priority = 0.0
            description = ""
            
            # Button detection (small, square-ish objects)
            if (obj_type in ['square', 'rectangle'] and 
                5 <= area <= 50 and 
                0.7 <= bbox[2] / bbox[3] <= 1.3):  # Roughly square aspect ratio
                is_actionable = True
                action_type = "click"
                priority = 0.8
                description = f"Button-like object (area: {area}, color: {color})"
            
            # Portal detection (larger, distinctive objects)
            elif (obj_type in ['circle', 'rectangle'] and 
                  20 <= area <= 200 and 
                  color in [1, 2, 3, 4, 5]):  # Non-background colors
                is_actionable = True
                action_type = "move_to"
                priority = 0.7
                description = f"Portal-like object (area: {area}, color: {color})"
            
            # Interactive element detection (medium-sized objects)
            elif (obj_type in ['rectangle', 'blob'] and 
                  10 <= area <= 100 and 
                  color != 0):  # Non-background color
                is_actionable = True
                action_type = "interact"
                priority = 0.6
                description = f"Interactive element (area: {area}, color: {color})"
            
            # High-priority targets (specific patterns)
            elif self._is_high_priority_target(obj, grid):
                is_actionable = True
                action_type = "click"
                priority = 0.9
                description = f"High-priority target (area: {area}, color: {color})"
            
            if is_actionable:
                return ActionableTarget(
                    id=obj.get('id', 0),
                    coordinates=(int(centroid[0]), int(centroid[1])),
                    object_type=obj_type,
                    action_type=action_type,
                    priority=priority,
                    confidence=0.8,  # Base confidence
                    description=description,
                    bounding_box=bbox,
                    color=color,
                    area=area
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Target classification failed for object {obj.get('id', 'unknown')}: {e}")
            return None
    
    def _is_high_priority_target(self, obj: Dict[str, Any], grid: List[List[int]]) -> bool:
        """Check if an object is a high-priority target based on context."""
        try:
            # Check for objects in corners (often buttons)
            centroid = obj.get('centroid', (0, 0))
            bbox = obj.get('bounding_box', (0, 0, 0, 0))
            height, width = len(grid), len(grid[0])
            
            # Corner objects are often interactive
            if (centroid[0] < width * 0.2 or centroid[0] > width * 0.8 or
                centroid[1] < height * 0.2 or centroid[1] > height * 0.8):
                return True
            
            # Check for objects with high color contrast
            color = obj.get('color', 0)
            if color in [1, 2, 3, 4, 5, 6, 7, 8, 9]:  # Non-zero colors
                # Check surrounding area for contrast
                x, y, w, h = bbox
                if self._has_high_contrast(grid, x, y, w, h, color):
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"High priority check failed: {e}")
            return False
    
    def _has_high_contrast(self, grid: List[List[int]], x: int, y: int, w: int, h: int, color: int) -> bool:
        """Check if an object has high contrast with its surroundings."""
        try:
            height, width = len(grid), len(grid[0])
            
            # Check surrounding pixels
            surrounding_colors = []
            for dy in range(max(0, y-2), min(height, y+h+2)):
                for dx in range(max(0, x-2), min(width, x+w+2)):
                    if not (x <= dx < x+w and y <= dy < y+h):  # Outside the object
                        surrounding_colors.append(grid[dy][dx])
            
            if not surrounding_colors:
                return False
            
            # Check if object color is different from surroundings
            unique_surrounding = set(surrounding_colors)
            return color not in unique_surrounding and len(unique_surrounding) > 1
            
        except Exception as e:
            logger.debug(f"Contrast check failed: {e}")
            return False
    
    def _detect_actionable_patterns(self, grid: List[List[int]]) -> List[ActionableTarget]:
        """Detect actionable patterns in the grid."""
        targets = []
        
        try:
            height, width = len(grid), len(grid[0])
            
            # Look for button patterns (small squares in rows/columns)
            button_patterns = self._detect_button_patterns(grid)
            targets.extend(button_patterns)
            
            # Look for portal patterns (larger distinctive areas)
            portal_patterns = self._detect_portal_patterns(grid)
            targets.extend(portal_patterns)
            
            # Look for interactive element patterns
            interactive_patterns = self._detect_interactive_patterns(grid)
            targets.extend(interactive_patterns)
            
        except Exception as e:
            logger.debug(f"Pattern detection failed: {e}")
        
        return targets
    
    def _detect_button_patterns(self, grid: List[List[int]]) -> List[ActionableTarget]:
        """Detect button-like patterns in the grid."""
        targets = []
        
        try:
            height, width = len(grid), len(grid[0])
            
            # Look for small square patterns
            for y in range(height - 2):
                for x in range(width - 2):
                    # Check 3x3 region for button pattern
                    region = []
                    for dy in range(3):
                        for dx in range(3):
                            region.append(grid[y + dy][x + dx])
                    
                    # Check if this is a button pattern (uniform color, small size)
                    if len(set(region)) == 1 and region[0] != 0:  # Uniform non-background
                        target = ActionableTarget(
                            id=len(targets),
                            coordinates=(x + 1, y + 1),  # Center of 3x3 region
                            object_type="button",
                            action_type="click",
                            priority=0.7,
                            confidence=0.8,
                            description=f"Button pattern at ({x+1}, {y+1})",
                            bounding_box=(x, y, 3, 3),
                            color=region[0],
                            area=9
                        )
                        targets.append(target)
            
        except Exception as e:
            logger.debug(f"Button pattern detection failed: {e}")
        
        return targets
    
    def _detect_portal_patterns(self, grid: List[List[int]]) -> List[ActionableTarget]:
        """Detect portal-like patterns in the grid."""
        targets = []
        
        try:
            height, width = len(grid), len(grid[0])
            
            # Look for larger distinctive areas
            for y in range(height - 5):
                for x in range(width - 5):
                    # Check 5x5 region for portal pattern
                    region = []
                    for dy in range(5):
                        for dx in range(5):
                            region.append(grid[y + dy][x + dx])
                    
                    # Check if this is a portal pattern (distinctive color, larger size)
                    unique_colors = set(region)
                    if (len(unique_colors) <= 2 and  # Mostly uniform
                        0 not in unique_colors and  # No background
                        region.count(region[0]) >= 15):  # At least 15/25 pixels same color
                        
                        target = ActionableTarget(
                            id=len(targets),
                            coordinates=(x + 2, y + 2),  # Center of 5x5 region
                            object_type="portal",
                            action_type="move_to",
                            priority=0.6,
                            confidence=0.7,
                            description=f"Portal pattern at ({x+2}, {y+2})",
                            bounding_box=(x, y, 5, 5),
                            color=region[0],
                            area=25
                        )
                        targets.append(target)
            
        except Exception as e:
            logger.debug(f"Portal pattern detection failed: {e}")
        
        return targets
    
    def _detect_interactive_patterns(self, grid: List[List[int]]) -> List[ActionableTarget]:
        """Detect interactive element patterns in the grid."""
        targets = []
        
        try:
            height, width = len(grid), len(grid[0])
            
            # Look for interactive element patterns (medium-sized, distinctive)
            for y in range(height - 4):
                for x in range(width - 4):
                    # Check 4x4 region for interactive pattern
                    region = []
                    for dy in range(4):
                        for dx in range(4):
                            region.append(grid[y + dy][x + dx])
                    
                    # Check if this is an interactive pattern
                    unique_colors = set(region)
                    if (len(unique_colors) <= 3 and  # Limited color variation
                        0 not in unique_colors and  # No background
                        region.count(region[0]) >= 8):  # At least 8/16 pixels same color
                        
                        target = ActionableTarget(
                            id=len(targets),
                            coordinates=(x + 2, y + 2),  # Center of 4x4 region
                            object_type="interactive",
                            action_type="interact",
                            priority=0.5,
                            confidence=0.6,
                            description=f"Interactive pattern at ({x+2}, {y+2})",
                            bounding_box=(x, y, 4, 4),
                            color=region[0],
                            area=16
                        )
                        targets.append(target)
            
        except Exception as e:
            logger.debug(f"Interactive pattern detection failed: {e}")
        
        return targets
    
    def get_actionable_targets_for_action6(self, grid: List[List[int]], game_id: str = "unknown") -> List[Tuple[int, int]]:
        """
        Get coordinates of actionable targets suitable for ACTION6 commands.
        
        This is a convenience method that returns just the coordinates
        of the highest-priority actionable targets.
        
        Args:
            grid: 2D list representing the puzzle grid
            game_id: Identifier for the game
            
        Returns:
            List of (x, y) coordinate tuples for ACTION6 commands
        """
        try:
            targets = self.identify_actionable_targets(grid, game_id)
            
            # Return coordinates of targets with priority > 0.5
            high_priority_targets = [t for t in targets if t.priority > 0.5]
            coordinates = [(t.coordinates[0], t.coordinates[1]) for t in high_priority_targets]
            
            logger.info(f"Found {len(coordinates)} high-priority ACTION6 targets for {game_id}")
            return coordinates
            
        except Exception as e:
            logger.error(f"Failed to get ACTION6 targets for {game_id}: {e}")
            return []

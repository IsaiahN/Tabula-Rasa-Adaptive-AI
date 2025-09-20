"""
Object Detector

Detects and analyzes objects in ARC puzzle grids.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

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

class ObjectDetector:
    """Detects and analyzes objects in ARC puzzle grids."""
    
    def __init__(self, min_contour_area: int = 5, max_contour_area: int = 10000):
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
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
    
    def detect_objects(self, grid: List[List[int]]) -> List[DetectedObject]:
        """Detect objects in the grid using simple contour detection."""
        try:
            # Convert grid to OpenCV format
            cv_image = self._grid_to_opencv(grid)
            
            # Find contours
            contours, _ = cv2.findContours(cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objects = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area < self.min_contour_area or area > self.max_contour_area:
                    continue
                
                # Calculate object properties
                object_data = self._analyze_contour(contour, i, grid)
                if object_data:
                    objects.append(object_data)
            
            logger.debug(f"Detected {len(objects)} objects in grid")
            return objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def _grid_to_opencv(self, grid: List[List[int]]) -> np.ndarray:
        """Convert ARC grid to OpenCV format."""
        if not grid or not grid[0]:
            return np.zeros((1, 1), dtype=np.uint8)
        
        # Convert to numpy array and scale to 0-255
        grid_array = np.array(grid, dtype=np.uint8)
        
        # Scale values to 0-255 range
        if grid_array.max() > 0:
            grid_array = (grid_array * 255 / grid_array.max()).astype(np.uint8)
        
        return grid_array
    
    def _analyze_contour(self, contour: np.ndarray, object_id: int, grid: List[List[int]]) -> Optional[DetectedObject]:
        """Analyze a contour to extract object properties."""
        try:
            # Calculate basic properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bounding_box = (x, y, w, h)
            
            # Calculate centroid
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                centroid = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
            else:
                centroid = (x + w/2, y + h/2)
            
            # Calculate Hu moments for shape description
            hu_moments = cv2.HuMoments(moments).flatten().tolist()
            
            # Determine object type
            object_type = self._classify_object_type(contour, area, perimeter)
            
            # Get dominant color
            color = self._get_dominant_color(contour, grid)
            
            # Determine if hollow
            is_hollow = self._is_hollow(contour, area)
            
            # Determine if actionable
            is_actionable, action_type, priority = self._assess_actionability(contour, object_type, area)
            
            return DetectedObject(
                id=object_id,
                type=object_type,
                points=contour.reshape(-1, 2).tolist(),
                color=color,
                area=int(area),
                centroid=centroid,
                bounding_box=bounding_box,
                hu_moments=hu_moments,
                is_hollow=is_hollow,
                is_actionable=is_actionable,
                action_type=action_type,
                priority=priority
            )
            
        except Exception as e:
            logger.error(f"Error analyzing contour: {e}")
            return None
    
    def _classify_object_type(self, contour: np.ndarray, area: float, perimeter: float) -> str:
        """Classify the type of object based on its properties."""
        try:
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1.0
            
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Classify based on properties
            if circularity > 0.8:
                return 'circle'
            elif 0.9 <= aspect_ratio <= 1.1:
                return 'square'
            elif aspect_ratio > 3 or aspect_ratio < 0.33:
                return 'line'
            else:
                return 'rectangle'
                
        except Exception as e:
            logger.error(f"Error classifying object type: {e}")
            return 'unknown'
    
    def _get_dominant_color(self, contour: np.ndarray, grid: List[List[int]]) -> int:
        """Get the dominant color of the object."""
        try:
            # Create mask for the contour
            mask = np.zeros((len(grid), len(grid[0])), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Get colors within the contour
            colors = []
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if mask[i, j] > 0:
                        colors.append(grid[i][j])
            
            # Return most common color
            if colors:
                return max(set(colors), key=colors.count)
            return 0
            
        except Exception as e:
            logger.error(f"Error getting dominant color: {e}")
            return 0
    
    def _is_hollow(self, contour: np.ndarray, area: float) -> bool:
        """Determine if the object is hollow."""
        try:
            # Simple heuristic: if the contour area is much smaller than bounding box area
            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = w * h
            
            if bbox_area > 0:
                fill_ratio = area / bbox_area
                return fill_ratio < 0.7  # Less than 70% filled
            return False
            
        except Exception as e:
            logger.error(f"Error determining if hollow: {e}")
            return False
    
    def _assess_actionability(self, contour: np.ndarray, object_type: str, area: float) -> Tuple[bool, str, float]:
        """Assess if the object is actionable and determine action type."""
        try:
            # Simple heuristics for actionability
            if object_type in ['circle', 'square'] and area > 20:
                return True, 'click', 0.8
            elif object_type == 'line' and area > 50:
                return True, 'drag', 0.6
            elif object_type == 'rectangle' and area > 100:
                return True, 'select', 0.7
            else:
                return False, 'none', 0.0
                
        except Exception as e:
            logger.error(f"Error assessing actionability: {e}")
            return False, 'none', 0.0

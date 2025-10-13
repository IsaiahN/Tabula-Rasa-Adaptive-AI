"""
Enhanced Pattern Detection System for ARC Games

This module provides advanced pattern detection capabilities including:
- Color sequence detection
- Shape recognition
- Directional pattern detection
- Enhanced grid analysis
- Symmetry detection

The system integrates with the existing GamePatternAnalyzer to provide more
sophisticated pattern recognition capabilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import cv2

# Disable pycache
import sys
sys.dont_write_bytecode = True

@dataclass
class ColorSequence:
    """Represents a detected color sequence pattern."""
    colors: List[int]
    repetitions: int
    confidence: float
    direction: str  # 'horizontal', 'vertical', 'diagonal'

@dataclass
class ShapePattern:
    """Represents a detected shape pattern."""
    shape_type: str  # 'rectangle', 'triangle', 'line', etc.
    size: Tuple[int, int]
    color: int
    position: Tuple[int, int]
    confidence: float

@dataclass
class DirectionalPattern:
    """Represents a detected directional pattern."""
    direction: str
    strength: float
    repetitions: int
    elements: List[Any]

@dataclass
class GridPattern:
    """Represents a detected grid pattern."""
    grid_size: Tuple[int, int]
    cell_contents: List[List[Any]]
    pattern_type: str
    confidence: float

@dataclass
class SymmetryPattern:
    """Represents detected symmetry in the game."""
    symmetry_type: str  # 'horizontal', 'vertical', 'rotational'
    axis_position: Optional[int]
    confidence: float
    elements: List[Any]

class EnhancedPatternDetector:
    """Advanced pattern detection system for ARC games."""

    def __init__(self):
        self.color_threshold = 0.85
        self.shape_threshold = 0.80
        self.direction_threshold = 0.75
        self.symmetry_threshold = 0.90

    def detect_color_sequences(self, grid: np.ndarray) -> List[ColorSequence]:
        """Detect color sequences in the grid."""
        sequences = []
        
        # Check horizontal sequences
        for row in range(grid.shape[0]):
            colors = []
            current_color = None
            count = 0
            
            for col in range(grid.shape[1]):
                color = grid[row, col]
                if color == current_color:
                    count += 1
                else:
                    if count > 1:
                        colors.append((current_color, count))
                    current_color = color
                    count = 1
                    
            if count > 1:
                colors.append((current_color, count))
                
            if len(colors) > 1:
                sequences.append(ColorSequence(
                    colors=[c[0] for c in colors],
                    repetitions=len(colors),
                    confidence=self._calculate_sequence_confidence(colors),
                    direction='horizontal'
                ))
                
        # Similar checks for vertical and diagonal sequences
        # [Implementation similar to horizontal but for different directions]
        
        return sequences

    def detect_shapes(self, grid: np.ndarray) -> List[ShapePattern]:
        """Detect shapes in the grid using contour detection and shape analysis."""
        shapes = []
        
        # Convert grid to binary image for contour detection
        binary = np.where(grid > 0, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Analyze contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            # Determine shape type based on vertices
            shape_type = self._classify_shape(approx, area, perimeter)
            if shape_type:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                # Get color (most common color in the shape region)
                color = self._get_dominant_color(grid[y:y+h, x:x+w])
                
                shapes.append(ShapePattern(
                    shape_type=shape_type,
                    size=(w, h),
                    color=color,
                    position=(x, y),
                    confidence=self._calculate_shape_confidence(approx, area, perimeter)
                ))
                
        return shapes

    def detect_directional_patterns(self, grid: np.ndarray) -> List[DirectionalPattern]:
        """Detect directional patterns in the grid."""
        patterns = []
        
        # Check for horizontal patterns
        horizontal_patterns = self._find_directional_patterns(grid, 'horizontal')
        patterns.extend(horizontal_patterns)
        
        # Check for vertical patterns
        vertical_patterns = self._find_directional_patterns(grid, 'vertical')
        patterns.extend(vertical_patterns)
        
        # Check for diagonal patterns
        diagonal_patterns = self._find_directional_patterns(grid, 'diagonal')
        patterns.extend(diagonal_patterns)
        
        return patterns

    def analyze_grid_structure(self, grid: np.ndarray) -> List[GridPattern]:
        """Analyze grid structure for patterns."""
        patterns = []
        
        # Detect grid subdivisions
        subdivisions = self._detect_grid_subdivisions(grid)
        
        # Analyze cell patterns
        for subdivision in subdivisions:
            pattern_type = self._analyze_cell_pattern(subdivision)
            if pattern_type:
                patterns.append(GridPattern(
                    grid_size=subdivision.shape,
                    cell_contents=subdivision.tolist(),
                    pattern_type=pattern_type,
                    confidence=self._calculate_grid_confidence(subdivision)
                ))
                
        return patterns

    def detect_symmetry(self, grid: np.ndarray) -> List[SymmetryPattern]:
        """Detect symmetrical patterns in the grid."""
        symmetries = []
        
        # Check for horizontal symmetry
        for axis in range(1, grid.shape[0]):
            if self._check_horizontal_symmetry(grid, axis):
                symmetries.append(SymmetryPattern(
                    symmetry_type='horizontal',
                    axis_position=axis,
                    confidence=self._calculate_symmetry_confidence(grid, 'horizontal', axis),
                    elements=self._get_symmetric_elements(grid, 'horizontal', axis)
                ))
                
        # Check for vertical symmetry
        for axis in range(1, grid.shape[1]):
            if self._check_vertical_symmetry(grid, axis):
                symmetries.append(SymmetryPattern(
                    symmetry_type='vertical',
                    axis_position=axis,
                    confidence=self._calculate_symmetry_confidence(grid, 'vertical', axis),
                    elements=self._get_symmetric_elements(grid, 'vertical', axis)
                ))
                
        # Check for rotational symmetry
        if self._check_rotational_symmetry(grid):
            symmetries.append(SymmetryPattern(
                symmetry_type='rotational',
                axis_position=None,
                confidence=self._calculate_symmetry_confidence(grid, 'rotational', None),
                elements=self._get_symmetric_elements(grid, 'rotational', None)
            ))
                
        return symmetries

    def _calculate_sequence_confidence(self, colors: List[Tuple[int, int]]) -> float:
        """Calculate confidence score for a color sequence."""
        if not colors:
            return 0.0
            
        # Consider factors like:
        # - Regularity of repetitions
        # - Clarity of color transitions
        # - Consistency of pattern
        
        repetition_scores = []
        for i in range(len(colors) - 1):
            if colors[i][1] == colors[i+1][1]:  # Same count of repetitions
                repetition_scores.append(1.0)
            else:
                ratio = min(colors[i][1], colors[i+1][1]) / max(colors[i][1], colors[i+1][1])
                repetition_scores.append(ratio)
                
        return sum(repetition_scores) / len(repetition_scores) if repetition_scores else 0.0

    def _classify_shape(self, approx: np.ndarray, area: float, perimeter: float) -> Optional[str]:
        """Classify shape based on its properties."""
        num_vertices = len(approx)
        
        if num_vertices == 3:
            return 'triangle'
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.95 <= aspect_ratio <= 1.05:
                return 'square'
            else:
                return 'rectangle'
        elif num_vertices == 5:
            return 'pentagon'
        elif num_vertices == 6:
            return 'hexagon'
        elif num_vertices > 10:
            # Circularity check
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.8:
                return 'circle'
                
        return None

    def _calculate_shape_confidence(self, approx: np.ndarray, area: float, perimeter: float) -> float:
        """Calculate confidence score for shape detection."""
        # Consider factors like:
        # - Regularity of shape
        # - Clarity of edges
        # - Noise level
        # [Implementation details]
        return 0.9  # Placeholder

    def _find_directional_patterns(self, grid: np.ndarray, direction: str) -> List[DirectionalPattern]:
        """Find patterns in specified direction."""
        # [Implementation details for finding directional patterns]
        return []  # Placeholder

    def _detect_grid_subdivisions(self, grid: np.ndarray) -> List[np.ndarray]:
        """Detect logical subdivisions in the grid."""
        # [Implementation details for grid subdivision detection]
        return []  # Placeholder

    def _analyze_cell_pattern(self, subdivision: np.ndarray) -> Optional[str]:
        """Analyze pattern type in grid cells."""
        # [Implementation details for cell pattern analysis]
        return None  # Placeholder

    def _calculate_grid_confidence(self, subdivision: np.ndarray) -> float:
        """Calculate confidence score for grid pattern."""
        # [Implementation details for grid confidence calculation]
        return 0.9  # Placeholder

    def _check_horizontal_symmetry(self, grid: np.ndarray, axis: int) -> bool:
        """Check for horizontal symmetry around specified axis."""
        # [Implementation details for horizontal symmetry check]
        return False  # Placeholder

    def _check_vertical_symmetry(self, grid: np.ndarray, axis: int) -> bool:
        """Check for vertical symmetry around specified axis."""
        # [Implementation details for vertical symmetry check]
        return False  # Placeholder

    def _check_rotational_symmetry(self, grid: np.ndarray) -> bool:
        """Check for rotational symmetry."""
        # [Implementation details for rotational symmetry check]
        return False  # Placeholder

    def _calculate_symmetry_confidence(self, grid: np.ndarray, symmetry_type: str, axis: Optional[int]) -> float:
        """Calculate confidence score for symmetry detection."""
        # [Implementation details for symmetry confidence calculation]
        return 0.9  # Placeholder

    def _get_symmetric_elements(self, grid: np.ndarray, symmetry_type: str, axis: Optional[int]) -> List[Any]:
        """Get list of symmetric elements."""
        # [Implementation details for getting symmetric elements]
        return []  # Placeholder

    def _get_dominant_color(self, region: np.ndarray) -> int:
        """Get the dominant color in a region."""
        if region.size == 0:
            return 0
        unique, counts = np.unique(region, return_counts=True)
        return unique[np.argmax(counts)]
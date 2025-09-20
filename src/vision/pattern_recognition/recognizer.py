"""
Pattern Recognizer

Detects and recognizes patterns in ARC puzzle grids.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PatternInfo:
    """Represents a detected pattern in the ARC puzzle grid."""
    pattern_type: str  # 'repetition', 'symmetry', 'progression', 'sequence'
    description: str
    confidence: float  # 0.0 to 1.0
    locations: List[Tuple[int, int]]  # Grid coordinates where pattern occurs
    properties: Dict[str, Any]  # Pattern-specific properties
    size: Tuple[int, int]  # Pattern dimensions

class PatternRecognizer:
    """Recognizes patterns in ARC puzzle grids."""
    
    def __init__(self, min_pattern_size: int = 2, max_pattern_size: int = 10):
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.pattern_templates = self._initialize_pattern_templates()
    
    def _initialize_pattern_templates(self) -> Dict[str, np.ndarray]:
        """Initialize pattern templates for recognition."""
        templates = {}
        
        # Repetition patterns
        templates['horizontal_stripe'] = np.array([[1, 0, 1, 0]])
        templates['vertical_stripe'] = np.array([[1], [0], [1], [0]])
        templates['checkerboard'] = np.array([[1, 0], [0, 1]])
        
        # Symmetry patterns
        templates['horizontal_symmetry'] = np.array([[1, 2, 2, 1]])
        templates['vertical_symmetry'] = np.array([[1], [2], [2], [1]])
        
        return templates
    
    def detect_patterns(self, grid: List[List[int]]) -> List[PatternInfo]:
        """Detect all patterns in the grid."""
        try:
            patterns = []
            
            # Convert grid to numpy array
            grid_array = np.array(grid)
            
            # Detect different types of patterns
            patterns.extend(self._detect_repetition_patterns(grid_array))
            patterns.extend(self._detect_symmetry_patterns(grid_array))
            patterns.extend(self._detect_progression_patterns(grid_array))
            patterns.extend(self._detect_sequence_patterns(grid_array))
            
            logger.debug(f"Detected {len(patterns)} patterns in grid")
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return []
    
    def _detect_repetition_patterns(self, grid: np.ndarray) -> List[PatternInfo]:
        """Detect repetition patterns in the grid."""
        patterns = []
        
        try:
            # Check for horizontal repetitions
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1] - self.min_pattern_size + 1):
                    pattern = self._extract_pattern(grid, row, col, 1, self.min_pattern_size)
                    if self._is_repetition_pattern(pattern):
                        patterns.append(PatternInfo(
                            pattern_type='repetition',
                            description=f"Horizontal repetition at ({row}, {col})",
                            confidence=0.8,
                            locations=[(row, col)],
                            properties={'direction': 'horizontal', 'pattern': pattern.tolist()},
                            size=(1, len(pattern))
                        ))
            
            # Check for vertical repetitions
            for row in range(grid.shape[0] - self.min_pattern_size + 1):
                for col in range(grid.shape[1]):
                    pattern = self._extract_pattern(grid, row, col, self.min_pattern_size, 1)
                    if self._is_repetition_pattern(pattern):
                        patterns.append(PatternInfo(
                            pattern_type='repetition',
                            description=f"Vertical repetition at ({row}, {col})",
                            confidence=0.8,
                            locations=[(row, col)],
                            properties={'direction': 'vertical', 'pattern': pattern.tolist()},
                            size=(len(pattern), 1)
                        ))
            
        except Exception as e:
            logger.error(f"Error detecting repetition patterns: {e}")
        
        return patterns
    
    def _detect_symmetry_patterns(self, grid: np.ndarray) -> List[PatternInfo]:
        """Detect symmetry patterns in the grid."""
        patterns = []
        
        try:
            # Check for horizontal symmetry
            if self._has_horizontal_symmetry(grid):
                patterns.append(PatternInfo(
                    pattern_type='symmetry',
                    description="Horizontal symmetry",
                    confidence=0.9,
                    locations=[(0, 0)],
                    properties={'axis': 'horizontal'},
                    size=grid.shape
                ))
            
            # Check for vertical symmetry
            if self._has_vertical_symmetry(grid):
                patterns.append(PatternInfo(
                    pattern_type='symmetry',
                    description="Vertical symmetry",
                    confidence=0.9,
                    locations=[(0, 0)],
                    properties={'axis': 'vertical'},
                    size=grid.shape
                ))
            
        except Exception as e:
            logger.error(f"Error detecting symmetry patterns: {e}")
        
        return patterns
    
    def _detect_progression_patterns(self, grid: np.ndarray) -> List[PatternInfo]:
        """Detect progression patterns in the grid."""
        patterns = []
        
        try:
            # Check for numerical progressions
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1] - 2):
                    sequence = grid[row, col:col+3]
                    if self._is_progression(sequence):
                        patterns.append(PatternInfo(
                            pattern_type='progression',
                            description=f"Numerical progression at ({row}, {col})",
                            confidence=0.7,
                            locations=[(row, col)],
                            properties={'sequence': sequence.tolist(), 'type': 'numerical'},
                            size=(1, 3)
                        ))
            
        except Exception as e:
            logger.error(f"Error detecting progression patterns: {e}")
        
        return patterns
    
    def _detect_sequence_patterns(self, grid: np.ndarray) -> List[PatternInfo]:
        """Detect sequence patterns in the grid."""
        patterns = []
        
        try:
            # Check for repeating sequences
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1] - 3):
                    sequence = grid[row, col:col+4]
                    if self._is_repeating_sequence(sequence):
                        patterns.append(PatternInfo(
                            pattern_type='sequence',
                            description=f"Repeating sequence at ({row}, {col})",
                            confidence=0.6,
                            locations=[(row, col)],
                            properties={'sequence': sequence.tolist()},
                            size=(1, 4)
                        ))
            
        except Exception as e:
            logger.error(f"Error detecting sequence patterns: {e}")
        
        return patterns
    
    def _extract_pattern(self, grid: np.ndarray, row: int, col: int, height: int, width: int) -> np.ndarray:
        """Extract a pattern from the grid at the specified location."""
        try:
            return grid[row:row+height, col:col+width]
        except Exception as e:
            logger.error(f"Error extracting pattern: {e}")
            return np.array([])
    
    def _is_repetition_pattern(self, pattern: np.ndarray) -> bool:
        """Check if a pattern represents a repetition."""
        try:
            if len(pattern) < 2:
                return False
            
            # Check if pattern repeats
            if len(pattern) % 2 == 0:
                half = len(pattern) // 2
                return np.array_equal(pattern[:half], pattern[half:])
            
            return False
        except Exception as e:
            logger.error(f"Error checking repetition pattern: {e}")
            return False
    
    def _has_horizontal_symmetry(self, grid: np.ndarray) -> bool:
        """Check if the grid has horizontal symmetry."""
        try:
            if grid.shape[0] < 2:
                return False
            
            mid = grid.shape[0] // 2
            top = grid[:mid]
            bottom = grid[mid:][::-1]  # Reverse bottom half
            
            return np.array_equal(top, bottom)
        except Exception as e:
            logger.error(f"Error checking horizontal symmetry: {e}")
            return False
    
    def _has_vertical_symmetry(self, grid: np.ndarray) -> bool:
        """Check if the grid has vertical symmetry."""
        try:
            if grid.shape[1] < 2:
                return False
            
            mid = grid.shape[1] // 2
            left = grid[:, :mid]
            right = grid[:, mid:][:, ::-1]  # Reverse right half
            
            return np.array_equal(left, right)
        except Exception as e:
            logger.error(f"Error checking vertical symmetry: {e}")
            return False
    
    def _is_progression(self, sequence: np.ndarray) -> bool:
        """Check if a sequence represents a progression."""
        try:
            if len(sequence) < 3:
                return False
            
            # Check for arithmetic progression
            diffs = np.diff(sequence)
            if len(set(diffs)) == 1 and diffs[0] != 0:
                return True
            
            # Check for geometric progression
            if all(x != 0 for x in sequence[:-1]):
                ratios = sequence[1:] / sequence[:-1]
                if len(set(ratios)) == 1 and ratios[0] != 1:
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking progression: {e}")
            return False
    
    def _is_repeating_sequence(self, sequence: np.ndarray) -> bool:
        """Check if a sequence has repeating elements."""
        try:
            if len(sequence) < 4:
                return False
            
            # Check for 2-element repetition
            if len(sequence) >= 4:
                return np.array_equal(sequence[:2], sequence[2:4])
            
            return False
        except Exception as e:
            logger.error(f"Error checking repeating sequence: {e}")
            return False

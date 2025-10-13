"""
Enhanced pattern detection system for identifying complex patterns in ARC tasks.

This module provides advanced pattern detection capabilities including:
- Color sequence analysis
- Shape pattern recognition  
- Directional pattern detection
- Grid-based pattern analysis
- Symmetry detection
- Pattern abstraction and transfer
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of patterns that can be detected."""
    COLOR_SEQUENCE = "color_sequence"
    SHAPE_PATTERN = "shape_pattern"
    DIRECTIONAL = "directional"
    GRID_BASED = "grid_based"
    SYMMETRICAL = "symmetrical"
    TRANSFORMATION = "transformation"
    COMPOSITE = "composite"

@dataclass
class PatternFeatures:
    """Features extracted from a detected pattern."""
    pattern_type: PatternType
    confidence: float
    elements: List[Any]
    metadata: Dict[str, Any]
    abstracted_form: Optional[str] = None
    transformation_rules: Optional[Dict[str, Any]] = None

class EnhancedPatternDetector:
    """Advanced pattern detection system."""
    
    def __init__(self):
        """Initialize the pattern detector."""
        self.detection_stats = {
            "patterns_found": 0,
            "patterns_by_type": {},
            "confidence_scores": []
        }
        
    def detect_patterns(self, frame: List[List[int]]) -> List[PatternFeatures]:
        """Detect all types of patterns in a frame."""
        patterns = []
        
        # Convert frame to numpy array for easier processing
        frame_array = np.array(frame)
        
        # Run all pattern detection methods
        patterns.extend(self._detect_color_sequences(frame_array))
        patterns.extend(self._detect_shape_patterns(frame_array))
        patterns.extend(self._detect_directional_patterns(frame_array))
        patterns.extend(self._detect_grid_patterns(frame_array))
        patterns.extend(self._detect_symmetry(frame_array))
        
        # Update statistics
        self._update_detection_stats(patterns)
        
        return patterns
        
    def _detect_color_sequences(self, frame: np.ndarray) -> List[PatternFeatures]:
        """Detect color sequences and repeating color patterns."""
        patterns = []
        height, width = frame.shape
        
        # Scan rows and columns for color sequences
        for i in range(height):
            row = frame[i]
            sequences = self._find_repeating_sequences(row)
            for seq in sequences:
                pattern = PatternFeatures(
                    pattern_type=PatternType.COLOR_SEQUENCE,
                    confidence=self._calculate_sequence_confidence(seq),
                    elements=seq["elements"],
                    metadata={
                        "orientation": "horizontal",
                        "position": {"row": i},
                        "length": len(seq["elements"]),
                        "repeating": seq["repeating"]
                    }
                )
                patterns.append(pattern)
                
        for j in range(width):
            col = frame[:, j]
            sequences = self._find_repeating_sequences(col)
            for seq in sequences:
                pattern = PatternFeatures(
                    pattern_type=PatternType.COLOR_SEQUENCE,
                    confidence=self._calculate_sequence_confidence(seq),
                    elements=seq["elements"],
                    metadata={
                        "orientation": "vertical",
                        "position": {"col": j},
                        "length": len(seq["elements"]),
                        "repeating": seq["repeating"]
                    }
                )
                patterns.append(pattern)
                
        return patterns
        
    def _detect_shape_patterns(self, frame: np.ndarray) -> List[PatternFeatures]:
        """Detect recurring shapes and shape transformations."""
        patterns = []
        height, width = frame.shape
        
        # Scan for common shapes using sliding windows
        window_sizes = [(2,2), (3,3), (4,4)]
        for w_height, w_width in window_sizes:
            for i in range(height - w_height + 1):
                for j in range(width - w_width + 1):
                    window = frame[i:i+w_height, j:j+w_width]
                    shape_features = self._analyze_shape(window)
                    if shape_features:
                        pattern = PatternFeatures(
                            pattern_type=PatternType.SHAPE_PATTERN,
                            confidence=shape_features["confidence"],
                            elements=shape_features["elements"],
                            metadata={
                                "position": {"row": i, "col": j},
                                "size": (w_height, w_width),
                                "shape_type": shape_features["type"]
                            }
                        )
                        patterns.append(pattern)
                        
        return patterns
        
    def _detect_directional_patterns(self, frame: np.ndarray) -> List[PatternFeatures]:
        """Detect patterns that follow directional rules."""
        patterns = []
        
        # Check for gradients and directional changes
        gradients = self._calculate_gradients(frame)
        
        for direction, gradient in gradients.items():
            if self._is_significant_gradient(gradient):
                pattern = PatternFeatures(
                    pattern_type=PatternType.DIRECTIONAL,
                    confidence=self._calculate_gradient_confidence(gradient),
                    elements=gradient["elements"],
                    metadata={
                        "direction": direction,
                        "strength": gradient["strength"],
                        "consistency": gradient["consistency"]
                    }
                )
                patterns.append(pattern)
                
        return patterns
        
    def _detect_grid_patterns(self, frame: np.ndarray) -> List[PatternFeatures]:
        """Detect patterns in grid layouts."""
        patterns = []
        height, width = frame.shape
        
        # Try different grid sizes
        grid_sizes = [(2,2), (3,3), (4,4), (5,5)]
        for rows, cols in grid_sizes:
            if height % rows == 0 and width % cols == 0:
                cell_height = height // rows
                cell_width = width // cols
                
                grid_pattern = self._analyze_grid(frame, rows, cols, cell_height, cell_width)
                if grid_pattern:
                    pattern = PatternFeatures(
                        pattern_type=PatternType.GRID_BASED,
                        confidence=grid_pattern["confidence"],
                        elements=grid_pattern["elements"],
                        metadata={
                            "grid_size": (rows, cols),
                            "cell_size": (cell_height, cell_width),
                            "regularity": grid_pattern["regularity"],
                            "pattern_type": grid_pattern["type"]
                        }
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _detect_symmetry(self, frame: np.ndarray) -> List[PatternFeatures]:
        """Detect various types of symmetry."""
        patterns = []
        height, width = frame.shape
        
        # Check horizontal symmetry
        h_sym = self._check_horizontal_symmetry(frame)
        if h_sym["is_symmetric"]:
            pattern = PatternFeatures(
                pattern_type=PatternType.SYMMETRICAL,
                confidence=h_sym["confidence"],
                elements=h_sym["elements"],
                metadata={
                    "symmetry_type": "horizontal",
                    "axis": h_sym["axis"],
                    "quality": h_sym["quality"]
                }
            )
            patterns.append(pattern)
            
        # Check vertical symmetry
        v_sym = self._check_vertical_symmetry(frame)
        if v_sym["is_symmetric"]:
            pattern = PatternFeatures(
                pattern_type=PatternType.SYMMETRICAL,
                confidence=v_sym["confidence"],
                elements=v_sym["elements"],
                metadata={
                    "symmetry_type": "vertical",
                    "axis": v_sym["axis"],
                    "quality": v_sym["quality"]
                }
            )
            patterns.append(pattern)
            
        return patterns
        
    def _find_repeating_sequences(self, arr: np.ndarray) -> List[Dict[str, Any]]:
        """Find repeating sequences in an array."""
        sequences = []
        n = len(arr)
        
        # Try different sequence lengths
        for length in range(2, n//2 + 1):
            for start in range(n - length + 1):
                sequence = arr[start:start+length]
                # Look for repetitions
                repetitions = self._find_sequence_repetitions(arr, sequence)
                if repetitions > 1:
                    sequences.append({
                        "elements": sequence.tolist(),
                        "start": start,
                        "length": length,
                        "repetitions": repetitions,
                        "repeating": True
                    })
                    
        return sequences
        
    def _find_sequence_repetitions(self, arr: np.ndarray, sequence: np.ndarray) -> int:
        """Count how many times a sequence repeats in array."""
        n, seq_len = len(arr), len(sequence)
        count = 0
        
        for i in range(0, n - seq_len + 1, seq_len):
            if np.array_equal(arr[i:i+seq_len], sequence):
                count += 1
            else:
                break
                
        return count
        
    def _calculate_sequence_confidence(self, sequence: Dict[str, Any]) -> float:
        """Calculate confidence score for a sequence pattern."""
        # Factors that increase confidence:
        # - More repetitions
        # - Longer sequence length
        # - Clean repetitions without noise
        base_confidence = 0.5
        repetition_factor = min(sequence["repetitions"] / 2, 1.0)
        length_factor = min(sequence["length"] / 5, 1.0)
        
        confidence = base_confidence + (repetition_factor * 0.3) + (length_factor * 0.2)
        return min(confidence, 1.0)
        
    def _analyze_shape(self, window: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze a window for shape patterns."""
        unique_values = np.unique(window)
        if len(unique_values) < 2:
            return None
            
        # Analyze shape characteristics
        area = np.count_nonzero(window)
        perimeter = self._calculate_perimeter(window)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        shape_type = self._classify_shape(area, perimeter, compactness)
        if not shape_type:
            return None
            
        return {
            "type": shape_type,
            "elements": window.tolist(),
            "confidence": self._calculate_shape_confidence(area, compactness),
            "area": area,
            "perimeter": perimeter,
            "compactness": compactness
        }
        
    def _calculate_perimeter(self, window: np.ndarray) -> int:
        """Calculate perimeter of shape in window."""
        edges_h = np.diff(window, axis=1)
        edges_v = np.diff(window, axis=0)
        return np.count_nonzero(edges_h) + np.count_nonzero(edges_v)
        
    def _classify_shape(self, area: int, perimeter: int, compactness: float) -> Optional[str]:
        """Classify shape based on its properties."""
        if compactness > 0.8:
            return "circle"
        elif 0.4 <= compactness <= 0.6:
            return "rectangle"
        elif compactness < 0.4:
            return "irregular"
        return None
        
    def _calculate_shape_confidence(self, area: int, compactness: float) -> float:
        """Calculate confidence score for shape classification."""
        base_confidence = 0.5
        area_factor = min(area / 9, 1.0) * 0.3  # Larger shapes are more reliable
        compactness_factor = compactness * 0.2   # Clear geometric shapes boost confidence
        
        return base_confidence + area_factor + compactness_factor
        
    def _calculate_gradients(self, frame: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Calculate gradients in different directions."""
        height, width = frame.shape
        gradients = {}
        
        # Horizontal gradient
        h_grad = np.diff(frame, axis=1)
        gradients["horizontal"] = {
            "elements": h_grad.tolist(),
            "strength": np.mean(np.abs(h_grad)),
            "consistency": np.std(h_grad),
            "direction": "left_to_right" if np.mean(h_grad) > 0 else "right_to_left"
        }
        
        # Vertical gradient
        v_grad = np.diff(frame, axis=0)
        gradients["vertical"] = {
            "elements": v_grad.tolist(),
            "strength": np.mean(np.abs(v_grad)),
            "consistency": np.std(v_grad),
            "direction": "top_to_bottom" if np.mean(v_grad) > 0 else "bottom_to_top"
        }
        
        return gradients
        
    def _is_significant_gradient(self, gradient: Dict[str, Any]) -> bool:
        """Check if gradient is significant enough to be a pattern."""
        return gradient["strength"] > 0.1 and gradient["consistency"] < gradient["strength"] * 2
        
    def _calculate_gradient_confidence(self, gradient: Dict[str, Any]) -> float:
        """Calculate confidence score for gradient pattern."""
        strength_factor = min(gradient["strength"], 1.0) * 0.5
        consistency_factor = (1.0 - min(gradient["consistency"], 1.0)) * 0.5
        return strength_factor + consistency_factor
        
    def _analyze_grid(self, frame: np.ndarray, rows: int, cols: int,
                     cell_height: int, cell_width: int) -> Optional[Dict[str, Any]]:
        """Analyze grid structure for patterns."""
        cells = []
        cell_values = []
        
        # Extract all cells
        for i in range(rows):
            for j in range(cols):
                cell = frame[i*cell_height:(i+1)*cell_height, 
                           j*cell_width:(j+1)*cell_width]
                cells.append(cell)
                cell_values.append(np.mean(cell))
                
        # Check for grid patterns
        regularity = np.std(cell_values)
        if regularity > 0.5:  # Too irregular
            return None
            
        pattern_type = self._identify_grid_pattern_type(cell_values, rows, cols)
        if not pattern_type:
            return None
            
        return {
            "elements": cells,
            "confidence": self._calculate_grid_confidence(regularity),
            "regularity": 1.0 - regularity,
            "type": pattern_type
        }
        
    def _identify_grid_pattern_type(self, cell_values: List[float],
                                  rows: int, cols: int) -> Optional[str]:
        """Identify type of pattern in grid."""
        values = np.array(cell_values).reshape(rows, cols)
        
        # Check alternating pattern
        if self._is_alternating_pattern(values):
            return "alternating"
            
        # Check checkerboard pattern
        if self._is_checkerboard_pattern(values):
            return "checkerboard"
            
        # Check gradient pattern
        if self._is_gradient_pattern(values):
            return "gradient"
            
        return None
        
    def _is_alternating_pattern(self, values: np.ndarray) -> bool:
        """Check if values form alternating pattern."""
        rows, cols = values.shape
        
        # Check rows
        for i in range(rows):
            row = values[i]
            if len(set(row[::2])) > 1 or len(set(row[1::2])) > 1:
                return False
                
        # Check columns
        for j in range(cols):
            col = values[:, j]
            if len(set(col[::2])) > 1 or len(set(col[1::2])) > 1:
                return False
                
        return True
        
    def _is_checkerboard_pattern(self, values: np.ndarray) -> bool:
        """Check if values form checkerboard pattern."""
        rows, cols = values.shape
        
        # Get values at even/odd coordinates
        even_coords = values[::2, ::2]
        odd_coords = values[1::2, 1::2]
        
        # Check if they're consistent
        return (len(set(even_coords.flat)) == 1 and
                len(set(odd_coords.flat)) == 1 and
                set(even_coords.flat) != set(odd_coords.flat))
        
    def _is_gradient_pattern(self, values: np.ndarray) -> bool:
        """Check if values form gradient pattern."""
        # Check horizontal gradient
        h_diff = np.diff(values, axis=1)
        h_consistent = np.allclose(h_diff, h_diff[0,0], rtol=0.1)
        
        # Check vertical gradient
        v_diff = np.diff(values, axis=0)
        v_consistent = np.allclose(v_diff, v_diff[0,0], rtol=0.1)
        
        return h_consistent or v_consistent
        
    def _calculate_grid_confidence(self, regularity: float) -> float:
        """Calculate confidence score for grid pattern."""
        return 0.6 + (0.4 * (1.0 - regularity))
        
    def _check_horizontal_symmetry(self, frame: np.ndarray) -> Dict[str, Any]:
        """Check for horizontal symmetry."""
        height = frame.shape[0]
        best_axis = -1
        best_quality = 0
        
        # Try different potential symmetry axes
        for axis in range(1, height):
            upper = frame[:axis]
            lower = frame[axis:]
            min_height = min(len(upper), len(lower))
            
            # Compare mirrored sections
            upper_section = upper[-min_height:]
            lower_section = np.flipud(lower[:min_height])
            
            quality = np.mean(upper_section == lower_section)
            if quality > best_quality:
                best_quality = quality
                best_axis = axis
                
        is_symmetric = best_quality > 0.8
        return {
            "is_symmetric": is_symmetric,
            "axis": best_axis if is_symmetric else None,
            "quality": best_quality,
            "confidence": best_quality if is_symmetric else 0,
            "elements": frame.tolist()
        }
        
    def _check_vertical_symmetry(self, frame: np.ndarray) -> Dict[str, Any]:
        """Check for vertical symmetry."""
        width = frame.shape[1]
        best_axis = -1
        best_quality = 0
        
        # Try different potential symmetry axes
        for axis in range(1, width):
            left = frame[:, :axis]
            right = frame[:, axis:]
            min_width = min(left.shape[1], right.shape[1])
            
            # Compare mirrored sections
            left_section = left[:, -min_width:]
            right_section = np.fliplr(right[:, :min_width])
            
            quality = np.mean(left_section == right_section)
            if quality > best_quality:
                best_quality = quality
                best_axis = axis
                
        is_symmetric = best_quality > 0.8
        return {
            "is_symmetric": is_symmetric,
            "axis": best_axis if is_symmetric else None,
            "quality": best_quality,
            "confidence": best_quality if is_symmetric else 0,
            "elements": frame.tolist()
        }
        
    def _update_detection_stats(self, patterns: List[PatternFeatures]) -> None:
        """Update pattern detection statistics."""
        self.detection_stats["patterns_found"] += len(patterns)
        
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            if pattern_type not in self.detection_stats["patterns_by_type"]:
                self.detection_stats["patterns_by_type"][pattern_type] = 0
            self.detection_stats["patterns_by_type"][pattern_type] += 1
            
            self.detection_stats["confidence_scores"].append(pattern.confidence)
            
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get current pattern detection statistics."""
        stats = self.detection_stats.copy()
        if self.detection_stats["confidence_scores"]:
            stats["average_confidence"] = np.mean(self.detection_stats["confidence_scores"])
        else:
            stats["average_confidence"] = 0.0
        return stats
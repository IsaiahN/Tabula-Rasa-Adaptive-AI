"""
Game Pattern Analyzer for Visual Pattern Detection and Game Mechanics Analysis

This module analyzes visual patterns in ARC games to detect game mechanics and support
hypothesis generation. Works alongside the existing GameTypeClassifier to provide
deeper visual analysis beyond game ID classification.

Key Features:
- Visual pattern detection (grids, shapes, colors)
- Game mechanics classification (pattern completion, physics, spatial puzzles)
- Grid analysis and feature extraction
- Support for hypothesis generation system
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json
import cv2
from datetime import datetime

# Disable pycache
import sys
sys.dont_write_bytecode = True

logger = logging.getLogger(__name__)

class GameMechanic(Enum):
    """Enumeration of different game mechanics that can be detected."""
    PATTERN_COMPLETION = "pattern_completion"
    PHYSICS_SIMULATION = "physics_simulation"
    SPATIAL_PUZZLE = "spatial_puzzle"
    OBJECT_MANIPULATION = "object_manipulation"
    SORTING_ORGANIZING = "sorting_organizing"
    NAVIGATION_PATHFINDING = "navigation_pathfinding"
    COLOR_MATCHING = "color_matching"
    SHAPE_TRANSFORMATION = "shape_transformation"
    SEQUENCE_PREDICTION = "sequence_prediction"
    SYMMETRY_PATTERN = "symmetry_pattern"
    UNKNOWN = "unknown"

@dataclass
class VisualPattern:
    """Represents a detected visual pattern in a game."""
    pattern_id: str
    pattern_type: str
    confidence: float
    location: Tuple[int, int]
    size: Tuple[int, int]
    features: Dict[str, Any]
    timestamp: datetime

@dataclass
class GameMechanicsProfile:
    """Profile of game mechanics detected in a game."""
    primary_mechanic: GameMechanic
    secondary_mechanics: List[GameMechanic]
    mechanic_confidence: Dict[GameMechanic, float]
    visual_patterns: List[VisualPattern]
    grid_features: Dict[str, Any]
    complexity_score: float
    timestamp: datetime

class GamePatternAnalyzer:
    """
    Analyzes visual patterns and game mechanics in ARC games.

    Provides detailed visual pattern analysis to support hypothesis generation
    and complement the existing GameTypeClassifier.
    """

    def __init__(self):
        self.pattern_cache: Dict[str, GameMechanicsProfile] = {}
        self.mechanic_patterns = self._initialize_mechanic_patterns()
        logger.info("Game Pattern Analyzer initialized")

    def _initialize_mechanic_patterns(self) -> Dict[GameMechanic, Dict[str, Any]]:
        """Initialize patterns for detecting different game mechanics."""
        return {
            GameMechanic.PATTERN_COMPLETION: {
                "indicators": ["incomplete_grid", "missing_elements", "regular_spacing"],
                "grid_features": ["partial_symmetry", "repetitive_elements"],
                "color_patterns": ["consistent_palette", "predictable_sequence"]
            },
            GameMechanic.PHYSICS_SIMULATION: {
                "indicators": ["gravity_effects", "collision_detection", "momentum"],
                "grid_features": ["falling_objects", "stacked_elements"],
                "color_patterns": ["object_trails", "impact_indicators"]
            },
            GameMechanic.SPATIAL_PUZZLE: {
                "indicators": ["geometric_shapes", "spatial_relationships", "fit_together"],
                "grid_features": ["complex_shapes", "interlocking_pieces"],
                "color_patterns": ["shape_coding", "boundary_definition"]
            },
            GameMechanic.OBJECT_MANIPULATION: {
                "indicators": ["moveable_objects", "transformation", "repositioning"],
                "grid_features": ["discrete_objects", "clear_boundaries"],
                "color_patterns": ["object_highlighting", "state_changes"]
            },
            GameMechanic.SORTING_ORGANIZING: {
                "indicators": ["categorization", "grouping", "arrangement"],
                "grid_features": ["clustered_elements", "ordered_sequences"],
                "color_patterns": ["category_coding", "gradient_organization"]
            },
            GameMechanic.NAVIGATION_PATHFINDING: {
                "indicators": ["path_creation", "obstacle_avoidance", "route_optimization"],
                "grid_features": ["maze_like", "connected_paths"],
                "color_patterns": ["path_marking", "obstacle_coding"]
            }
        }

    def analyze_game_screenshot(self, screenshot_array: np.ndarray, game_id: str = None) -> GameMechanicsProfile:
        """
        Analyze a game screenshot to detect visual patterns and game mechanics.

        Args:
            screenshot_array: Numpy array of the game screenshot
            game_id: Optional game identifier for caching

        Returns:
            GameMechanicsProfile with detected mechanics and patterns
        """
        if game_id and game_id in self.pattern_cache:
            logger.debug(f"Using cached analysis for game {game_id}")
            return self.pattern_cache[game_id]

        # Extract grid features
        grid_features = self._extract_grid_features(screenshot_array)

        # Detect visual patterns
        visual_patterns = self._detect_visual_patterns(screenshot_array)

        # Classify game mechanics
        mechanic_scores = self._classify_game_mechanics(grid_features, visual_patterns)

        # Determine primary and secondary mechanics
        primary_mechanic = max(mechanic_scores.items(), key=lambda x: x[1])[0]
        secondary_mechanics = [
            mechanic for mechanic, score in mechanic_scores.items()
            if score > 0.3 and mechanic != primary_mechanic
        ]

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(grid_features, visual_patterns)

        profile = GameMechanicsProfile(
            primary_mechanic=primary_mechanic,
            secondary_mechanics=secondary_mechanics,
            mechanic_confidence=mechanic_scores,
            visual_patterns=visual_patterns,
            grid_features=grid_features,
            complexity_score=complexity_score,
            timestamp=datetime.now()
        )

        if game_id:
            self.pattern_cache[game_id] = profile

        logger.info(f"Analyzed game mechanics: Primary={primary_mechanic.value}, "
                   f"Secondary={[m.value for m in secondary_mechanics]}, "
                   f"Complexity={complexity_score:.2f}")

        return profile

    def _extract_grid_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract features from the game grid."""
        features = {}

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Grid structure analysis
        features["grid_detected"] = self._detect_grid_structure(gray)
        features["grid_size"] = self._estimate_grid_size(gray)
        features["cell_uniformity"] = self._analyze_cell_uniformity(gray)

        # Color analysis
        if len(image.shape) == 3:
            features["color_count"] = self._count_unique_colors(image)
            features["color_distribution"] = self._analyze_color_distribution(image)

        # Shape analysis
        features["shape_count"] = self._count_shapes(gray)
        features["shape_complexity"] = self._analyze_shape_complexity(gray)

        # Pattern analysis
        features["symmetry_detected"] = self._detect_symmetry(gray)
        features["repetition_detected"] = self._detect_repetition(gray)
        features["gradient_detected"] = self._detect_gradients(image if len(image.shape) == 3 else gray)

        return features

    def _detect_visual_patterns(self, image: np.ndarray) -> List[VisualPattern]:
        """Detect specific visual patterns in the image."""
        patterns = []

        # Convert to grayscale for pattern detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Detect geometric patterns
        geometric_patterns = self._detect_geometric_patterns(gray)
        patterns.extend(geometric_patterns)

        # Detect color patterns
        if len(image.shape) == 3:
            color_patterns = self._detect_color_patterns(image)
            patterns.extend(color_patterns)

        # Detect sequence patterns
        sequence_patterns = self._detect_sequence_patterns(gray)
        patterns.extend(sequence_patterns)

        return patterns

    def _classify_game_mechanics(self, grid_features: Dict[str, Any],
                               visual_patterns: List[VisualPattern]) -> Dict[GameMechanic, float]:
        """Classify the type of game mechanics based on features and patterns."""
        scores = {mechanic: 0.0 for mechanic in GameMechanic}

        # Analyze each mechanic type
        for mechanic, pattern_def in self.mechanic_patterns.items():
            score = 0.0

            # Check grid feature indicators
            for indicator in pattern_def["grid_features"]:
                if self._check_grid_indicator(grid_features, indicator):
                    score += 0.2

            # Check visual pattern indicators
            for pattern in visual_patterns:
                if self._pattern_matches_mechanic(pattern, mechanic):
                    score += 0.3

            # Apply mechanic-specific scoring logic
            score += self._apply_mechanic_specific_scoring(mechanic, grid_features, visual_patterns)

            scores[mechanic] = min(score, 1.0)  # Cap at 1.0

        return scores

    def _detect_grid_structure(self, gray: np.ndarray) -> bool:
        """Detect if the image contains a grid structure."""
        # Use Hough lines to detect grid lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

        if lines is None:
            return False

        # Check for both horizontal and vertical lines
        horizontal_lines = 0
        vertical_lines = 0

        for line in lines[:20]:  # Check first 20 lines
            rho, theta = line[0]
            if abs(theta) < np.pi/4 or abs(theta - np.pi) < np.pi/4:
                horizontal_lines += 1
            elif abs(theta - np.pi/2) < np.pi/4:
                vertical_lines += 1

        return horizontal_lines >= 2 and vertical_lines >= 2

    def _estimate_grid_size(self, gray: np.ndarray) -> Tuple[int, int]:
        """Estimate the size of the grid in the image."""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

        if lines is None:
            return (0, 0)

        # Find grid spacing
        horizontal_positions = []
        vertical_positions = []

        for line in lines:
            rho, theta = line[0]
            if abs(theta) < np.pi/4 or abs(theta - np.pi) < np.pi/4:
                horizontal_positions.append(rho)
            elif abs(theta - np.pi/2) < np.pi/4:
                vertical_positions.append(rho)

        # Estimate grid size based on line spacing
        if len(horizontal_positions) > 1 and len(vertical_positions) > 1:
            h_spacing = np.median(np.diff(sorted(horizontal_positions)))
            v_spacing = np.median(np.diff(sorted(vertical_positions)))

            if h_spacing > 0 and v_spacing > 0:
                grid_h = int(gray.shape[0] / h_spacing)
                grid_w = int(gray.shape[1] / v_spacing)
                return (grid_h, grid_w)

        return (0, 0)

    def _analyze_cell_uniformity(self, gray: np.ndarray) -> float:
        """Analyze uniformity of grid cells."""
        # Simple approach: check variance in cell regions
        if gray.shape[0] < 10 or gray.shape[1] < 10:
            return 0.0

        # Divide into approximate grid cells
        cell_h = gray.shape[0] // 10
        cell_w = gray.shape[1] // 10

        cell_variances = []
        for i in range(0, gray.shape[0] - cell_h, cell_h):
            for j in range(0, gray.shape[1] - cell_w, cell_w):
                cell = gray[i:i+cell_h, j:j+cell_w]
                cell_variances.append(np.var(cell))

        if len(cell_variances) == 0:
            return 0.0

        # Uniformity is inverse of variance spread
        variance_spread = np.std(cell_variances)
        return max(0.0, 1.0 - variance_spread / 255.0)

    def _count_unique_colors(self, image: np.ndarray) -> int:
        """Count unique colors in the image."""
        if len(image.shape) == 3:
            # Reshape to list of pixels
            pixels = image.reshape(-1, image.shape[-1])
            unique_colors = np.unique(pixels, axis=0)
            return len(unique_colors)
        else:
            return len(np.unique(image))

    def _analyze_color_distribution(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze distribution of colors in the image."""
        distribution = {}

        if len(image.shape) == 3:
            # Calculate color histogram
            hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

            distribution["red_entropy"] = self._calculate_entropy(hist_r)
            distribution["green_entropy"] = self._calculate_entropy(hist_g)
            distribution["blue_entropy"] = self._calculate_entropy(hist_b)
        else:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            distribution["gray_entropy"] = self._calculate_entropy(hist)

        return distribution

    def _calculate_entropy(self, histogram: np.ndarray) -> float:
        """Calculate entropy of a histogram."""
        histogram = histogram.flatten()
        histogram = histogram[histogram > 0]  # Remove zeros
        if len(histogram) <= 1:
            return 0.0

        prob = histogram / histogram.sum()
        entropy = -np.sum(prob * np.log2(prob))
        return entropy

    def _count_shapes(self, gray: np.ndarray) -> int:
        """Count distinct shapes in the image."""
        # Use contour detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out very small contours
        significant_contours = [c for c in contours if cv2.contourArea(c) > 50]

        return len(significant_contours)

    def _analyze_shape_complexity(self, gray: np.ndarray) -> float:
        """Analyze complexity of shapes in the image."""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return 0.0

        complexities = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                # Calculate complexity as ratio of contour length to bounding rect perimeter
                perimeter = cv2.arcLength(contour, True)
                x, y, w, h = cv2.boundingRect(contour)
                rect_perimeter = 2 * (w + h)

                if rect_perimeter > 0:
                    complexity = perimeter / rect_perimeter
                    complexities.append(complexity)

        return np.mean(complexities) if complexities else 0.0

    def _detect_symmetry(self, gray: np.ndarray) -> Dict[str, bool]:
        """Detect various types of symmetry in the image."""
        symmetry = {}

        # Horizontal symmetry
        flipped_h = cv2.flip(gray, 0)
        h_diff = np.mean(np.abs(gray.astype(float) - flipped_h.astype(float)))
        symmetry["horizontal"] = h_diff < 50

        # Vertical symmetry
        flipped_v = cv2.flip(gray, 1)
        v_diff = np.mean(np.abs(gray.astype(float) - flipped_v.astype(float)))
        symmetry["vertical"] = v_diff < 50

        # Diagonal symmetry (simple approximation)
        if gray.shape[0] == gray.shape[1]:
            transposed = gray.T
            d_diff = np.mean(np.abs(gray.astype(float) - transposed.astype(float)))
            symmetry["diagonal"] = d_diff < 50
        else:
            symmetry["diagonal"] = False

        return symmetry

    def _detect_repetition(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect repetitive patterns in the image."""
        repetition = {}

        # Template matching for repetitive patterns
        h, w = gray.shape
        template_size = min(h//4, w//4, 50)  # Use small template

        if template_size > 10:
            template = gray[:template_size, :template_size]
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

            # Find strong matches (excluding the template location itself)
            threshold = 0.8
            locations = np.where(result >= threshold)
            matches = len(locations[0])

            repetition["pattern_matches"] = matches
            repetition["repetition_detected"] = matches > 1
        else:
            repetition["pattern_matches"] = 0
            repetition["repetition_detected"] = False

        return repetition

    def _detect_gradients(self, image: np.ndarray) -> Dict[str, bool]:
        """Detect color gradients in the image."""
        gradients = {}

        if len(image.shape) == 3:
            # Convert to grayscale for gradient detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Analyze gradient strength
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = np.mean(grad_magnitude)

        gradients["strong_gradients"] = mean_gradient > 50
        gradients["gradient_strength"] = mean_gradient

        return gradients

    def _detect_geometric_patterns(self, gray: np.ndarray) -> List[VisualPattern]:
        """Detect geometric patterns in the image."""
        patterns = []

        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=5, maxRadius=100)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                pattern = VisualPattern(
                    pattern_id=f"circle_{x}_{y}_{r}",
                    pattern_type="circle",
                    confidence=0.8,
                    location=(x, y),
                    size=(r*2, r*2),
                    features={"radius": r},
                    timestamp=datetime.now()
                )
                patterns.append(pattern)

        # Detect rectangles/squares
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 100:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:  # Rectangle/square
                    x, y, w, h = cv2.boundingRect(contour)
                    pattern = VisualPattern(
                        pattern_id=f"rectangle_{i}_{x}_{y}",
                        pattern_type="rectangle",
                        confidence=0.7,
                        location=(x, y),
                        size=(w, h),
                        features={"width": w, "height": h, "is_square": abs(w-h) < 5},
                        timestamp=datetime.now()
                    )
                    patterns.append(pattern)

        return patterns

    def _detect_color_patterns(self, image: np.ndarray) -> List[VisualPattern]:
        """Detect color-based patterns in the image."""
        patterns = []

        # Simple color region detection
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Define color ranges (simplified)
        color_ranges = {
            "red": ([0, 50, 50], [10, 255, 255]),
            "blue": ([100, 50, 50], [130, 255, 255]),
            "green": ([40, 50, 50], [80, 255, 255]),
            "yellow": ([20, 50, 50], [40, 255, 255])
        }

        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    pattern = VisualPattern(
                        pattern_id=f"color_{color_name}_{i}_{x}_{y}",
                        pattern_type="color_region",
                        confidence=0.6,
                        location=(x, y),
                        size=(w, h),
                        features={"color": color_name, "area": cv2.contourArea(contour)},
                        timestamp=datetime.now()
                    )
                    patterns.append(pattern)

        return patterns

    def _detect_sequence_patterns(self, gray: np.ndarray) -> List[VisualPattern]:
        """Detect sequence-based patterns in the image."""
        patterns = []

        # Analyze rows and columns for sequences
        h, w = gray.shape

        # Row analysis
        for i in range(0, h, max(1, h//10)):  # Sample every 10th of height
            row = gray[i, :]
            if self._is_sequence_pattern(row):
                pattern = VisualPattern(
                    pattern_id=f"row_sequence_{i}",
                    pattern_type="row_sequence",
                    confidence=0.5,
                    location=(0, i),
                    size=(w, 1),
                    features={"sequence_type": "row", "position": i},
                    timestamp=datetime.now()
                )
                patterns.append(pattern)

        # Column analysis
        for j in range(0, w, max(1, w//10)):  # Sample every 10th of width
            col = gray[:, j]
            if self._is_sequence_pattern(col):
                pattern = VisualPattern(
                    pattern_id=f"col_sequence_{j}",
                    pattern_type="col_sequence",
                    confidence=0.5,
                    location=(j, 0),
                    size=(1, h),
                    features={"sequence_type": "column", "position": j},
                    timestamp=datetime.now()
                )
                patterns.append(pattern)

        return patterns

    def _is_sequence_pattern(self, array: np.ndarray) -> bool:
        """Check if an array contains a sequence pattern."""
        if len(array) < 4:
            return False

        # Check for arithmetic progression
        diffs = np.diff(array)
        if len(set(diffs)) <= 2:  # Allow for some noise
            return True

        # Check for repeating pattern
        for period in range(2, len(array)//2):
            if np.array_equal(array[:period], array[period:2*period]):
                return True

        return False

    def _check_grid_indicator(self, grid_features: Dict[str, Any], indicator: str) -> bool:
        """Check if a grid feature indicator is present."""
        indicator_map = {
            "partial_symmetry": lambda f: any(f.get("symmetry_detected", {}).values()),
            "repetitive_elements": lambda f: f.get("repetition_detected", {}).get("repetition_detected", False),
            "falling_objects": lambda f: f.get("gradient_detected", {}).get("strong_gradients", False),
            "stacked_elements": lambda f: f.get("shape_count", 0) > 3,
            "complex_shapes": lambda f: f.get("shape_complexity", 0) > 1.5,
            "interlocking_pieces": lambda f: f.get("shape_complexity", 0) > 2.0,
            "discrete_objects": lambda f: f.get("shape_count", 0) > 1,
            "clear_boundaries": lambda f: f.get("gradient_detected", {}).get("strong_gradients", False),
            "clustered_elements": lambda f: f.get("shape_count", 0) > 2,
            "ordered_sequences": lambda f: f.get("repetition_detected", {}).get("repetition_detected", False),
            "maze_like": lambda f: f.get("grid_detected", False) and f.get("shape_complexity", 0) > 1.0,
            "connected_paths": lambda f: f.get("grid_detected", False)
        }

        check_func = indicator_map.get(indicator)
        return check_func(grid_features) if check_func else False

    def _pattern_matches_mechanic(self, pattern: VisualPattern, mechanic: GameMechanic) -> bool:
        """Check if a visual pattern matches a specific mechanic."""
        mechanic_patterns = {
            GameMechanic.PATTERN_COMPLETION: ["rectangle", "row_sequence", "col_sequence"],
            GameMechanic.PHYSICS_SIMULATION: ["circle", "color_region"],
            GameMechanic.SPATIAL_PUZZLE: ["rectangle", "circle"],
            GameMechanic.OBJECT_MANIPULATION: ["color_region", "rectangle"],
            GameMechanic.SORTING_ORGANIZING: ["color_region", "row_sequence", "col_sequence"],
            GameMechanic.NAVIGATION_PATHFINDING: ["rectangle", "color_region"]
        }

        return pattern.pattern_type in mechanic_patterns.get(mechanic, [])

    def _apply_mechanic_specific_scoring(self, mechanic: GameMechanic,
                                       grid_features: Dict[str, Any],
                                       visual_patterns: List[VisualPattern]) -> float:
        """Apply mechanic-specific scoring logic."""
        score = 0.0

        if mechanic == GameMechanic.PATTERN_COMPLETION:
            # Look for incomplete grids and regular patterns
            if grid_features.get("grid_detected", False):
                score += 0.3
            if grid_features.get("cell_uniformity", 0) > 0.7:
                score += 0.2

        elif mechanic == GameMechanic.PHYSICS_SIMULATION:
            # Look for objects that could fall or interact
            if grid_features.get("shape_count", 0) > 2:
                score += 0.2
            if any(p.pattern_type == "circle" for p in visual_patterns):
                score += 0.3

        elif mechanic == GameMechanic.SPATIAL_PUZZLE:
            # Look for complex shapes and geometric patterns
            if grid_features.get("shape_complexity", 0) > 1.5:
                score += 0.3
            geometric_patterns = [p for p in visual_patterns if p.pattern_type in ["rectangle", "circle"]]
            if len(geometric_patterns) > 1:
                score += 0.2

        elif mechanic == GameMechanic.OBJECT_MANIPULATION:
            # Look for discrete, moveable objects
            if grid_features.get("shape_count", 0) > 1:
                score += 0.2
            color_regions = [p for p in visual_patterns if p.pattern_type == "color_region"]
            if len(color_regions) > 1:
                score += 0.3

        elif mechanic == GameMechanic.SORTING_ORGANIZING:
            # Look for multiple similar objects
            color_regions = [p for p in visual_patterns if p.pattern_type == "color_region"]
            if len(color_regions) > 3:
                score += 0.4
            if grid_features.get("color_count", 0) > 3:
                score += 0.2

        elif mechanic == GameMechanic.NAVIGATION_PATHFINDING:
            # Look for maze-like structures
            if grid_features.get("grid_detected", False) and grid_features.get("shape_complexity", 0) > 1.0:
                score += 0.4

        return min(score, 0.5)  # Cap additional scoring

    def _calculate_complexity_score(self, grid_features: Dict[str, Any],
                                  visual_patterns: List[VisualPattern]) -> float:
        """Calculate overall complexity score for the game."""
        complexity = 0.0

        # Grid complexity
        complexity += grid_features.get("shape_complexity", 0) * 0.3
        complexity += min(grid_features.get("shape_count", 0) / 10.0, 1.0) * 0.2
        complexity += min(grid_features.get("color_count", 0) / 10.0, 1.0) * 0.2

        # Pattern complexity
        complexity += min(len(visual_patterns) / 10.0, 1.0) * 0.3

        return min(complexity, 1.0)

    def get_hypothesis_indicators(self, profile: GameMechanicsProfile) -> Dict[str, Any]:
        """
        Extract indicators that can be used for hypothesis generation.

        Args:
            profile: GameMechanicsProfile from analysis

        Returns:
            Dictionary of indicators for hypothesis generation
        """
        indicators = {
            "primary_mechanic": profile.primary_mechanic.value,
            "secondary_mechanics": [m.value for m in profile.secondary_mechanics],
            "complexity_level": "high" if profile.complexity_score > 0.7 else "medium" if profile.complexity_score > 0.3 else "low",
            "grid_based": profile.grid_features.get("grid_detected", False),
            "color_dependent": profile.grid_features.get("color_count", 0) > 3,
            "shape_dependent": profile.grid_features.get("shape_count", 0) > 2,
            "symmetry_present": any(profile.grid_features.get("symmetry_detected", {}).values()),
            "repetition_present": profile.grid_features.get("repetition_detected", {}).get("repetition_detected", False),
            "suggested_approaches": self._suggest_approaches(profile),
            "visual_patterns": [p.pattern_type for p in profile.visual_patterns],
            "confidence_scores": profile.mechanic_confidence
        }

        return indicators

    def _suggest_approaches(self, profile: GameMechanicsProfile) -> List[str]:
        """Suggest approaches based on detected mechanics."""
        approaches = []

        if profile.primary_mechanic == GameMechanic.PATTERN_COMPLETION:
            approaches.extend(["complete_missing_elements", "extend_pattern", "fill_gaps"])
        elif profile.primary_mechanic == GameMechanic.PHYSICS_SIMULATION:
            approaches.extend(["simulate_gravity", "predict_collisions", "trace_movement"])
        elif profile.primary_mechanic == GameMechanic.SPATIAL_PUZZLE:
            approaches.extend(["fit_pieces", "rotate_objects", "spatial_reasoning"])
        elif profile.primary_mechanic == GameMechanic.OBJECT_MANIPULATION:
            approaches.extend(["move_objects", "transform_shapes", "reposition_elements"])
        elif profile.primary_mechanic == GameMechanic.SORTING_ORGANIZING:
            approaches.extend(["group_by_color", "sort_by_size", "organize_by_pattern"])
        elif profile.primary_mechanic == GameMechanic.NAVIGATION_PATHFINDING:
            approaches.extend(["find_path", "avoid_obstacles", "connect_points"])

        # Add secondary mechanic approaches
        for secondary in profile.secondary_mechanics:
            if secondary == GameMechanic.COLOR_MATCHING:
                approaches.append("match_colors")
            elif secondary == GameMechanic.SYMMETRY_PATTERN:
                approaches.append("maintain_symmetry")

        return approaches

# Module initialization
def create_game_pattern_analyzer() -> GamePatternAnalyzer:
    """Factory function to create a GamePatternAnalyzer instance."""
    return GamePatternAnalyzer()

# Singleton instance for global use
_analyzer_instance = None

def get_game_pattern_analyzer() -> GamePatternAnalyzer:
    """Get the singleton GamePatternAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = create_game_pattern_analyzer()
    return _analyzer_instance
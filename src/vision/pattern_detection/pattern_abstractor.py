"""
Pattern abstraction and rule generation for ARC task transfer learning.

This module takes concrete patterns detected by the EnhancedPatternDetector
and converts them into abstract rules that can be applied to new situations.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
from .enhanced_pattern_detector import PatternFeatures, PatternType

logger = logging.getLogger(__name__)

class AbstractionLevel(Enum):
    """Levels of pattern abstraction."""
    CONCRETE = "concrete"         # Direct pattern with specific values
    RELATIVE = "relative"         # Pattern described in relative terms
    CATEGORICAL = "categorical"   # Pattern described by categories
    STRUCTURAL = "structural"     # Pattern described by structure
    CONCEPTUAL = "conceptual"     # High-level concept pattern

class RelationType(Enum):
    """Types of relationships between pattern elements."""
    SEQUENTIAL = "sequential"     # Elements follow a sequence
    SPATIAL = "spatial"          # Elements have spatial relationship
    TRANSFORM = "transform"      # Elements transform into others
    HIERARCHICAL = "hierarchical" # Elements have hierarchy
    FUNCTIONAL = "functional"    # Elements serve a function

@dataclass
class AbstractRule:
    """An abstract rule derived from a pattern."""
    rule_id: str
    pattern_type: PatternType
    abstraction_level: AbstractionLevel
    relation_type: RelationType
    rule_components: Dict[str, Any]
    application_constraints: List[Dict[str, Any]]
    confidence: float
    generalization_score: float
    metadata: Dict[str, Any]

class PatternAbstractor:
    """Converts concrete patterns into abstract rules."""
    
    def __init__(self):
        """Initialize the pattern abstractor."""
        self.abstraction_stats = {
            "patterns_processed": 0,
            "rules_generated": 0,
            "confidence_scores": [],
            "generalization_scores": []
        }
    
    def abstract_pattern(self, pattern: PatternFeatures) -> List[AbstractRule]:
        """Convert a concrete pattern into abstract rules."""
        rules = []
        
        # Generate rules at different abstraction levels
        concrete_rule = self._create_concrete_rule(pattern)
        if concrete_rule:
            rules.append(concrete_rule)
            
        relative_rule = self._create_relative_rule(pattern)
        if relative_rule:
            rules.append(relative_rule)
            
        categorical_rule = self._create_categorical_rule(pattern)
        if categorical_rule:
            rules.append(categorical_rule)
            
        structural_rule = self._create_structural_rule(pattern)
        if structural_rule:
            rules.append(structural_rule)
            
        # Update statistics
        self._update_abstraction_stats(rules)
        
        return rules
        
    def _create_concrete_rule(self, pattern: PatternFeatures) -> Optional[AbstractRule]:
        """Create a concrete-level rule from pattern."""
        try:
            rule_components = {
                "elements": pattern.elements,
                "exact_values": True,
                "pattern_sequence": self._extract_sequence(pattern)
            }
            
            constraints = [{
                "type": "exact_match",
                "elements": pattern.elements
            }]
            
            return AbstractRule(
                rule_id=f"concrete_{pattern.pattern_type.value}_{id(pattern)}",
                pattern_type=pattern.pattern_type,
                abstraction_level=AbstractionLevel.CONCRETE,
                relation_type=self._determine_relation_type(pattern),
                rule_components=rule_components,
                application_constraints=constraints,
                confidence=pattern.confidence,
                generalization_score=0.2,  # Low generalization for concrete rules
                metadata=pattern.metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating concrete rule: {e}")
            return None
            
    def _create_relative_rule(self, pattern: PatternFeatures) -> Optional[AbstractRule]:
        """Create a relative-level rule from pattern."""
        try:
            relative_components = self._extract_relative_components(pattern)
            if not relative_components:
                return None
                
            constraints = [{
                "type": "relative_match",
                "relations": relative_components["relations"]
            }]
            
            return AbstractRule(
                rule_id=f"relative_{pattern.pattern_type.value}_{id(pattern)}",
                pattern_type=pattern.pattern_type,
                abstraction_level=AbstractionLevel.RELATIVE,
                relation_type=self._determine_relation_type(pattern),
                rule_components=relative_components,
                application_constraints=constraints,
                confidence=pattern.confidence * 0.9,  # Slightly lower confidence
                generalization_score=0.4,
                metadata=pattern.metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating relative rule: {e}")
            return None
            
    def _create_categorical_rule(self, pattern: PatternFeatures) -> Optional[AbstractRule]:
        """Create a categorical-level rule from pattern."""
        try:
            categorical_components = self._extract_categorical_components(pattern)
            if not categorical_components:
                return None
                
            constraints = [{
                "type": "category_match",
                "categories": categorical_components["categories"]
            }]
            
            return AbstractRule(
                rule_id=f"categorical_{pattern.pattern_type.value}_{id(pattern)}",
                pattern_type=pattern.pattern_type,
                abstraction_level=AbstractionLevel.CATEGORICAL,
                relation_type=self._determine_relation_type(pattern),
                rule_components=categorical_components,
                application_constraints=constraints,
                confidence=pattern.confidence * 0.8,
                generalization_score=0.6,
                metadata=pattern.metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating categorical rule: {e}")
            return None
            
    def _create_structural_rule(self, pattern: PatternFeatures) -> Optional[AbstractRule]:
        """Create a structural-level rule from pattern."""
        try:
            structural_components = self._extract_structural_components(pattern)
            if not structural_components:
                return None
                
            constraints = [{
                "type": "structure_match",
                "structure": structural_components["structure"]
            }]
            
            return AbstractRule(
                rule_id=f"structural_{pattern.pattern_type.value}_{id(pattern)}",
                pattern_type=pattern.pattern_type,
                abstraction_level=AbstractionLevel.STRUCTURAL,
                relation_type=self._determine_relation_type(pattern),
                rule_components=structural_components,
                application_constraints=constraints,
                confidence=pattern.confidence * 0.7,
                generalization_score=0.8,
                metadata=pattern.metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating structural rule: {e}")
            return None
            
    def _extract_sequence(self, pattern: PatternFeatures) -> List[Any]:
        """Extract sequential elements from pattern."""
        if pattern.pattern_type == PatternType.COLOR_SEQUENCE:
            return pattern.elements
            
        elif pattern.pattern_type == PatternType.DIRECTIONAL:
            return self._extract_directional_sequence(pattern)
            
        return []
        
    def _extract_directional_sequence(self, pattern: PatternFeatures) -> List[Any]:
        """Extract sequence from directional pattern."""
        if "direction" not in pattern.metadata:
            return []
            
        direction = pattern.metadata["direction"]
        elements = pattern.elements
        
        if direction in ["horizontal", "left_to_right"]:
            return elements[0] if isinstance(elements[0], list) else elements
            
        elif direction in ["vertical", "top_to_bottom"]:
            return [row[0] if isinstance(row, list) else row for row in elements]
            
        return []
        
    def _extract_relative_components(self, pattern: PatternFeatures) -> Optional[Dict[str, Any]]:
        """Extract relative relationships between pattern elements."""
        try:
            relations = []
            elements = np.array(pattern.elements)
            
            if pattern.pattern_type == PatternType.COLOR_SEQUENCE:
                # Find relative color changes
                for i in range(len(elements) - 1):
                    diff = int(elements[i+1]) - int(elements[i])
                    relations.append({
                        "type": "color_change",
                        "from_idx": i,
                        "to_idx": i + 1,
                        "delta": diff
                    })
                    
            elif pattern.pattern_type == PatternType.GRID_BASED:
                # Extract grid cell relationships
                grid_size = pattern.metadata.get("grid_size", (0, 0))
                if all(grid_size):
                    rows, cols = grid_size
                    grid = elements.reshape(rows, cols)
                    
                    # Horizontal relationships
                    for i in range(rows):
                        for j in range(cols - 1):
                            diff = int(grid[i, j+1]) - int(grid[i, j])
                            relations.append({
                                "type": "grid_horizontal",
                                "position": (i, j),
                                "delta": diff
                            })
                            
                    # Vertical relationships
                    for i in range(rows - 1):
                        for j in range(cols):
                            diff = int(grid[i+1, j]) - int(grid[i, j])
                            relations.append({
                                "type": "grid_vertical",
                                "position": (i, j),
                                "delta": diff
                            })
                            
            return {
                "relations": relations,
                "element_count": len(elements),
                "relative_positions": self._calculate_relative_positions(elements)
            }
            
        except Exception as e:
            logger.error(f"Error extracting relative components: {e}")
            return None
            
    def _extract_categorical_components(self, pattern: PatternFeatures) -> Optional[Dict[str, Any]]:
        """Extract categorical components from pattern."""
        try:
            elements = np.array(pattern.elements)
            unique_values = np.unique(elements)
            
            categories = {
                "low": [],
                "medium": [],
                "high": [],
                "background": [],
                "foreground": []
            }
            
            if len(unique_values) > 0:
                min_val, max_val = np.min(unique_values), np.max(unique_values)
                range_size = (max_val - min_val) / 3
                
                for val in unique_values:
                    # Categorize by value range
                    if val <= min_val + range_size:
                        categories["low"].append(val)
                    elif val <= min_val + 2 * range_size:
                        categories["medium"].append(val)
                    else:
                        categories["high"].append(val)
                        
                    # Categorize by frequency
                    freq = np.sum(elements == val) / elements.size
                    if freq > 0.4:  # High frequency suggests background
                        categories["background"].append(val)
                    elif freq < 0.1:  # Low frequency suggests foreground
                        categories["foreground"].append(val)
                        
            return {
                "categories": categories,
                "value_ranges": {
                    "min": min_val,
                    "max": max_val,
                    "mean": np.mean(unique_values)
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting categorical components: {e}")
            return None
            
    def _extract_structural_components(self, pattern: PatternFeatures) -> Optional[Dict[str, Any]]:
        """Extract structural components from pattern."""
        try:
            elements = np.array(pattern.elements)
            
            structure = {
                "shape": self._analyze_shape_structure(elements),
                "connectivity": self._analyze_connectivity(elements),
                "density": self._calculate_density(elements),
                "distribution": self._analyze_distribution(elements)
            }
            
            if pattern.pattern_type == PatternType.GRID_BASED:
                structure.update(self._analyze_grid_structure(elements, pattern.metadata))
                
            elif pattern.pattern_type == PatternType.SYMMETRICAL:
                structure.update(self._analyze_symmetry_structure(elements, pattern.metadata))
                
            return {
                "structure": structure,
                "complexity": self._calculate_complexity(elements),
                "hierarchy": self._analyze_hierarchy(elements)
            }
            
        except Exception as e:
            logger.error(f"Error extracting structural components: {e}")
            return None
            
    def _analyze_shape_structure(self, elements: np.ndarray) -> Dict[str, Any]:
        """Analyze structural properties of shape."""
        try:
            if elements.size == 0:
                return {}
                
            # Calculate basic shape properties
            height, width = elements.shape if len(elements.shape) > 1 else (1, len(elements))
            aspect_ratio = width / height if height > 0 else 0
            
            # Find contiguous regions
            regions = self._find_contiguous_regions(elements)
            
            return {
                "dimensions": (height, width),
                "aspect_ratio": aspect_ratio,
                "region_count": len(regions),
                "region_sizes": [len(region) for region in regions],
                "region_shapes": [self._classify_region_shape(region) for region in regions]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing shape structure: {e}")
            return {}
            
    def _find_contiguous_regions(self, elements: np.ndarray) -> List[np.ndarray]:
        """Find contiguous regions in elements."""
        regions = []
        visited = np.zeros_like(elements, dtype=bool)
        
        def flood_fill(i, j, value):
            if (i < 0 or i >= elements.shape[0] or 
                j < 0 or j >= elements.shape[1] or
                visited[i, j] or
                elements[i, j] != value):
                return []
            
            visited[i, j] = True
            region = [(i, j)]
            
            # Check neighbors
            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                region.extend(flood_fill(i + di, j + dj, value))
                
            return region
        
        # Find regions
        for i in range(elements.shape[0]):
            for j in range(elements.shape[1]):
                if not visited[i, j]:
                    region = flood_fill(i, j, elements[i, j])
                    if region:
                        regions.append(np.array(region))
                        
        return regions
        
    def _classify_region_shape(self, region: np.ndarray) -> str:
        """Classify shape of a contiguous region."""
        if len(region) < 4:
            return "point"
            
        # Calculate bounding box
        min_i = np.min(region[:, 0])
        max_i = np.max(region[:, 0])
        min_j = np.min(region[:, 1])
        max_j = np.max(region[:, 1])
        
        height = max_i - min_i + 1
        width = max_j - min_j + 1
        
        # Calculate shape metrics
        area = len(region)
        expected_rect_area = height * width
        fill_ratio = area / expected_rect_area
        
        if fill_ratio > 0.9:  # Almost completely filled
            if abs(height - width) <= 1:
                return "square"
            return "rectangle"
            
        elif fill_ratio > 0.7:  # Mostly filled
            return "blob"
            
        elif fill_ratio > 0.4:  # Partially filled
            return "irregular"
            
        return "sparse"
        
    def _analyze_connectivity(self, elements: np.ndarray) -> Dict[str, Any]:
        """Analyze connectivity between elements."""
        try:
            if elements.size == 0:
                return {}
                
            # Find adjacent elements
            connections = 0
            strong_connections = 0  # Elements with same value
            
            for i in range(elements.shape[0]):
                for j in range(elements.shape[1]):
                    val = elements[i, j]
                    
                    # Check right neighbor
                    if j + 1 < elements.shape[1]:
                        connections += 1
                        if elements[i, j+1] == val:
                            strong_connections += 1
                            
                    # Check bottom neighbor
                    if i + 1 < elements.shape[0]:
                        connections += 1
                        if elements[i+1, j] == val:
                            strong_connections += 1
                            
            connectivity = strong_connections / connections if connections > 0 else 0
            
            return {
                "connectivity_score": connectivity,
                "total_connections": connections,
                "strong_connections": strong_connections
            }
            
        except Exception as e:
            logger.error(f"Error analyzing connectivity: {e}")
            return {}
            
    def _calculate_density(self, elements: np.ndarray) -> float:
        """Calculate density of non-zero elements."""
        try:
            return np.count_nonzero(elements) / elements.size
        except Exception:
            return 0.0
            
    def _analyze_distribution(self, elements: np.ndarray) -> Dict[str, Any]:
        """Analyze distribution of elements."""
        try:
            if elements.size == 0:
                return {}
                
            # Calculate center of mass
            indices = np.nonzero(elements)
            if len(indices[0]) == 0:
                return {"uniformity": 1.0}
                
            center_i = np.mean(indices[0])
            center_j = np.mean(indices[1])
            
            # Calculate distances from center
            distances = np.sqrt((indices[0] - center_i)**2 + (indices[1] - center_j)**2)
            
            return {
                "center": (float(center_i), float(center_j)),
                "mean_distance": float(np.mean(distances)),
                "std_distance": float(np.std(distances)),
                "uniformity": float(1.0 - np.std(distances) / np.max(distances))
                if np.max(distances) > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing distribution: {e}")
            return {}
            
    def _analyze_grid_structure(self, elements: np.ndarray,
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze grid-specific structural properties."""
        try:
            grid_size = metadata.get("grid_size", (0, 0))
            if not all(grid_size):
                return {}
                
            rows, cols = grid_size
            cells = elements.reshape(rows, cols)
            
            # Analyze row/column patterns
            row_patterns = []
            for i in range(rows):
                row = cells[i]
                row_patterns.append({
                    "repeating": self._is_repeating_sequence(row),
                    "alternating": self._is_alternating_sequence(row),
                    "monotonic": self._is_monotonic_sequence(row)
                })
                
            col_patterns = []
            for j in range(cols):
                col = cells[:, j]
                col_patterns.append({
                    "repeating": self._is_repeating_sequence(col),
                    "alternating": self._is_alternating_sequence(col),
                    "monotonic": self._is_monotonic_sequence(col)
                })
                
            return {
                "grid_size": grid_size,
                "row_patterns": row_patterns,
                "column_patterns": col_patterns,
                "regularity": metadata.get("regularity", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing grid structure: {e}")
            return {}
            
    def _is_repeating_sequence(self, arr: np.ndarray) -> bool:
        """Check if sequence is repeating."""
        if len(arr) < 2:
            return False
            
        # Try different periods
        for period in range(1, len(arr) // 2 + 1):
            is_repeating = True
            pattern = arr[:period]
            
            for i in range(period, len(arr), period):
                if not np.array_equal(arr[i:i+period], pattern):
                    is_repeating = False
                    break
                    
            if is_repeating:
                return True
                
        return False
        
    def _is_alternating_sequence(self, arr: np.ndarray) -> bool:
        """Check if sequence is alternating."""
        if len(arr) < 2:
            return False
            
        return np.array_equal(arr[::2], arr[0]) and np.array_equal(arr[1::2], arr[1])
        
    def _is_monotonic_sequence(self, arr: np.ndarray) -> bool:
        """Check if sequence is monotonic (increasing or decreasing)."""
        diffs = np.diff(arr)
        return np.all(diffs >= 0) or np.all(diffs <= 0)
        
    def _analyze_symmetry_structure(self, elements: np.ndarray,
                                  metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze symmetry-specific structural properties."""
        try:
            symmetry_type = metadata.get("symmetry_type")
            if not symmetry_type:
                return {}
                
            axis = metadata.get("axis", 0)
            quality = metadata.get("quality", 0.0)
            
            # Analyze symmetry components
            if symmetry_type == "horizontal":
                upper = elements[:axis]
                lower = np.flipud(elements[axis:])
                deviation = np.mean(upper != lower[:len(upper)])
                
            else:  # vertical
                left = elements[:, :axis]
                right = np.fliplr(elements[:, axis:])
                deviation = np.mean(left != right[:, :left.shape[1]])
                
            return {
                "symmetry_type": symmetry_type,
                "axis_position": axis,
                "quality": quality,
                "deviation": float(deviation),
                "perfect": deviation == 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing symmetry structure: {e}")
            return {}
            
    def _calculate_complexity(self, elements: np.ndarray) -> float:
        """Calculate structural complexity score."""
        try:
            if elements.size == 0:
                return 0.0
                
            # Factors that contribute to complexity:
            # 1. Number of unique values
            unique_count = len(np.unique(elements))
            value_complexity = min(unique_count / 10, 1.0)
            
            # 2. Spatial complexity (changes between adjacent elements)
            changes_h = np.sum(np.diff(elements, axis=1) != 0)
            changes_v = np.sum(np.diff(elements, axis=0) != 0)
            spatial_complexity = min((changes_h + changes_v) / elements.size, 1.0)
            
            # 3. Distribution complexity
            distribution = self._analyze_distribution(elements)
            distribution_complexity = 1.0 - distribution.get("uniformity", 0.0)
            
            # Weighted combination
            return (0.3 * value_complexity + 
                   0.4 * spatial_complexity +
                   0.3 * distribution_complexity)
            
        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
            return 0.0
            
    def _analyze_hierarchy(self, elements: np.ndarray) -> Dict[str, Any]:
        """Analyze hierarchical relationships in pattern."""
        try:
            if elements.size == 0:
                return {}
                
            # Find nested structures
            regions = self._find_contiguous_regions(elements)
            nested = []
            
            for i, region_i in enumerate(regions):
                for j, region_j in enumerate(regions):
                    if i != j:
                        # Check if region_j is contained within region_i
                        if self._is_region_contained(region_i, region_j):
                            nested.append({
                                "parent": i,
                                "child": j,
                                "parent_size": len(region_i),
                                "child_size": len(region_j)
                            })
                            
            return {
                "nested_relationships": nested,
                "hierarchy_depth": self._calculate_hierarchy_depth(nested),
                "branching_factor": len(nested) / len(regions) if regions else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing hierarchy: {e}")
            return {}
            
    def _is_region_contained(self, parent: np.ndarray, child: np.ndarray) -> bool:
        """Check if one region is contained within another."""
        parent_bbox = (np.min(parent[:, 0]), np.max(parent[:, 0]),
                      np.min(parent[:, 1]), np.max(parent[:, 1]))
        
        child_bbox = (np.min(child[:, 0]), np.max(child[:, 0]),
                     np.min(child[:, 1]), np.max(child[:, 1]))
        
        return (parent_bbox[0] <= child_bbox[0] and
                parent_bbox[1] >= child_bbox[1] and
                parent_bbox[2] <= child_bbox[2] and
                parent_bbox[3] >= child_bbox[3])
                
    def _calculate_hierarchy_depth(self, nested: List[Dict[str, Any]]) -> int:
        """Calculate maximum depth of nested relationships."""
        if not nested:
            return 0
            
        # Build parent-child relationships
        children = {}
        for rel in nested:
            parent = rel["parent"]
            child = rel["child"]
            if parent not in children:
                children[parent] = []
            children[parent].append(child)
            
        # Find max depth using DFS
        def get_depth(node: int) -> int:
            if node not in children:
                return 1
            return 1 + max(get_depth(child) for child in children[node])
            
        return max(get_depth(node) for node in children.keys())
        
    def _determine_relation_type(self, pattern: PatternFeatures) -> RelationType:
        """Determine the primary relationship type in pattern."""
        if pattern.pattern_type == PatternType.COLOR_SEQUENCE:
            return RelationType.SEQUENTIAL
            
        elif pattern.pattern_type in [PatternType.GRID_BASED, PatternType.SYMMETRICAL]:
            return RelationType.SPATIAL
            
        elif pattern.pattern_type == PatternType.TRANSFORMATION:
            return RelationType.TRANSFORM
            
        return RelationType.FUNCTIONAL
        
    def _calculate_relative_positions(self, elements: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate relative positions between elements."""
        positions = []
        
        if len(elements.shape) == 2:
            height, width = elements.shape
            for i in range(height):
                for j in range(width):
                    if elements[i, j] != 0:  # Non-zero element
                        # Calculate relative positions to other non-zero elements
                        for ii in range(height):
                            for jj in range(width):
                                if elements[ii, jj] != 0 and (i != ii or j != jj):
                                    positions.append({
                                        "from": (i, j),
                                        "to": (ii, jj),
                                        "delta": (ii - i, jj - j)
                                    })
                                    
        return positions
        
    def _update_abstraction_stats(self, rules: List[AbstractRule]) -> None:
        """Update abstraction statistics."""
        self.abstraction_stats["patterns_processed"] += 1
        self.abstraction_stats["rules_generated"] += len(rules)
        
        for rule in rules:
            self.abstraction_stats["confidence_scores"].append(rule.confidence)
            self.abstraction_stats["generalization_scores"].append(rule.generalization_score)
            
    def get_abstraction_stats(self) -> Dict[str, Any]:
        """Get current abstraction statistics."""
        stats = self.abstraction_stats.copy()
        
        if self.abstraction_stats["confidence_scores"]:
            stats["average_confidence"] = np.mean(self.abstraction_stats["confidence_scores"])
            stats["average_generalization"] = np.mean(self.abstraction_stats["generalization_scores"])
        else:
            stats["average_confidence"] = 0.0
            stats["average_generalization"] = 0.0
            
        return stats
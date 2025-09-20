"""
Change Detector

Detects changes between input and output grids.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChangeInfo:
    """Represents a detected change between input and output grids."""
    change_type: str  # 'added', 'removed', 'moved', 'modified', 'color_change'
    location: Tuple[int, int]  # Grid coordinates where change occurred
    old_value: int  # Original value
    new_value: int  # New value
    confidence: float  # 0.0 to 1.0
    description: str  # Human-readable description

class ChangeDetector:
    """Detects changes between input and output grids."""
    
    def __init__(self, change_threshold: float = 0.1):
        self.change_threshold = change_threshold
    
    def detect_changes(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> List[ChangeInfo]:
        """Detect all changes between input and output grids."""
        try:
            # Convert grids to numpy arrays
            input_array = np.array(input_grid)
            output_array = np.array(output_grid)
            
            # Ensure grids have the same shape
            if input_array.shape != output_array.shape:
                logger.warning("Input and output grids have different shapes")
                return []
            
            changes = []
            
            # Detect pixel-level changes
            changes.extend(self._detect_pixel_changes(input_array, output_array))
            
            # Detect structural changes
            changes.extend(self._detect_structural_changes(input_array, output_array))
            
            logger.debug(f"Detected {len(changes)} changes between grids")
            return changes
            
        except Exception as e:
            logger.error(f"Change detection failed: {e}")
            return []
    
    def _detect_pixel_changes(self, input_array: np.ndarray, output_array: np.ndarray) -> List[ChangeInfo]:
        """Detect pixel-level changes between grids."""
        changes = []
        
        try:
            # Find positions where values differ
            diff_mask = input_array != output_array
            
            for row in range(diff_mask.shape[0]):
                for col in range(diff_mask.shape[1]):
                    if diff_mask[row, col]:
                        old_value = input_array[row, col]
                        new_value = output_array[row, col]
                        
                        # Determine change type
                        change_type = self._classify_pixel_change(old_value, new_value)
                        
                        changes.append(ChangeInfo(
                            change_type=change_type,
                            location=(row, col),
                            old_value=int(old_value),
                            new_value=int(new_value),
                            confidence=1.0,
                            description=f"Pixel changed from {old_value} to {new_value} at ({row}, {col})"
                        ))
            
        except Exception as e:
            logger.error(f"Error detecting pixel changes: {e}")
        
        return changes
    
    def _detect_structural_changes(self, input_array: np.ndarray, output_array: np.ndarray) -> List[ChangeInfo]:
        """Detect structural changes between grids."""
        changes = []
        
        try:
            # Detect added objects (new non-zero regions)
            added_regions = self._find_added_regions(input_array, output_array)
            for region in added_regions:
                changes.append(ChangeInfo(
                    change_type='added',
                    location=region['center'],
                    old_value=0,
                    new_value=region['value'],
                    confidence=region['confidence'],
                    description=f"Object added at {region['center']}"
                ))
            
            # Detect removed objects (disappeared non-zero regions)
            removed_regions = self._find_removed_regions(input_array, output_array)
            for region in removed_regions:
                changes.append(ChangeInfo(
                    change_type='removed',
                    location=region['center'],
                    old_value=region['value'],
                    new_value=0,
                    confidence=region['confidence'],
                    description=f"Object removed at {region['center']}"
                ))
            
            # Detect moved objects
            moved_objects = self._find_moved_objects(input_array, output_array)
            for move in moved_objects:
                changes.append(ChangeInfo(
                    change_type='moved',
                    location=move['from'],
                    old_value=move['value'],
                    new_value=move['value'],
                    confidence=move['confidence'],
                    description=f"Object moved from {move['from']} to {move['to']}"
                ))
            
        except Exception as e:
            logger.error(f"Error detecting structural changes: {e}")
        
        return changes
    
    def _classify_pixel_change(self, old_value: int, new_value: int) -> str:
        """Classify the type of pixel change."""
        if old_value == 0 and new_value != 0:
            return 'added'
        elif old_value != 0 and new_value == 0:
            return 'removed'
        elif old_value != new_value:
            return 'color_change'
        else:
            return 'unchanged'
    
    def _find_added_regions(self, input_array: np.ndarray, output_array: np.ndarray) -> List[Dict[str, Any]]:
        """Find regions that were added in the output."""
        regions = []
        
        try:
            # Find positions where input is 0 but output is not
            added_mask = (input_array == 0) & (output_array != 0)
            
            # Find connected components
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(added_mask)
            
            for i in range(1, num_features + 1):
                region_mask = labeled_array == i
                region_coords = np.where(region_mask)
                
                if len(region_coords[0]) > 0:
                    center = (int(np.mean(region_coords[0])), int(np.mean(region_coords[1])))
                    value = int(np.median(output_array[region_mask]))
                    
                    regions.append({
                        'center': center,
                        'value': value,
                        'confidence': min(1.0, len(region_coords[0]) / 10.0)
                    })
            
        except ImportError:
            # Fallback if scipy is not available
            logger.warning("scipy not available, using simple region detection")
            regions = self._find_added_regions_simple(input_array, output_array)
        except Exception as e:
            logger.error(f"Error finding added regions: {e}")
        
        return regions
    
    def _find_removed_regions(self, input_array: np.ndarray, output_array: np.ndarray) -> List[Dict[str, Any]]:
        """Find regions that were removed in the output."""
        regions = []
        
        try:
            # Find positions where input is not 0 but output is 0
            removed_mask = (input_array != 0) & (output_array == 0)
            
            # Find connected components
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(removed_mask)
            
            for i in range(1, num_features + 1):
                region_mask = labeled_array == i
                region_coords = np.where(region_mask)
                
                if len(region_coords[0]) > 0:
                    center = (int(np.mean(region_coords[0])), int(np.mean(region_coords[1])))
                    value = int(np.median(input_array[region_mask]))
                    
                    regions.append({
                        'center': center,
                        'value': value,
                        'confidence': min(1.0, len(region_coords[0]) / 10.0)
                    })
            
        except ImportError:
            # Fallback if scipy is not available
            logger.warning("scipy not available, using simple region detection")
            regions = self._find_removed_regions_simple(input_array, output_array)
        except Exception as e:
            logger.error(f"Error finding removed regions: {e}")
        
        return regions
    
    def _find_moved_objects(self, input_array: np.ndarray, output_array: np.ndarray) -> List[Dict[str, Any]]:
        """Find objects that moved between input and output."""
        moves = []
        
        try:
            # Simple heuristic: find objects that appear in different locations
            # This is a simplified implementation
            input_objects = self._find_objects(input_array)
            output_objects = self._find_objects(output_array)
            
            # Match objects by value and size
            for in_obj in input_objects:
                for out_obj in output_objects:
                    if (in_obj['value'] == out_obj['value'] and 
                        abs(in_obj['size'] - out_obj['size']) < 2):
                        
                        distance = np.sqrt((in_obj['center'][0] - out_obj['center'][0])**2 + 
                                         (in_obj['center'][1] - out_obj['center'][1])**2)
                        
                        if distance > 2:  # Object moved significantly
                            moves.append({
                                'from': in_obj['center'],
                                'to': out_obj['center'],
                                'value': in_obj['value'],
                                'confidence': 0.7
                            })
            
        except Exception as e:
            logger.error(f"Error finding moved objects: {e}")
        
        return moves
    
    def _find_objects(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Find objects in a grid (simplified implementation)."""
        objects = []
        
        try:
            # Find non-zero regions
            non_zero_mask = grid != 0
            
            # Simple object detection by connected components
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(non_zero_mask)
            
            for i in range(1, num_features + 1):
                region_mask = labeled_array == i
                region_coords = np.where(region_mask)
                
                if len(region_coords[0]) > 0:
                    center = (int(np.mean(region_coords[0])), int(np.mean(region_coords[1])))
                    value = int(np.median(grid[region_mask]))
                    size = len(region_coords[0])
                    
                    objects.append({
                        'center': center,
                        'value': value,
                        'size': size
                    })
            
        except ImportError:
            # Fallback if scipy is not available
            logger.warning("scipy not available, using simple object detection")
            objects = self._find_objects_simple(grid)
        except Exception as e:
            logger.error(f"Error finding objects: {e}")
        
        return objects
    
    def _find_added_regions_simple(self, input_array: np.ndarray, output_array: np.ndarray) -> List[Dict[str, Any]]:
        """Simple fallback for finding added regions."""
        regions = []
        
        try:
            for row in range(input_array.shape[0]):
                for col in range(input_array.shape[1]):
                    if input_array[row, col] == 0 and output_array[row, col] != 0:
                        regions.append({
                            'center': (row, col),
                            'value': int(output_array[row, col]),
                            'confidence': 0.5
                        })
        except Exception as e:
            logger.error(f"Error in simple added regions detection: {e}")
        
        return regions
    
    def _find_removed_regions_simple(self, input_array: np.ndarray, output_array: np.ndarray) -> List[Dict[str, Any]]:
        """Simple fallback for finding removed regions."""
        regions = []
        
        try:
            for row in range(input_array.shape[0]):
                for col in range(input_array.shape[1]):
                    if input_array[row, col] != 0 and output_array[row, col] == 0:
                        regions.append({
                            'center': (row, col),
                            'value': int(input_array[row, col]),
                            'confidence': 0.5
                        })
        except Exception as e:
            logger.error(f"Error in simple removed regions detection: {e}")
        
        return regions
    
    def _find_objects_simple(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Simple fallback for finding objects."""
        objects = []
        
        try:
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    if grid[row, col] != 0:
                        objects.append({
                            'center': (row, col),
                            'value': int(grid[row, col]),
                            'size': 1
                        })
        except Exception as e:
            logger.error(f"Error in simple object detection: {e}")
        
        return objects

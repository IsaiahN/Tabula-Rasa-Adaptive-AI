#!/usr/bin/env python3
"""
Frame Processor - Processes and normalizes ARC-AGI-3 frames.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

class FrameProcessor:
    """Processes and normalizes ARC-AGI-3 frames."""
    
    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.frame_history = deque(maxlen=10)
        self.normalization_cache = {}
        
    def normalize_frame(self, frame: Any) -> np.ndarray:
        """Normalize a frame to a standard format."""
        if isinstance(frame, list):
            # Convert list to numpy array
            frame_array = np.array(frame)
        elif isinstance(frame, np.ndarray):
            frame_array = frame
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}")
        
        # Ensure the frame has the correct shape
        if len(frame_array.shape) == 2:
            # Grayscale frame, convert to RGB
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2RGB)
        elif len(frame_array.shape) == 3:
            # RGB frame, ensure it's in the correct format
            if frame_array.shape[2] == 3:
                # Already RGB
                pass
            elif frame_array.shape[2] == 4:
                # RGBA, convert to RGB
                frame_array = frame_array[:, :, :3]
            else:
                raise ValueError(f"Unsupported frame shape: {frame_array.shape}")
        else:
            raise ValueError(f"Unsupported frame dimensions: {frame_array.shape}")
        
        # Resize to standard grid size if needed
        if frame_array.shape[:2] != (self.grid_size, self.grid_size):
            frame_array = cv2.resize(frame_array, (self.grid_size, self.grid_size))
        
        # Ensure data type is correct
        if frame_array.dtype != np.uint8:
            frame_array = frame_array.astype(np.uint8)
        
        return frame_array
    
    def preprocess_frame(self, frame: Any) -> Dict[str, Any]:
        """Preprocess a frame for analysis."""
        # Normalize the frame
        normalized_frame = self.normalize_frame(frame)
        
        # Add to history
        self.frame_history.append(normalized_frame)
        
        # Calculate frame properties
        properties = self._calculate_frame_properties(normalized_frame)
        
        # Detect changes from previous frame
        changes = self._detect_frame_changes(normalized_frame)
        
        return {
            'frame': normalized_frame,
            'properties': properties,
            'changes': changes,
            'history_length': len(self.frame_history)
        }
    
    def _calculate_frame_properties(self, frame: np.ndarray) -> Dict[str, Any]:
        """Calculate properties of a frame."""
        # Basic properties
        height, width, channels = frame.shape
        
        # Color statistics
        mean_color = np.mean(frame, axis=(0, 1))
        std_color = np.std(frame, axis=(0, 1))
        
        # Brightness and contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Color diversity
        unique_colors = len(np.unique(frame.reshape(-1, channels), axis=0))
        color_diversity = unique_colors / (height * width)
        
        return {
            'dimensions': (height, width, channels),
            'mean_color': mean_color.tolist(),
            'std_color': std_color.tolist(),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'edge_density': float(edge_density),
            'color_diversity': float(color_diversity),
            'unique_colors': int(unique_colors)
        }
    
    def _detect_frame_changes(self, current_frame: np.ndarray) -> Dict[str, Any]:
        """Detect changes from the previous frame."""
        if len(self.frame_history) < 2:
            return {
                'has_changes': False,
                'change_magnitude': 0.0,
                'changed_pixels': 0,
                'change_regions': []
            }
        
        previous_frame = self.frame_history[-2]
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(current_frame, previous_frame)
        change_magnitude = np.mean(frame_diff)
        
        # Count changed pixels
        changed_pixels = np.sum(frame_diff > 0)
        total_pixels = current_frame.size
        change_ratio = changed_pixels / total_pixels
        
        has_changes = change_magnitude > 10.0  # Threshold for significant changes
        
        # Find changed regions
        change_regions = self._find_change_regions(frame_diff)
        
        return {
            'has_changes': has_changes,
            'change_magnitude': float(change_magnitude),
            'changed_pixels': int(changed_pixels),
            'change_ratio': float(change_ratio),
            'change_regions': change_regions
        }
    
    def _find_change_regions(self, frame_diff: np.ndarray) -> List[Dict[str, Any]]:
        """Find regions where changes occurred."""
        # Convert to grayscale if needed
        if len(frame_diff.shape) == 3:
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_RGB2GRAY)
        else:
            gray_diff = frame_diff
        
        # Threshold the difference
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': float(area),
                    'center': (int(x + w/2), int(y + h/2))
                })
        
        return regions
    
    def get_frame_history(self) -> List[np.ndarray]:
        """Get the frame history."""
        return list(self.frame_history)
    
    def clear_history(self):
        """Clear the frame history."""
        self.frame_history.clear()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about frame processing."""
        return {
            'frames_processed': len(self.frame_history),
            'grid_size': self.grid_size,
            'cache_size': len(self.normalization_cache),
            'history_capacity': self.frame_history.maxlen
        }
    
    def set_grid_size(self, size: int):
        """Set the grid size for frame processing."""
        self.grid_size = size
        self.clear_history()  # Clear history when changing grid size
    
    def optimize_processing(self):
        """Optimize frame processing performance."""
        # Clear old cache entries
        if len(self.normalization_cache) > 100:
            # Keep only the most recent 50 entries
            cache_items = list(self.normalization_cache.items())
            self.normalization_cache = dict(cache_items[-50:])
        
        # Trim frame history if it's getting too large
        if len(self.frame_history) > 8:
            # Keep only the most recent 5 frames
            recent_frames = list(self.frame_history)[-5:]
            self.frame_history.clear()
            self.frame_history.extend(recent_frames)
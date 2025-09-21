#!/usr/bin/env python3
"""
Frame Analyzer for ARC-AGI-3
Analyzes game frames to extract visual information and suggest intelligent actions.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class FrameAnalyzer:
    """Analyzes ARC-AGI-3 game frames to extract visual information."""
    
    def __init__(self):
        self.color_map = {
            0: (0, 0, 0),      # Black
            1: (255, 255, 255), # White
            2: (255, 0, 0),    # Red
            3: (0, 255, 0),    # Green
            4: (0, 0, 255),    # Blue
            5: (255, 255, 0),  # Yellow
            6: (255, 0, 255),  # Magenta
            7: (0, 255, 255),  # Cyan
            8: (128, 128, 128), # Gray
            9: (128, 0, 0),    # Dark Red
            10: (0, 128, 0),   # Dark Green
            11: (0, 0, 128),   # Dark Blue
            12: (128, 128, 0), # Dark Yellow
            13: (128, 0, 128), # Dark Magenta
            14: (0, 128, 128), # Dark Cyan
            15: (192, 192, 192) # Light Gray
        }
    
    def analyze_frame(self, frame_data: List[List[List[int]]]) -> Dict[str, Any]:
        """Analyze a game frame and extract visual information."""
        if not frame_data or len(frame_data) == 0:
            return self._empty_analysis()
        
        # Get the most recent frame
        current_frame = frame_data[0] if isinstance(frame_data[0], list) else frame_data
        
        # Convert to numpy array
        grid = np.array(current_frame, dtype=np.uint8)
        
        analysis = {
            'grid_size': grid.shape,
            'unique_colors': np.unique(grid).tolist(),
            'color_distribution': self._get_color_distribution(grid),
            'objects': self._detect_objects(grid),
            'patterns': self._detect_patterns(grid),
            'interactive_elements': self._detect_interactive_elements(grid),
            'suggested_actions': []
        }
        
        # Generate action suggestions based on analysis
        analysis['suggested_actions'] = self._suggest_actions(analysis)
        
        return analysis
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis when no frame data is available."""
        return {
            'grid_size': (0, 0),
            'unique_colors': [],
            'color_distribution': {},
            'objects': [],
            'patterns': [],
            'interactive_elements': [],
            'suggested_actions': []
        }
    
    def _get_color_distribution(self, grid: np.ndarray) -> Dict[int, int]:
        """Get the distribution of colors in the grid."""
        unique, counts = np.unique(grid, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def _detect_objects(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in the grid using contour detection."""
        objects = []
        
        # Convert to BGR for OpenCV
        bgr_image = self._grid_to_bgr(grid)
        
        # Convert to grayscale
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 5:  # Minimum area threshold
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                # Get dominant color in the object
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                dominant_color = self._get_dominant_color_in_region(grid, mask)
                
                objects.append({
                    'id': i,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'dominant_color': dominant_color,
                    'contour': contour.tolist()
                })
        
        return objects
    
    def _detect_patterns(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect patterns in the grid."""
        patterns = []
        
        # Detect lines
        lines = self._detect_lines(grid)
        if lines:
            patterns.append({'type': 'lines', 'count': len(lines), 'data': lines})
        
        # Detect rectangles
        rectangles = self._detect_rectangles(grid)
        if rectangles:
            patterns.append({'type': 'rectangles', 'count': len(rectangles), 'data': rectangles})
        
        # Detect symmetry
        symmetry = self._detect_symmetry(grid)
        if symmetry:
            patterns.append({'type': 'symmetry', 'data': symmetry})
        
        return patterns
    
    def _detect_interactive_elements(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect potentially interactive elements."""
        interactive = []
        
        # Look for bright colors (often interactive)
        bright_colors = [1, 5, 6, 7, 15]  # White, Yellow, Magenta, Cyan, Light Gray
        for color in bright_colors:
            mask = (grid == color)
            if np.any(mask):
                # Find connected components
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 10:  # Minimum size for interactive elements
                        x, y, w, h = cv2.boundingRect(contour)
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = x + w//2, y + h//2
                        
                        interactive.append({
                            'color': color,
                            'area': area,
                            'centroid': (cx, cy),
                            'bbox': (x, y, w, h),
                            'confidence': min(area / 100, 1.0)  # Confidence based on size
                        })
        
        return interactive
    
    def _suggest_actions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest actions based on frame analysis."""
        suggestions = []
        
        # If we have interactive elements, suggest ACTION6 targeting them
        for element in analysis.get('interactive_elements', []):
            if element['confidence'] > 0.3:  # Only suggest high-confidence elements
                suggestions.append({
                    'action': 'ACTION6',
                    'x': element['centroid'][0],
                    'y': element['centroid'][1],
                    'confidence': element['confidence'],
                    'reason': f"Interactive element detected at ({element['centroid'][0]}, {element['centroid'][1]})"
                })
        
        # If we have objects, suggest exploring them
        for obj in analysis.get('objects', []):
            if obj['area'] > 20:  # Only suggest large objects
                suggestions.append({
                    'action': 'ACTION6',
                    'x': obj['centroid'][0],
                    'y': obj['centroid'][1],
                    'confidence': min(obj['area'] / 200, 1.0),
                    'reason': f"Large object detected at ({obj['centroid'][0]}, {obj['centroid'][1]})"
                })
        
        # If no specific targets, suggest movement actions
        if not suggestions:
            suggestions.extend([
                {'action': 'ACTION1', 'reason': 'No specific targets, try moving up'},
                {'action': 'ACTION2', 'reason': 'No specific targets, try moving down'},
                {'action': 'ACTION3', 'reason': 'No specific targets, try moving left'},
                {'action': 'ACTION4', 'reason': 'No specific targets, try moving right'}
            ])
        
        return suggestions
    
    def _grid_to_bgr(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid to BGR image for OpenCV processing."""
        h, w = grid.shape
        bgr_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                color_idx = grid[y, x]
                if color_idx in self.color_map:
                    bgr_image[y, x] = self.color_map[color_idx]
        
        return bgr_image
    
    def _get_dominant_color_in_region(self, grid: np.ndarray, mask: np.ndarray) -> int:
        """Get the dominant color in a masked region."""
        masked_region = grid[mask > 0]
        if len(masked_region) > 0:
            unique, counts = np.unique(masked_region, return_counts=True)
            return unique[np.argmax(counts)]
        return 0
    
    def _detect_lines(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect lines in the grid."""
        # Convert to BGR for line detection
        bgr_image = self._grid_to_bgr(grid)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Use Hough line detection
        lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=50, minLineLength=10, maxLineGap=5)
        
        if lines is not None:
            return [{'start': (line[0][0], line[0][1]), 'end': (line[0][2], line[0][3])} for line in lines]
        return []
    
    def _detect_rectangles(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect rectangles in the grid."""
        # Convert to BGR for rectangle detection
        bgr_image = self._grid_to_bgr(grid)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                rectangles.append({'bbox': (x, y, w, h), 'corners': approx.tolist()})
        
        return rectangles
    
    def _detect_symmetry(self, grid: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect symmetry in the grid."""
        h, w = grid.shape
        
        # Check horizontal symmetry
        top_half = grid[:h//2, :]
        bottom_half = np.flipud(grid[h//2:, :])
        h_symmetry = np.allclose(top_half, bottom_half, atol=1)
        
        # Check vertical symmetry
        left_half = grid[:, :w//2]
        right_half = np.fliplr(grid[:, w//2:])
        v_symmetry = np.allclose(left_half, right_half, atol=1)
        
        if h_symmetry or v_symmetry:
            return {
                'horizontal': h_symmetry,
                'vertical': v_symmetry,
                'center': (w//2, h//2)
            }
        
        return None
    
    # Penalty Decay System Integration
    def get_penalty_aware_avoidance_scores(self) -> Dict[Tuple[int, int], float]:
        """Get penalty-aware avoidance scores for coordinates."""
        # For now, return empty dict - this would be implemented with actual penalty system
        # In a full implementation, this would integrate with the penalty decay system
        return {}
    
    async def record_coordinate_attempt(self, x: int, y: int, success: bool, game_id: str) -> None:
        """Record a coordinate attempt for penalty tracking."""
        # For now, this is a no-op - in a full implementation, this would update the penalty system
        pass
    
    async def decay_penalties(self) -> None:
        """Decay penalties over time."""
        # For now, this is a no-op - in a full implementation, this would decay penalties
        pass

def create_frame_analyzer() -> FrameAnalyzer:
    """Create a frame analyzer instance."""
    return FrameAnalyzer()

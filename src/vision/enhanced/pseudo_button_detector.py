"""
Enhanced Pseudo-Button Detection for Action 6 Games

This module provides advanced computer vision capabilities for detecting pseudo-buttons
in Action 6-only games. It analyzes frames to find clickable elements and tests their
effectiveness through frame difference analysis.

Features:
- Edge-based button detection
- Contrast-based detection
- Pattern-based recognition
- Grid layout analysis
- Frame difference analysis for effectiveness testing
- Learning integration with coordinate intelligence
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class PseudoButtonDetector:
    """Advanced pseudo-button detection system for Action 6 games."""

    def __init__(self, db_interface=None):
        """Initialize the pseudo-button detector.

        Args:
            db_interface: Database interface for storing learning data
        """
        self.db_interface = db_interface
        self.stats = {
            'buttons_detected': 0,
            'effective_buttons_found': 0,
            'frame_analyses_performed': 0,
            'detection_sessions': 0
        }

    async def detect_pseudo_buttons(self, frame: List[List[int]], game_id: str) -> List[Dict[str, Any]]:
        """Detect pseudo-buttons in the frame using computer vision.

        Pseudo-buttons are visual elements that look clickable and may control gameplay.
        This method analyzes the frame to find rectangular regions, color boundaries,
        and other visual patterns that suggest interactive elements.

        Args:
            frame: Current game frame
            game_id: Game identifier for caching results

        Returns:
            List of detected button candidates with coordinates and confidence
        """
        try:
            if not frame or len(frame) == 0:
                return []

            height, width = len(frame), len(frame[0]) if frame else 0
            if height == 0 or width == 0:
                return []

            self.stats['detection_sessions'] += 1
            button_candidates = []

            # Method 1: Edge detection for rectangular shapes
            edge_buttons = self._detect_edge_based_buttons(frame)
            button_candidates.extend(edge_buttons)

            # Method 2: Color contrast analysis
            contrast_buttons = self._detect_contrast_based_buttons(frame)
            button_candidates.extend(contrast_buttons)

            # Method 3: Pattern recognition for common button shapes
            pattern_buttons = self._detect_pattern_based_buttons(frame)
            button_candidates.extend(pattern_buttons)

            # Method 4: Grid-based analysis for systematic layouts
            grid_buttons = self._detect_grid_based_buttons(frame)
            button_candidates.extend(grid_buttons)

            # Remove duplicates and sort by confidence
            unique_buttons = self._deduplicate_buttons(button_candidates)
            unique_buttons.sort(key=lambda b: b['confidence'], reverse=True)

            self.stats['buttons_detected'] += len(unique_buttons)
            logger.debug(f"Detected {len(unique_buttons)} potential pseudo-buttons")
            return unique_buttons[:10]  # Return top 10 candidates

        except Exception as e:
            logger.error(f"Error detecting pseudo-buttons: {e}")
            return []

    def _detect_edge_based_buttons(self, frame: List[List[int]]) -> List[Dict[str, Any]]:
        """Detect buttons based on edge patterns."""
        buttons = []
        height, width = len(frame), len(frame[0])

        # Simple edge detection by looking for rectangular patterns
        for y in range(1, height - 1, 3):  # Sample every 3 pixels for efficiency
            for x in range(1, width - 1, 3):
                # Check for potential rectangular button boundaries
                current = frame[y][x]

                # Look for edge patterns (significant color changes)
                edges = 0
                neighbors = [
                    frame[y-1][x], frame[y+1][x],  # Vertical neighbors
                    frame[y][x-1], frame[y][x+1]   # Horizontal neighbors
                ]

                for neighbor in neighbors:
                    if abs(current - neighbor) > 30:  # Threshold for edge detection
                        edges += 1

                # If we found strong edges, this might be a button corner/edge
                if edges >= 2:
                    # Check if this forms a rectangular region
                    button_info = self._analyze_potential_button_region(frame, x, y)
                    if button_info:
                        buttons.append(button_info)

        return buttons

    def _detect_contrast_based_buttons(self, frame: List[List[int]]) -> List[Dict[str, Any]]:
        """Detect buttons based on color contrast."""
        buttons = []
        height, width = len(frame), len(frame[0])

        # Look for high-contrast regions that might be buttons
        for y in range(5, height - 5, 5):
            for x in range(5, width - 5, 5):
                # Calculate local contrast
                region_values = []
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if 0 <= y + dy < height and 0 <= x + dx < width:
                            region_values.append(frame[y + dy][x + dx])

                if len(region_values) > 0:
                    avg_value = sum(region_values) / len(region_values)
                    contrast = max(region_values) - min(region_values)

                    # High contrast might indicate a button
                    if contrast > 50:
                        buttons.append({
                            'x': x,
                            'y': y,
                            'confidence': min(contrast / 255.0, 1.0),
                            'type': 'contrast',
                            'size': 10,  # Default button size
                            'priority': 0.6
                        })

        return buttons

    def _detect_pattern_based_buttons(self, frame: List[List[int]]) -> List[Dict[str, Any]]:
        """Detect buttons based on common visual patterns."""
        buttons = []
        height, width = len(frame), len(frame[0])

        # Look for common button patterns (darker/lighter rectangles)
        for y in range(10, height - 10, 8):
            for x in range(10, width - 10, 8):
                # Analyze 20x20 regions for button-like patterns
                region_size = min(20, height - y, width - x)
                if region_size < 10:
                    continue

                # Check if region has button-like characteristics
                center_value = frame[y + region_size//2][x + region_size//2]
                edge_values = []

                # Sample edge pixels
                for i in range(region_size):
                    if y + i < height and x < width:
                        edge_values.append(frame[y + i][x])  # Left edge
                    if y + i < height and x + region_size - 1 < width:
                        edge_values.append(frame[y + i][x + region_size - 1])  # Right edge
                    if y < height and x + i < width:
                        edge_values.append(frame[y][x + i])  # Top edge
                    if y + region_size - 1 < height and x + i < width:
                        edge_values.append(frame[y + region_size - 1][x + i])  # Bottom edge

                if edge_values:
                    avg_edge = sum(edge_values) / len(edge_values)
                    # Button-like if center is significantly different from edges
                    if abs(center_value - avg_edge) > 25:
                        buttons.append({
                            'x': x + region_size // 2,
                            'y': y + region_size // 2,
                            'confidence': min(abs(center_value - avg_edge) / 100.0, 1.0),
                            'type': 'pattern',
                            'size': region_size,
                            'priority': 0.7
                        })

        return buttons

    def _detect_grid_based_buttons(self, frame: List[List[int]]) -> List[Dict[str, Any]]:
        """Detect buttons in systematic grid layouts."""
        buttons = []
        height, width = len(frame), len(frame[0])

        # Common grid sizes for button layouts
        grid_sizes = [(3, 3), (4, 4), (5, 5), (2, 3), (3, 2)]

        for rows, cols in grid_sizes:
            cell_height = height // rows
            cell_width = width // cols

            # Skip if cells are too small
            if cell_height < 10 or cell_width < 10:
                continue

            for row in range(rows):
                for col in range(cols):
                    center_x = col * cell_width + cell_width // 2
                    center_y = row * cell_height + cell_height // 2

                    # Analyze this grid cell for button characteristics
                    if self._is_grid_cell_button_like(frame, center_x, center_y, cell_width, cell_height):
                        buttons.append({
                            'x': center_x,
                            'y': center_y,
                            'confidence': 0.5 + (rows * cols) * 0.05,  # Higher confidence for larger grids
                            'type': 'grid',
                            'size': min(cell_width, cell_height),
                            'priority': 0.8,
                            'grid_info': {'rows': rows, 'cols': cols, 'row': row, 'col': col}
                        })

        return buttons

    def _analyze_potential_button_region(self, frame: List[List[int]], x: int, y: int) -> Dict[str, Any]:
        """Analyze a region to determine if it's a button."""
        height, width = len(frame), len(frame[0])

        # Check for rectangular patterns around this point
        region_size = 15  # 15x15 analysis region

        if (x - region_size//2 < 0 or x + region_size//2 >= width or
            y - region_size//2 < 0 or y + region_size//2 >= height):
            return None

        # Analyze uniformity and edges in the region
        center_val = frame[y][x]
        uniform_count = 0
        edge_strength = 0

        for dy in range(-region_size//2, region_size//2 + 1):
            for dx in range(-region_size//2, region_size//2 + 1):
                val = frame[y + dy][x + dx]

                # Check uniformity (similar values indicate button interior)
                if abs(val - center_val) < 20:
                    uniform_count += 1

                # Check edge strength at region boundary
                if abs(dy) == region_size//2 or abs(dx) == region_size//2:
                    edge_strength += abs(val - center_val)

        total_pixels = region_size * region_size
        uniformity = uniform_count / total_pixels
        avg_edge_strength = edge_strength / (4 * region_size)  # 4 edges

        # Button-like if reasonably uniform interior with strong edges
        if uniformity > 0.6 and avg_edge_strength > 15:
            return {
                'x': x,
                'y': y,
                'confidence': min((uniformity * avg_edge_strength) / 50.0, 1.0),
                'type': 'edge',
                'size': region_size,
                'priority': 0.8
            }

        return None

    def _is_grid_cell_button_like(self, frame: List[List[int]], center_x: int, center_y: int,
                                 cell_width: int, cell_height: int) -> bool:
        """Check if a grid cell has button-like characteristics."""
        height, width = len(frame), len(frame[0])

        # Ensure we're within bounds
        if (center_x - cell_width//4 < 0 or center_x + cell_width//4 >= width or
            center_y - cell_height//4 < 0 or center_y + cell_height//4 >= height):
            return False

        # Sample center and edge regions
        center_samples = []
        edge_samples = []

        # Sample center region
        for dy in range(-cell_height//6, cell_height//6 + 1):
            for dx in range(-cell_width//6, cell_width//6 + 1):
                if (0 <= center_y + dy < height and 0 <= center_x + dx < width):
                    center_samples.append(frame[center_y + dy][center_x + dx])

        # Sample edge regions
        edge_offsets = [
            (-cell_height//4, 0), (cell_height//4, 0),  # Top/bottom
            (0, -cell_width//4), (0, cell_width//4)     # Left/right
        ]

        for dy, dx in edge_offsets:
            if (0 <= center_y + dy < height and 0 <= center_x + dx < width):
                edge_samples.append(frame[center_y + dy][center_x + dx])

        if not center_samples or not edge_samples:
            return False

        # Calculate averages
        avg_center = sum(center_samples) / len(center_samples)
        avg_edge = sum(edge_samples) / len(edge_samples)

        # Button-like if there's sufficient contrast between center and edges
        return abs(avg_center - avg_edge) > 20

    def _deduplicate_buttons(self, buttons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate button candidates that are too close to each other."""
        if not buttons:
            return []

        unique_buttons = []
        min_distance = 15  # Minimum distance between button centers

        for button in buttons:
            is_duplicate = False
            for existing in unique_buttons:
                distance = ((button['x'] - existing['x'])**2 + (button['y'] - existing['y'])**2)**0.5
                if distance < min_distance:
                    # Keep the one with higher confidence
                    if button['confidence'] > existing['confidence']:
                        unique_buttons.remove(existing)
                        unique_buttons.append(button)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_buttons.append(button)

        return unique_buttons

    def calculate_frame_differences(self, frame_before: List[List[int]],
                                   frame_after: List[List[int]]) -> Dict[str, float]:
        """Calculate various metrics for frame differences."""
        if not frame_before or not frame_after:
            return {'total_diff': 0.0, 'significant_changes': 0.0, 'change_ratio': 0.0}

        height = min(len(frame_before), len(frame_after))
        width = min(len(frame_before[0]) if frame_before else 0,
                   len(frame_after[0]) if frame_after else 0)

        if height == 0 or width == 0:
            return {'total_diff': 0.0, 'significant_changes': 0.0, 'change_ratio': 0.0}

        total_diff = 0.0
        significant_changes = 0
        total_pixels = height * width
        threshold = 30  # Threshold for "significant" change

        for y in range(height):
            for x in range(width):
                diff = abs(frame_before[y][x] - frame_after[y][x])
                total_diff += diff

                if diff > threshold:
                    significant_changes += 1

        self.stats['frame_analyses_performed'] += 1

        return {
            'total_diff': total_diff,
            'avg_diff': total_diff / total_pixels,
            'significant_changes': significant_changes,
            'change_ratio': significant_changes / total_pixels,
            'total_pixels': total_pixels
        }

    def evaluate_click_effectiveness(self, change_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate if a click was effective based on frame change metrics."""
        change_ratio = change_metrics.get('change_ratio', 0.0)
        avg_diff = change_metrics.get('avg_diff', 0.0)

        # Scoring system for click effectiveness
        score = 0.0

        # Significant pixel changes (more is better, up to a point)
        if change_ratio > 0.1:  # More than 10% of pixels changed significantly
            score += 0.6
        elif change_ratio > 0.05:  # 5-10% changed
            score += 0.4
        elif change_ratio > 0.01:  # 1-5% changed
            score += 0.2

        # Average difference intensity
        if avg_diff > 50:  # High intensity changes
            score += 0.3
        elif avg_diff > 20:  # Medium intensity
            score += 0.2
        elif avg_diff > 5:   # Low intensity
            score += 0.1

        # Determine effectiveness
        effective = score > 0.3  # Threshold for considering a click "effective"
        confidence = min(score, 1.0)

        if effective:
            self.stats['effective_buttons_found'] += 1

        return {
            'effective': effective,
            'confidence': confidence,
            'score': score,
            'change_ratio': change_ratio,
            'avg_diff': avg_diff,
            'analysis': self._get_effectiveness_reason(score, change_ratio, avg_diff)
        }

    def _get_effectiveness_reason(self, score: float, change_ratio: float, avg_diff: float) -> str:
        """Get a human-readable reason for the effectiveness evaluation."""
        if score > 0.7:
            return "high_impact_changes"
        elif score > 0.3:
            return "moderate_changes_detected"
        elif change_ratio > 0.001:
            return "minimal_visual_changes"
        else:
            return "no_significant_changes"

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about button detection performance."""
        return {
            'detection_sessions': self.stats['detection_sessions'],
            'buttons_detected': self.stats['buttons_detected'],
            'effective_buttons_found': self.stats['effective_buttons_found'],
            'frame_analyses_performed': self.stats['frame_analyses_performed'],
            'avg_buttons_per_session': (
                self.stats['buttons_detected'] / max(self.stats['detection_sessions'], 1)
            ),
            'effectiveness_rate': (
                self.stats['effective_buttons_found'] / max(self.stats['buttons_detected'], 1)
            )
        }


def create_pseudo_button_detector(db_interface=None) -> PseudoButtonDetector:
    """Factory function to create a pseudo-button detector.

    Args:
        db_interface: Optional database interface for learning integration

    Returns:
        Configured PseudoButtonDetector instance
    """
    return PseudoButtonDetector(db_interface=db_interface)
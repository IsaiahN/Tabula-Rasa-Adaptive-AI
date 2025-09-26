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
        
        # Penalty / position tracking integration
        try:
            from src.vision.position_tracking.tracker import PositionTracker
            self.position_tracker = PositionTracker()
        except Exception:
            # If position tracker isn't available, keep a minimal fallback
            self.position_tracker = None

        # Local avoidance score cache (string key "x,y" -> float)
        self.avoidance_scores = {}
        
        # AGI INTEGRATION: Initialize AGI Rapid Puzzle Solver
        try:
            from src.core.agi_rapid_puzzle_solver import create_agi_rapid_solver
            self.agi_solver = create_agi_rapid_solver()
            logger.info("[AGI] Initialized AGI Rapid Puzzle Solver")
        except Exception as e:
            logger.warning(f"[AGI] Could not initialize AGI solver: {e}")
            self.agi_solver = None
    
    def analyze_frame(self, frame_data: List[List[List[int]]]) -> Dict[str, Any]:
        """Analyze a game frame and extract visual information + AGI insights."""
        try:
            logger.info(f" DEBUG: FrameAnalyzer.analyze_frame called with frame_data type: {type(frame_data)}")
            if hasattr(frame_data, 'shape'):
                logger.info(f" DEBUG: frame_data shape: {frame_data.shape}")
            elif isinstance(frame_data, list):
                logger.info(f" DEBUG: frame_data length: {len(frame_data)}")
            
            if frame_data is None or len(frame_data) == 0:
                return self._empty_analysis()
            
            # Get the most recent frame
            current_frame = frame_data[0] if isinstance(frame_data[0], list) else frame_data
            logger.info(f" DEBUG: current_frame type: {type(current_frame)}")
        except Exception as e:
            logger.error(f" DEBUG: Error in FrameAnalyzer.analyze_frame: {e}")
            import traceback
            logger.error(f" DEBUG: FrameAnalyzer traceback: {traceback.format_exc()}")
            raise
        
        # Convert to numpy array
        grid = np.array(current_frame, dtype=np.uint8)
        
        # TRADITIONAL ANALYSIS
        analysis = {
            'grid_size': grid.shape,
            'unique_colors': np.unique(grid).tolist(),
            'color_distribution': self._get_color_distribution(grid),
            'objects': self._detect_objects(grid),
            'patterns': self._detect_patterns(grid),
            'interactive_elements': self._detect_interactive_elements(grid),
            'suggested_actions': []
        }
        
        # AGI RAPID ANALYSIS INTEGRATION
        if self.agi_solver:
            try:
                logger.info("[AGI] Running rapid puzzle analysis...")
                
                # Prepare puzzle state for AGI analysis
                puzzle_state = {
                    'frame': grid,
                    'traditional_analysis': analysis,
                    'grid_shape': grid.shape,
                    'colors': analysis['unique_colors']
                }
                
                # Get AGI insights in seconds (human-like speed)
                agi_insights = self.agi_solver.solve_puzzle_rapidly(puzzle_state)
                
                # Integrate AGI insights into analysis
                analysis['agi_insights'] = agi_insights
                analysis['agi_understanding'] = agi_insights.get('understanding', {})
                analysis['agi_strategy'] = agi_insights.get('solution_strategy', {})
                analysis['agi_confidence'] = agi_insights.get('confidence', 0.0)
                
                # CRITICAL: Use AGI strategy to enhance action suggestions
                if agi_insights.get('solution_strategy'):
                    agi_actions = self._convert_agi_strategy_to_actions(agi_insights['solution_strategy'])
                    analysis['agi_suggested_actions'] = agi_actions
                    
                    # If AGI has high confidence, prioritize AGI actions
                    if agi_insights.get('confidence', 0.0) > 0.7:
                        logger.info(f"[AGI] High confidence ({agi_insights['confidence']:.2f}) - prioritizing AGI strategy")
                        analysis['suggested_actions'] = agi_actions + analysis['suggested_actions']
                    else:
                        # Lower confidence: blend traditional and AGI suggestions
                        analysis['suggested_actions'] = self._suggest_actions(analysis) + agi_actions
                
                logger.info(f"[AGI] Analysis complete - confidence: {agi_insights.get('confidence', 0.0):.2f}")
                if agi_insights.get('best_analogy'):
                    logger.info(f"[AGI] Best analogy: {agi_insights['best_analogy']}")
                
            except Exception as e:
                logger.error(f"[AGI] AGI analysis failed: {e}")
                # Fall back to traditional analysis only
                analysis['suggested_actions'] = self._suggest_actions(analysis)
        else:
            # No AGI solver - use traditional analysis only
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
            'suggested_actions': [],
            'agi_insights': None,
            'agi_understanding': {},
            'agi_strategy': {},
            'agi_confidence': 0.0,
            'agi_suggested_actions': []
        }
    
    def _get_color_distribution(self, grid: np.ndarray) -> Dict[int, int]:
        """Get the distribution of colors in the grid."""
        unique, counts = np.unique(grid, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def _detect_objects(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in the grid using contour detection."""
        objects = []
        
        # For ARC puzzles, we need to detect objects for each non-background color separately
        unique_colors = np.unique(grid).tolist()
        
        # Process each color separately to get better object detection
        for color_value in unique_colors:
            if color_value == 0:  # Skip background (black)
                continue
                
            # Create binary mask for this color
            color_mask = (grid == color_value).astype(np.uint8) * 255
            
            # Find contours for this specific color
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                # Much lower threshold for ARC puzzles - even single pixels matter
                if area >= 1:  
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w//2, y + h//2
                    
                    objects.append({
                        'id': f'{color_value}_{i}',  # Include color in ID for uniqueness
                        'area': area,
                        'bbox': (x, y, w, h),
                        'centroid': (cx, cy),
                        'dominant_color': color_value,
                        'color': color_value,  # Make color easily accessible
                        'contour': contour.tolist()
                    })
        
        logger.info(f"🔍 Detected {len(objects)} objects across colors: {[obj['color'] for obj in objects]}")
        
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
        
        # ARC color palette (0-9): 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=magenta, 7=orange, 8=light_blue, 9=brown
        # Look for ALL non-background colors as potentially interactive
        unique_colors = np.unique(grid).tolist()
        
        # Consider all colors except pure black (0) as potentially interactive
        # In ARC puzzles, any colored object could be clickable
        interactive_colors = [color for color in unique_colors if color != 0]
        
        logger.info(f"🎨 Detected colors in frame: {unique_colors}")
        logger.info(f"🎯 Treating as interactive colors: {interactive_colors}")
        
        for color in interactive_colors:
            mask = (grid == color)
            if np.any(mask).item():  # Convert to Python bool to avoid ambiguous truth value
                # Find connected components
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    # Lower minimum size threshold - some ARC elements can be small but important
                    if area >= 1:  # Even single pixels can be interactive in ARC
                        x, y, w, h = cv2.boundingRect(contour)
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = x + w//2, y + h//2
                        
                        # Higher confidence for larger, more distinct objects
                        # But don't exclude small objects entirely
                        base_confidence = 0.6  # Base confidence for any colored object
                        size_bonus = min(area / 50, 0.4)  # Up to 0.4 bonus for larger objects
                        
                        confidence = base_confidence + size_bonus
                        
                        interactive.append({
                            'color': color,
                            'area': area,
                            'centroid': (cx, cy),
                            'bbox': (x, y, w, h),
                            'confidence': confidence
                        })
        
        logger.info(f"🎯 Detected {len(interactive)} interactive elements: {[(elem['color'], elem['centroid'], f'{elem['confidence']:.2f}') for elem in interactive]}")
        
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

    
    def _convert_agi_strategy_to_actions(self, agi_strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert AGI solution strategy to executable actions."""
        actions = []
        
        if not agi_strategy:
            return actions
            
        # Extract primary action from AGI strategy
        primary_action = agi_strategy.get('primary_action')
        if primary_action and primary_action.get('action') == 'ACTION6':
            # AGI recommends a specific click action
            coords = primary_action.get('coordinates')
            if coords and coords != "calculated_rotation_center":  # Skip placeholder coordinates
                try:
                    if isinstance(coords, tuple) and len(coords) == 2:
                        x, y = coords
                    elif isinstance(coords, str) and ',' in coords:
                        x, y = map(int, coords.split(','))
                    else:
                        x, y = 5, 5  # Fallback coordinates
                    
                    actions.append({
                        'action': 'ACTION6',
                        'x': x,
                        'y': y,
                        'confidence': agi_strategy.get('confidence', 0.8),
                        'reason': f"AGI strategy: {agi_strategy.get('rationale', 'AGI-recommended action')}",
                        'agi_insight': True
                    })
                except Exception as e:
                    logger.warning(f"[AGI] Could not parse coordinates from AGI strategy: {e}")
        
        # Add backup plans from AGI
        backup_plans = agi_strategy.get('backup_plans', [])
        for i, backup in enumerate(backup_plans[:2]):  # Limit to top 2 backups
            backup_action = backup.get('action')
            if backup_action and backup_action.get('action') == 'ACTION6':
                coords = backup_action.get('coordinates')
                if coords and coords != "calculated_rotation_center":
                    try:
                        if isinstance(coords, tuple) and len(coords) == 2:
                            x, y = coords
                        elif isinstance(coords, str) and ',' in coords:
                            x, y = map(int, coords.split(','))
                        else:
                            x, y = 5, 5
                        
                        actions.append({
                            'action': 'ACTION6',
                            'x': x,
                            'y': y,
                            'confidence': backup.get('confidence', 0.6) * 0.8,  # Reduce confidence for backup
                            'reason': f"AGI backup plan {i+1}: {backup.get('rationale', 'AGI backup strategy')}",
                            'agi_insight': True
                        })
                    except Exception:
                        continue
        
        # If AGI strategy suggests exploration, add movement actions
        if agi_strategy.get('approach') == 'systematic_exploration':
            actions.extend([
                {'action': 'ACTION1', 'reason': 'AGI: systematic exploration (up)', 'confidence': 0.5, 'agi_insight': True},
                {'action': 'ACTION2', 'reason': 'AGI: systematic exploration (down)', 'confidence': 0.5, 'agi_insight': True},
                {'action': 'ACTION3', 'reason': 'AGI: systematic exploration (left)', 'confidence': 0.5, 'agi_insight': True},
                {'action': 'ACTION4', 'reason': 'AGI: systematic exploration (right)', 'confidence': 0.5, 'agi_insight': True}
            ])
        
        return actions
    
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
    async def get_penalty_aware_avoidance_scores(self, candidate_coordinates: List[Tuple[int, int]] = None, game_id: str = "unknown") -> Dict[Tuple[int, int], float]:
        """Get penalty-aware avoidance scores for coordinates by delegating to PositionTracker.

        Args:
            candidate_coordinates: list of (x,y) tuples to score. If None, uses tracked avoidance_scores keys.
            game_id: game identifier passed through to penalty system.
        Returns:
            Dict mapping (x,y) -> avoidance score (0.0-1.0)
        """
        # If no specific candidates given, return local avoidance_scores mapping
        try:
            if candidate_coordinates is None:
                # Return current local avoidance scores as a mapping
                return {tuple(map(int, k.split(','))): float(v) for k, v in self.avoidance_scores.items()}

            # Lazily ensure we have a PositionTracker instance
            if getattr(self, 'position_tracker', None) is None:
                try:
                    from src.vision.position_tracking.tracker import PositionTracker
                    self.position_tracker = PositionTracker()
                except Exception as ie:
                    logger.debug(f"Could not instantiate PositionTracker lazily: {ie}")
                    self.position_tracker = None

            if self.position_tracker is not None:
                try:
                    return await self.position_tracker.get_penalty_aware_avoidance_scores(candidate_coordinates, game_id)
                except Exception as e:
                    logger.error(f"PositionTracker failed to provide avoidance scores: {e}")

            # Fallback to local avoidance scores when PositionTracker unavailable or failed
            return {(x, y): float(self.avoidance_scores.get(f"{x},{y}", 0.0)) for x, y in (candidate_coordinates or [])}

        except Exception as e:
            logger.error(f"Failed to get penalty-aware avoidance scores: {e}")
            # Final fallback
            try:
                return {(x, y): float(self.avoidance_scores.get(f"{x},{y}", 0.0)) for x, y in (candidate_coordinates or [])}
            except Exception:
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

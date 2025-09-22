#!/usr/bin/env python3
"""
Systematic Button Discovery System

This system ensures that every object in the game frame is tested with Action 6
to identify which ones are buttons and which ones provide score increases.

Key Features:
- Exhaustive testing of all objects
- Priority system for buttons based on score increases
- Systematic exploration to ensure no object is missed
- Persistence of button knowledge across games
"""

import logging
import json
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from src.database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class ButtonType(Enum):
    """Types of buttons based on their effects."""
    SCORE_BUTTON = "score_button"  # Increases score
    ACTION_BUTTON = "action_button"  # Unlocks new actions
    VISUAL_BUTTON = "visual_button"  # Changes visual state
    UNKNOWN_BUTTON = "unknown_button"  # Has some effect but unclear
    NON_BUTTON = "non_button"  # No effect detected

@dataclass
class ButtonTestResult:
    """Result of testing a coordinate for button behavior."""
    coordinate: Tuple[int, int]
    test_count: int = 0
    score_increases: int = 0
    action_unlocks: int = 0
    visual_changes: int = 0
    total_score_change: float = 0.0
    button_type: ButtonType = ButtonType.NON_BUTTON
    confidence: float = 0.0
    last_tested: Optional[str] = None
    is_confirmed_button: bool = False
    priority_score: float = 0.0

class SystematicButtonDiscovery:
    """Systematic discovery and testing of buttons in the game."""
    
    def __init__(self):
        self.integration = get_system_integration()
        self.logger = logging.getLogger(__name__)
        
        # Button testing state
        self.tested_coordinates: Dict[Tuple[int, int], ButtonTestResult] = {}
        self.confirmed_buttons: Set[Tuple[int, int]] = set()
        self.pending_tests: deque = deque()  # Queue of coordinates to test
        self.current_game_id: Optional[str] = None
        
        # Testing parameters
        self.min_tests_per_object = 3  # Minimum tests before confirming button status
        self.exploration_phase = True  # Whether we're in systematic exploration phase
        self.exploration_complete = False
        
        # Button priority system
        self.button_priorities: Dict[Tuple[int, int], float] = {}
        self.score_button_priority = 1.0  # Highest priority
        self.action_button_priority = 0.8
        self.visual_button_priority = 0.6
        self.unknown_button_priority = 0.4
        self.non_button_priority = 0.1
        
    async def start_new_game(self, game_id: str, frame_data: List[List[int]]) -> None:
        """Initialize button discovery for a new game."""
        self.current_game_id = game_id
        self.exploration_phase = True
        self.exploration_complete = False
        self.tested_coordinates.clear()
        self.confirmed_buttons.clear()
        self.pending_tests.clear()
        self.button_priorities.clear()
        
        # Find all potential objects in the frame
        potential_objects = self._find_potential_objects(frame_data)
        
        # Add all objects to testing queue
        for coord in potential_objects:
            self.pending_tests.append(coord)
            self.tested_coordinates[coord] = ButtonTestResult(coordinate=coord)
        
        self.logger.info(f"🎯 BUTTON DISCOVERY STARTED for game {game_id}")
        self.logger.info(f"   Found {len(potential_objects)} potential objects to test")
        self.logger.info(f"   Testing queue initialized with {len(self.pending_tests)} coordinates")
    
    def _find_potential_objects(self, frame_data: List[List[int]]) -> List[Tuple[int, int]]:
        """Find all potential interactive objects in the frame."""
        objects = []
        
        if not frame_data or not isinstance(frame_data, list):
            return objects
        
        # Convert to numpy array for easier processing
        frame_array = np.array(frame_data)
        
        # Find all non-zero cells (potential objects)
        non_zero_coords = np.where(frame_array != 0)
        
        for y, x in zip(non_zero_coords[0], non_zero_coords[1]):
            # Only consider cells that might be interactive
            if frame_array[y, x] > 0:  # Non-zero values might be objects
                objects.append((int(x), int(y)))
        
        # Also check for patterns that might indicate buttons
        # Look for clusters of non-zero cells
        for y in range(len(frame_data)):
            for x in range(len(frame_data[y])):
                if frame_array[y, x] != 0:
                    # Check if this might be part of a larger object
                    if self._is_likely_object_center(frame_array, x, y):
                        objects.append((x, y))
        
        # Remove duplicates and sort for systematic testing
        objects = list(set(objects))
        objects.sort(key=lambda coord: (coord[1], coord[0]))  # Sort by row, then column
        
        return objects
    
    def _is_likely_object_center(self, frame_array: np.ndarray, x: int, y: int) -> bool:
        """Check if a coordinate is likely the center of an interactive object."""
        # Simple heuristic: check if there are non-zero cells around this position
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < frame_array.shape[0] and 0 <= nx < frame_array.shape[1]:
                    if frame_array[ny, nx] != 0:
                        return True
        return False
    
    async def get_next_test_coordinate(self) -> Optional[Tuple[int, int]]:
        """Get the next coordinate to test for button behavior."""
        if not self.exploration_phase:
            # If exploration is complete, prioritize confirmed buttons
            return self._get_priority_button()
        
        # During exploration, test untested objects first
        while self.pending_tests:
            coord = self.pending_tests.popleft()
            if coord not in self.tested_coordinates or self.tested_coordinates[coord].test_count == 0:
                return coord
        
        # If all objects have been tested once, mark exploration as complete
        if not self.exploration_complete:
            self.exploration_complete = True
            self.exploration_phase = False
            self.logger.info("🎯 EXPLORATION PHASE COMPLETE - All objects tested at least once")
            self.logger.info(f"   Confirmed buttons: {len(self.confirmed_buttons)}")
            
            # Log button summary
            await self._log_button_summary()
        
        # Return priority button for continued testing
        return self._get_priority_button()
    
    def _get_priority_button(self) -> Optional[Tuple[int, int]]:
        """Get the highest priority button for testing."""
        if not self.confirmed_buttons:
            return None
        
        # Sort by priority score (higher is better)
        sorted_buttons = sorted(
            self.confirmed_buttons,
            key=lambda coord: self.button_priorities.get(coord, 0.0),
            reverse=True
        )
        
        return sorted_buttons[0] if sorted_buttons else None
    
    async def record_test_result(
        self, 
        coordinate: Tuple[int, int], 
        score_change: float,
        action_unlocks: int = 0,
        visual_changes: int = 0,
        frame_changed: bool = False
    ) -> None:
        """Record the result of testing a coordinate."""
        if coordinate not in self.tested_coordinates:
            self.tested_coordinates[coordinate] = ButtonTestResult(coordinate=coordinate)
        
        result = self.tested_coordinates[coordinate]
        result.test_count += 1
        result.last_tested = str(self.test_count) if hasattr(self, 'test_count') else "unknown"
        
        # Record changes
        if score_change > 0:
            result.score_increases += 1
            result.total_score_change += score_change
        
        if action_unlocks > 0:
            result.action_unlocks += action_unlocks
        
        if visual_changes > 0 or frame_changed:
            result.visual_changes += 1
        
        # Update button type and confidence
        await self._update_button_classification(result)
        
        # Update priority
        self._update_button_priority(coordinate, result)
        
        # Log significant discoveries
        if result.is_confirmed_button and result.test_count == self.min_tests_per_object:
            self.confirmed_buttons.add(coordinate)
            self.logger.info(f"✅ BUTTON CONFIRMED: {coordinate} - {result.button_type.value}")
            self.logger.info(f"   Score increases: {result.score_increases}, Total score change: {result.total_score_change:.2f}")
    
    async def _update_button_classification(self, result: ButtonTestResult) -> None:
        """Update button classification based on test results."""
        if result.test_count < self.min_tests_per_object:
            # Not enough tests yet
            return
        
        # Determine button type based on effects
        if result.score_increases > 0:
            result.button_type = ButtonType.SCORE_BUTTON
            result.confidence = min(1.0, result.score_increases / result.test_count)
        elif result.action_unlocks > 0:
            result.button_type = ButtonType.ACTION_BUTTON
            result.confidence = min(1.0, result.action_unlocks / result.test_count)
        elif result.visual_changes > 0:
            result.button_type = ButtonType.VISUAL_BUTTON
            result.confidence = min(1.0, result.visual_changes / result.test_count)
        else:
            result.button_type = ButtonType.NON_BUTTON
            result.confidence = 0.0
        
        # Confirm as button if it has any positive effects
        result.is_confirmed_button = result.button_type != ButtonType.NON_BUTTON
    
    def _update_button_priority(self, coordinate: Tuple[int, int], result: ButtonTestResult) -> None:
        """Update button priority based on its effectiveness."""
        if not result.is_confirmed_button:
            self.button_priorities[coordinate] = self.non_button_priority
            return
        
        # Base priority by button type
        if result.button_type == ButtonType.SCORE_BUTTON:
            base_priority = self.score_button_priority
        elif result.button_type == ButtonType.ACTION_BUTTON:
            base_priority = self.action_button_priority
        elif result.button_type == ButtonType.VISUAL_BUTTON:
            base_priority = self.visual_button_priority
        else:
            base_priority = self.unknown_button_priority
        
        # Boost priority based on effectiveness
        effectiveness_boost = 0.0
        if result.total_score_change > 0:
            effectiveness_boost += min(0.3, result.total_score_change / 10.0)  # Cap at 0.3
        if result.score_increases > 0:
            effectiveness_boost += min(0.2, result.score_increases * 0.1)  # Cap at 0.2
        
        self.button_priorities[coordinate] = min(1.0, base_priority + effectiveness_boost)
    
    async def _log_button_summary(self) -> None:
        """Log a summary of discovered buttons."""
        if not self.tested_coordinates:
            return
        
        # Count by button type
        type_counts = defaultdict(int)
        total_score_change = 0.0
        
        for result in self.tested_coordinates.values():
            if result.is_confirmed_button:
                type_counts[result.button_type.value] += 1
                total_score_change += result.total_score_change
        
        self.logger.info("📊 BUTTON DISCOVERY SUMMARY:")
        self.logger.info(f"   Total objects tested: {len(self.tested_coordinates)}")
        self.logger.info(f"   Confirmed buttons: {len(self.confirmed_buttons)}")
        self.logger.info(f"   Score buttons: {type_counts['score_button']}")
        self.logger.info(f"   Action buttons: {type_counts['action_button']}")
        self.logger.info(f"   Visual buttons: {type_counts['visual_button']}")
        self.logger.info(f"   Total score change: {total_score_change:.2f}")
        
        # Log top buttons by priority
        top_buttons = sorted(
            self.confirmed_buttons,
            key=lambda coord: self.button_priorities.get(coord, 0.0),
            reverse=True
        )[:5]
        
        if top_buttons:
            self.logger.info("🏆 TOP PRIORITY BUTTONS:")
            for i, coord in enumerate(top_buttons, 1):
                result = self.tested_coordinates[coord]
                priority = self.button_priorities.get(coord, 0.0)
                self.logger.info(f"   {i}. {coord} - {result.button_type.value} (priority: {priority:.2f})")
    
    def get_button_suggestions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get button suggestions for action selection."""
        suggestions = []
        
        # Get confirmed buttons sorted by priority
        sorted_buttons = sorted(
            self.confirmed_buttons,
            key=lambda coord: self.button_priorities.get(coord, 0.0),
            reverse=True
        )
        
        for coord in sorted_buttons[:limit]:
            result = self.tested_coordinates[coord]
            priority = self.button_priorities.get(coord, 0.0)
            
            suggestions.append({
                'action': 'ACTION6',
                'x': coord[0],
                'y': coord[1],
                'confidence': result.confidence,
                'priority': priority,
                'button_type': result.button_type.value,
                'reason': f"Confirmed {result.button_type.value} (priority: {priority:.2f})",
                'source': 'button_discovery'
            })
        
        return suggestions
    
    def is_exploration_complete(self) -> bool:
        """Check if systematic exploration is complete."""
        return self.exploration_complete
    
    def get_exploration_progress(self) -> Dict[str, Any]:
        """Get exploration progress information."""
        total_objects = len(self.tested_coordinates)
        tested_objects = sum(1 for result in self.tested_coordinates.values() if result.test_count > 0)
        confirmed_buttons = len(self.confirmed_buttons)
        
        return {
            'total_objects': total_objects,
            'tested_objects': tested_objects,
            'confirmed_buttons': confirmed_buttons,
            'exploration_complete': self.exploration_complete,
            'progress_percentage': (tested_objects / total_objects * 100) if total_objects > 0 else 0
        }

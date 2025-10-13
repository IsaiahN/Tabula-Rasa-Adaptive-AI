"""
Intelligent action selection for the Continuous Learning Loop.
This module handles smart action selection with a focus on using appropriate
actions for different game mechanics (movement vs targeting).
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import random

logger = logging.getLogger(__name__)

def choose_smart_action(available_actions: List[int], game_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Choose a smart action based on available actions and game state."""
    try:
        # Always check for movement actions first (1-4)
        movement_actions = [a for a in available_actions if 1 <= a <= 4]
        if movement_actions:
            # Pick movement direction intelligently
            frame = game_response.get('frame', [])
            if frame:
                # Try to detect bar position and choose direction
                bar_position = detect_bar_position(frame[0]) if frame else None
                if bar_position is not None:
                    y_pos = bar_position[1]
                    # If near bottom, prioritize moving up
                    if y_pos > len(frame[0]) * 0.7:
                        return {'id': 1}  # Move up
                    # If near top, prioritize moving down 
                    elif y_pos < len(frame[0]) * 0.3:
                        return {'id': 2}  # Move down
            
            # If no specific direction needed, cycle through movements
            return {'id': random.choice(movement_actions)}
        
        # Only use ACTION6 for targeting if no movement actions available
        if 6 in available_actions:
            # Only use ACTION6 with smart targeting
            frame_data = game_response.get('frame', [])
            if frame_data and len(frame_data) > 0:
                target = find_target_coordinates(frame_data[0])
                if target:
                    return {
                        'id': 6,
                        'x': target[0],
                        'y': target[1]
                    }
        
        # Fallback to any available simple action
        simple_actions = [a for a in available_actions if a in [5, 7]]
        if simple_actions:
            return {'id': random.choice(simple_actions)}
        
        return None
        
    except Exception as e:
        logger.error(f"Error choosing action: {e}")
        return None

def detect_bar_position(frame: List[List[int]]) -> Optional[Tuple[int, int]]:
    """
    Detect position of a vertical bar in the frame.
    Returns (x, y) coordinates of bar center, or None if not found.
    """
    if not frame or not frame[0]:
        return None
        
    height = len(frame)
    width = len(frame[0])
    
    # Look for vertical sequences of same color
    for x in range(width):
        vertical_runs = []
        current_run = []
        
        for y in range(height):
            if not current_run:
                current_run = [(x, y)]
            elif frame[y][x] == frame[current_run[0][1]][current_run[0][0]]:
                current_run.append((x, y))
            else:
                if len(current_run) > 3:  # Minimum bar height
                    vertical_runs.append(current_run)
                current_run = [(x, y)]
                
        # Check last run
        if len(current_run) > 3:
            vertical_runs.append(current_run)
            
        # Return center of longest vertical run
        if vertical_runs:
            longest_run = max(vertical_runs, key=len)
            center_y = sum(y for _, y in longest_run) // len(longest_run)
            return (x, center_y)
            
    return None

def find_target_coordinates(frame: List[List[int]]) -> Optional[Tuple[int, int]]:
    """Find target coordinates for ACTION6 based on frame analysis."""
    if not frame or not frame[0]:
        return None
        
    height = len(frame)
    width = len(frame[0])
    
    # Look for distinct objects or patterns
    for y in range(height):
        for x in range(width):
            # Check for interesting features that might be worth targeting
            if is_interesting_point(frame, x, y):
                return (x, y)
                
    return None

def is_interesting_point(frame: List[List[int]], x: int, y: int) -> bool:
    """Determine if a point is interesting for targeting."""
    if not frame or not frame[0]:
        return False
        
    height = len(frame)
    width = len(frame[0])
    
    # Skip edge coordinates
    if x == 0 or x == width-1 or y == 0 or y == height-1:
        return False
        
    current_value = frame[y][x]
    
    # Check if point is different from surroundings
    neighbors = [
        frame[y-1][x], frame[y+1][x],  # Above and below
        frame[y][x-1], frame[y][x+1],  # Left and right
    ]
    
    return current_value != 0 and any(n != current_value for n in neighbors)
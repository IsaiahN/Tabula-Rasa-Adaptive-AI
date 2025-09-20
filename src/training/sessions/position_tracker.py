"""
Position Tracker

Tracks agent positions and movement patterns during training sessions.
"""

import logging
from typing import List, Tuple, Set, Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)

class PositionTracker:
    """Tracks agent positions and movement patterns."""
    
    def __init__(self, max_history: int = 1000):
        self.visited_positions: Set[Tuple[int, int]] = set()
        self.movement_history: deque = deque(maxlen=max_history)
        self.current_position: Optional[Tuple[int, int]] = None
        self.last_position: Optional[Tuple[int, int]] = None
        self.stuck_threshold = 5  # Consider stuck after 5 moves in same area
        self.stuck_positions: Dict[Tuple[int, int], int] = {}
    
    def update_position(self, position: Tuple[int, int]) -> None:
        """Update current position and track movement."""
        try:
            self.last_position = self.current_position
            self.current_position = position
            
            # Add to visited positions
            self.visited_positions.add(position)
            
            # Track movement
            if self.last_position is not None:
                movement = {
                    'from': self.last_position,
                    'to': position,
                    'timestamp': self._get_timestamp()
                }
                self.movement_history.append(movement)
            
            # Track stuck positions
            self._update_stuck_tracking(position)
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def is_stuck(self) -> bool:
        """Check if agent is stuck in the same area."""
        if self.current_position is None:
            return False
        
        return self.stuck_positions.get(self.current_position, 0) >= self.stuck_threshold
    
    def get_movement_direction(self) -> Optional[Tuple[int, int]]:
        """Get the direction of the last movement."""
        if self.last_position is None or self.current_position is None:
            return None
        
        return (
            self.current_position[0] - self.last_position[0],
            self.current_position[1] - self.last_position[1]
        )
    
    def get_visited_positions(self) -> Set[Tuple[int, int]]:
        """Get all visited positions."""
        return self.visited_positions.copy()
    
    def get_movement_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get movement history."""
        if limit is None:
            return list(self.movement_history)
        return list(self.movement_history)[-limit:]
    
    def get_stuck_positions(self) -> Dict[Tuple[int, int], int]:
        """Get positions where agent got stuck."""
        return {pos: count for pos, count in self.stuck_positions.items() if count >= self.stuck_threshold}
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.visited_positions.clear()
        self.movement_history.clear()
        self.current_position = None
        self.last_position = None
        self.stuck_positions.clear()
        logger.debug("Position tracker reset")
    
    def get_position_stats(self) -> Dict[str, Any]:
        """Get position tracking statistics."""
        return {
            'total_positions_visited': len(self.visited_positions),
            'total_movements': len(self.movement_history),
            'current_position': self.current_position,
            'last_position': self.last_position,
            'is_stuck': self.is_stuck(),
            'stuck_positions_count': len(self.get_stuck_positions()),
            'movement_direction': self.get_movement_direction()
        }
    
    def _update_stuck_tracking(self, position: Tuple[int, int]) -> None:
        """Update stuck position tracking."""
        # Increment count for current position
        self.stuck_positions[position] = self.stuck_positions.get(position, 0) + 1
        
        # Decrease count for other positions (decay)
        for pos in list(self.stuck_positions.keys()):
            if pos != position:
                self.stuck_positions[pos] = max(0, self.stuck_positions[pos] - 1)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()

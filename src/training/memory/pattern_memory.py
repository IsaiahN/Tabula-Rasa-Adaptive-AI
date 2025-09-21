"""
Pattern Memory Manager

Handles pattern learning, coordinate tracking, and boundary detection.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ..core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class PatternMemoryManager:
    """Manages pattern learning and coordinate tracking."""
    
    def __init__(self, memory_manager: 'MemoryManager'):
        self.memory_manager = memory_manager
        self.pattern_confidence_threshold = 0.7
        self.boundary_detection_threshold = 0.5
    
    def update_coordinate_patterns(self, game_id: str, coordinates: List[Tuple[int, int]], 
                                 success: bool) -> None:
        """Update coordinate pattern tracking."""
        try:
            patterns = self.memory_manager.get_memory_key('coordinate_patterns', {})
            if game_id not in patterns:
                patterns[game_id] = {
                    'successful_coordinates': [],
                    'failed_coordinates': [],
                    'coordinate_sequences': [],
                    'success_rate': 0.0
                }
            
            if success:
                patterns[game_id]['successful_coordinates'].extend(coordinates)
            else:
                patterns[game_id]['failed_coordinates'].extend(coordinates)
            
            # Update success rate
            total_attempts = len(patterns[game_id]['successful_coordinates']) + len(patterns[game_id]['failed_coordinates'])
            if total_attempts > 0:
                patterns[game_id]['success_rate'] = len(patterns[game_id]['successful_coordinates']) / total_attempts
            
            self.memory_manager.update_memory_key('coordinate_patterns', patterns)
            
        except Exception as e:
            logger.error(f"Error updating coordinate patterns: {e}")
    
    def get_successful_coordinates(self, game_id: str) -> List[Tuple[int, int]]:
        """Get successful coordinates for a game."""
        patterns = self.memory_manager.get_memory_key('coordinate_patterns', {})
        return patterns.get(game_id, {}).get('successful_coordinates', [])
    
    def update_boundary_detection(self, game_id: str, boundary_data: Dict[str, Any]) -> None:
        """Update boundary detection data."""
        try:
            universal_boundary = self.memory_manager.get_memory_key('universal_boundary_detection', {})
            universal_boundary[game_id] = boundary_data
            self.memory_manager.update_memory_key('universal_boundary_detection', universal_boundary)
            
        except Exception as e:
            logger.error(f"Error updating boundary detection: {e}")
    
    def get_boundary_data(self, game_id: str) -> Dict[str, Any]:
        """Get boundary detection data for a game."""
        universal_boundary = self.memory_manager.get_memory_key('universal_boundary_detection', {})
        return universal_boundary.get(game_id, {})
    
    def update_learned_patterns(self, pattern_type: str, pattern_data: Dict[str, Any]) -> None:
        """Update learned patterns."""
        try:
            patterns = self.memory_manager.get_memory_key('learned_patterns', {})
            if pattern_type not in patterns:
                patterns[pattern_type] = []
            
            patterns[pattern_type].append({
                'data': pattern_data,
                'confidence': pattern_data.get('confidence', 0.0),
                'timestamp': self._get_timestamp()
            })
            
            # Keep only high-confidence patterns
            patterns[pattern_type] = [
                p for p in patterns[pattern_type] 
                if p['confidence'] >= self.pattern_confidence_threshold
            ]
            
            self.memory_manager.update_memory_key('learned_patterns', patterns)
            
        except Exception as e:
            logger.error(f"Error updating learned patterns: {e}")
    
    def get_learned_patterns(self, pattern_type: str) -> List[Dict[str, Any]]:
        """Get learned patterns of a specific type."""
        patterns = self.memory_manager.get_memory_key('learned_patterns', {})
        return patterns.get(pattern_type, [])
    
    def update_directional_system(self, game_id: str, direction_data: Dict[str, Any]) -> None:
        """Update directional system tracking."""
        try:
            universal_boundary = self.memory_manager.get_memory_key('universal_boundary_detection', {})
            if 'directional_systems' not in universal_boundary:
                universal_boundary['directional_systems'] = {}
            
            if game_id not in universal_boundary['directional_systems']:
                universal_boundary['directional_systems'][game_id] = {
                    'current_direction': {},
                    'direction_history': []
                }
            
            universal_boundary['directional_systems'][game_id].update(direction_data)
            self.memory_manager.update_memory_key('universal_boundary_detection', universal_boundary)
            
        except Exception as e:
            logger.error(f"Error updating directional system: {e}")
    
    def get_directional_system(self, game_id: str) -> Dict[str, Any]:
        """Get directional system data for a game."""
        universal_boundary = self.memory_manager.get_memory_key('universal_boundary_detection', {})
        directional_systems = universal_boundary.get('directional_systems', {})
        return directional_systems.get(game_id, {
            'current_direction': {},
            'direction_history': []
        })
    
    def update_stuck_patterns(self, game_id: str, stuck_data: Dict[str, Any]) -> None:
        """Update stuck pattern tracking."""
        try:
            universal_boundary = self.memory_manager.get_memory_key('universal_boundary_detection', {})
            if 'stuck_patterns' not in universal_boundary:
                universal_boundary['stuck_patterns'] = {}
            
            universal_boundary['stuck_patterns'][game_id] = stuck_data
            self.memory_manager.update_memory_key('universal_boundary_detection', universal_boundary)
            
        except Exception as e:
            logger.error(f"Error updating stuck patterns: {e}")
    
    def get_stuck_patterns(self, game_id: str) -> Dict[str, Any]:
        """Get stuck patterns for a game."""
        universal_boundary = self.memory_manager.get_memory_key('universal_boundary_detection', {})
        stuck_patterns = universal_boundary.get('stuck_patterns', {})
        return stuck_patterns.get(game_id, {})
    
    def generate_pattern_insights(self) -> List[str]:
        """Generate insights from pattern memory."""
        insights = []
        
        try:
            # Analyze coordinate patterns
            patterns = self.memory_manager.get_memory_key('coordinate_patterns', {})
            if patterns:
                total_games = len(patterns)
                high_success_games = sum(1 for p in patterns.values() if p.get('success_rate', 0) > 0.8)
                insights.append(f"Coordinate patterns: {high_success_games}/{total_games} games with high success rate")
            
            # Analyze learned patterns
            learned_patterns = self.memory_manager.get_memory_key('learned_patterns', {})
            if learned_patterns:
                total_patterns = sum(len(patterns) for patterns in learned_patterns.values())
                insights.append(f"Learned patterns: {total_patterns} high-confidence patterns")
            
            # Analyze boundary detection
            universal_boundary = self.memory_manager.get_memory_key('universal_boundary_detection', {})
            if universal_boundary:
                boundary_games = len([k for k in universal_boundary.keys() if k != 'directional_systems' and k != 'stuck_patterns'])
                insights.append(f"Boundary detection active for {boundary_games} games")
            
        except Exception as e:
            logger.error(f"Error generating pattern insights: {e}")
        
        return insights
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def cleanup_old_patterns(self, max_age_days: int = 7) -> int:
        """Clean up old patterns to prevent memory bloat."""
        cleaned_count = 0
        
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            # Clean up learned patterns
            patterns = self.memory_manager.get_memory_key('learned_patterns', {})
            for pattern_type in list(patterns.keys()):
                original_count = len(patterns[pattern_type])
                patterns[pattern_type] = [
                    p for p in patterns[pattern_type]
                    if datetime.fromisoformat(p['timestamp']) > cutoff_date
                ]
                cleaned_count += original_count - len(patterns[pattern_type])
            
            self.memory_manager.update_memory_key('learned_patterns', patterns)
            
        except Exception as e:
            logger.error(f"Error cleaning up old patterns: {e}")
        
        return cleaned_count

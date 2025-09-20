"""
Memory Manager

Central memory management system for the training loop.
Handles initialization, storage, and retrieval of all memory types.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryManager:
    """Central memory management system."""
    
    def __init__(self):
        self.available_actions_memory = self._initialize_available_actions_memory()
        self.memory_compression_active = False
        self.memory_hierarchy_status = {}
        self.memory_consolidation_status = {}
    
    def _initialize_available_actions_memory(self) -> Dict[str, Any]:
        """Initialize the available actions memory with all required keys and proper structure."""
        return {
            'current_game_id': None,
            'current_actions': [],
            'action_history': [],
            'action_effectiveness': {},
            'action_relevance_scores': {},
            'action_sequences': [],
            'winning_action_sequences': [],
            'coordinate_patterns': {},
            'learned_patterns': {},
            'action_learning_stats': {
                'total_observations': 0,
                'pattern_confidence_threshold': 0.7,
                'movements_tracked': 0,
                'effects_catalogued': 0,
                'game_contexts_learned': 0
            },
            'action_semantic_mapping': {},
            'sequence_in_progress': [],
            'initial_actions': [],
            'action_stagnation': {},
            'universal_boundary_detection': {
                'boundary_data': {},
                'coordinate_attempts': {},
                'action_coordinate_history': {},
                'stuck_patterns': {},
                'success_zone_mapping': {},
                'danger_zones': {},
                'coordinate_clusters': {},
                'directional_systems': {
                    6: {'current_direction': {}, 'direction_history': {}}
                },
                'current_direction': {},
                'last_coordinates': {},
                'stuck_count': {},
                'coordinate_attempts': {}
            },
            'action6_boundary_detection': {
                'boundary_data': {},
                'coordinate_attempts': {},
                'last_coordinates': {},
                'stuck_count': {},
                'current_direction': {}
            }
        }
    
    def ensure_available_actions_memory(self) -> None:
        """Ensure available_actions_memory is properly initialized with all required keys."""
        if not hasattr(self, 'available_actions_memory'):
            self.available_actions_memory = self._initialize_available_actions_memory()
            return
        
        # Ensure all required keys exist
        required_keys = self._initialize_available_actions_memory()
        for key, default_value in required_keys.items():
            if key not in self.available_actions_memory:
                self.available_actions_memory[key] = default_value.copy() if isinstance(default_value, dict) else default_value
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status."""
        return {
            'available_actions_memory_keys': list(self.available_actions_memory.keys()),
            'memory_compression_active': self.memory_compression_active,
            'hierarchy_status': self.memory_hierarchy_status,
            'consolidation_status': self.memory_consolidation_status
        }
    
    def reset_memory(self) -> None:
        """Reset all memory to initial state."""
        self.available_actions_memory = self._initialize_available_actions_memory()
        self.memory_compression_active = False
        self.memory_hierarchy_status = {}
        self.memory_consolidation_status = {}
        logger.info("Memory reset to initial state")
    
    def update_memory_key(self, key: str, value: Any) -> None:
        """Update a specific memory key."""
        if key in self.available_actions_memory:
            self.available_actions_memory[key] = value
        else:
            logger.warning(f"Attempted to update unknown memory key: {key}")
    
    def get_memory_key(self, key: str, default: Any = None) -> Any:
        """Get a specific memory key value."""
        return self.available_actions_memory.get(key, default)
    
    def is_memory_system_healthy(self) -> bool:
        """Check if the memory system is in a healthy state."""
        try:
            # Check if all required keys exist
            required_keys = self._initialize_available_actions_memory()
            for key in required_keys:
                if key not in self.available_actions_memory:
                    logger.warning(f"Missing required memory key: {key}")
                    return False
            
            # Check if critical memory structures are valid
            if not isinstance(self.available_actions_memory.get('action_learning_stats'), dict):
                logger.warning("Invalid action_learning_stats structure")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking memory system health: {e}")
            return False

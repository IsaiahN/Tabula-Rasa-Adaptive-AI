"""
Action Memory Manager

Handles action-specific memory including effectiveness tracking,
sequences, and pattern learning.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ..core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class ActionMemoryManager:
    """Manages action-specific memory and learning."""
    
    def __init__(self, memory_manager: 'MemoryManager'):
        self.memory_manager = memory_manager
        self.action_effectiveness_threshold = 0.1
        self.sequence_learning_threshold = 0.7
    
    async def update_action_effectiveness(self, action: Dict[str, Any], effectiveness: float) -> None:
        """Update action effectiveness tracking and persist to DB."""
        action_id = self._get_action_id(action)
        if action_id:
            current_effectiveness = self.memory_manager.get_memory_key('action_effectiveness', {})
            current_effectiveness[action_id] = effectiveness
            self.memory_manager.update_memory_key('action_effectiveness', current_effectiveness)
            # Persist to database
            try:
                from src.database.api import get_database
                db = get_database()
                # Use a generic game_id for now, or extract from action if available
                game_id = action.get('game_id', 'unknown')
                action_number = action.get('id', 0)
                await db.update_action_effectiveness(game_id, action_number, 1, int(effectiveness > 0.5), effectiveness)
            except Exception as e:
                logger.error(f"Failed to persist action effectiveness: {e}")
    
    def get_action_effectiveness(self, action: Dict[str, Any]) -> float:
        """Get effectiveness score for an action."""
        action_id = self._get_action_id(action)
        if action_id:
            effectiveness = self.memory_manager.get_memory_key('action_effectiveness', {})
            return effectiveness.get(action_id, 0.0)
        return 0.0
    
    def add_action_sequence(self, sequence: List[Dict[str, Any]], effectiveness: float) -> None:
        """Add an action sequence to memory."""
        if effectiveness > self.sequence_learning_threshold:
            sequences = self.memory_manager.get_memory_key('action_sequences', [])
            sequences.append({
                'sequence': sequence,
                'effectiveness': effectiveness,
                'timestamp': self._get_timestamp()
            })
            self.memory_manager.update_memory_key('action_sequences', sequences)
    
    def get_winning_sequences(self) -> List[Dict[str, Any]]:
        """Get all winning action sequences."""
        return self.memory_manager.get_memory_key('winning_action_sequences', [])
    
    async def add_winning_sequence(self, game_id: str, sequence: List[int]) -> None:
        """Add a winning sequence to memory and persist to DB."""
        winning_sequences = self.memory_manager.get_memory_key('winning_action_sequences', [])
        winning_sequences.append({
            'game_id': game_id,
            'sequence': sequence,
            'timestamp': self._get_timestamp()
        })
        self.memory_manager.update_memory_key('winning_action_sequences', winning_sequences)
        # Persist to database and log errors
        try:
            from src.database.persistence_helpers import persist_winning_sequence
            await persist_winning_sequence(game_id, sequence)
        except Exception as e:
            logger.error(f"Failed to persist winning sequence: {e}")
    
    def update_action_learning_stats(self, stat_name: str, increment: int = 1) -> None:
        """Update action learning statistics."""
        stats = self.memory_manager.get_memory_key('action_learning_stats', {})
        stats[stat_name] = stats.get(stat_name, 0) + increment
        self.memory_manager.update_memory_key('action_learning_stats', stats)
    
    def get_effective_actions(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get actions above effectiveness threshold."""
        if threshold is None:
            threshold = self.action_effectiveness_threshold
        
        effectiveness = self.memory_manager.get_memory_key('action_effectiveness', {})
        return [
            {'action_id': action_id, 'effectiveness': eff}
            for action_id, eff in effectiveness.items()
            if eff >= threshold
        ]
    
    def strengthen_action_memory(self, action: Dict[str, Any], context_memories: List[Dict[str, Any]]) -> None:
        """Strengthen action memory with context."""
        try:
            action_id = self._get_action_id(action)
            if not action_id:
                return
            
            # Update effectiveness based on context
            current_effectiveness = self.get_action_effectiveness(action)
            context_boost = len(context_memories) * 0.1
            new_effectiveness = min(current_effectiveness + context_boost, 1.0)
            
            self.update_action_effectiveness(action, new_effectiveness)
            
            # Update learning stats
            self.update_action_learning_stats('effects_catalogued')
            
            logger.debug(f"Strengthened memory for action {action_id} with {len(context_memories)} context memories")
            
        except Exception as e:
            logger.error(f"Error strengthening action memory: {e}")
    
    def _get_action_id(self, action: Dict[str, Any]) -> Optional[str]:
        """Generate a unique ID for an action."""
        try:
            # Create a hashable representation of the action
            action_type = action.get('type', 'unknown')
            coordinates = action.get('coordinates', [])
            return f"{action_type}_{hash(tuple(coordinates))}"
        except Exception as e:
            logger.error(f"Error generating action ID: {e}")
            return None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_memory_insights(self) -> List[str]:
        """Generate insights from action memory patterns."""
        insights = []
        
        try:
            # Analyze effectiveness patterns
            effectiveness = self.memory_manager.get_memory_key('action_effectiveness', {})
            if effectiveness:
                avg_effectiveness = sum(effectiveness.values()) / len(effectiveness)
                insights.append(f"Average action effectiveness: {avg_effectiveness:.2f}")
            
            # Analyze sequence patterns
            sequences = self.memory_manager.get_memory_key('action_sequences', [])
            if sequences:
                high_eff_sequences = [s for s in sequences if s.get('effectiveness', 0) > 0.8]
                insights.append(f"High effectiveness sequences: {len(high_eff_sequences)}/{len(sequences)}")
            
            # Analyze learning stats
            stats = self.memory_manager.get_memory_key('action_learning_stats', {})
            if stats:
                total_observations = stats.get('total_observations', 0)
                insights.append(f"Total action observations: {total_observations}")
            
        except Exception as e:
            logger.error(f"Error generating memory insights: {e}")
        
        return insights

"""
ARC Action Mapper

Maps between ARC actions and Adaptive Learning Agent actions.
"""

import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ActionMapping:
    """Represents a mapping between ARC and agent actions."""
    arc_action: str
    agent_action: torch.Tensor
    action_type: str
    description: str

class ARCActionMapper:
    """
    Enhanced ARC Action Mapper with coordinate-aware pathway learning integration.
    Maps between ARC actions and Adaptive Learning Agent actions with intelligent
    coordinate selection and pathway learning.
    """
    
    def __init__(self, game_id: str = None):
        self.game_id = game_id or "default"
        
        # Initialize action mappings
        self.action_mapping = self._initialize_action_mappings()
        
        # Coordinate strategies for ACTION6
        self.coordinate_strategies = ['center_start', 'corner_exploration', 'edge_scanning', 'random_intelligent']
        self.current_strategy = 'center_start'
        
        # Action history for learning
        self.action_history = []
        self.successful_actions = []
    
    def _initialize_action_mappings(self) -> Dict[str, torch.Tensor]:
        """Initialize action mappings between ARC and agent actions."""
        return {
            'RESET': torch.tensor([0.0, 0.0, 0.0]),
            'ACTION1': torch.tensor([1.0, 0.0, 0.0]),
            'ACTION2': torch.tensor([0.0, 1.0, 0.0]),
            'ACTION3': torch.tensor([0.0, 0.0, 1.0]),
            'ACTION4': torch.tensor([-1.0, 0.0, 0.0]),
            'ACTION5': torch.tensor([0.0, -1.0, 0.0]),
            'ACTION6': torch.tensor([0.0, 0.0, -1.0]),  # Enhanced with coordinates
            'ACTION7': torch.tensor([0.5, 0.5, 0.0]),
        }
    
    def arc_to_agent_action(self, arc_action: str, x: int = 0, y: int = 0, 
                           frame_data: dict = None) -> torch.Tensor:
        """Convert ARC action to agent action with coordinate analysis."""
        try:
            # Get base action
            base_action = self.action_mapping.get(arc_action, torch.zeros(3))
            
            # Handle coordinate-based actions
            if arc_action == 'ACTION6' and frame_data:
                enhanced_action = self._enhance_coordinate_action(base_action, x, y, frame_data)
                return enhanced_action
            
            # Record action for learning
            self._record_action(arc_action, base_action, x, y, frame_data)
            
            return base_action
            
        except Exception as e:
            logger.error(f"Error converting ARC action: {e}")
            return torch.zeros(3)
    
    def agent_to_arc_action(self, agent_action: torch.Tensor) -> str:
        """Convert agent action to ARC action."""
        try:
            # Find closest matching ARC action
            best_match = None
            best_distance = float('inf')
            
            for arc_action, arc_tensor in self.action_mapping.items():
                distance = torch.norm(agent_action - arc_tensor).item()
                if distance < best_distance:
                    best_distance = distance
                    best_match = arc_action
            
            return best_match or 'RESET'
            
        except Exception as e:
            logger.error(f"Error converting agent action: {e}")
            return 'RESET'
    
    def _enhance_coordinate_action(self, base_action: torch.Tensor, x: int, y: int, 
                                 frame_data: dict) -> torch.Tensor:
        """Enhance coordinate-based actions with intelligent selection."""
        try:
            # Get frame dimensions
            frame_height = frame_data.get('height', 64)
            frame_width = frame_data.get('width', 64)
            
            # Normalize coordinates
            norm_x = x / max(frame_width, 1)
            norm_y = y / max(frame_height, 1)
            
            # Create enhanced action with coordinate information
            enhanced_action = torch.cat([
                base_action,
                torch.tensor([norm_x, norm_y])
            ])
            
            return enhanced_action
            
        except Exception as e:
            logger.error(f"Error enhancing coordinate action: {e}")
            return base_action
    
    def _record_action(self, arc_action: str, agent_action: torch.Tensor, 
                      x: int, y: int, frame_data: dict = None):
        """Record action for learning purposes."""
        try:
            action_record = {
                'arc_action': arc_action,
                'agent_action': agent_action.tolist(),
                'coordinates': (x, y),
                'timestamp': self._get_timestamp(),
                'frame_data': frame_data
            }
            
            self.action_history.append(action_record)
            
            # Keep only recent history
            if len(self.action_history) > 1000:
                self.action_history = self.action_history[-500:]
                
        except Exception as e:
            logger.error(f"Error recording action: {e}")
    
    def update_action_success(self, arc_action: str, success: bool, 
                            reward: float = 0.0):
        """Update action success for learning."""
        try:
            if success:
                self.successful_actions.append({
                    'arc_action': arc_action,
                    'reward': reward,
                    'timestamp': self._get_timestamp()
                })
                
                # Keep only recent successful actions
                if len(self.successful_actions) > 500:
                    self.successful_actions = self.successful_actions[-250:]
                    
        except Exception as e:
            logger.error(f"Error updating action success: {e}")
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about action usage and success."""
        try:
            if not self.action_history:
                return {'total_actions': 0, 'success_rate': 0.0}
            
            # Count actions
            action_counts = {}
            for record in self.action_history:
                action = record['arc_action']
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Calculate success rate
            total_actions = len(self.action_history)
            successful_actions = len(self.successful_actions)
            success_rate = successful_actions / max(total_actions, 1)
            
            return {
                'total_actions': total_actions,
                'successful_actions': successful_actions,
                'success_rate': success_rate,
                'action_counts': action_counts,
                'most_used_action': max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else 'none'
            }
            
        except Exception as e:
            logger.error(f"Error getting action statistics: {e}")
            return {'error': str(e)}
    
    def get_coordinate_strategy(self) -> str:
        """Get current coordinate strategy."""
        return self.current_strategy
    
    def set_coordinate_strategy(self, strategy: str):
        """Set coordinate strategy."""
        if strategy in self.coordinate_strategies:
            self.current_strategy = strategy
            logger.info(f"Set coordinate strategy to: {strategy}")
        else:
            logger.warning(f"Unknown coordinate strategy: {strategy}")
    
    def get_available_actions(self) -> List[str]:
        """Get list of available ARC actions."""
        return list(self.action_mapping.keys())
    
    def add_custom_action(self, action_name: str, agent_tensor: torch.Tensor):
        """Add a custom action mapping."""
        try:
            self.action_mapping[action_name] = agent_tensor
            logger.info(f"Added custom action: {action_name}")
        except Exception as e:
            logger.error(f"Error adding custom action: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        import time
        return str(time.time())
